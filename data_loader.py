import json
import asyncio
from sentence_transformers import SentenceTransformer
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text
import os
from typing import List, Dict
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError
import time
from dotenv import load_dotenv

load_dotenv()

# -----------------------------------------------------
# Database Setup
# -----------------------------------------------------

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql+asyncpg://postgres:postgres@localhost:5432/mercil_db"
)

engine = create_async_engine(DATABASE_URL, echo=True)
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# -----------------------------------------------------
# Models
# -----------------------------------------------------

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
geolocator = Nominatim(user_agent="mercil_geocoder")

# Convert vector list → pgvector format
def embedding_to_pgvector(vec: List[float]) -> str:
    return "[" + ",".join(str(x) for x in vec) + "]"

# -----------------------------------------------------
# Initialize DB
# -----------------------------------------------------

async def init_db():
    async with engine.begin() as conn:
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))
        await conn.execute(text("CREATE EXTENSION IF NOT EXISTS postgis;"))

        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS assets (
                id SERIAL PRIMARY KEY,
                name TEXT NOT NULL,
                description TEXT,
                address TEXT,
                category TEXT,
                location GEOGRAPHY(POINT, 4326),
                embedding vector(384),
                metadata JSONB,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        """))

        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS assets_embedding_idx
            ON assets USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        """))

        await conn.execute(text("""
            CREATE INDEX IF NOT EXISTS assets_location_idx
            ON assets USING GIST (location);
        """))

    print("✅ Database initialized successfully!")

# -----------------------------------------------------
# Geocode
# -----------------------------------------------------

def geocode_address(address: str, max_retries: int = 3) -> tuple:
    for attempt in range(max_retries):
        try:
            location = geolocator.geocode(address, timeout=10)
            if location:
                return (location.latitude, location.longitude)
            print(f"⚠️ Could not geocode: {address}")
            return None
        except (GeocoderTimedOut, GeocoderServiceError):
            if attempt < max_retries - 1:
                time.sleep(2)
                continue
            return None

# -----------------------------------------------------
# Load + Insert
# -----------------------------------------------------

async def load_assets_from_json(json_path: str):

    with open(json_path, "r", encoding="utf-8") as f:
        assets = json.load(f)

    print(f"📂 Loaded {len(assets)} assets")

    async with async_session() as session:

        for idx, asset in enumerate(assets, 1):
            try:
                name = asset.get("name", "")
                description = asset.get("description", "")
                address = asset.get("address", "")
                category = asset.get("category", "general")

                # -------------------------------------------------
                # Embed
                # -------------------------------------------------
                embed_list = model.encode(f"{name}. {description}").tolist()
                embed_str = embedding_to_pgvector(embed_list)

                # -------------------------------------------------
                # Geocode
                # -------------------------------------------------
                coords = geocode_address(address)
                location_wkt = f"POINT({coords[1]} {coords[0]})" if coords else None

                # -------------------------------------------------
                # INSERT (FIXED VERSION)
                # -------------------------------------------------
                if location_wkt:
                    q = text("""
                        INSERT INTO assets
                        (name, description, address, category, location, embedding, metadata)
                        VALUES (
                            :name,
                            :description,
                            :address,
                            :category,
                            ST_GeogFromText(:loc),
                            cast(:embed AS vector),
                            :meta
                        )
                    """)
                    params = {
                        "name": name,
                        "description": description,
                        "address": address,
                        "category": category,
                        "loc": location_wkt,
                        "embed": embed_str,
                        "meta": json.dumps(asset)
                    }
                else:
                    q = text("""
                        INSERT INTO assets
                        (name, description, address, category, embedding, metadata)
                        VALUES (
                            :name,
                            :description,
                            :address,
                            :category,
                            cast(:embed AS vector),
                            :meta
                        )
                    """)
                    params = {
                        "name": name,
                        "description": description,
                        "address": address,
                        "category": category,
                        "embed": embed_str,
                        "meta": json.dumps(asset)
                    }

                await session.execute(q, params)
                print(f"✅ [{idx}] Inserted {name}")

                time.sleep(1)

            except Exception as e:
                print(f"❌ Error processing {name}: {e}")

        await session.commit()
        print("🎉 Done inserting all assets!")

# -----------------------------------------------------
# Search Function (FIXED)
# -----------------------------------------------------

async def search_similar_assets(query: str, limit: int = 5):

    embed_list = model.encode(query).tolist()
    embed_str = embedding_to_pgvector(embed_list)

    async with async_session() as session:
        q = text("""
            SELECT
                id, name, description, address, category,
                ST_Y(location::geometry) AS latitude,
                ST_X(location::geometry) AS longitude,
                1 - (embedding <=> cast(:embed AS vector)) AS similarity
            FROM assets
            ORDER BY embedding <=> cast(:embed AS vector)
            LIMIT :limit;
        """)

        rows = await session.execute(q, {"embed": embed_str, "limit": limit})
        return [dict(r._mapping) for r in rows]

# -----------------------------------------------------
# Main
# -----------------------------------------------------

async def main():
    print("🚀 Starting loader…")

    await init_db()

    if os.path.exists("assets_rows.json"):
        await load_assets_from_json("assets_rows.json")
    else:
        print("❌ Missing assets_rows.json")

    print("\n🔍 Test search = 'โรงพยาบาล'")
    results = await search_similar_assets("โรงพยาบาล", limit=3)

    for r in results:
        print(f"- {r['name']}  (score={r['similarity']:.3f})")

if __name__ == "__main__":
    asyncio.run(main())
