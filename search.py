import os
import json
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession
from sentence_transformers import SentenceTransformer
from google import genai

# Import DB dependency
from database import get_db

# --- ประกาศตัวแปร router (ตัวที่ Error หาไม่เจอ) ---
router = APIRouter(prefix="/api/search", tags=["Search"])

# ---------------------------------------------------------
# 1. Load Models
# ---------------------------------------------------------
print("⏳ Loading Embedding Model...")
embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print("✅ Embedding Model Loaded")

# ดึง Key จาก Environment (ต้องมีใน .env)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") 
client = genai.Client(api_key=GEMINI_API_KEY)

# ---------------------------------------------------------
# 2. Helper Functions
# ---------------------------------------------------------
def embedding_to_pgvector(vec: List[float]) -> str:
    return "[" + ",".join(str(x) for x in vec) + "]"

def llm_understand_query(raw_query: str) -> Dict[str, Any]:
    # ✅ แก้ Prompt ให้ตรงกับข้อมูลจริง (Hospital, School, Temple...)
    prompt = f"""
    คุณคือระบบ AI Search Engine สำหรับหาสถานที่ชุมชน
    ผู้ใช้พิมพ์คำค้นว่า: "{raw_query}"
    
    หน้าที่ของคุณ:
    1. วิเคราะห์ว่าผู้ใช้ต้องการหาสถานที่หมวดหมู่ไหน จาก list นี้เท่านั้น: 
       [hospital, school, temple, shopping_mall, park, transit, market]
    2. ถ้าผู้ใช้ไม่ได้ระบุชัดเจน ให้ type_name เป็น null
    
    ตอบเป็น JSON เท่านั้น (ห้ามมีคำอธิบายอื่น):
    {{
      "clean_query": "keyword สำหรับค้นหา (ตัดคำขยายทิ้ง)",
      "type_name": "หมวดหมู่จาก list ข้างบน หรือ null",
      "min_price": null, 
      "max_price": null,
      "location": "ชื่อทำเล/ถนน/เขต หรือ null"
    }}
    """
    
    try:
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )
        text_resp = resp.text.strip()
        if text_resp.startswith("```"):
            text_resp = text_resp.split("```")[1].replace("json", "").strip()
        return json.loads(text_resp)
    except Exception as e:
        print(f"LLM Error: {e}")
        return {}

# ---------------------------------------------------------
# 3. Schemas
# ---------------------------------------------------------
class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    type_name: Optional[str] = None

# ---------------------------------------------------------
# 4. Search Endpoint
# ---------------------------------------------------------
@router.post("/")
async def search_assets(req: SearchRequest, db: AsyncSession = Depends(get_db)):
    # 1. LLM Process
    llm_info = llm_understand_query(req.query)
    
    # 2. Merge Filters
    final_query = llm_info.get("clean_query") or req.query
    final_type = req.type_name or llm_info.get("type_name")
    
    def _pick(val1, val2): return val1 if val1 is not None else val2
    final_min = _pick(req.min_price, llm_info.get("min_price"))
    final_max = _pick(req.max_price, llm_info.get("max_price"))
    location_boost = llm_info.get("location")

    # 3. Embed
    query_vec = embed_model.encode(final_query).tolist()
    query_vec_str = embedding_to_pgvector(query_vec)

    # 4. SQL Logic
    boost_sql = "0"
    if location_boost:
        boost_sql = """
            CASE 
                WHEN (address ILIKE :loc_boost OR name ILIKE :loc_boost) THEN 0.5 
                ELSE 0 
            END
        """

    sql = f"""
        SELECT 
            id, name, description, address, category, metadata,
            (1 - (embedding <=> cast(:embed AS vector))) + {boost_sql} as final_score
        FROM assets
        WHERE 1=1
    """
    
    params = {"embed": query_vec_str, "limit": req.top_k}

    if final_type:
        sql += " AND category = :category"
        params["category"] = final_type
        
    if final_min is not None:
        sql += " AND COALESCE((metadata->>'asset_details_selling_price')::numeric, 0) >= :min_price"
        params["min_price"] = final_min

    if final_max is not None:
        sql += " AND COALESCE((metadata->>'asset_details_selling_price')::numeric, 999999999999) <= :max_price"
        params["max_price"] = final_max
        
    if location_boost:
        params["loc_boost"] = f"%{location_boost}%"

    sql += " ORDER BY final_score DESC LIMIT :limit"

    try:
        result = await db.execute(text(sql), params)
        rows = result.fetchall()
        
        results = []
        for row in rows:
            results.append({
                "id": row.id,
                "name": row.name,
                "address": row.address,
                "category": row.category,
                "score": float(row.final_score),
                "price": row.metadata.get("asset_details_selling_price") if row.metadata else None
            })

        return {
            "intent": llm_info,
            "results": results
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))