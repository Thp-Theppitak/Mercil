import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

# ดึงค่าจาก .env หรือใช้ค่า default
DATABASE_URL = os.getenv(
    "DATABASE_URL", 
    "postgresql+asyncpg://postgres:postgres@localhost:5432/mercil_db"
)

# สร้าง Engine
engine = create_async_engine(DATABASE_URL, echo=False)
# สร้าง Session Factory
async_session = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# Dependency Function สำหรับ FastAPI
async def get_db():
    async with async_session() as session:
        yield session