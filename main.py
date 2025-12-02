from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import uvicorn
import os
from dotenv import load_dotenv

load_dotenv()

# Import Router ที่เราเพิ่งสร้าง
from api.search import router as search_router


app = FastAPI(
    title="Mercil Backend API",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# --- Register Router ---
# บรรทัดนี้จะดึง Logic การค้นหาทั้งหมดเข้ามาทำงาน
app.include_router(search_router)

@app.get("/")
async def root():
    return {"message": "Mercil API is running", "docs": "/docs"}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)