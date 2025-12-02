# 🏥 Mercil Backend - AI Hybrid Search API

Backend Service สำหรับระบบค้นหาสถานที่ชุมชน (Mercil) ด้วยเทคโนโลยี **Hybrid Search** ที่ผสานการทำงานระหว่าง:
1. **Semantic Search** (Vector Embedding) - ค้นหาด้วยความหมายของประโยค
2. **SQL Filtering** (PostgreSQL) - กรองหมวดหมู่และราคา
3. **AI Intent Parsing** (Google Gemini) - ใช้ AI แกะความต้องการของผู้ใช้ (เช่น แยกสถานที่ vs ทำเล)

---

## 🛠️ Tech Stack
- **Language:** Python 3.10+
- **Framework:** FastAPI
- **Database:** PostgreSQL 15 (with `pgvector` & `PostGIS`)
- **AI/LLM:** - `sentence-transformers` (Embedding)
  - `Google Gemini Flash 2.5` (Intent Understanding)
- **ORM:** SQLAlchemy (Async)

---

## ⚙️ Installation & Setup

### 1. Prerequisites
ต้องติดตั้งโปรแกรมเหล่านี้ก่อน:
- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (สำหรับจำลอง Database)
- [Python 3.10+](https://www.python.org/downloads/)

### 2. Setup Environment
สร้างไฟล์ `.env` ที่ root folder และใส่ค่า config ดังนี้:

3. Install Dependencies
Bash

### สร้างและเข้า Virtual Environment (ถ้ายังไม่มี) python -m venv .venv
### Windows: .venv\Scripts\activate
### Mac/Linux: source .venv/bin/activate

# ลง Library
pip install -r requirements.txt

🚀 How to Run
Step 1: Start Database
รัน Docker Compose เพื่อเปิดใช้งาน PostgreSQL (พร้อม pgvector + postgis)

Bash

docker-compose up -d --build
Step 2: Load Initial Data
รันสคริปต์เพื่อสร้างตารางและโหลดข้อมูลตัวอย่างจาก assets_rows.json เข้า Database (ทำแค่ครั้งแรก หรือเมื่อต้องการ Reset ข้อมูล)

Bash

python data_loader.py
Step 3: Start API Server
Bash

python main.py
เมื่อรันสำเร็จ Server จะทำงานที่: http://localhost:8000

API Documentation
สามารถดูเอกสาร API และทดลองยิง Request ได้ที่ Swagger UI: 👉 http://localhost:8000/docs

📂 Project Structure
Mercil/
├── api/                # Logic ของ API ทั้งหมด
│   └── search.py       # ระบบ Search (Hybrid Logic อยู่ที่นี่)
├── uploads/            # โฟลเดอร์เก็บไฟล์อัปโหลด (ต้องสร้างไว้)
├── database.py         # Config การเชื่อมต่อ Database
├── data_loader.py      # Script สำหรับโหลดข้อมูลเข้า DB
├── main.py             # จุดเริ่มต้นของ Server (Entry point)
├── docker-compose.yaml # Config สำหรับ Docker
├── Dockerfile          # Config Image ของ Database (PostGIS+pgvector)
└── requirements.txt    # รายชื่อ Library ที่ต้องใช้

```env
# Database Config
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5432/mercil_db

# AI Config (ใส่ Key ของ Gemini)
GEMINI_API_KEY=AIzaSyxxxxxxxxxxxxxxxxxxxxxxxxxxxx

