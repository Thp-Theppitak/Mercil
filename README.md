# Mercil
MERCIL Hybrid Search API  
Search Engine สำหรับค้นหาทรัพย์ NPA ด้วย Hybrid Semantic Search

โปรเจกต์นี้เป็น API ที่ใช้สำหรับค้นหาทรัพย์โดยผสาน  
**Semantic Vector Search (Gemini Embedding)** + **Filter (ราคา/ประเภท)*

Features

- Hybrid Search (Semantic Similarity + Filters)
- รองรับคำค้นแบบธรรมชาติ เช่น  
  - "คอนโด ลาดพร้าว"  
  - "บ้านเดี่ยว ราคาต่ำกว่า 3 ล้าน"
- กรองผลลัพธ์ด้วย  
  - `type_name` (ประเภททรัพย์)  
  - `min_price` / `max_price`
- คืนผลลัพธ์แบบ JSON พร้อม score สำหรับจัด ranking
- พร้อมใช้งานกับหน้า Search ของ MERCIL

ก่อนใช้งานต้องใส่ API Key Gimini ก่อน

Run 
python -m uvicorn Mercil.HybridSearch:app --reload
เปิด Swagger UI ได้ที่:
http://127.0.0.1:8000/docs
