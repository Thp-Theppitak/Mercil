import os
import json
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
from google import genai
from google.genai import types
from fastapi import FastAPI
from pydantic import BaseModel

# -------------------------------------------------
# 1) ตั้งค่า Gemini API Key 
# -------------------------------------------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "API")  #ใส่ API Gimini ที่บอก
client = genai.Client(api_key=GEMINI_API_KEY)

# -------------------------------------------------
# Global variables
# -------------------------------------------------
documents: List[Dict[str, Any]] = []
doc_embeddings: np.ndarray = np.array([])
asset_type_names: List[str] = []   # เก็บชื่อประเภททรัพย์สำหรับ Auto Filter


# -------------------------------------------------
# 2) ฟังก์ชันโหลดข้อมูลจาก JSON
# -------------------------------------------------
def load_data() -> List[Dict[str, Any]]:
    """
    โหลดข้อมูลทรัพย์ + ประเภท แล้วรวมเป็น text เดียวต่อ 1 ทรัพย์
    เพิ่มฟิลด์ price_value (ตัวเลข) ไว้ใช้ filter ราคา
    """
    global asset_type_names

    try:
        # โหลดประเภททรัพย์
        with open("Mercil/asset_type_rows.json", "r", encoding="utf-8") as f:
            asset_types = json.load(f)

        # เก็บชื่อประเภทไว้ใช้สำหรับ Auto Filter / อ้างอิงให้ LLM ก็ได้
        asset_type_names = [t.get("name_th", "").strip() for t in asset_types]

        # โหลดรายการทรัพย์
        with open("Mercil/assets_rows.json", "r", encoding="utf-8") as f:
            assets = json.load(f)

    except FileNotFoundError:
        print(
            "ไม่พบไฟล์ข้อมูล กรุณาตรวจสอบว่า Mercil/asset_type_rows.json "
            "และ Mercil/assets_rows.json อยู่ในโฟลเดอร์โปรเจกต์"
        )
        return []

    # แปลง asset_type_id -> ชื่อประเภท
    type_map = {int(t["id"]): t.get("name_th", "") for t in asset_types}

    docs: List[Dict[str, Any]] = []

    for row in assets:
        type_id = row.get("asset_type_id")
        asset_type_name = type_map.get(type_id, "ไม่ระบุประเภท")

        name = row.get("name_th", "") or ""
        raw_price = row.get("asset_details_selling_price")

        # แปลงราคาจาก string -> float สำหรับ filter
        price_value: Optional[float]
        if raw_price is None:
            price_value = None
            price_str = "ไม่ระบุ"
        else:
            try:
                price_value = float(raw_price)
                price_str = str(raw_price)
            except (TypeError, ValueError):
                price_value = None
                price_str = str(raw_price)

        road = row.get("location_road_th") or "-"
        village = row.get("location_village_th") or "-"
        desc = row.get("asset_details_description_th") or ""

        # ข้อความที่ใช้ทำ Semantic Search
        text = (
            f"ชื่อทรัพย์: {name} | "
            f"ประเภท: {asset_type_name} | "
            f"ราคา: {price_str} บาท | "
            f"ทำเล/ถนน: {road} | "
            f"โครงการ: {village} | "
            f"รายละเอียด: {desc}"
        )

        docs.append(
            {
                "id": row.get("id"),
                "asset_code": row.get("asset_code"),
                "text": text,
                "price": price_str,
                "price_value": price_value,
                "type_name": asset_type_name,
                "metadata": {
                    "name": name,
                    "road": road,
                    "village": village,
                },
            }
        )

    return docs


# -------------------------------------------------
# 3) ฟังก์ชันสร้าง Embedding ด้วย Gemini
# -------------------------------------------------
def embed_texts(
    texts: List[str],
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> np.ndarray:
    """
    รับ list ของ text แล้วคืนค่าเป็น matrix [n_docs, dim] โดยใช้ Gemini API
    """
    if not texts:
        return np.array([])

    try:
        resp = client.models.embed_content(
            model="text-embedding-004",
            contents=texts,
            config=types.EmbedContentConfig(task_type=task_type),
        )

        if hasattr(resp, "embeddings"):
            vectors = [e.values for e in resp.embeddings]
        else:
            vectors = []

        return np.array(vectors, dtype="float32")
    except Exception as e:
        print(f"เกิดข้อผิดพลาดในการทำ Embedding: {e}")
        return np.array([])


# -------------------------------------------------
# 3.1) ฟังก์ชันให้ LLM ช่วยตีความ Query (LLM-Boosted)
# -------------------------------------------------
def llm_understand_query(raw_query: str) -> Dict[str, Any]:
    """
    ใช้ Gemini 2.5 Flash ช่วยตีความ query ภาษาคน
    ให้กลายเป็น:
      - clean_query: ข้อความสำหรับ semantic search
      - type_name: ประเภททรัพย์ (ถ้าระบุ)
      - min_price, max_price: ช่วงราคา (ถ้ามี)
      - location: ทำเล / เขต / ถนน (ถ้ามี)
    ถ้า parse ไม่ได้ จะคืนค่าที่ปลอดภัยโดยใช้ raw_query เดิม
    """

    # prompt แบบบังคับให้ออก JSON อย่างเดียว
    prompt = f"""
คุณคือระบบช่วยค้นหาอสังหาริมทรัพย์จากฐานข้อมูล NPA

ผู้ใช้พิมพ์คำค้นว่า: "{raw_query}"

ให้คุณสรุป Intent ของผู้ใช้ แล้วตอบเป็น JSON ล้วน ๆ (ห้ามมีคำอธิบายอื่น)
โครงสร้าง JSON:

{{
  "clean_query": "ข้อความที่เหมาะสำหรับ semantic search",
  "type_name": "ชื่อประเภททรัพย์ เช่น บ้านเดี่ยว, ทาวน์เฮ้าส์, ห้องชุดพักอาศัย หรือเว้นเป็นค่าว่างถ้าไม่ชัดเจน",
  "min_price": ตัวเลขราคาต่ำสุด หรือ null,
  "max_price": ตัวเลขราคาสูงสุด หรือ null,
  "location": "ทำเลหรือถนน เช่น ลาดพร้าว, บางแค, บางขุนเทียน หรือค่าว่าง"
}}

- ถ้าผู้ใช้บอกว่า "ราคาไม่เกิน X" ให้ใส่ใน max_price
- ถ้าบอกว่า "อย่างน้อย X" หรือ "มากกว่า X" ให้ใส่ใน min_price
- ถ้าไม่รู้ให้ใส่ null ใน field ราคานั้น ๆ
- ถ้าไม่รู้ประเภททรัพย์ ให้ type_name เป็นค่าว่าง "" 
- อย่าใส่ comment หรือข้อความอื่น นอกจาก JSON
"""

    try:
        resp = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        text = resp.text.strip()

        # ดึงเฉพาะส่วนที่เป็น JSON เผื่อ LLM เผลอใส่อะไรเกินมา
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            json_str = text[start : end + 1]
        else:
            json_str = text

        data = json.loads(json_str)

        # สร้างผลลัพธ์แบบปลอดภัย
        clean_query = data.get("clean_query") or raw_query

        type_name = data.get("type_name") or None
        if isinstance(type_name, str):
            type_name = type_name.strip() or None

        def _to_float(v):
            if v is None:
                return None
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        min_price = _to_float(data.get("min_price"))
        max_price = _to_float(data.get("max_price"))

        location = data.get("location") or None
        if isinstance(location, str):
            location = location.strip() or None

        return {
            "clean_query": clean_query,
            "type_name": type_name,
            "min_price": min_price,
            "max_price": max_price,
            "location": location,
        }

    except Exception as e:
        print(f"LLM understand error: {e}")
        # ถ้า LLM พัง ให้ใช้ query เดิม ป้องกันระบบล่ม
        return {
            "clean_query": raw_query,
            "type_name": None,
            "min_price": None,
            "max_price": None,
            "location": None,
        }


# -------------------------------------------------
# 4) เตรียมข้อมูล (โหลด docs + สร้าง embeddings ตอนเริ่มรัน)
# -------------------------------------------------
print("กำลังโหลดข้อมูลจาก JSON...")
documents = load_data()
doc_texts = [d["text"] for d in documents]

if documents:
    print(f"โหลดข้อมูลสำเร็จ {len(documents)} รายการ")
    print("กำลังสร้าง Embeddings (ขั้นตอนนี้อาจใช้เวลาสักครู่)...")
    doc_embeddings = embed_texts(doc_texts, task_type="RETRIEVAL_DOCUMENT")
    print(f"สร้าง Embeddings เสร็จสิ้น (Shape: {doc_embeddings.shape})")
else:
    print("ไม่พบข้อมูลเอกสาร")


# -------------------------------------------------
# 5) ฟังก์ชัน Hybrid Semantic Search (+ Filters & Boosting)
# -------------------------------------------------
def semantic_search(
    query: str,
    top_k: int = 10,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    type_name: Optional[str] = None,
    location: Optional[str] = None,
) -> List[Tuple[Dict[str, Any], float]]:
    """
    ค้นหาเอกสารแบบ Hybrid:
    - ใช้ semantic similarity จาก embedding (Base Score)
    - เพิ่มคะแนน (Boosting) ถ้าทำเลตรงกัน
    - filter (ตัดทิ้ง) ด้วยราคาและประเภททรัพย์
    """

    if len(doc_embeddings) == 0:
        return []

    # 1) แปลง query เป็น vector และคำนวณ similarity (dot product)
    query_vec = embed_texts([query], task_type="RETRIEVAL_QUERY")
    if len(query_vec) == 0:
        return []

    scores = np.dot(doc_embeddings, query_vec[0])

    # -------------------------------------------------------
    # 2) Location Boosting (ดันคะแนนทำเลที่ตรงกัน)
    # -------------------------------------------------------
    if location:
        # ตรวจสอบว่าทรัพย์ไหนมีคำว่า location (เช่น "ลาดพร้าว") อยู่ในข้อความบ้าง
        is_match_location = np.array([
            (location in doc["text"]) or 
            (location in doc.get("metadata", {}).get("road", "")) or
            (location in doc.get("metadata", {}).get("village", ""))
            for doc in documents
        ], dtype=float)
        
        # บวกคะแนนเพิ่ม 0.5 คะแนน ให้กับตัวที่เจอทำเลที่ระบุ
        scores += (is_match_location * 0.5) 
        
    # 3) สร้าง mask สำหรับ Hard Filter (ราคาและประเภทเท่านั้น)
    mask = np.ones(len(documents), dtype=bool)

    # Type Filter
    if type_name:
        mask &= np.array(
            [(doc.get("type_name") == type_name) for doc in documents]
        )

    # Price Filters
    if min_price is not None:
        mask &= np.array(
            [
                (doc.get("price_value") is not None and doc["price_value"] >= min_price)
                for doc in documents
            ]
        )

    if max_price is not None:
        mask &= np.array(
            [
                (doc.get("price_value") is not None and doc["price_value"] <= max_price)
                for doc in documents
            ]
        )
    
    if not mask.any():
        # ถ้าไม่มีทรัพย์สินที่ผ่าน Hard Filter เลย
        return []

    # 4) ตัดตัวที่ไม่ผ่าน Hard Filter ออก โดยให้คะแนนเป็นค่าต่ำมาก
    filtered_scores = scores.copy()
    filtered_scores[~mask] = -1e9

    # 5) เลือก top_k เรียงจากคะแนนมาก -> น้อย
    top_indices = np.argsort(filtered_scores)[::-1][:top_k]

    results: List[Tuple[Dict[str, Any], float]] = []
    for idx in top_indices:
        # ไม่ต้องแสดงผลที่ถูกตัดทิ้งด้วย Hard Filter
        if filtered_scores[idx] <= -1e8:
            continue
        # นำคะแนนที่รวม Boosting แล้วมาใช้
        results.append((documents[idx], float(filtered_scores[idx])))

    return results


# -------------------------------------------------
# 6) FastAPI Models & Endpoints
# -------------------------------------------------
app = FastAPI(title="NPA Hybrid Search API")


class SearchRequest(BaseModel):
    query: str
    top_k: int = 10
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    type_name: Optional[str] = None


class SearchItem(BaseModel):
    id: Optional[int] = None
    asset_code: Optional[str] = None
    text: str
    score: float
    price: Optional[str] = None
    type_name: Optional[str] = None


class SearchResponse(BaseModel):
    results: List[SearchItem]


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/")
def root():
    return {"message": "NPA Hybrid Search API is running"}


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    # 1) LLM Extract Intent
    llm_info = llm_understand_query(req.query)
    """
    Hybrid Search API:
    - รับ query + filters
    - ใช้ LLM ช่วยตีความ query ภาษาคน
    - คืนผลลัพธ์ดิบ (รายการทรัพย์ + คะแนนความเกี่ยวข้อง)
    """

    # ---------------------------------------------------
    # 1) ใช้ LLM ช่วยตีความ Query (LLM-Boosted)
    # ---------------------------------------------------
    llm_info = llm_understand_query(req.query)

    # ปรับ query ให้เหมาะกับ semantic search มากขึ้น
    if llm_info.get("clean_query"):
        req.query = llm_info["clean_query"]

    # ถ้า LLM เจอประเภท และ user ไม่ได้กำหนดเอง → ใช้เป็น type_name
    if llm_info.get("type_name") and not req.type_name:
        req.type_name = llm_info["type_name"]

    # ถ้า LLM ดึงช่วงราคาได้ และ user ไม่ได้ส่ง filter เอง
    if llm_info.get("min_price") is not None and req.min_price is None:
        req.min_price = llm_info["min_price"]

    if llm_info.get("max_price") is not None and req.max_price is None:
        req.max_price = llm_info["max_price"]

    # ถ้าเจอทำเล ให้ใส่เพิ่มใน query เพื่อเพิ่ม context
    if llm_info.get("location"):
        loc = llm_info["location"]
        if loc not in req.query:
            req.query = f"{req.query} ทำเล {loc}"

    detected_location = llm_info.get("location")
    # ---------------------------------------------------
    # 2) Auto Filter ประเภทจากชื่อประเภทที่มีในระบบ (backup อีกชั้น)
    # ---------------------------------------------------
    q = req.query.strip()
    if q in asset_type_names and not req.type_name:
        req.type_name = q

    # ---------------------------------------------------
    # 3) เรียก Hybrid Semantic Search
    # ---------------------------------------------------
    docs = semantic_search(
        query=req.query,
        top_k=req.top_k,
        min_price=req.min_price,
        max_price=req.max_price,
        type_name=req.type_name,
        location=detected_location, 
    )

    results: List[SearchItem] = []
    for doc, score in docs:
        results.append(
            SearchItem(
                id=doc.get("id"),
                asset_code=doc.get("asset_code"),
                text=doc["text"],
                score=score,
                price=str(doc.get("price")),
                type_name=doc.get("type_name"),
            )
        )

    return SearchResponse(results=results)
