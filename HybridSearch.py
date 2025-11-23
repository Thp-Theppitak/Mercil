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
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "ใส่ API Key") #"API KEY" ให้ใส่ API KEY ของ Gimini 
client = genai.Client(api_key=GEMINI_API_KEY)

# -------------------------------------------------
# 2) ฟังก์ชันโหลดข้อมูลจาก JSON
# -------------------------------------------------
def load_data() -> List[Dict[str, Any]]:
    """
    โหลดข้อมูลทรัพย์ + ประเภท แล้วรวมเป็น text เดียวต่อ 1 ทรัพย์
    เพิ่มฟิลด์ price_value (ตัวเลข) ไว้ใช้ filter ราคา
    """
    try:
        with open("Mercil/asset_type_rows.json", "r", encoding="utf-8") as f:
            asset_types = json.load(f)
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
# 4) เตรียมข้อมูล (โหลด docs + สร้าง embeddings ตอนเริ่มรัน)
# -------------------------------------------------
print("กำลังโหลดข้อมูลจาก JSON...")
documents: List[Dict[str, Any]] = load_data()
doc_texts = [d["text"] for d in documents]

doc_embeddings = np.array([])

if documents:
    print(f"โหลดข้อมูลสำเร็จ {len(documents)} รายการ")
    print("กำลังสร้าง Embeddings (ขั้นตอนนี้อาจใช้เวลาสักครู่)...")
    doc_embeddings = embed_texts(doc_texts, task_type="RETRIEVAL_DOCUMENT")
    print(f"สร้าง Embeddings เสร็จสิ้น (Shape: {doc_embeddings.shape})")
else:
    print("ไม่พบข้อมูลเอกสาร")


# -------------------------------------------------
# 5) ฟังก์ชัน Hybrid Semantic Search (+ Filters)
# -------------------------------------------------
def semantic_search(
    query: str,
    top_k: int = 10,
    min_price: Optional[float] = None,
    max_price: Optional[float] = None,
    type_name: Optional[str] = None,
) -> List[Tuple[Dict[str, Any], float]]:
    """
    ค้นหาเอกสารแบบ Hybrid:
    - ใช้ semantic similarity จาก embedding
    - filter ด้วยราคาและประเภททรัพย์
    """

    if len(doc_embeddings) == 0:
        return []

    # 1) แปลง query เป็น vector
    query_vec = embed_texts([query], task_type="RETRIEVAL_QUERY")
    if len(query_vec) == 0:
        return []

    # 2) คำนวณ similarity (dot product)
    scores = np.dot(doc_embeddings, query_vec[0])

    # 3) สร้าง mask สำหรับ filter
    mask = np.ones(len(documents), dtype=bool)

    if type_name:
        mask &= np.array(
            [(doc.get("type_name") == type_name) for doc in documents]
        )

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
        return []

    # 4) ตัดตัวที่ไม่ผ่าน filter ออก โดยให้คะแนนเป็นค่าต่ำมาก
    filtered_scores = scores.copy()
    filtered_scores[~mask] = -1e9

    # 5) เลือก top_k เรียงจากคะแนนมาก -> น้อย
    top_indices = np.argsort(filtered_scores)[::-1][:top_k]

    results: List[Tuple[Dict[str, Any], float]] = []
    for idx in top_indices:
        if filtered_scores[idx] <= -1e8:
            continue
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
    """
    Hybrid Search API:
    - รับ query + filters
    - คืนผลลัพธ์ดิบ (รายการทรัพย์ + คะแนนความเกี่ยวข้อง)
    """

    docs = semantic_search(
        query=req.query,
        top_k=req.top_k,
        min_price=req.min_price,
        max_price=req.max_price,
        type_name=req.type_name,
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
