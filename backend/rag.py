# rag.py - RAG Pipeline using Endee Vector Database
# Handles: embedding, ingestion, and semantic search

import os
import math
import hashlib
import httpx
from data import CRICKET_KNOWLEDGE

# ── Config ────────────────────────────────────────────────────────────────────
ENDEE_BASE_URL  = os.getenv("ENDEE_BASE_URL", "http://localhost:8080/api/v1")
ENDEE_API_TOKEN = os.getenv("ENDEE_API_TOKEN", "")
INDEX_NAME      = "cricketiq_index"
VECTOR_DIM      = 384

# ── Endee HTTP Helpers ────────────────────────────────────────────────────────
def endee_headers():
    h = {"Content-Type": "application/json"}
    if ENDEE_API_TOKEN:
        h["Authorization"] = ENDEE_API_TOKEN
    return h

async def endee_get(path: str):
    async with httpx.AsyncClient() as c:
        r = await c.get(f"{ENDEE_BASE_URL}{path}", headers=endee_headers(), timeout=30)
        r.raise_for_status()
        return r.json()

async def endee_post(path: str, body: dict):
    async with httpx.AsyncClient() as c:
        r = await c.post(f"{ENDEE_BASE_URL}{path}", headers=endee_headers(), json=body, timeout=60)
        r.raise_for_status()
        return r.json()

# ── Embedding ─────────────────────────────────────────────────────────────────
def get_embedding(text: str) -> list:
    """
    Generate a 384-dim embedding.
    Uses sentence-transformers if installed, else falls back to deterministic mock.
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("all-MiniLM-L6-v2")
        return model.encode(text).tolist()
    except ImportError:
        # Deterministic mock embedding based on text content
        h = hashlib.sha256(text.lower().encode()).digest()
        vec = []
        for i in range(VECTOR_DIM):
            b1 = h[i % 32]
            b2 = h[(i + 7) % 32]
            vec.append(math.sin(b1 + i * 0.1) * math.cos(b2 * 0.05))
        norm = math.sqrt(sum(x * x for x in vec)) or 1.0
        return [x / norm for x in vec]

# ── Index Management ──────────────────────────────────────────────────────────
async def index_exists() -> bool:
    try:
        await endee_get(f"/index/{INDEX_NAME}")
        return True
    except Exception:
        return False

async def create_index():
    await endee_post("/index", {
        "name": INDEX_NAME,
        "dimension": VECTOR_DIM,
        "space_type": "cosine",
        "precision": "int8"
    })
    print(f"[Endee] Created index: {INDEX_NAME}")

# ── Ingestion ─────────────────────────────────────────────────────────────────
async def ingest_knowledge():
    """Embed all cricket knowledge and upsert into Endee."""
    if await index_exists():
        return {"status": "skipped", "message": "Index already exists"}

    await create_index()

    vectors = []
    for doc in CRICKET_KNOWLEDGE:
        vec = get_embedding(doc["text"])
        vectors.append({
            "id": doc["id"],
            "vector": vec,
            "meta": {
                "title": doc["title"],
                "text": doc["text"],
                "category": doc["category"]
            },
            "filter": {"category": doc["category"]}
        })

    # Upsert in batches of 10
    batch_size = 10
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        await endee_post(f"/index/{INDEX_NAME}/upsert", {"vectors": batch})
        print(f"[Endee] Upserted batch {i // batch_size + 1}")

    print(f"[Endee] Ingested {len(vectors)} cricket knowledge chunks")
    return {"status": "ok", "ingested": len(vectors)}

# ── Semantic Search ───────────────────────────────────────────────────────────
async def semantic_search(query: str, category: str = None, top_k: int = 4) -> list:
    """
    Search Endee for relevant cricket knowledge using semantic similarity.
    Optionally filter by category: player, team, worldcup, strategy, record
    """
    query_vec = get_embedding(query)

    body = {
        "vector": query_vec,
        "top_k": top_k,
        "ef": 128,
        "include_vectors": False
    }

    # Apply category filter if provided
    if category:
        body["filter"] = [{"category": {"$eq": category}}]

    results = await endee_post(f"/index/{INDEX_NAME}/query", body)

    # Extract and return relevant chunks
    chunks = []
    for item in results.get("results", []):
        meta = item.get("meta", {})
        chunks.append({
            "id": item.get("id"),
            "title": meta.get("title", ""),
            "text": meta.get("text", ""),
            "category": meta.get("category", ""),
            "similarity": round(item.get("similarity", 0), 3)
        })

    return chunks
