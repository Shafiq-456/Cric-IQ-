# rag.py - RAG Pipeline using Endee Vector Database
# Handles: embedding, ingestion, and semantic search
# FALLBACK: If Endee is unavailable, uses mock search with string matching

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
ENDEE_AVAILABLE = True  # Will be set to False if Endee fails

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
    try:
        await endee_post("/index", {
            "name": INDEX_NAME,
            "dimension": VECTOR_DIM,
            "space_type": "cosine",
            "precision": "int8"
        })
        print(f"[Endee] Created index: {INDEX_NAME}")
    except Exception as e:
        global ENDEE_AVAILABLE
        ENDEE_AVAILABLE = False
        print(f"[Endee] Connection failed - switching to mock search mode: {str(e)}")

# ── Mock Search (Fallback) ────────────────────────────────────────────────────
def mock_search(query: str, category: str = None, top_k: int = 4) -> list:
    """
    Fallback search using simple string matching on CRICKET_KNOWLEDGE.
    Returns top_k matches based on keyword presence and text similarity.
    """
    query_lower = query.lower()
    query_words = set(query_lower.split())
    
    scored_results = []
    
    for doc in CRICKET_KNOWLEDGE:
        # Filter by category if provided
        if category and doc.get("category") != category:
            continue
        
        text_lower = doc.get("text", "").lower()
        title_lower = doc.get("title", "").lower()
        
        # Calculate match score based on word overlaps
        text_words = set(text_lower.split())
        title_words = set(title_lower.split())
        
        # Count matching words (higher score = better match)
        text_matches = len(query_words & text_words)
        title_matches = len(query_words & title_words) * 2  # Title matches weighted higher
        
        total_score = text_matches + title_matches
        
        # Only include if there's at least one word match
        if total_score > 0:
            scored_results.append({
                "id": doc.get("id"),
                "title": doc.get("title", ""),
                "text": doc.get("text", ""),
                "category": doc.get("category", ""),
                "similarity": round(total_score / len(query_words) if query_words else 0, 3),
                "score": total_score
            })
    
    # Sort by score descending and take top_k
    scored_results.sort(key=lambda x: x["score"], reverse=True)
    results = scored_results[:top_k]
    
    # Remove the internal score field
    for r in results:
        del r["score"]
    
    return results

# ── Ingestion ─────────────────────────────────────────────────────────────────
async def ingest_knowledge():
    """Embed all cricket knowledge and upsert into Endee. Falls back to mock mode if Endee unavailable."""
    global ENDEE_AVAILABLE
    
    try:
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
        return {"status": "ok", "ingested": len(vectors), "mode": "endee"}
    
    except Exception as e:
        ENDEE_AVAILABLE = False
        print(f"[Fallback] Endee unavailable - using mock search mode: {str(e)}")
        return {
            "status": "ok", 
            "message": f"Using fallback mock search mode - Endee not available",
            "mode": "mock",
            "ingested": len(CRICKET_KNOWLEDGE)
        }

# ── Semantic Search ───────────────────────────────────────────────────────────
async def semantic_search(query: str, category: str = None, top_k: int = 4) -> list:
    """
    Search Endee for relevant cricket knowledge using semantic similarity.
    Falls back to mock string matching if Endee is unavailable.
    Optionally filter by category: player, team, worldcup, strategy, record
    """
    global ENDEE_AVAILABLE
    
    # Use mock search if Endee is not available
    if not ENDEE_AVAILABLE:
        return mock_search(query, category, top_k)
    
    try:
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
    
    except Exception as e:
        # If Endee call fails, fall back to mock search
        print(f"[Fallback] Endee search failed, using mock search: {str(e)}")
        ENDEE_AVAILABLE = False
        return mock_search(query, category, top_k)
