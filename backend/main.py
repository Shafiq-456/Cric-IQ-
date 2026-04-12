# main.py - CricketIQ FastAPI Server
# Endpoints: /init (ingest data), /ask (agentic query), /health

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from rag import ingest_knowledge
from agent import run_agent

app = FastAPI(title="CricketIQ API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Request Models ────────────────────────────────────────────────────────────
class AskRequest(BaseModel):
    query: str

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok", "service": "CricketIQ"}

@app.post("/init")
async def init_database():
    """Initialize and ingest cricket knowledge into Endee vector database."""
    try:
        result = await ingest_knowledge()
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask")
async def ask(request: AskRequest):
    """
    Main agentic endpoint.
    Agent searches cricket knowledge base and returns an answer.
    Falls back to direct search if Gemini API fails.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")
    try:
        result = run_agent(request.query)
        return result
    except Exception as e:
        print(f"[Main] Error in /ask endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# ── Serve Frontend ────────────────────────────────────────────────────────────
frontend_path = os.path.join(os.path.dirname(__file__), "..", "frontend")
if os.path.exists(frontend_path):
    app.mount("/static", StaticFiles(directory=frontend_path), name="static")

    @app.get("/")
    async def serve_frontend():
        return FileResponse(os.path.join(frontend_path, "index.html"))
