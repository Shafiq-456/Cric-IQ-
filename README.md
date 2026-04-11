
# 🏏 CricIQ — Agentic AI Cricket Analyst

> An AI-powered cricket assistant combining **Semantic Search**, **RAG (Retrieval Augmented Generation)**, and **Agentic AI** — all backed by the **Endee vector database**.

---

## 🎯 Project Overview

CricIQ lets you ask any cricket question in plain English and get intelligent, data-grounded answers. The system uses an AI agent that autonomously decides which tool to use, searches the Endee vector database semantically, and generates accurate answers using Claude AI.

**Example questions:**
- *"Who is the best batsman against fast bowlers?"*
- *"Compare Virat Kohli and Rohit Sharma"*
- *"What is the best death bowling strategy in T20?"*
- *"Which team has won the most IPL titles?"*
- *"Tell me about the 2024 T20 World Cup"*

---

## 🧠 How the 3 Domains Are Used

### 1. 🔍 Semantic Search (via Endee)
Cricket knowledge is embedded as 384-dimensional vectors and stored in Endee. When a user asks a question, the query is also embedded and Endee finds the most semantically similar knowledge chunks — not by keyword matching, but by **meaning**.

### 2. 📚 RAG (Retrieval Augmented Generation)
Retrieved knowledge chunks are passed as context to Claude AI. The AI generates answers **grounded in the retrieved data**, preventing hallucination and ensuring factual accuracy.

### 3. 🤖 Agentic AI
The AI agent has 6 tools it can autonomously choose from:

| Tool | When Used |
|------|-----------|
| `search_player` | Questions about specific players |
| `search_team` | Questions about IPL/national teams |
| `search_worldcup` | World Cup history and results |
| `search_strategy` | Batting/bowling tactics |
| `search_records` | Cricket statistics and records |
| `compare_players` | Comparing two or more players |

The agent picks the right tool on its own — **no user input needed** for tool selection.

---

## 🏗️ System Architecture

```
User Question
     │
     ▼
┌─────────────────────┐
│   FastAPI Backend   │
│     (main.py)       │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   AI Agent          │   ← Picks tool autonomously
│   (agent.py)        │   ← Calls Claude API with tools
└────────┬────────────┘
         │  Tool chosen
         ▼
┌─────────────────────┐
│   RAG Pipeline      │   ← Embeds query (384-dim)
│   (rag.py)          │   ← Searches Endee semantically
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Endee Vector DB    │   ← Returns top-K similar chunks
│  (port 8080)        │
└────────┬────────────┘
         │  Context retrieved
         ▼
┌─────────────────────┐
│   Claude AI         │   ← Generates grounded answer
│   (claude-sonnet)   │
└────────┬────────────┘
         │
         ▼
    Final Answer
  shown to the user
```

---

## 📁 Project Structure

```
Cric-IQ-/
├── backend/
│   ├── main.py          # FastAPI server with /ask and /init endpoints
│   ├── agent.py         # Agentic AI with 6 tools using Claude API
│   ├── rag.py           # RAG pipeline: embedding + Endee search
│   ├── data.py          # Cricket knowledge base (players, teams, records)
│   ├── requirements.txt # Python dependencies
│   └── Dockerfile
├── frontend/
│   └── index.html       # Cricket-themed chat UI
├── docker-compose.yml   # Runs Endee + Backend together
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
- Docker Desktop installed
- An Anthropic API key ([get one here](https://console.anthropic.com))

### Step 1 — Clone the repo
```bash
git clone https://github.com/Shafiq-456/Cric-IQ-.git
cd Cric-IQ-
```

### Step 2 — Set your API key
```bash
export ANTHROPIC_API_KEY=your_key_here
```

### Step 3 — Run everything
```bash
docker-compose up
```

### Step 4 — Open the app
Go to 👉 **http://localhost:8000**

### Step 5 — Load the database
Click **"Load Database"** in the UI to ingest cricket knowledge into Endee.

### Step 6 — Ask anything!
Type your cricket question and let the AI agent work its magic 🏏

---

## 🛠️ Manual Setup (without Docker)

```bash
# Terminal 1 — Run Endee
docker run -p 8080:8080 endeeio/endee-server:latest

# Terminal 2 — Run Backend
cd backend
pip install -r requirements.txt
export ANTHROPIC_API_KEY=your_key_here
uvicorn main:app --reload --port 8000
```

---

## 🧪 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET    | `/health` | Health check |
| POST   | `/init`   | Ingest cricket knowledge into Endee |
| POST   | `/ask`    | Ask the AI agent a cricket question |

### Example Request
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"query": "Who is the best T20 bowler in India?"}'
```

### Example Response
```json
{
  "answer": "Jasprit Bumrah is widely regarded as India's best T20 bowler...",
  "tool_used": "search_player",
  "sources": ["Jasprit Bumrah", "Hardik Pandya"]
}
```

---

## 🔗 Built With

- **[Endee](https://github.com/endee-io/endee)** — High-performance vector database
- **[Claude AI](https://anthropic.com)** — Agentic AI and answer generation
- **[FastAPI](https://fastapi.tiangolo.com)** — Python web framework
- **[Sentence Transformers](https://sbert.net)** — Text embeddings (all-MiniLM-L6-v2)
- **Docker** — Containerization

---

## 👨‍💻 Author

**Shafiq** — Built for Endee Internship Evaluation
- GitHub: [@Shafiq-456](https://github.com/Shafiq-456)
- Forked from: [endee-io/endee](https://github.com/endee-io/endee)

---

## 📄 License

Apache 2.0 — see [LICENSE](LICENSE)
