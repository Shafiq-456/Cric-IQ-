# 🏏 CricIQ — Agentic AI Cricket Analyst

> An AI-powered cricket assistant combining **Semantic Search**, **RAG (Retrieval Augmented Generation)**, and **Agentic AI** — all backed by the **Endee.io vector database**.

![Python](https://img.shields.io/badge/Python-3.11-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![Endee](https://img.shields.io/badge/Vector_DB-Endee.io-orange)
![Gemini](https://img.shields.io/badge/AI-Google_Gemini-red)
![License](https://img.shields.io/badge/License-Apache_2.0-yellow)

---

## 🎯 What is CricIQ?

CricketIQ is an intelligent cricket assistant that lets you ask any cricket question in plain English and get smart, data-grounded answers instantly.

**Ask anything like:**
- *"Who is the best batsman against fast bowlers?"*
- *"Compare Virat Kohli and Rohit Sharma"*
- *"Which team has won the most IPL titles?"*
- *"What is the best death bowling strategy in T20?"*
- *"Tell me about the 2024 T20 World Cup"*
- *"What are the highest individual scores in cricket history?"*

---

## 🧠 How the 3 AI Domains Work Together

### 1. 🔍 Semantic Search (via Endee.io)
Cricket knowledge is embedded as **384-dimensional vectors** and stored in Endee. When a user asks a question, the query is also embedded and Endee finds the most **semantically similar** knowledge chunks — not by keyword matching, but by **meaning**.

### 2. 📚 RAG — Retrieval Augmented Generation
Retrieved knowledge chunks from Endee are passed as **context to Google Gemini AI**. The AI generates answers **grounded in the retrieved data**, preventing hallucination and ensuring factual accuracy.

### 3. 🤖 Agentic AI — The Smart Part
The AI agent **autonomously decides** which tool to use based on the user's question. No manual selection needed — the agent thinks and acts on its own.

| Tool | Triggered When User Asks About |
|------|-------------------------------|
| `search_player` | A specific cricket player |
| `search_team` | IPL or national teams |
| `search_worldcup` | World Cup history and results |
| `search_strategy` | Batting or bowling tactics |
| `search_records` | Cricket statistics and records |
| `compare_players` | Comparing two or more players |

---

## 🏗️ System Architecture

```
User Types Question
        │
        ▼
┌──────────────────────┐
│   FastAPI Backend    │  ← Receives the question
│      (main.py)       │
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│     AI Agent         │  ← Autonomously picks the right tool
│     (agent.py)       │  ← Based on keywords in the question
└────────┬─────────────┘
         │ Tool selected
         ▼
┌──────────────────────┐
│    RAG Pipeline      │  ← Converts question to 384-dim vector
│      (rag.py)        │  ← Searches Endee semantically
└────────┬─────────────┘
         │
         ▼
┌──────────────────────┐
│  Endee Vector DB     │  ← Returns Top-K most relevant chunks
│   (port 8080)        │  ← Based on cosine similarity
└────────┬─────────────┘
         │ Context retrieved
         ▼
┌──────────────────────┐
│  Google Gemini AI    │  ← Generates grounded answer
│  (gemini-1.5-flash)  │  ← Using retrieved cricket facts
└────────┬─────────────┘
         │
         ▼
   Answer shown to User 🏏
```

---

## 📁 Project Structure

```
Cric-IQ-/
├── backend/
│   ├── main.py            # FastAPI server — /health, /init, /ask
│   ├── agent.py           # Agentic AI brain — picks tools, calls Gemini
│   ├── rag.py             # RAG pipeline — embeds query, searches Endee
│   ├── data.py            # Cricket knowledge base (25+ entries)
│   ├── requirements.txt   # Python dependencies
│   └── Dockerfile         # Backend container
├── frontend/
│   └── index.html         # Cricket-themed chat UI
├── docker-compose.yml     # Runs Endee + Backend together
└── README.md
```

---

## 🚀 Quick Start

### Prerequisites
- **Docker Desktop** installed → [Download here](https://www.docker.com/products/docker-desktop)
- **Google Gemini API Key** (free) → [Get here](https://aistudio.google.com)
- **Python 3.9+** → [Download here](https://python.org/downloads)

---

### Option 1 — Run with Docker (Recommended)

**Step 1 — Clone the repo**
```bash
git clone https://github.com/Shafiq-456/Cric-IQ-.git
cd Cric-IQ-
```

**Step 2 — Set your Gemini API key**
```bash
export GEMINI_API_KEY=your_key_here
```

**Step 3 — Start everything**
```bash
docker-compose up
```

**Step 4 — Open the app**
Go to 👉 **http://localhost:8000**

**Step 5 — Load the database**
Click **"Load Database"** button to ingest cricket knowledge into Endee

**Step 6 — Ask anything!** 🏏

---

### Option 2 — Manual Setup

```bash
# Terminal 1 — Start Endee vector database
docker run -p 8080:8080 endeeio/endee-server:latest

# Terminal 2 — Start the backend
cd backend
pip install -r requirements.txt
export GEMINI_API_KEY=your_key_here
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Then open **http://localhost:8000** in your browser.

---

## 🧪 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Check if server is running |
| POST | `/init` | Load cricket knowledge into Endee |
| POST | `/ask` | Ask the AI agent a cricket question |

### Example — Ask a question
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

## 📊 Cricket Knowledge Base

The system contains **25+ knowledge entries** across 5 categories:

| Category | Examples |
|----------|---------|
| 🏏 Players | Virat Kohli, Rohit Sharma, MS Dhoni, Bumrah, Sachin, Babar Azam |
| 🏟️ Teams | Mumbai Indians, CSK, RCB, KKR, Rajasthan Royals |
| 🌍 World Cup | 2024 T20 WC, 2023 ODI WC, 2011 ODI WC |
| 🎯 Strategy | Death bowling, Powerplay tactics, Batting vs spin/pace |
| 📊 Records | Highest scores, Most wickets, IPL records |

---

## 🔗 Built With

| Technology | Purpose |
|-----------|---------|
| **[Endee.io](https://github.com/endee-io/endee)** | High-performance vector database |
| **[Google Gemini](https://aistudio.google.com)** | Free AI for answer generation |
| **[FastAPI](https://fastapi.tiangolo.com)** | Python web framework |
| **[Sentence Transformers](https://sbert.net)** | Text embeddings (all-MiniLM-L6-v2, 384-dim) |
| **Docker** | Containerization |

---

## 🎓 Internship Submission Details

This project was built for the **Endee.io Internship Evaluation**.

| Requirement | Implementation |
|-------------|---------------|
| ✅ Semantic Search | Endee vector DB with cosine similarity search |
| ✅ RAG Pipeline | Retrieved chunks passed as context to Gemini AI |
| ✅ Agentic AI | Agent autonomously picks from 6 tools |
| ✅ Endee.io Database | Used as the primary vector store |
| ✅ GitHub Hosted | Full project with README |
| ✅ Forked Endee Repo | github.com/Shafiq-456/CricIQ |

---

## 👨‍💻 Author

**Mohammed Shafiq**
- GitHub: [@Shafiq-456](https://github.com/Shafiq-456)
- Forked Endee repo: [Shafiq-456/CricIQ](https://github.com/Shafiq-456/CricIQ)

---

## 📄 License

Apache License 2.0 — see [LICENSE](LICENSE) for details.

---

> Built with ❤️ for cricket and AI 🏏🤖
