# agent.py - Agentic AI Brain using Google Gemini (Free Tier)
# The agent reads the user query, decides which tool to use,
# searches Endee via RAG, and generates an answer using Gemini AI

import os
import httpx
import json
from rag import semantic_search

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

# ── Tool Definitions ──────────────────────────────────────────────────────────
# These are the tools the agent can choose from autonomously

TOOLS = [
    {
        "name": "search_player",
        "description": "Search for information about a specific cricket player - their stats, batting/bowling style, strengths, weaknesses, records, and career details.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The player-related search query"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "search_team",
        "description": "Search for information about cricket teams - IPL teams, national teams, their records, key players, home grounds, and titles won.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The team-related search query"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "search_worldcup",
        "description": "Search for World Cup history, results, records, and tournament information.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The World Cup related search query"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "search_strategy",
        "description": "Search for cricket strategies, tactics, batting techniques, bowling plans, and game plans.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The strategy-related search query"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "search_records",
        "description": "Search for cricket records - highest scores, most wickets, fastest centuries, IPL records, and other statistical achievements.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The records-related search query"}
            },
            "required": ["query"]
        }
    },
    {
        "name": "compare_players",
        "description": "Compare two or more cricket players based on their stats, styles, and achievements.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The comparison query mentioning the players to compare"}
            },
            "required": ["query"]
        }
    }
]

# ── Tool Execution ────────────────────────────────────────────────────────────
async def execute_tool(tool_name: str, tool_input: dict) -> str:
    """Execute the tool chosen by the agent and return results from Endee."""
    query = tool_input.get("query", "")

    category_map = {
        "search_player":   "player",
        "search_team":     "team",
        "search_worldcup": "worldcup",
        "search_strategy": "strategy",
        "search_records":  "record",
        "compare_players": None  # No filter - search across all categories
    }

    category = category_map.get(tool_name)
    top_k = 5 if tool_name == "compare_players" else 3

    chunks = await semantic_search(query, category=category, top_k=top_k)

    if not chunks:
        return "No relevant cricket information found for this query."

    # Format chunks as context string
    context = ""
    for i, chunk in enumerate(chunks, 1):
        context += f"\n[Source {i}: {chunk['title']}]\n{chunk['text']}\n"

    return context

# ── Main Agent ────────────────────────────────────────────────────────────────
async def run_agent(user_query: str) -> dict:
    """
    Agentic pipeline using Google Gemini:
    1. Analyze user query to determine which tool to use
    2. Execute tool (semantic search in Endee)
    3. Use Gemini to generate a grounded answer based on search results
    """

    if not GEMINI_API_KEY:
        return {
            "answer": "Please set your GEMINI_API_KEY environment variable to use CricketIQ.",
            "tool_used": "none",
            "sources": []
        }

    # Step 1: Analyze query and select appropriate tool
    tool_description = """
    You must respond with ONLY a JSON object (no markdown, no explanation) with this exact structure:
    {"tool": "tool_name", "query": "refined_search_query"}
    
    Choose ONE of these tools based on the user query:
    - search_player: For questions about cricket players
    - search_team: For questions about teams (IPL, national, etc.)
    - search_worldcup: For World Cup history and records
    - search_strategy: For cricket tactics and strategies
    - search_records: For cricket statistics and records
    - compare_players: For comparing multiple players
    
    User query: {query}
    """

    system_prompt = """You are CricketIQ, an expert AI cricket analyst and strategist. 
You have deep knowledge of cricket players, teams, strategies, records, and tournaments.
You use tools to search a cricket knowledge database and provide accurate, insightful answers.
Always be enthusiastic about cricket. Use cricket terminology naturally.
Keep answers concise but informative. Always cite sources clearly."""

    # Tool selection prompt
    headers = {"Content-Type": "application/json"}
    tool_selection_payload = {
        "contents": [{
            "role": "user",
            "parts": [{"text": tool_description.format(query=user_query)}]
        }],
        "generationConfig": {"temperature": 0.3, "maxOutputTokens": 200}
    }

    tool_used = "general"
    context = ""
    sources = []

    try:
        async with httpx.AsyncClient() as client:
            tool_response = await client.post(
                f"{GEMINI_URL}?key={GEMINI_API_KEY}",
                headers=headers,
                json=tool_selection_payload,
                timeout=30
            )
            tool_response.raise_for_status()
            tool_result = tool_response.json()

        # Extract tool selection
        tool_text = ""
        if "candidates" in tool_result and len(tool_result["candidates"]) > 0:
            for part in tool_result["candidates"][0].get("content", {}).get("parts", []):
                tool_text += part.get("text", "")

        # Parse tool selection (try to extract JSON)
        try:
            tool_json = json.loads(tool_text)
            tool_used = tool_json.get("tool", "general")
            search_query = tool_json.get("query", user_query)
        except (json.JSONDecodeError, TypeError):
            # Fallback: try to find tool name in response
            tool_text_lower = tool_text.lower()
            if "search_player" in tool_text_lower:
                tool_used = "search_player"
            elif "search_team" in tool_text_lower:
                tool_used = "search_team"
            elif "search_worldcup" in tool_text_lower:
                tool_used = "search_worldcup"
            elif "search_strategy" in tool_text_lower:
                tool_used = "search_strategy"
            elif "search_records" in tool_text_lower:
                tool_used = "search_records"
            elif "compare_players" in tool_text_lower:
                tool_used = "compare_players"
            search_query = user_query

        # Step 2: Execute tool (search Endee)
        if tool_used in [t["name"] for t in TOOLS]:
            context = await execute_tool(tool_used, {"query": search_query})

            # Extract sources
            for chunk in context.split("[Source"):
                if chunk.strip():
                    title_end = chunk.find("]")
                    if title_end > 0:
                        title = chunk[chunk.find(":")+1:title_end].strip()
                        if title and title not in sources:
                            sources.append(title)

        # Step 3: Generate answer using Gemini with context
        answer_prompt = f"""Based on this cricket information, answer the user's question concisely and accurately.

Cricket Information:
{context if context else "No specific information found. Provide a general answer based on cricket knowledge."}

User Question: {user_query}

Provide a helpful, enthusiastic cricket answer. Cite the sources provided above."""

        answer_payload = {
            "contents": [{
                "role": "user",
                "parts": [{"text": answer_prompt}]
            }],
            "generationConfig": {"temperature": 0.7, "maxOutputTokens": 500}
        }

        async with httpx.AsyncClient() as client:
            answer_response = await client.post(
                f"{GEMINI_URL}?key={GEMINI_API_KEY}",
                headers=headers,
                json=answer_payload,
                timeout=30
            )
            answer_response.raise_for_status()
            answer_result = answer_response.json()

        answer = ""
        if "candidates" in answer_result and len(answer_result["candidates"]) > 0:
            for part in answer_result["candidates"][0].get("content", {}).get("parts", []):
                answer += part.get("text", "")

        return {
            "answer": answer or "I couldn't generate an answer. Please try rephrasing your question.",
            "tool_used": tool_used,
            "sources": sources[:3]
        }

    except Exception as e:
        return {
            "answer": f"Error processing query: {str(e)}",
            "tool_used": "error",
            "sources": []
        }