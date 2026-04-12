# agent.py - Simplified Cricket AI Agent with Groq Integration
# Synchronous version using requests library
# Fallback to direct cricket knowledge base search if Groq fails

import os
import json
import requests
from dotenv import load_dotenv
from data import CRICKET_KNOWLEDGE

# Load environment variables from .env file
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"

print(f"[Agent] GROQ_API_KEY loaded: {bool(GROQ_API_KEY)}")
print(f"[Agent] Cricket Knowledge Base loaded: {len(CRICKET_KNOWLEDGE)} documents")

# ── Simple Tool Selection (Keyword Matching) ──────────────────────────────────
def select_tool(query: str) -> str:
    """Select appropriate tool based on simple keyword matching."""
    query_lower = query.lower()
    
    if any(word in query_lower for word in ["compare", "vs", "versus", "difference"]):
        return "compare_players"
    elif any(word in query_lower for word in ["player", "batsman", "bowler", "cricketer"]):
        return "search_player"
    elif any(word in query_lower for word in ["team", "ipl", "india", "england", "australia", "pakistan"]):
        return "search_team"
    elif any(word in query_lower for word in ["world cup", "worldcup", "tournament"]):
        return "search_worldcup"
    elif any(word in query_lower for word in ["strategy", "tactic", "batting", "bowling", "technique"]):
        return "search_strategy"
    elif any(word in query_lower for word in ["record", "highest", "most", "statistic", "average"]):
        return "search_records"
    else:
        return "search_player"  # Default

# ── Direct Search from Cricket Knowledge Base ─────────────────────────────────
def search_cricket_knowledge(query: str, category: str = None, top_k: int = 3) -> list:
    """
    Direct search through CRICKET_KNOWLEDGE using keyword matching.
    Returns top_k matches based on keyword overlap.
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
        
        # Calculate match score
        text_words = set(text_lower.split())
        title_words = set(title_lower.split())
        
        text_matches = len(query_words & text_words)
        title_matches = len(query_words & title_words) * 2  # Title matches weighted higher
        
        total_score = text_matches + title_matches
        
        if total_score > 0:
            scored_results.append({
                "title": doc.get("title", ""),
                "text": doc.get("text", ""),
                "category": doc.get("category", ""),
                "score": total_score
            })
    
    # Sort by score and return top_k
    scored_results.sort(key=lambda x: x["score"], reverse=True)
    return scored_results[:top_k]

# ── Fallback Answer Generator ─────────────────────────────────────────────────
def generate_fallback_answer(search_results: list, query: str) -> dict:
    """Generate answer directly from search results when Gemini fails."""
    if not search_results:
        return {
            "answer": f"I searched our cricket knowledge base for '{query}' but didn't find specific information. Try asking about players, teams, IPL, World Cup records, or cricket strategies!",
            "tool_used": "direct_search",
            "sources": []
        }
    
    answer_text = f"Based on our cricket knowledge base, here's what I found about '{query}':\n\n"
    sources = []
    
    for i, result in enumerate(search_results, 1):
        answer_text += f"✓ {result['title']}\n"
        answer_text += f"  {result['text'][:300]}{'...' if len(result['text']) > 300 else ''}\n\n"
        if result['title'] not in sources:
            sources.append(result['title'])
    
    return {
        "answer": answer_text,
        "tool_used": "direct_search",
        "sources": sources[:3]
    }

# ── Call Groq API Synchronously ──────────────────────────────────────────────
def call_groq(prompt: str) -> str:
    """Call Groq API synchronously and return generated text."""
    if not GROQ_API_KEY:
        print("[Agent] ERROR: GROQ_API_KEY not set!")
        return None
    
    try:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {GROQ_API_KEY}"
        }
        
        payload = {
            "model": "llama3-8b-8192",
            "messages": [
                {
                    "role": "system",
                    "content": "You are CricketIQ, an expert AI cricket analyst and strategist. You have deep knowledge of cricket players, teams, strategies, records, and tournaments. Always be enthusiastic about cricket. Use cricket terminology naturally. Keep answers concise but informative. Always cite sources clearly."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "max_tokens": 800,
            "temperature": 0.7
        }
        
        print(f"[Agent] Calling Groq API...")
        response = requests.post(
            GROQ_URL,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        print(f"[Agent] Groq response status: {response.status_code}")
        
        if response.status_code != 200:
            print(f"[Agent] ERROR: Groq API returned {response.status_code}")
            print(f"[Agent] Response: {response.text[:500]}")
            return None
        
        result = response.json()
        
        # Extract answer from response (OpenAI compatible format)
        if "choices" in result and len(result["choices"]) > 0:
            message = result["choices"][0].get("message", {})
            text = message.get("content", "")
            if text:
                print(f"[Agent] Groq answered successfully ({len(text)} chars)")
                return text
        
        print("[Agent] ERROR: No choices in Groq response")
        return None
        
    except requests.exceptions.Timeout:
        print("[Agent] ERROR: Groq API call timed out")
        return None
    except requests.exceptions.RequestException as e:
        print(f"[Agent] ERROR: Groq API request failed: {str(e)}")
        return None
    except Exception as e:
        print(f"[Agent] ERROR: Unexpected error calling Groq: {str(e)}")
        return None

# ── Main Agent Function ───────────────────────────────────────────────────────
def run_agent(user_query: str) -> dict:
    """
    Main agent function that:
    1. Selects appropriate tool/category
    2. Searches cricket knowledge base
    3. Tries to use Gemini to generate answer
    4. Falls back to direct search results if Gemini fails
    """
    print(f"\n[Agent] Processing query: '{user_query}'")
    
    # Step 1: Select tool
    tool = select_tool(user_query)
    print(f"[Agent] Selected tool: {tool}")
    
    # Map tool to category
    category_map = {
        "search_player": "player",
        "search_team": "team",
        "search_worldcup": "worldcup",
        "search_strategy": "strategy",
        "search_records": "record",
        "compare_players": None
    }
    
    category = category_map.get(tool)
    top_k = 5 if tool == "compare_players" else 3
    
    # Step 2: Search cricket knowledge base directly
    print(f"[Agent] Searching cricket knowledge base (category={category}, top_k={top_k})...")
    search_results = search_cricket_knowledge(user_query, category=category, top_k=top_k)
    print(f"[Agent] Found {len(search_results)} results")
    
    # Extract sources
    sources = [r["title"] for r in search_results]
    
    # Step 3: Try to use Groq for better answer
    if GROQ_API_KEY and search_results:
        # Format search results as context
        context = "\n".join([
            f"• {r['title']}: {r['text'][:200]}..."
            for r in search_results
        ])
        
        groq_prompt = f"""Answer this question based ONLY on the cricket information provided:

Cricket Information:
{context}

User Question: {user_query}

Provide a concise, helpful answer citing the sources. Be enthusiastic about cricket!"""
        
        groq_answer = call_groq(groq_prompt)
        
        if groq_answer and groq_answer.strip():
            print(f"[Agent] Successfully generated Groq answer")
            return {
                "answer": groq_answer,
                "tool_used": tool,
                "sources": sources[:3]
            }
        else:
            print(f"[Agent] Groq returned empty result, using fallback")
    else:
        if not GROQ_API_KEY:
            print("[Agent] No API key, using fallback")
        elif not search_results:
            print("[Agent] No search results, using fallback")
    
    # Step 4: Fallback to direct search results
    print(f"[Agent] Returning fallback answer from direct search")
    return generate_fallback_answer(search_results, user_query)