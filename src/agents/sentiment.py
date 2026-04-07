"""
agents/sentiment.py — The Sentiment Agent.

WHAT IT DOES:
    Searches the web for real audience reactions to a piece of content —
    reviews, Reddit discussions, forum posts — and summarises them into
    a structured sentiment score.

WHY THIS MATTERS:
    Unlike the analysis agent (which classifies the content itself),
    sentiment measures how audiences *actually feel* about it.
    This is live data — not LLM guesses about what people might think.

    A show can be accurately tagged as "Comedy, Family" (analysis)
    but still have 60% negative sentiment because audiences find it
    boring. These are different signals and both matter for a
    streaming platform making programming decisions.

OUTPUT:
    Populates state['sentiment'] with score, summary, positives,
    criticisms, source count, and confidence.
"""

import json
import logging
import os
import time

from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

from src.state import AgentState, SentimentData

logger = logging.getLogger(__name__)

_model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

llm = ChatOllama(model=_model, base_url=_base_url, temperature=0)
search_tool = TavilySearch(max_results=3)

SENTIMENT_PROMPT = """\
You are an audience sentiment analyst for a streaming platform.
Analyse the following web content about a TV show and extract audience sentiment.

Show title: {title}
Web content about audience reactions:
{snippets}

Based ONLY on the web content above, extract:
- An overall sentiment score from 0 to 100 (0 = very negative, 50 = mixed, 100 = very positive)
- A one-sentence summary of overall audience feeling
- What audiences specifically praise (2-3 points)
- What audiences specifically criticise (2-3 points, use "None found" if no criticism)
- How many distinct sources you found reactions in (integer)

Return ONLY a valid JSON object with exactly these keys:
{{
  "score": <integer 0-100>,
  "summary": "...",
  "positives": "...",
  "criticisms": "...",
  "source_count": <integer>
}}

No preamble. No explanation. JSON only.\
"""


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        lines = lines[1:] if lines[-1].strip() == "```" else lines[1:]
        text = "\n".join(lines).strip()
    return text


def sentiment_agent(state: AgentState) -> AgentState:
    """Sentiment agent node for the LangGraph pipeline."""
    start_time = time.time()
    title = state["title"]

    logger.info(f"[Sentiment] Starting for: {title!r}")

    try:
        # Search specifically for audience reactions, reviews, opinions
        query = f"{title} TV show audience review opinion rating"
        raw = search_tool.invoke({"query": query})
        snippets = [r["content"] for r in raw.get("results", []) if "content" in r]

        if not snippets:
            raise ValueError("No sentiment data found from web search.")

        logger.info(f"[Sentiment] Fetched {len(snippets)} reaction snippets.")

        prompt = SENTIMENT_PROMPT.format(
            title=title,
            snippets="\n\n---\n\n".join(snippets),
        )

        response = llm.invoke(prompt)
        data: dict = json.loads(_strip_fences(response.content))

        required = {"score", "summary", "positives", "criticisms", "source_count"}
        missing = required - set(data.keys())
        if missing:
            raise ValueError(f"LLM response missing keys: {missing}")

        # Validate score is in range
        score = max(0, min(100, int(data["score"])))

        # Confidence: based on how many sources we found
        source_count = int(data.get("source_count", len(snippets)))
        confidence = min(0.90, 0.60 + source_count * 0.08)

        sentiment_data: SentimentData = {
            "score": score,
            "summary": data["summary"],
            "positives": data["positives"],
            "criticisms": data["criticisms"],
            "source_count": source_count,
            "confidence": round(confidence, 2),
        }

        elapsed = round(time.time() - start_time, 2)
        logger.info(
            f"[Sentiment] Done in {elapsed}s. "
            f"Score: {score}/100. Confidence: {confidence:.0%}"
        )

        return {
            **state,
            "sentiment": sentiment_data,
            "completed_agents": state["completed_agents"] + ["sentiment"],
        }

    except Exception as exc:
        error_msg = f"Sentiment agent failed: {exc}"
        logger.error(f"[Sentiment] {error_msg}")
        return {
            **state,
            "errors": state["errors"] + [error_msg],
            "completed_agents": state["completed_agents"] + ["sentiment"],
        }