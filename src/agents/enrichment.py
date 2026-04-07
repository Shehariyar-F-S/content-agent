"""
agents/enrichment.py — The Enrichment Agent.

WHAT IT DOES:
    Takes a content title (and optional synopsis) and fetches real-world facts
    from the web using Tavily search. It then uses Claude to extract a clean,
    structured fact dictionary from the raw search snippets.

WHY IT RUNS FIRST:
    Every downstream agent depends on real facts, not LLM guesses.
    Without this agent, the analysis agent would invent viewer counts,
    the sentiment agent would hallucinate reviews, and the generation
    agent would write descriptions based on nothing real.

    Ground truth first. LLM reasoning second. Always.

OUTPUT:
    Populates state['enrichment'] with facts, raw snippets, and a confidence score.
"""

import json
import logging
import os
import time

from langchain_ollama import ChatOllama
from langchain_tavily import TavilySearch

from src.state import AgentState, EnrichmentData

logger = logging.getLogger(__name__)

# Ollama — local LLM, free, no API key needed.
# Make sure you have run: ollama pull llama3.1
_model = os.getenv("OLLAMA_MODEL", "llama3.1")
_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

llm = ChatOllama(model=_model, base_url=_base_url, temperature=0)

# Tavily — purpose-built for LLM applications, reliable, free tier (1000/month).
# At 2 searches per pipeline run you get 500 full runs on the free tier.
search_tool = TavilySearch(max_results=2)

EXTRACTION_PROMPT = """\
You are a fact extraction agent. Your only job is to read web search results
and extract key facts about a TV show or media content into a structured format.

Content title: {title}
Synopsis provided: {synopsis}

Web search results:
{snippets}

Extract the following facts. Use the string "unknown" for any fact not found in the results.
Do not guess or infer — only use what is explicitly stated in the results.

Return ONLY a valid JSON object with these exact keys:
{{
  "network_or_platform": "...",
  "seasons_count": "...",
  "average_viewers": "...",
  "format_origin": "...",
  "related_shows": ["...", "..."],
  "first_aired": "...",
  "genre_category": "..."
}}

No preamble. No explanation. JSON only.\
"""


def enrichment_agent(state: AgentState) -> AgentState:
    """
    Enrichment agent node for the LangGraph pipeline.

    LangGraph calls this function with the current state and expects
    a (partial) state dict back. Only the keys you return will be
    updated — everything else in state is preserved automatically.
    """
    start_time = time.time()
    title = state["title"]
    synopsis = state.get("synopsis") or "Not provided"

    logger.info(f"[Enrichment] Starting for: {title!r}")

    try:
        # Step 1: search the web for real facts
        query = f"{title} TV show streaming platform viewers seasons format"
        raw_results = search_tool.invoke({"query": query})
        snippets = [r["content"] for r in raw_results["results"] if "content" in r]

        if not snippets:
            raise ValueError("Tavily returned no results — check your API key.")

        logger.info(f"[Enrichment] Fetched {len(snippets)} web snippets.")

        # Step 2: extract structured facts with Claude
        prompt = EXTRACTION_PROMPT.format(
            title=title,
            synopsis=synopsis,
            snippets="\n\n---\n\n".join(snippets),
        )
        response = llm.invoke(prompt)
        raw_content = response.content.strip()

        # Strip markdown code fences if the model wraps its response
        if raw_content.startswith("```"):
            raw_content = raw_content.split("```")[1]
            if raw_content.startswith("json"):
                raw_content = raw_content[4:]

        facts: dict = json.loads(raw_content)
        logger.info(f"[Enrichment] Extracted facts: {facts}")

        # Confidence heuristic: more snippets = higher confidence
        confidence = min(0.95, 0.60 + len(snippets) * 0.07)

        enrichment_data: EnrichmentData = {
            "facts": facts,
            "web_snippets": snippets,
            "confidence": round(confidence, 2),
        }

        elapsed = round(time.time() - start_time, 2)
        logger.info(f"[Enrichment] Done in {elapsed}s. Confidence: {confidence:.0%}")

        return {
            **state,
            "enrichment": enrichment_data,
            "completed_agents": state["completed_agents"] + ["enrichment"],
        }

    except Exception as exc:
        # Non-fatal: log the error and let the graph continue.
        # Downstream agents will get None for state['enrichment'] and
        # should degrade gracefully.
        error_msg = f"Enrichment agent failed: {exc}"
        logger.error(f"[Enrichment] {error_msg}")

        return {
            **state,
            "errors": state["errors"] + [error_msg],
            "completed_agents": state["completed_agents"] + ["enrichment"],
        }