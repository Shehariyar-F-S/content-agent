"""
agents/generation.py — The Generation Agent.

WHAT IT DOES:
    The final content-producing agent. Takes everything the previous agents
    gathered — real facts, structured tags, sentiment — and writes:
        - SEO title
        - Meta description (max 155 chars)
        - Instagram post
        - Twitter/X post (max 280 chars)

WHY IT RUNS LAST (before evaluation):
    Generation is the most hallucination-prone step because it involves
    creative writing. Running it last means it always has grounded context
    to write from. It should never invent facts — only use what enrichment
    and analysis already established.

OUTPUT:
    Populates state['generation'] with copy + confidence score.
"""

import json
import logging
import os
import time

from langchain_groq import ChatGroq

from src.state import AgentState, GenerationData

logger = logging.getLogger(__name__)

#_model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
#_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

#llm = ChatOllama(model=_model, base_url=_base_url, temperature=0.3)
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.3)
# Note: temperature=0.3 here (not 0) — generation benefits from slight creativity
# while still being mostly deterministic. Pure temperature=0 produces flat copy.


GENERATION_PROMPT = """\
You are a content marketing expert for a streaming platform.
Write compelling copy for a TV show based on the information provided.

Title: {title}
Synopsis: {synopsis}
Genre tags: {genre_tags}
Mood tags: {mood_tags}
Target audience: {audience_tags}
Key facts: {facts}

Write the following. Follow the character limits strictly.

Rules:
- Only use facts that are provided above — do not invent information
- Write in an engaging, platform-appropriate tone
- The meta_description must be 155 characters or fewer
- The twitter_post must be 280 characters or fewer
- Do not use hashtags in seo_title or meta_description
- Do not include any URLs or links

Return ONLY a valid JSON object with exactly these keys:
{{
  "seo_title": "...",
  "meta_description": "...",
  "instagram_post": "...",
  "twitter_post": "..."
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


def _truncate(text: str, limit: int) -> str:
    """Hard-truncate a string to a character limit, adding ellipsis if needed."""
    if len(text) <= limit:
        return text
    return text[: limit - 3].rstrip() + "..."


def generation_agent(state: AgentState) -> AgentState:
    """Generation agent node for the LangGraph pipeline."""
    start_time = time.time()
    title = state["title"]
    synopsis = state.get("synopsis") or "Not provided"

    logger.info(f"[Generation] Starting for: {title!r}")

    # Gather context from previous agents
    enrichment = state.get("enrichment")
    analysis = state.get("analysis")

    facts_str = (
        json.dumps(enrichment["facts"], indent=2)
        if enrichment
        else "No enrichment data available."
    )
    genre_tags = analysis["genre_tags"] if analysis else ["Unknown"]
    mood_tags = analysis["mood_tags"] if analysis else ["Unknown"]
    audience_tags = analysis["audience_tags"] if analysis else ["General audience"]

    if not enrichment and not analysis:
        logger.warning("[Generation] No enrichment or analysis data — degraded mode.")

    try:
        prompt = GENERATION_PROMPT.format(
            title=title,
            synopsis=synopsis,
            genre_tags=", ".join(genre_tags),
            mood_tags=", ".join(mood_tags),
            audience_tags=", ".join(audience_tags),
            facts=facts_str,
        )

        response = llm.invoke(prompt)
        raw = _strip_fences(response.content)
        data: dict = json.loads(raw)

        required = {"seo_title", "meta_description", "instagram_post", "twitter_post"}
        missing = required - set(data.keys())
        if missing:
            raise ValueError(f"LLM response missing keys: {missing}")

        # Enforce character limits — LLMs often ignore them despite instructions
        meta = _truncate(data["meta_description"], 155)
        tweet = _truncate(data["twitter_post"], 280)

        # Confidence: lower than analysis because generation is more creative
        # and therefore more likely to drift from facts
        has_enrichment = enrichment is not None
        has_analysis = analysis is not None
        confidence = 0.70 + (0.10 if has_enrichment else 0) + (0.08 if has_analysis else 0)

        generation_data: GenerationData = {
            "seo_title": data["seo_title"],
            "meta_description": meta,
            "instagram_post": data["instagram_post"],
            "twitter_post": tweet,
            "confidence": round(confidence, 2),
        }

        elapsed = round(time.time() - start_time, 2)
        logger.info(
            f"[Generation] Done in {elapsed}s. "
            f"Meta desc: {len(meta)} chars. Confidence: {confidence:.0%}"
        )

        return {
            **state,
            "generation": generation_data,
            "completed_agents": state["completed_agents"] + ["generation"],
        }

    except Exception as exc:
        error_msg = f"Generation agent failed: {exc}"
        logger.error(f"[Generation] {error_msg}")
        return {
            **state,
            "errors": state["errors"] + [error_msg],
            "completed_agents": state["completed_agents"] + ["generation"],
        }