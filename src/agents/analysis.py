"""
agents/analysis.py — The Analysis Agent.

WHAT IT DOES:
    Takes the enrichment data (real facts from the web) and uses an LLM
    to classify the content into structured tags:
        - genre tags      (e.g. "Reality competition", "Music")
        - mood tags       (e.g. "Fun", "Suspenseful", "Light-hearted")
        - audience tags   (e.g. "Families", "Age 25-55")
        - content warnings (e.g. "Violence", "None")

WHY IT RUNS AFTER ENRICHMENT:
    It reads enrichment.facts so it classifies based on real data,
    not LLM assumptions. If enrichment failed (state['enrichment'] is None),
    it falls back to the raw title + synopsis — still useful, just less grounded.

OUTPUT:
    Populates state['analysis'] with structured tag lists + confidence score.
"""

import json
import logging
import os
import time

from langchain_groq import ChatGroq

from src.state import AgentState, AnalysisData

logger = logging.getLogger(__name__)

#_model = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
#_base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

#llm = ChatOllama(model=_model, base_url=_base_url, temperature=0)
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)

ANALYSIS_PROMPT = """\
You are a content classification expert for a streaming platform.
Your job is to classify a TV show into structured tags.

Content title: {title}
Synopsis: {synopsis}
Known facts: {facts}

Classify this content. Be specific and accurate.
Use between 2 and 5 tags per category.
For content_warnings use ["None"] if there are no warnings.

Return ONLY a valid JSON object with exactly these keys:
{{
  "genre_tags": ["...", "..."],
  "mood_tags": ["...", "..."],
  "audience_tags": ["...", "..."],
  "content_warnings": ["..."]
}}

No preamble. No explanation. JSON only.\
"""


def _strip_fences(text: str) -> str:
    """Strip markdown code fences that models sometimes add despite being told not to."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # remove first line (```json or ```) and last line (```)
        lines = lines[1:] if lines[-1].strip() == "```" else lines[1:]
        text = "\n".join(lines).strip()
    return text


def analysis_agent(state: AgentState) -> AgentState:
    """Analysis agent node for the LangGraph pipeline."""
    start_time = time.time()
    title = state["title"]
    synopsis = state.get("synopsis") or "Not provided"

    logger.info(f"[Analysis] Starting for: {title!r}")

    # Build context — use enrichment facts if available, otherwise work from title/synopsis only
    enrichment = state.get("enrichment")
    if enrichment:
        facts_str = json.dumps(enrichment["facts"], indent=2)
        logger.info("[Analysis] Using enrichment data as context.")
    else:
        facts_str = "No enrichment data available — classify from title and synopsis only."
        logger.warning("[Analysis] No enrichment data — degraded mode.")

    try:
        prompt = ANALYSIS_PROMPT.format(
            title=title,
            synopsis=synopsis,
            facts=facts_str,
        )

        response = llm.invoke(prompt)
        raw = _strip_fences(response.content)
        data: dict = json.loads(raw)

        # Validate expected keys are present
        required = {"genre_tags", "mood_tags", "audience_tags", "content_warnings"}
        missing = required - set(data.keys())
        if missing:
            raise ValueError(f"LLM response missing keys: {missing}")

        # Confidence: high if enrichment was available, lower if we worked blind
        confidence = 0.88 if enrichment else 0.65

        analysis_data: AnalysisData = {
            "genre_tags": data["genre_tags"],
            "mood_tags": data["mood_tags"],
            "audience_tags": data["audience_tags"],
            "content_warnings": data["content_warnings"],
            "confidence": confidence,
        }

        elapsed = round(time.time() - start_time, 2)
        logger.info(
            f"[Analysis] Done in {elapsed}s. "
            f"Genre: {data['genre_tags']}. Confidence: {confidence:.0%}"
        )

        return {
            **state,
            "analysis": analysis_data,
            "completed_agents": state["completed_agents"] + ["analysis"],
        }

    except Exception as exc:
        error_msg = f"Analysis agent failed: {exc}"
        logger.error(f"[Analysis] {error_msg}")
        return {
            **state,
            "errors": state["errors"] + [error_msg],
            "completed_agents": state["completed_agents"] + ["analysis"],
        }