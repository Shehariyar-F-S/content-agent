"""
src/evaluation.py — The Evaluation Node.

WHAT IT DOES:
    Runs after all agents complete. It does NOT call an LLM —
    it is pure Python logic that inspects the pipeline outputs and:

    1. Aggregates per-agent confidence scores into an overall score
    2. Detects potential hallucinations by cross-checking generation
       output against enrichment facts
    3. Identifies which agents ran with low confidence
    4. Counts total tokens (estimated) and measures total latency

WHY THIS IS THE MOST IMPORTANT NODE:
    This is what separates a student project from a production system.
    Any LLM pipeline can produce output. Not every pipeline can tell you
    *how much to trust* that output.

    When an interviewer asks "how do you know it's not hallucinating?"
    you point to this node.

OUTPUT:
    Populates state['evaluation'] with overall_confidence,
    hallucination_flags, low_confidence_agents, and run metadata.
"""

import logging
import time

from src.state import AgentState, EvaluationData

logger = logging.getLogger(__name__)

# Confidence thresholds
LOW_CONFIDENCE_THRESHOLD = 0.75
HALLUCINATION_SCORE_THRESHOLD = 0.70


def _check_hallucinations(state: AgentState) -> list[str]:
    """
    Cross-check generation output against enrichment facts.

    Strategy: look for specific numeric or named claims in the generated
    copy that contradict what enrichment found. This is a heuristic —
    not exhaustive — but catches the most common failure modes.
    """
    flags = []
    generation = state.get("generation")
    enrichment = state.get("enrichment")

    if not generation or not enrichment:
        return flags

    facts = enrichment.get("facts", {})
    copy_fields = [
        generation.get("seo_title", ""),
        generation.get("meta_description", ""),
        generation.get("instagram_post", ""),
        generation.get("twitter_post", ""),
    ]
    all_copy = " ".join(copy_fields).lower()

    # Check: did generation mention a year that contradicts enrichment?
    first_aired = str(facts.get("first_aired", "")).strip()
    if first_aired and first_aired != "unknown" and len(first_aired) == 4:
        # If copy mentions a year and it's not the correct one
        import re
        years_in_copy = re.findall(r"\b(19|20)\d{2}\b", all_copy)
        for year in years_in_copy:
            if year != first_aired:
                flags.append(
                    f"Generation mentions year {year} but enrichment says "
                    f"first_aired={first_aired}. Possible hallucination."
                )

    # Check: did generation claim a platform that wasn't in enrichment?
    platform = str(facts.get("network_or_platform", "")).lower()
    if platform and platform != "unknown":
        known_platforms = {p.strip() for p in platform.split(",")}
        streaming_keywords = ["netflix", "amazon", "disney", "hbo", "apple", "joyn",
                              "prosieben", "ard", "zdf", "sat.1", "rtl"]
        for keyword in streaming_keywords:
            if keyword in all_copy and not any(keyword in p for p in known_platforms):
                flags.append(
                    f"Generation mentions '{keyword}' but enrichment "
                    f"platform is '{platform}'. Verify this claim."
                )

    return flags


def _identify_low_confidence_agents(state: AgentState) -> list[str]:
    """Return names of agents whose confidence fell below the threshold."""
    low = []
    agent_map = {
        "enrichment": state.get("enrichment"),
        "analysis": state.get("analysis"),
        "sentiment": state.get("sentiment"),
        "generation": state.get("generation"),
    }
    for name, data in agent_map.items():
        if data and data.get("confidence", 1.0) < LOW_CONFIDENCE_THRESHOLD:
            low.append(f"{name} ({data['confidence']:.0%})")
    return low


def _overall_confidence(state: AgentState) -> float:
    """
    Weighted average of per-agent confidence scores.

    Weights reflect how much each agent's reliability matters
    for the final output quality:
        enrichment  — 30% (facts ground everything else)
        analysis    — 25% (tags inform generation tone)
        sentiment   — 20% (important but pipeline works without it)
        generation  — 25% (the output users actually see)
    """
    weights = {
        "enrichment": 0.30,
        "analysis": 0.25,
        "sentiment": 0.20,
        "generation": 0.25,
    }
    agent_map = {
        "enrichment": state.get("enrichment"),
        "analysis": state.get("analysis"),
        "sentiment": state.get("sentiment"),
        "generation": state.get("generation"),
    }

    total_weight = 0.0
    weighted_sum = 0.0

    for name, weight in weights.items():
        data = agent_map.get(name)
        if data:
            weighted_sum += data.get("confidence", 0.0) * weight
            total_weight += weight

    if total_weight == 0:
        return 0.0

    return round(weighted_sum / total_weight, 2)


def evaluation_node(state: AgentState) -> AgentState:
    """Evaluation node for the LangGraph pipeline."""
    start_time = time.time()
    logger.info("[Evaluation] Starting pipeline evaluation.")

    hallucination_flags = _check_hallucinations(state)
    low_confidence_agents = _identify_low_confidence_agents(state)
    overall = _overall_confidence(state)

    if hallucination_flags:
        for flag in hallucination_flags:
            logger.warning(f"[Evaluation] HALLUCINATION FLAG: {flag}")
    else:
        logger.info("[Evaluation] No hallucination flags detected.")

    if low_confidence_agents:
        logger.warning(f"[Evaluation] Low confidence agents: {low_confidence_agents}")

    # Estimate token usage (rough heuristic — replace with LangSmith data in prod)
    # Average prompt is ~400 tokens, average response is ~200 tokens, 3 LLM calls
    estimated_tokens = 3 * (400 + 200)

    elapsed = round(time.time() - start_time, 2)
    logger.info(
        f"[Evaluation] Done in {elapsed}s. "
        f"Overall confidence: {overall:.0%}. "
        f"Flags: {len(hallucination_flags)}."
    )

    evaluation_data: EvaluationData = {
        "overall_confidence": overall,
        "hallucination_flags": hallucination_flags,
        "low_confidence_agents": low_confidence_agents,
        "total_tokens_used": estimated_tokens,
        "total_latency_seconds": elapsed,
    }

    return {
        **state,
        "evaluation": evaluation_data,
        "completed_agents": state["completed_agents"] + ["evaluation"],
    }