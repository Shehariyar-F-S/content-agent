"""
state.py — The shared state that flows through every agent in the pipeline.

Think of this as the "contract" between agents. Each agent reads what it needs
from here and writes its output back into here. LangGraph passes this state
object through the graph automatically.

Design principle: make every field Optional so the graph can run partially
(useful for testing individual agents without running the full pipeline).
"""

from typing import Optional, TypedDict


class EnrichmentData(TypedDict):
    """Output of the Enrichment Agent."""
    facts: dict                  # structured facts extracted from web
    web_snippets: list[str]      # raw text snippets fetched from the web
    confidence: float            # 0.0 - 1.0


class AnalysisData(TypedDict):
    """Output of the Analysis Agent."""
    genre_tags: list[str]
    mood_tags: list[str]
    audience_tags: list[str]
    content_warnings: list[str]
    confidence: float


class SentimentData(TypedDict):
    """Output of the Sentiment Agent."""
    score: float                 # 0 (very negative) to 100 (very positive)
    summary: str
    positives: str
    criticisms: str
    source_count: int            # how many sources were analysed
    confidence: float


class GenerationData(TypedDict):
    """Output of the Generation Agent."""
    seo_title: str
    meta_description: str        # max 155 chars
    instagram_post: str
    twitter_post: str            # max 280 chars
    confidence: float


class EvaluationData(TypedDict):
    """Populated after all agents complete."""
    overall_confidence: float
    hallucination_flags: list[str]
    low_confidence_agents: list[str]
    total_tokens_used: int
    total_latency_seconds: float


class AgentState(TypedDict):
    """
    The single state object that flows through the entire LangGraph pipeline.

    Fields are populated progressively — enrichment fills 'enrichment',
    analysis fills 'analysis', and so on. Agents must never assume a
    downstream field is populated.

    'errors' accumulates non-fatal errors. The graph continues running
    even when an agent fails — it just logs the error and moves on.
    """

    # --- Input (required) ---
    title: str
    synopsis: Optional[str]

    # --- Agent outputs (populated progressively) ---
    enrichment: Optional[EnrichmentData]
    analysis: Optional[AnalysisData]
    sentiment: Optional[SentimentData]
    generation: Optional[GenerationData]
    evaluation: Optional[EvaluationData]

    # --- Pipeline metadata ---
    run_id: str
    errors: list[str]
    completed_agents: list[str]
