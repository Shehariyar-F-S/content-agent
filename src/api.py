"""
src/api.py — FastAPI backend.

Wraps the LangGraph pipeline in a REST API with two endpoints:

    POST /analyse          Run the full pipeline for a given title
    GET  /health           Health check — confirms API + Ollama are reachable

Auto-generated docs available at http://localhost:8000/docs when running.
"""

import os
import logging
import time
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

# LangSmith tracing — enable if API key is present
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.graph import run_pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Content Intelligence Agent",
    description="Multi-agent AI pipeline for streaming content metadata generation.",
    version="1.0.0",
)

# Allow Streamlit (running on port 8501) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class AnalyseRequest(BaseModel):
    title: str
    synopsis: Optional[str] = None

    model_config = {
        "json_schema_extra": {
            "example": {
                "title": "Turkish for Beginners",
                "synopsis": "A German comedy series about cultural clashes in a blended family.",
            }
        }
    }


class EnrichmentResponse(BaseModel):
    facts: dict
    snippets_count: int
    confidence: float


class AnalysisResponse(BaseModel):
    genre_tags: list[str]
    mood_tags: list[str]
    audience_tags: list[str]
    content_warnings: list[str]
    confidence: float


class SentimentResponse(BaseModel):
    score: int
    summary: str
    positives: str
    criticisms: str
    source_count: int
    confidence: float


class GenerationResponse(BaseModel):
    seo_title: str
    meta_description: str
    instagram_post: str
    twitter_post: str
    confidence: float


class EvaluationResponse(BaseModel):
    overall_confidence: float
    hallucination_flags: list[str]
    low_confidence_agents: list[str]
    total_tokens_used: int


class AnalyseResponse(BaseModel):
    run_id: str
    title: str
    completed_agents: list[str]
    errors: list[str]
    latency_seconds: float
    enrichment: Optional[EnrichmentResponse]
    analysis: Optional[AnalysisResponse]
    sentiment: Optional[SentimentResponse]
    generation: Optional[GenerationResponse]
    evaluation: Optional[EvaluationResponse]


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    """Health check — returns 200 if the API is running."""
    return {"status": "ok", "service": "content-intelligence-agent"}


@app.post("/analyse", response_model=AnalyseResponse)
def analyse(request: AnalyseRequest):
    """
    Run the full content intelligence pipeline.

    Takes a show title and optional synopsis, runs all 5 agents,
    and returns structured metadata with confidence scores.
    """
    logger.info(f"[API] /analyse called for: {request.title!r}")
    start = time.time()

    try:
        state = run_pipeline(
            title=request.title,
            synopsis=request.synopsis,
        )
    except Exception as exc:
        logger.error(f"[API] Pipeline crashed: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))

    latency = round(time.time() - start, 2)

    # Build typed response — handle missing/failed agents gracefully
    enrichment = None
    if state.get("enrichment"):
        e = state["enrichment"]
        enrichment = EnrichmentResponse(
            facts=e["facts"],
            snippets_count=len(e.get("web_snippets", [])),
            confidence=e["confidence"],
        )

    analysis = None
    if state.get("analysis"):
        a = state["analysis"]
        analysis = AnalysisResponse(
            genre_tags=a["genre_tags"],
            mood_tags=a["mood_tags"],
            audience_tags=a["audience_tags"],
            content_warnings=a["content_warnings"],
            confidence=a["confidence"],
        )

    sentiment = None
    if state.get("sentiment"):
        s = state["sentiment"]
        sentiment = SentimentResponse(
            score=s["score"],
            summary=s["summary"],
            positives=s["positives"],
            criticisms=s["criticisms"],
            source_count=s["source_count"],
            confidence=s["confidence"],
        )

    generation = None
    if state.get("generation"):
        g = state["generation"]
        generation = GenerationResponse(
            seo_title=g["seo_title"],
            meta_description=g["meta_description"],
            instagram_post=g["instagram_post"],
            twitter_post=g["twitter_post"],
            confidence=g["confidence"],
        )

    evaluation = None
    if state.get("evaluation"):
        ev = state["evaluation"]
        evaluation = EvaluationResponse(
            overall_confidence=ev["overall_confidence"],
            hallucination_flags=ev["hallucination_flags"],
            low_confidence_agents=ev["low_confidence_agents"],
            total_tokens_used=ev["total_tokens_used"],
        )

    return AnalyseResponse(
        run_id=state["run_id"],
        title=state["title"],
        completed_agents=state["completed_agents"],
        errors=state["errors"],
        latency_seconds=latency,
        enrichment=enrichment,
        analysis=analysis,
        sentiment=sentiment,
        generation=generation,
        evaluation=evaluation,
    )