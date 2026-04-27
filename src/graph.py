"""
graph.py — The LangGraph orchestrator.

This is where all agents are connected into a directed graph.
LangGraph manages the state transitions automatically — you just
define the nodes (agents) and edges (what runs after what).

LLM backend: Ollama (local, free — https://ollama.com)
Search:       DuckDuckGo (no key needed)
Tracing:      LangSmith (optional free tier)

Current graph shape (linear pipeline):

    enrichment → analysis → sentiment → generation → evaluation → END

Later you can make edges conditional — e.g. skip sentiment if
enrichment failed, or run analysis and sentiment in parallel.
"""

import os
from dotenv import load_dotenv
load_dotenv()

# Initialise LangSmith tracing
if os.getenv("LANGCHAIN_API_KEY"):
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGCHAIN_ENDPOINT"] = "https://api.smith.langchain.com"
    os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "content-agent")

import uuid
import logging
from typing import Optional

from langgraph.graph import StateGraph, END

from src.state import AgentState
from src.agents.enrichment import enrichment_agent
from src.agents.analysis import analysis_agent
from src.agents.sentiment import sentiment_agent
from src.agents.generation import generation_agent
from src.evaluation import evaluation_node

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------




# ---------------------------------------------------------------------------
# Graph assembly
# ---------------------------------------------------------------------------

def build_graph() -> StateGraph:
    """
    Construct and compile the LangGraph pipeline.

    Returns a compiled graph that can be invoked with an initial state dict.
    Call build_graph() once at startup and reuse the compiled graph.
    """
    graph = StateGraph(AgentState)

    # Register nodes — each node is a function: AgentState -> AgentState
    graph.add_node("enrichment", enrichment_agent)
    graph.add_node("analysis", analysis_agent)
    graph.add_node("sentiment", sentiment_agent)
    graph.add_node("generation", generation_agent)
    graph.add_node("evaluation", evaluation_node)

    # Wire the edges — defines execution order
    graph.set_entry_point("enrichment")
    graph.add_edge("enrichment", "analysis")
    graph.add_edge("analysis", "sentiment")
    graph.add_edge("sentiment", "generation")
    graph.add_edge("generation", "evaluation")
    graph.add_edge("evaluation", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Public interface
# ---------------------------------------------------------------------------

# Compile once at import time — reused across all requests
pipeline = build_graph()


def run_pipeline(
    title: str,
    synopsis: Optional[str] = None,
) -> AgentState:
    """
    Run the full content intelligence pipeline for a given title.

    Args:
        title:    The content title (e.g. "The Masked Singer Germany")
        synopsis: Optional short description to help the agents

    Returns:
        The final AgentState after all agents have run.
        Check state['errors'] for any non-fatal failures.
        Check state['completed_agents'] to see which agents ran.
    """
    initial_state: AgentState = {
        "title": title,
        "synopsis": synopsis,
        "enrichment": None,
        "analysis": None,
        "sentiment": None,
        "generation": None,
        "evaluation": None,
        "run_id": str(uuid.uuid4()),
        "errors": [],
        "completed_agents": [],
    }

    logger.info(f"[Pipeline] Starting run {initial_state['run_id']} for: {title!r}")
    result = pipeline.invoke(initial_state)
    logger.info(
        f"[Pipeline] Completed. Agents run: {result['completed_agents']}. "
        f"Errors: {result['errors']}"
    )

    return result