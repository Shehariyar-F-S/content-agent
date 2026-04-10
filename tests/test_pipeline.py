"""
tests/test_pipeline.py — Pytest test suite with mocked external calls.

Run locally:
    pytest tests/test_pipeline.py -v

Mocking strategy:
    llm and search_tool are instantiated at module level when the agent
    module is first imported. We import the agent module first, then
    patch the already-instantiated objects in place using their full
    module path. This is why all agent imports happen INSIDE the test
    methods after the patches are applied.
"""

import json
import os
import uuid
from unittest.mock import MagicMock, patch

import pytest

from src.state import AgentState

# Force all agent modules to import now so patch targets exist
import src.agents.enrichment
import src.agents.analysis
import src.agents.sentiment
import src.agents.generation
import src.evaluation


def make_state(title="Test Show", synopsis="A test synopsis.") -> AgentState:
    return {
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


def mock_response(content: str) -> MagicMock:
    m = MagicMock()
    m.content = content
    return m


FACTS = {
    "network_or_platform": "ProSieben",
    "seasons_count": "3",
    "average_viewers": "7.5/10",
    "format_origin": "German",
    "related_shows": [],
    "first_aired": "2006",
    "genre_category": "Comedy, Family, Drama",
}

ANALYSIS = {
    "genre_tags": ["Comedy", "Drama"],
    "mood_tags": ["Light-hearted", "Warm"],
    "audience_tags": ["Families", "General Audience"],
    "content_warnings": ["None"],
}

SENTIMENT = {
    "score": 75,
    "summary": "Generally positive.",
    "positives": "Strong acting",
    "criticisms": "Quality decline in season 3",
    "source_count": 2,
}

GENERATION = {
    "seo_title": "Test Show: Comedy and Drama",
    "meta_description": "A compelling German comedy-drama series for families.",
    "instagram_post": "Watch Test Show now!",
    "twitter_post": "Test Show — great German comedy-drama. 3 seasons!",
}


# ---------------------------------------------------------------------------
# State schema tests
# ---------------------------------------------------------------------------

class TestStateSchema:

    def test_has_required_keys(self):
        state = make_state()
        for key in ["title", "synopsis", "enrichment", "analysis", "sentiment",
                    "generation", "evaluation", "run_id", "errors", "completed_agents"]:
            assert key in state

    def test_agent_outputs_start_as_none(self):
        state = make_state()
        for key in ["enrichment", "analysis", "sentiment", "generation", "evaluation"]:
            assert state[key] is None

    def test_lists_start_empty(self):
        state = make_state()
        assert state["errors"] == []
        assert state["completed_agents"] == []

    def test_run_id_is_valid_uuid(self):
        state = make_state()
        uuid.UUID(state["run_id"])


# ---------------------------------------------------------------------------
# Enrichment agent tests
# ---------------------------------------------------------------------------

class TestEnrichmentAgent:

    def test_populates_state_on_success(self):
        with patch.object(src.agents.enrichment, "search_tool") as mock_search, \
             patch.object(src.agents.enrichment, "llm") as mock_llm:

            mock_search.invoke.return_value = {
                "results": [
                    {"content": "Test Show is a German comedy from 2006."},
                    {"content": "It ran for 3 seasons on ProSieben."},
                ]
            }
            mock_llm.invoke.return_value = mock_response(json.dumps(FACTS))

            result = src.agents.enrichment.enrichment_agent(make_state())

        assert result["enrichment"] is not None
        assert result["enrichment"]["facts"] == FACTS
        assert len(result["enrichment"]["web_snippets"]) >= 2
        assert result["enrichment"]["confidence"] > 0
        assert "enrichment" in result["completed_agents"]
        assert result["errors"] == []

    def test_handles_no_results_gracefully(self):
        with patch.object(src.agents.enrichment, "search_tool") as mock_search, \
             patch.object(src.agents.enrichment, "llm"):

            mock_search.invoke.return_value = {"results": []}

            result = src.agents.enrichment.enrichment_agent(make_state())

        assert result["enrichment"] is None
        assert len(result["errors"]) == 1
        assert "enrichment" in result["completed_agents"]

    def test_handles_invalid_json_gracefully(self):
        with patch.object(src.agents.enrichment, "search_tool") as mock_search, \
             patch.object(src.agents.enrichment, "llm") as mock_llm:

            mock_search.invoke.return_value = {
                "results": [{"content": "Some content."}]
            }
            mock_llm.invoke.return_value = mock_response("not valid json")

            result = src.agents.enrichment.enrichment_agent(make_state())

        assert result["enrichment"] is None
        assert len(result["errors"]) == 1


# ---------------------------------------------------------------------------
# Analysis agent tests
# ---------------------------------------------------------------------------

class TestAnalysisAgent:

    def state_with_enrichment(self) -> AgentState:
        state = make_state()
        state["enrichment"] = {
            "facts": FACTS,
            "web_snippets": ["s1", "s2"],
            "confidence": 0.74,
        }
        state["completed_agents"] = ["enrichment"]
        return state

    def test_populates_tags_from_enrichment(self):
        with patch.object(src.agents.analysis, "llm") as mock_llm:
            mock_llm.invoke.return_value = mock_response(json.dumps(ANALYSIS))
            result = src.agents.analysis.analysis_agent(self.state_with_enrichment())

        assert result["analysis"] is not None
        assert result["analysis"]["genre_tags"] == ["Comedy", "Drama"]
        assert result["analysis"]["confidence"] == 0.88
        assert "analysis" in result["completed_agents"]
        assert result["errors"] == []

    def test_degraded_confidence_without_enrichment(self):
        with patch.object(src.agents.analysis, "llm") as mock_llm:
            mock_llm.invoke.return_value = mock_response(json.dumps(ANALYSIS))
            result = src.agents.analysis.analysis_agent(make_state())

        assert result["analysis"]["confidence"] == 0.65

    def test_handles_missing_keys_gracefully(self):
        with patch.object(src.agents.analysis, "llm") as mock_llm:
            mock_llm.invoke.return_value = mock_response(
                json.dumps({"genre_tags": ["Comedy"]})
            )
            result = src.agents.analysis.analysis_agent(self.state_with_enrichment())

        assert result["analysis"] is None
        assert len(result["errors"]) == 1


# ---------------------------------------------------------------------------
# Sentiment agent tests
# ---------------------------------------------------------------------------

class TestSentimentAgent:

    def test_populates_sentiment_data(self):
        with patch.object(src.agents.sentiment, "search_tool") as mock_search, \
             patch.object(src.agents.sentiment, "llm") as mock_llm:

            mock_search.invoke.return_value = {
                "results": [
                    {"content": "Great show."},
                    {"content": "Season 3 was disappointing."},
                ]
            }
            mock_llm.invoke.return_value = mock_response(json.dumps(SENTIMENT))

            result = src.agents.sentiment.sentiment_agent(make_state())

        assert result["sentiment"] is not None
        assert 0 <= result["sentiment"]["score"] <= 100
        assert "sentiment" in result["completed_agents"]
        assert result["errors"] == []

    def test_score_clamped_to_100(self):
        with patch.object(src.agents.sentiment, "search_tool") as mock_search, \
             patch.object(src.agents.sentiment, "llm") as mock_llm:

            mock_search.invoke.return_value = {
                "results": [{"content": "Amazing show."}]
            }
            mock_llm.invoke.return_value = mock_response(
                json.dumps({**SENTIMENT, "score": 150})
            )

            result = src.agents.sentiment.sentiment_agent(make_state())

        assert result["sentiment"]["score"] == 100

    def test_score_clamped_to_0(self):
        with patch.object(src.agents.sentiment, "search_tool") as mock_search, \
             patch.object(src.agents.sentiment, "llm") as mock_llm:

            mock_search.invoke.return_value = {
                "results": [{"content": "Terrible show."}]
            }
            mock_llm.invoke.return_value = mock_response(
                json.dumps({**SENTIMENT, "score": -20})
            )

            result = src.agents.sentiment.sentiment_agent(make_state())

        assert result["sentiment"]["score"] == 0


# ---------------------------------------------------------------------------
# Generation agent tests
# ---------------------------------------------------------------------------

class TestGenerationAgent:

    def state_with_context(self) -> AgentState:
        state = make_state()
        state["enrichment"] = {"facts": FACTS, "web_snippets": [], "confidence": 0.74}
        state["analysis"] = {**ANALYSIS, "confidence": 0.88}
        state["completed_agents"] = ["enrichment", "analysis"]
        return state

    def test_populates_copy_fields(self):
        with patch.object(src.agents.generation, "llm") as mock_llm:
            mock_llm.invoke.return_value = mock_response(json.dumps(GENERATION))
            result = src.agents.generation.generation_agent(self.state_with_context())

        assert result["generation"] is not None
        assert result["generation"]["seo_title"] == GENERATION["seo_title"]
        assert "generation" in result["completed_agents"]
        assert result["errors"] == []

    def test_meta_description_truncated_to_155_chars(self):
        with patch.object(src.agents.generation, "llm") as mock_llm:
            mock_llm.invoke.return_value = mock_response(
                json.dumps({**GENERATION, "meta_description": "A" * 200})
            )
            result = src.agents.generation.generation_agent(self.state_with_context())

        assert len(result["generation"]["meta_description"]) <= 155

    def test_twitter_post_truncated_to_280_chars(self):
        with patch.object(src.agents.generation, "llm") as mock_llm:
            mock_llm.invoke.return_value = mock_response(
                json.dumps({**GENERATION, "twitter_post": "B" * 300})
            )
            result = src.agents.generation.generation_agent(self.state_with_context())

        assert len(result["generation"]["twitter_post"]) <= 280

    def test_higher_confidence_with_full_context(self):
        with patch.object(src.agents.generation, "llm") as mock_llm:
            mock_llm.invoke.return_value = mock_response(json.dumps(GENERATION))
            full = src.agents.generation.generation_agent(self.state_with_context())
            empty = src.agents.generation.generation_agent(make_state())

        assert full["generation"]["confidence"] > empty["generation"]["confidence"]


# ---------------------------------------------------------------------------
# Evaluation node tests
# ---------------------------------------------------------------------------

class TestEvaluationNode:

    def full_state(self) -> AgentState:
        state = make_state()
        state["enrichment"] = {"facts": FACTS, "web_snippets": [], "confidence": 0.74}
        state["analysis"] = {**ANALYSIS, "confidence": 0.88}
        state["sentiment"] = {**SENTIMENT, "confidence": 0.76}
        state["generation"] = {**GENERATION, "confidence": 0.88}
        state["completed_agents"] = ["enrichment", "analysis", "sentiment", "generation"]
        return state

    def test_populates_evaluation_data(self):
        result = src.evaluation.evaluation_node(self.full_state())
        ev = result["evaluation"]
        assert ev is not None
        assert 0 <= ev["overall_confidence"] <= 1
        assert isinstance(ev["hallucination_flags"], list)
        assert isinstance(ev["low_confidence_agents"], list)
        assert "evaluation" in result["completed_agents"]

    def test_flags_low_confidence_agent(self):
        state = self.full_state()
        state["enrichment"]["confidence"] = 0.50
        result = src.evaluation.evaluation_node(state)
        low = result["evaluation"]["low_confidence_agents"]
        assert any("enrichment" in a for a in low)

    def test_no_hallucination_flags_on_clean_output(self):
        state = self.full_state()
        state["generation"] = {
            **GENERATION,
            "seo_title": "Clean title",
            "meta_description": "Safe description.",
            "instagram_post": "Safe post.",
            "twitter_post": "Safe tweet.",
            "confidence": 0.88,
        }
        result = src.evaluation.evaluation_node(state)
        assert result["evaluation"]["hallucination_flags"] == []

    def test_weighted_confidence_calculation(self):
        result = src.evaluation.evaluation_node(self.full_state())
        expected = round(0.74*0.30 + 0.88*0.25 + 0.76*0.20 + 0.88*0.25, 2)
        assert result["evaluation"]["overall_confidence"] == expected


# ---------------------------------------------------------------------------
# End-to-end pipeline tests (fully mocked)
# ---------------------------------------------------------------------------

class TestFullPipeline:

    def test_runs_end_to_end_with_no_errors(self):
        with patch.object(src.agents.enrichment, "search_tool") as mock_es, \
             patch.object(src.agents.enrichment, "llm") as mock_el, \
             patch.object(src.agents.analysis, "llm") as mock_al, \
             patch.object(src.agents.sentiment, "search_tool") as mock_ss, \
             patch.object(src.agents.sentiment, "llm") as mock_sl, \
             patch.object(src.agents.generation, "llm") as mock_gl:

            mock_es.invoke.return_value = {
                "results": [{"content": "Test Show is a German comedy from 2006."}]
            }
            mock_el.invoke.return_value = mock_response(json.dumps(FACTS))
            mock_al.invoke.return_value = mock_response(json.dumps(ANALYSIS))
            mock_ss.invoke.return_value = {
                "results": [{"content": "Great show with strong acting."}]
            }
            mock_sl.invoke.return_value = mock_response(json.dumps(SENTIMENT))
            mock_gl.invoke.return_value = mock_response(json.dumps(GENERATION))

            from src.graph import run_pipeline
            result = run_pipeline(title="Test Show", synopsis="A test synopsis.")

        assert result["errors"] == []
        assert set(result["completed_agents"]) == {
            "enrichment", "analysis", "sentiment", "generation", "evaluation"
        }
        for key in ["enrichment", "analysis", "sentiment", "generation", "evaluation"]:
            assert result[key] is not None

    def test_pipeline_completes_even_when_enrichment_fails(self):
        with patch.object(src.agents.enrichment, "search_tool") as mock_es, \
             patch.object(src.agents.enrichment, "llm"), \
             patch.object(src.agents.analysis, "llm") as mock_al, \
             patch.object(src.agents.sentiment, "search_tool") as mock_ss, \
             patch.object(src.agents.sentiment, "llm") as mock_sl, \
             patch.object(src.agents.generation, "llm") as mock_gl:

            mock_es.invoke.return_value = {"results": []}  # enrichment fails
            mock_al.invoke.return_value = mock_response(json.dumps(ANALYSIS))
            mock_ss.invoke.return_value = {
                "results": [{"content": "Some review."}]
            }
            mock_sl.invoke.return_value = mock_response(json.dumps(SENTIMENT))
            mock_gl.invoke.return_value = mock_response(json.dumps(GENERATION))

            from src.graph import run_pipeline
            result = run_pipeline(title="Test Show")

        assert set(result["completed_agents"]) == {
            "enrichment", "analysis", "sentiment", "generation", "evaluation"
        }
        assert result["enrichment"] is None
        assert result["analysis"] is not None
        assert len(result["errors"]) == 1