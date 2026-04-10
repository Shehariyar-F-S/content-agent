# 🎬 Content Intelligence Agent

<p align="center">
  <img src="https://github.com/Shehariyar-F-S/content-agent/actions/workflows/ci.yml/badge.svg" alt="CI" />
  <img src="https://img.shields.io/badge/Python-3.11-blue?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/LangGraph-0.2-purple?logo=chainlink&logoColor=white" />
  <img src="https://img.shields.io/badge/Ollama-Qwen2.5_7B-green" />
  <img src="https://img.shields.io/badge/FastAPI-0.115-teal?logo=fastapi&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-1.40-red?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/Docker-ready-blue?logo=docker&logoColor=white" />
</p>

<p align="center">
  A production-grade multi-agent AI pipeline that automates metadata generation
  for streaming content — replacing hours of manual editorial work with a
  reliable, auditable system costing $0.004 per run.
</p>

---

## The problem

Streaming platforms like Joyn have thousands of shows. Every show needs genre
tags, mood labels, audience classification, SEO metadata, and social media copy.
Done manually, one show takes ~45 minutes. At scale this becomes thousands of
editor-hours per month — expensive, slow, and inconsistent.

A single LLM call does not solve it: no live data, no audit trail, no way to
know when it hallucinated.

## The solution

Five specialised agents run in sequence, each doing one job:

```
Input title
     │
     ▼
┌─────────────┐     Tavily       ┌──────────────────────────────────────┐
│ Enrichment  │──── web search ──│  Facts: platform, seasons, premiere  │
└─────────────┘                  └──────────────────────────────────────┘
     │
     ▼
┌─────────────┐     Groq API    ┌──────────────────────────────────────┐
│  Analysis   │────(Llama 3.1 8B) ────│  Tags: genre · mood · audience       │
└─────────────┘                  └──────────────────────────────────────┘
     │
     ▼
┌─────────────┐     Tavily       ┌──────────────────────────────────────┐
│  Sentiment  │──── web search ──│  Score: 75/100 · positives · issues  │
└─────────────┘                  └──────────────────────────────────────┘
     │
     ▼
┌─────────────┐     Groq API     ┌──────────────────────────────────────┐
│ Generation  │────(Llama 3.1 8B)────│  SEO title · meta desc · social copy │
└─────────────┘                  └──────────────────────────────────────┘
     │
     ▼
┌─────────────┐     Python       ┌──────────────────────────────────────┐
│ Evaluation  │──── logic ───────│  Confidence · hallucination flags    │
└─────────────┘                  └──────────────────────────────────────┘
     │
     ▼
Structured JSON output with per-agent confidence scores
```

## What makes this production-grade

- **Typed state schema** — all agents share a single `TypedDict` state. No silent key errors between agents.
- **Graceful degradation** — if any agent fails, the pipeline logs to `state["errors"]` and continues. Always reaches evaluation.
- **Confidence scoring** — each agent returns a confidence score. Evaluation aggregates into an overall pipeline score.
- **Hallucination detection** — generation copy is cross-checked against enrichment facts. Contradictions are flagged.
- **Fully observable** — LangSmith tracing captures every LLM call, input, output, and latency.

## Live demo output

Input: `"Turkish for Beginners"`

```
Overall confidence    80%
Hallucination flags   0
Low confidence agents enrichment (74%), sentiment (68%)

Genre tags     Comedy · Drama
Mood tags      Educational · Lighthearted
Audience       Beginners · Families

Sentiment      75/100 — Generally positive. Engaging narrative and
               relatable characters. Decline in later seasons noted.

SEO title      Turkish for Beginners: Comedy, Drama & Learning Made Easy
Meta desc      Join us on a lighthearted journey to learn Turkish with
               our comedy-drama series! Perfect for families. (122 chars)
```

## Tech stack

| Layer | Tool | Why |
|---|---|---|
| Agent framework | LangGraph 0.2 | Typed shared state, proper state machine |
| LLM | Ollama + Qwen 2.5 7B | Local, free, unlimited, swap-ready |
| Web search | Tavily | LLM-optimised snippets, reliable free tier |
| API | FastAPI | Pydantic models, auto-generated Swagger docs |
| Frontend | Streamlit | Demo-ready UI |
| Observability | LangSmith | Tracing + confidence (optional) |
| Deployment | Docker + Compose | One command to run everything |

## Quick start

```bash
# Get free Groq API key at console.groq.com
# Add GROQ_API_KEY to .env

# 2. Set up
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # add TAVILY_API_KEY from tavily.com

# 3. Test the pipeline
python -m tests.test_enrichment

# 4. Start API (terminal 1)
python -m uvicorn src.api:app --reload
# Docs → http://localhost:8000/docs

# 5. Start UI (terminal 2)
streamlit run ui/app.py
# UI → http://localhost:8501
```

## Docker

```bash
# Option 1 — pull pre-built image from GHCR (fastest)
docker pull ghcr.io/shehariyar-f-s/content-agent:latest
docker compose up

# Option 2 — build locally
docker compose up --build
docker compose exec ollama ollama pull qwen2.5:7b
```

UI at `localhost:8501` · API docs at `localhost:8000/docs`

## Project structure

```
├── src/
│   ├── state.py          ← shared TypedDict state schema
│   ├── graph.py          ← LangGraph orchestrator
│   ├── evaluation.py     ← confidence scoring + hallucination detection
│   ├── api.py            ← FastAPI backend
│   └── agents/
│       ├── enrichment.py ← Tavily web search
│       ├── analysis.py   ← content classification
│       ├── sentiment.py  ← audience sentiment
│       └── generation.py ← SEO + social copy
├── ui/app.py             ← Streamlit frontend
├── docker-compose.yml
└── requirements.txt
```

## API

```bash
curl -X POST http://localhost:8000/analyse \
  -H "Content-Type: application/json" \
  -d '{"title": "Dark", "synopsis": "German sci-fi thriller."}'
```

Full docs at `http://localhost:8000/docs`.

## Cost

| Tool | Cost |
|---|---|
| Groq API (Llama 3.1 8B) | Free tier — 14,400 req/day |
| Tavily | Free — 1000 searches/month (~200 runs) |
| LangSmith | Free tier — optional |

## Adaptability

Same architecture, different domain:
- **Customer service** — swap content for tickets, generation for response drafts
- **Medical documents** — swap shows for clinical reports, tags for ICD codes
- **Legal** — swap sentiment for risk scoring, generation for clause summaries

---

© 2026 Shehariyar Firdous.Shaikh. · Content Intelligence Agent · Built as a portfolio project demonstrating production-grade multi-agent AI engineering.