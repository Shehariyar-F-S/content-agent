# Content Intelligence Agent

A multi-agent AI pipeline that automates metadata generation for streaming content.
Built with LangGraph, Ollama (Qwen 2.5 7B), and Tavily.

## What it does

Given a show title, the pipeline runs five specialised agents in sequence:

| Agent | Job | Tool |
|---|---|---|
| Enrichment | Fetches real-world facts from the web | Tavily search |
| Analysis | Tags content: genre, mood, audience, warnings | Qwen 2.5 7B |
| Sentiment | Gauges audience reaction from live web signals | Tavily + Qwen |
| Generation | Writes SEO metadata and social media copy | Qwen 2.5 7B |
| Evaluation | Scores confidence, flags hallucinations | Pure Python |

## Quick start — local development

```bash
# 1. Install Ollama and pull the model
#    Download from https://ollama.com
ollama pull qwen2.5:7b

# 2. Clone and set up
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env — add your TAVILY_API_KEY (free at tavily.com)

# 3. Run the pipeline test
python -m tests.test_enrichment

# 4. Start the API (terminal 1)
uvicorn src.api:app --reload
# API at http://localhost:8000
# Docs at http://localhost:8000/docs

# 5. Start the UI (terminal 2)
streamlit run ui/app.py
# UI at http://localhost:8501
```

## Docker deployment

```bash
docker compose up --build
docker compose exec ollama ollama pull qwen2.5:7b
# UI  → http://localhost:8501
# API → http://localhost:8000/docs
```

## API reference

### `POST /analyse`
```json
{ "title": "Turkish for Beginners", "synopsis": "Optional" }
```

### `GET /health`
Returns `{"status": "ok"}` if the API is running.

## Project structure

```
content-agent/
├── src/
│   ├── state.py          ← shared state schema
│   ├── graph.py          ← LangGraph orchestrator
│   ├── evaluation.py     ← confidence scoring + hallucination detection
│   ├── api.py            ← FastAPI backend
│   └── agents/
│       ├── enrichment.py
│       ├── analysis.py
│       ├── sentiment.py
│       └── generation.py
├── ui/
│   └── app.py            ← Streamlit frontend
├── tests/
│   └── test_enrichment.py
├── Dockerfile
├── Dockerfile.ui
├── docker-compose.yml
├── requirements.txt
└── .env.example
```

## Cost

| Tool | Cost |
|---|---|
| Ollama + Qwen 2.5 7B | Free — runs locally |
| Tavily | Free tier — 1000 searches/month |
| LangSmith | Free tier — optional observability |