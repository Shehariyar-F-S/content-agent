"""
tests/conftest.py

Sets required environment variables before any test module is imported.
This prevents Tavily and Ollama from failing during module-level instantiation
when running tests without real API keys.

pytest loads conftest.py before collecting any test files, so env vars
are set before `import src.agents.enrichment` runs in test_pipeline.py.
"""

import os
from dotenv import load_dotenv

# Load real .env first — real keys take priority
load_dotenv()

# Set dummy fallbacks for CI where real keys don't exist
os.environ.setdefault("TAVILY_API_KEY", "dummy-for-testing")
os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434")
os.environ.setdefault("OLLAMA_MODEL", "qwen2.5:7b")
os.environ.setdefault("GROQ_API_KEY", "dummy-for-testing")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")