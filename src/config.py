"""
config.py
----------
Centralized configuration for the RAG pipeline. Keeping every tunable
in one place makes it trivial to swap models, chunk sizes, or the
number of retrieved documents without touching pipeline logic.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "pdfs"
INDEX_DIR = ROOT_DIR / "vector_index"
BENCHMARK_PATH = ROOT_DIR / "benchmark" / "benchmark_queries.json"

DATA_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Document chunking
# ---------------------------------------------------------------------------
CHUNK_SIZE = 1000          # characters per chunk
CHUNK_OVERLAP = 150        # overlap between consecutive chunks
SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
# Small, fast, strong general-purpose sentence embedding model.
# Runs comfortably on CPU; swap for "BAAI/bge-base-en-v1.5" for higher quality.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
TOP_K = 4                 # number of chunks retrieved per query
SCORE_THRESHOLD = None    # set e.g. 0.35 to filter out weak matches (cosine distance)

# ---------------------------------------------------------------------------
# Local LLM
# ---------------------------------------------------------------------------
# Backend can be switched between "huggingface" and "ollama" without touching
# rag_pipeline.py, since llm_handler.py exposes a single get_llm() factory.
# Defaults to "ollama" since that's the primary local setup this project is
# built and demoed against. Override with RAG_LLM_BACKEND=huggingface if
# Ollama isn't installed.
LLM_BACKEND = os.environ.get("RAG_LLM_BACKEND", "ollama")

# Lightweight instruction-tuned model that runs on a single consumer GPU
# or even CPU (slower). Swap for a larger model if you have the hardware.
HF_MODEL_NAME = os.environ.get("RAG_HF_MODEL", "Qwen/Qwen2.5-1.5B-Instruct")
HF_MAX_NEW_TOKENS = 512
HF_TEMPERATURE = 0.2

# Used only when LLM_BACKEND == "ollama"
OLLAMA_MODEL_NAME = os.environ.get("RAG_OLLAMA_MODEL", "llama3")
OLLAMA_TEMPERATURE = 0.2

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = (
    "You are a precise research assistant. Answer the question using ONLY "
    "the provided context extracted from the user's PDF documents. "
    "If the answer is not contained in the context, say you don't know "
    "instead of guessing. Always be concise and cite which part of the "
    "context you used when relevant."
)
