"""
vector_store.py
----------------
Wraps FAISS index creation, persistence, and loading behind a small
API so the rest of the pipeline never has to know embedding details.
"""

from pathlib import Path
from typing import List, Optional

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from src import config

_embeddings_cache: Optional[HuggingFaceEmbeddings] = None


def get_embedding_model() -> HuggingFaceEmbeddings:
    """
    Lazily instantiate (and cache) the sentence-embedding model so it's
    only loaded into memory once per process.
    """
    global _embeddings_cache
    if _embeddings_cache is None:
        _embeddings_cache = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL_NAME,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings_cache


def build_vector_store(chunks: List[Document]) -> FAISS:
    """Embed all chunks and build a fresh in-memory FAISS index."""
    embeddings = get_embedding_model()
    store = FAISS.from_documents(chunks, embeddings)
    print(f"Built FAISS index with {store.index.ntotal} vectors.")
    return store


def save_vector_store(store: FAISS, index_dir: Path = config.INDEX_DIR) -> None:
    """Persist the FAISS index + docstore to disk for reuse across sessions."""
    index_dir = Path(index_dir)
    index_dir.mkdir(parents=True, exist_ok=True)
    store.save_local(str(index_dir))
    print(f"Saved vector index to {index_dir}")


def load_vector_store(index_dir: Path = config.INDEX_DIR) -> FAISS:
    """Load a previously persisted FAISS index from disk."""
    embeddings = get_embedding_model()
    store = FAISS.load_local(
        str(index_dir),
        embeddings,
        allow_dangerous_deserialization=True,  # safe: we created this index ourselves
    )
    print(f"Loaded FAISS index with {store.index.ntotal} vectors from {index_dir}")
    return store


def index_exists(index_dir: Path = config.INDEX_DIR) -> bool:
    index_dir = Path(index_dir)
    return (index_dir / "index.faiss").exists() and (index_dir / "index.pkl").exists()


def get_or_build_vector_store(chunks: Optional[List[Document]] = None) -> FAISS:
    """
    Convenience helper used by the Streamlit app:
    - If an index already exists on disk, load it.
    - Otherwise build one from `chunks` and persist it.
    """
    if index_exists():
        return load_vector_store()
    if chunks is None:
        raise ValueError("No existing index found and no chunks provided to build one.")
    store = build_vector_store(chunks)
    save_vector_store(store)
    return store
