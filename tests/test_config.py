"""Tests for the configuration module."""

from src import config
from pathlib import Path


def test_config_paths_exist():
    """Config paths should be Path objects."""
    assert isinstance(config.ROOT_DIR, Path)
    assert isinstance(config.DATA_DIR, Path)
    assert isinstance(config.INDEX_DIR, Path)


def test_chunk_settings_are_reasonable():
    """Chunk settings should have sensible defaults."""
    assert config.CHUNK_SIZE > 0
    assert config.CHUNK_OVERLAP > 0
    assert config.CHUNK_OVERLAP < config.CHUNK_SIZE


def test_top_k_positive():
    assert config.TOP_K > 0


def test_embedding_model_name_set():
    assert config.EMBEDDING_MODEL_NAME
    assert isinstance(config.EMBEDDING_MODEL_NAME, str)


def test_llm_backend_valid():
    assert config.LLM_BACKEND in ("ollama", "huggingface")


def test_system_prompt_nonempty():
    assert config.SYSTEM_PROMPT
    assert len(config.SYSTEM_PROMPT) > 20
