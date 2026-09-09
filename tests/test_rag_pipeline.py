"""Tests for the RAG pipeline module."""

from src.rag_pipeline import RAGResult
from langchain_core.documents import Document


def test_rag_result_formatted_sources():
    """RAGResult.formatted_sources should produce readable source citations."""
    docs = [
        Document(page_content="The quick brown fox jumps over the lazy dog.", metadata={"source": "test.pdf", "page": 1}),
        Document(page_content="Machine learning is a subset of AI.", metadata={"source": "ml.pdf", "page": 3}),
    ]
    result = RAGResult(question="What is ML?", answer="A subset of AI", sources=docs)
    formatted = result.formatted_sources()

    assert "test.pdf" in formatted
    assert "ml.pdf" in formatted
    assert "page 1" in formatted
    assert "page 3" in formatted


def test_rag_result_empty_sources():
    """RAGResult with no sources should return empty string."""
    result = RAGResult(question="test", answer="answer", sources=[])
    assert result.formatted_sources() == ""


def test_rag_result_dataclass_fields():
    """RAGResult should have the expected fields."""
    result = RAGResult(question="q", answer="a")
    assert result.question == "q"
    assert result.answer == "a"
    assert result.sources == []
