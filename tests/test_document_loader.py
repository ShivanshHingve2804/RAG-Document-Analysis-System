"""Tests for the document loader module."""

import os
import tempfile
from unittest.mock import patch, MagicMock
from src.document_loader import split_documents


def test_split_documents_creates_chunks():
    """split_documents should produce chunks from a list of Documents."""
    from langchain_core.documents import Document

    docs = [
        Document(page_content="A" * 2000, metadata={"source": "test.pdf", "page": 0}),
        Document(page_content="B" * 500, metadata={"source": "test.pdf", "page": 1}),
    ]
    chunks = split_documents(docs)
    # First doc is 2000 chars with 1000-char chunks, so should produce 2+ chunks
    assert len(chunks) >= 2


def test_split_preserves_metadata():
    """Chunks should retain the source metadata from original documents."""
    from langchain_core.documents import Document

    docs = [Document(page_content="Hello world " * 200, metadata={"source": "report.pdf", "page": 5})]
    chunks = split_documents(docs)
    for chunk in chunks:
        assert chunk.metadata["source"] == "report.pdf"


def test_split_empty_input():
    """Splitting an empty list should return an empty list."""
    result = split_documents([])
    assert result == []
