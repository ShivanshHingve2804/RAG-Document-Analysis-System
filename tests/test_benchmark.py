"""Tests for benchmark queries file."""

import json
from pathlib import Path


def test_benchmark_file_is_valid_json():
    """benchmark_queries.json should be valid JSON."""
    path = Path(__file__).parent.parent / "benchmark" / "benchmark_queries.json"
    with open(path) as f:
        data = json.load(f)
    assert isinstance(data, list)


def test_benchmark_queries_have_required_fields():
    """Each benchmark query should have question, expected_keywords, and reference_answer."""
    path = Path(__file__).parent.parent / "benchmark" / "benchmark_queries.json"
    with open(path) as f:
        data = json.load(f)

    for i, entry in enumerate(data):
        assert "question" in entry, f"Entry {i} missing 'question'"
        assert "expected_keywords" in entry, f"Entry {i} missing 'expected_keywords'"
        assert "reference_answer" in entry, f"Entry {i} missing 'reference_answer'"
        assert len(entry["question"]) > 0, f"Entry {i} has empty question"
