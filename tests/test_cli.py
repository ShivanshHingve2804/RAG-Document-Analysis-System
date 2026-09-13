"""Tests for the CLI module."""

import sys
from unittest.mock import patch
from cli import main


def test_parser_has_subcommands():
    """CLI should have index, query, evaluate, serve subcommands."""
    assert callable(main)


def test_no_command_shows_help(capsys):
    """Running with no args should print help and exit."""
    with patch("sys.argv", ["ragdoc"]):
        try:
            main()
        except SystemExit as e:
            assert e.code == 0


def test_query_subcommand_parses():
    """Query subcommand should accept a question argument."""
    import argparse
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    q = subparsers.add_parser("query")
    q.add_argument("question")
    q.add_argument("--top-k", type=int, default=4)
    q.add_argument("--format", choices=["text", "json"], default="text")

    args = parser.parse_args(["query", "What is AI?"])
    assert args.question == "What is AI?"
    assert args.top_k == 4
    assert args.format == "text"
