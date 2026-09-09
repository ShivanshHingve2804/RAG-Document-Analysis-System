"""CLI entry point for RAG Document Analysis System.

Provides commands to build indexes, query documents, and run benchmarks
from the terminal — no Streamlit UI required.
"""

import argparse
import sys
import json
from pathlib import Path


def cmd_index(args):
    """Build or rebuild the FAISS vector index from PDFs."""
    from src import config
    from src.document_loader import load_pdfs, split_documents
    from src.vector_store import build_vector_store, save_vector_store

    pdf_dir = args.pdf_dir or str(config.DATA_DIR)
    print(f"📂 Loading PDFs from: {pdf_dir}")

    docs = load_pdfs(pdf_dir)
    if not docs:
        print("❌ No PDF documents found.", file=sys.stderr)
        return 1

    print(f"📄 Loaded {len(docs)} pages")

    chunks = split_documents(docs)
    print(f"✂️  Split into {len(chunks)} chunks (size={config.CHUNK_SIZE}, overlap={config.CHUNK_OVERLAP})")

    print("🔨 Building FAISS index with embeddings...")
    store = build_vector_store(chunks)
    save_vector_store(store)
    print(f"✅ Index saved to {config.INDEX_DIR}")
    return 0


def cmd_query(args):
    """Query the RAG pipeline with a question."""
    from src.rag_pipeline import build_pipeline_from_index

    question = args.question
    print(f"❓ Question: {question}\n")

    pipeline = build_pipeline_from_index(top_k=args.top_k)
    result = pipeline.query(question)

    if args.format == "json":
        output = {
            "question": result.question,
            "answer": result.answer,
            "sources": [
                {
                    "source": doc.metadata.get("source", "unknown"),
                    "page": doc.metadata.get("page", "?"),
                    "snippet": doc.page_content[:200],
                }
                for doc in result.sources
            ],
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"💡 Answer:\n{result.answer}\n")
        print(f"📚 Sources:\n{result.formatted_sources()}")

    return 0


def cmd_evaluate(args):
    """Run the benchmark evaluation suite."""
    from src.evaluate import run_benchmark
    run_benchmark()
    return 0


def cmd_serve(args):
    """Launch the Streamlit web UI."""
    import subprocess
    app_path = Path(__file__).parent / "app.py"
    print(f"🚀 Launching Streamlit UI...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(app_path)])
    return 0


def main():
    parser = argparse.ArgumentParser(
        prog="ragdoc",
        description="RAG Document Analysis System — CLI for PDF Q&A with local LLMs",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # index
    idx_parser = subparsers.add_parser("index", help="Build FAISS vector index from PDFs")
    idx_parser.add_argument("--pdf-dir", help="Directory containing PDF files (default: data/pdfs/)")

    # query
    q_parser = subparsers.add_parser("query", help="Ask a question about your documents")
    q_parser.add_argument("question", help="The question to ask")
    q_parser.add_argument("--top-k", type=int, default=4, help="Number of chunks to retrieve (default: 4)")
    q_parser.add_argument("--format", choices=["text", "json"], default="text", help="Output format")

    # evaluate
    subparsers.add_parser("evaluate", help="Run benchmark evaluation")

    # serve
    subparsers.add_parser("serve", help="Launch the Streamlit web UI")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(0)

    handlers = {
        "index": cmd_index,
        "query": cmd_query,
        "evaluate": cmd_evaluate,
        "serve": cmd_serve,
    }
    sys.exit(handlers[args.command](args))


if __name__ == "__main__":
    main()
