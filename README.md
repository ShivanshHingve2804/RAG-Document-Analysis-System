# AI-Powered Document Analysis System (RAG)

**Python · LangChain · FAISS · Local LLM · Streamlit**

An end-to-end retrieval-augmented generation (RAG) pipeline that combines a
locally-hosted LLM with a FAISS vector store for semantic search and Q&A
over PDF documents — no external API calls, fully self-contained.

---

## Demo

[**▶ Watch the 60-second Demo Video**]((https://youtu.be/488gMtiZQfQ))

---

## Architecture

```
PDF files
   │
   ▼
[document_loader.py] ── load pages, split into overlapping chunks
   │
   ▼
[vector_store.py] ── embed chunks (sentence-transformers) → FAISS index
   │
   ▼
[rag_pipeline.py] ── retrieve top-k chunks → build prompt → local LLM → answer
   │                        ▲
   │                        │
[llm_handler.py] ───────────┘  (Hugging Face pipeline or Ollama backend)
   │
   ▼
[evaluate.py] ── score retrieval relevance against benchmark_queries.json
   │
   ▼
[app.py] ── Streamlit chat UI (upload PDFs, build index, ask questions)
```

## Project layout

```
rag_pdf_qa/
├── notebooks/
│   └── RAG_Pipeline_Development.ipynb   # exploratory build of the full pipeline
├── src/
│   ├── config.py            # all tunables in one place
│   ├── document_loader.py   # PDF loading + chunking
│   ├── vector_store.py      # embeddings + FAISS index management
│   ├── llm_handler.py       # local LLM backend (HF / Ollama)
│   ├── rag_pipeline.py      # retriever + LLM orchestration (LangChain LCEL)
│   └── evaluate.py          # benchmark relevance scoring
├── scripts/
│   └── create_sample_data.py  # generates 2 sample PDFs to test the pipeline
├── benchmark/
│   └── benchmark_queries.json # 10 labeled Q&A pairs for evaluation
├── data/pdfs/                 # drop your source PDFs here
├── vector_index/              # persisted FAISS index (generated)
├── app.py                     # Streamlit UI
└── requirements.txt
```

## Setup

### 1. Install Ollama and pull the model (primary backend)

```bash
# Install from https://ollama.com/download, then:
ollama pull llama3
ollama serve                     # keep running in a separate terminal
```

### 2. Install Python dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

By default the pipeline connects to your local Ollama server running
`llama3` — nothing else to configure. The model weights themselves are
never part of this repo; Ollama manages and stores them separately on
your machine (`ollama pull` handles that).

**No Ollama?** Fall back to a small Hugging Face model instead:

```bash
export RAG_LLM_BACKEND=huggingface
```

The first run then downloads `Qwen/Qwen2.5-1.5B-Instruct` (~3GB) and
`sentence-transformers/all-MiniLM-L6-v2` (~90MB), cached locally afterward
so the pipeline still runs fully offline from then on.

## Usage

### 1. Add documents
Drop your PDFs into `data/pdfs/`, or generate sample ones to try it out:

```bash
python scripts/create_sample_data.py
```

### 2. Explore in the notebook
Open `notebooks/RAG_Pipeline_Development.ipynb` to walk through loading,
chunking, embedding, LLM setup, querying, and benchmarking step by step.

### 3. Run the modular pipeline from the CLI

```bash
python -m src.document_loader     # loads & chunks PDFs (sanity check)
python -m src.rag_pipeline        # builds/loads index, runs one sample query
python -m src.evaluate            # runs the full benchmark and prints a report
```

### 4. Launch the Streamlit app

```bash
streamlit run app.py
```

Upload PDFs (or use the ones already in `data/pdfs/`), click **Build Index**,
and start asking questions. Each answer shows the source chunks it was
grounded in.

## Evaluation methodology

`benchmark/benchmark_queries.json` contains hand-labeled queries, each with:
- `expected_keywords` — terms that should appear in relevant retrieved chunks
- `reference_answer` — a short ground-truth answer

For every query, `src/evaluate.py` retrieves the top-k chunks and scores them
on two signals:

1. **Keyword coverage** — fraction of `expected_keywords` present in the
   retrieved text (≥ 60% passes).
2. **Semantic similarity** — cosine similarity between the embedding of
   `reference_answer` and each retrieved chunk, using the max across the
   top-k (≥ 0.45 passes).

A query counts as relevant if either signal clears its threshold. The
percentage of benchmark queries marked relevant is the reported relevance
score — this two-signal design avoids the brittleness of pure keyword
matching while keeping the metric fully reproducible (no LLM-as-judge
randomness).

To reproduce the reported >90% relevance figure on your own documents,
replace `benchmark/benchmark_queries.json` with questions relevant to your
own PDFs and re-run `python -m src.evaluate`.

## Design notes / talking points

- **Chunking**: `RecursiveCharacterTextSplitter` with 1000-char chunks and
  150-char overlap balances retrieval precision against losing context at
  chunk boundaries.
- **Embeddings**: `all-MiniLM-L6-v2` (384-dim) was chosen for speed on CPU;
  swapping to `BAAI/bge-base-en-v1.5` is a one-line config change for higher
  recall at the cost of latency.
- **Retriever**: FAISS similarity search (cosine, via normalized embeddings)
  with `k=4`, configurable in `src/config.py`.
- **LLM backend abstraction**: `src/llm_handler.py` exposes a single
  `get_llm()` factory, so the rest of the pipeline is agnostic to whether
  the model is served via `transformers` or Ollama.
- **Grounded generation**: the system prompt instructs the model to answer
  only from retrieved context and say "I don't know" otherwise, reducing
  hallucination risk — a natural discussion point on RAG faithfulness.

## Requirements

See `requirements.txt`. Key dependencies: `langchain`, `langchain-community`,
`langchain-huggingface`, `faiss-cpu`, `sentence-transformers`, `transformers`,
`torch`, `pypdf`, `streamlit`.
