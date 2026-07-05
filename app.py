"""
app.py
------
Streamlit front-end for the RAG PDF Q&A system.

Run with:
    streamlit run app.py

Flow:
1. Upload one or more PDFs (or use the pre-loaded data/pdfs/ folder).
2. Build (or load a cached) FAISS index.
3. Ask questions in a chat-style interface; answers are grounded in
   the retrieved chunks, with sources shown for transparency.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

import streamlit as st

from src import config
from src.document_loader import load_pdfs, split_documents
from src.vector_store import build_vector_store, save_vector_store, index_exists, load_vector_store
from src.rag_pipeline import RAGPipeline

st.set_page_config(page_title="PDF RAG Q&A", page_icon="📄", layout="wide")

st.title("📄 AI-Powered Document Analysis (RAG)")
st.caption(
    "Local LLM + FAISS semantic search over your PDFs — LangChain-based "
    "retrieval-augmented generation, fully offline once models are downloaded."
)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# ---------------------------------------------------------------------------
# Sidebar: index management
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Index Setup")

    uploaded_files = st.file_uploader(
        "Upload PDFs", type=["pdf"], accept_multiple_files=True
    )

    if uploaded_files:
        config.DATA_DIR.mkdir(parents=True, exist_ok=True)
        for uf in uploaded_files:
            out_path = config.DATA_DIR / uf.name
            out_path.write_bytes(uf.getbuffer())
        st.success(f"Saved {len(uploaded_files)} file(s) to {config.DATA_DIR}")

    col1, col2 = st.columns(2)
    build_clicked = col1.button("🔨 Build / Rebuild Index")
    load_clicked = col2.button("📂 Load Existing Index")

    if build_clicked:
        with st.spinner("Loading PDFs, chunking, and embedding..."):
            docs = load_pdfs()
            chunks = split_documents(docs)
            store = build_vector_store(chunks)
            save_vector_store(store)
            st.session_state.pipeline = RAGPipeline(store)
        st.success("Index built and pipeline ready.")

    if load_clicked:
        if index_exists():
            with st.spinner("Loading saved index and local LLM..."):
                store = load_vector_store()
                st.session_state.pipeline = RAGPipeline(store)
            st.success("Existing index loaded and pipeline ready.")
        else:
            st.error("No saved index found. Build one first.")

    st.divider()
    st.caption(
        f"**Embedding model:** {config.EMBEDDING_MODEL_NAME}\n\n"
        f"**LLM backend:** {config.LLM_BACKEND}\n\n"
        f"**LLM model:** "
        f"{config.HF_MODEL_NAME if config.LLM_BACKEND == 'huggingface' else config.OLLAMA_MODEL_NAME}\n\n"
        f"**Top-K retrieved chunks:** {config.TOP_K}"
    )

# ---------------------------------------------------------------------------
# Main chat area
# ---------------------------------------------------------------------------
if st.session_state.pipeline is None:
    st.info("👈 Build or load an index from the sidebar to start asking questions.")
else:
    for turn in st.session_state.chat_history:
        with st.chat_message(turn["role"]):
            st.markdown(turn["content"])
            if turn.get("sources"):
                with st.expander("Sources"):
                    st.text(turn["sources"])

    question = st.chat_input("Ask a question about your documents...")

    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving relevant chunks and generating an answer..."):
                result = st.session_state.pipeline.query(question)
            st.markdown(result.answer)
            sources_text = result.formatted_sources()
            with st.expander("Sources"):
                st.text(sources_text)

        st.session_state.chat_history.append(
            {"role": "assistant", "content": result.answer, "sources": sources_text}
        )
