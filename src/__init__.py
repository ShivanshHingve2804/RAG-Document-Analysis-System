"""
src package - modular components of the RAG pipeline.

    document_loader  -> load & chunk PDFs
    vector_store      -> embeddings + FAISS index management
    llm_handler       -> local LLM backend loader (HF / Ollama)
    rag_pipeline      -> retriever + LLM orchestration (LangChain LCEL)
    evaluate          -> benchmark relevance scoring
    config            -> shared settings
"""
