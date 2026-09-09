"""
rag_pipeline.py
----------------
Wires the retriever (FAISS) and the local LLM together into a single
retrieval-augmented generation chain, and exposes a simple `.query()`
API that returns both the answer and the source chunks used, so the
UI layer (Streamlit / notebook) can show citations.
"""

from dataclasses import dataclass, field
from typing import List

from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src import config
from src.vector_store import FAISS
from src.llm_handler import get_llm


@dataclass
class RAGResult:
    question: str
    answer: str
    sources: List[Document] = field(default_factory=list)

    def formatted_sources(self) -> str:
        lines = []
        for doc in self.sources:
            src = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "?")
            snippet = doc.page_content[:150].replace("\n", " ")
            lines.append(f"- {src} (page {page}): \"{snippet}...\"")
        return "\n".join(lines)


def _format_docs(docs: List[Document]) -> str:
    return "\n\n".join(
        f"[Source: {d.metadata.get('source', 'unknown')} | "
        f"page {d.metadata.get('page', '?')}]\n{d.page_content}"
        for d in docs
    )


class RAGPipeline:
    """
    End-to-end retrieval-augmented Q&A pipeline over a FAISS vector store.
    """

    def __init__(self, vector_store: FAISS, top_k: int = config.TOP_K):
        self.vector_store = vector_store
        self.retriever = vector_store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": top_k},
        )
        self.llm = get_llm()
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", config.SYSTEM_PROMPT),
            ("human", "Context:\n{context}\n\nQuestion: {question}\n\nAnswer:"),
        ])
        self._chain = self.prompt | self.llm | StrOutputParser()

    def retrieve(self, question: str) -> List[Document]:
        """Return the raw retrieved chunks for a question (no generation)."""
        return self.retriever.invoke(question)

    def query(self, question: str) -> RAGResult:
        """Run the full retrieve -> augment -> generate pipeline."""
        sources = self.retrieve(question)
        context = _format_docs(sources)
        answer = self._chain.invoke({"context": context, "question": question})
        return RAGResult(question=question, answer=answer.strip(), sources=sources)


def build_pipeline_from_index(top_k: int = config.TOP_K) -> RAGPipeline:
    """Convenience factory: load the persisted FAISS index and build a pipeline."""
    from src.vector_store import load_vector_store

    store = load_vector_store()
    return RAGPipeline(store, top_k=top_k)


if __name__ == "__main__":
    pipeline = build_pipeline_from_index()
    result = pipeline.query("What is this document about?")
    print("\nANSWER:\n", result.answer)
    print("\nSOURCES:\n", result.formatted_sources())
