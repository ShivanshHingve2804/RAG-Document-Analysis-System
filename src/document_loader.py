"""
document_loader.py
-------------------
Handles ingestion of raw PDFs and splitting them into retrieval-sized
chunks. Kept isolated from embeddings/LLM logic so the ingestion step
can be tested, swapped (e.g. to add .docx or .txt support), or reused
by the Streamlit app without pulling in model dependencies.
"""

from pathlib import Path
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from src import config


def load_pdfs(pdf_dir: Path = config.DATA_DIR) -> List[Document]:
    """
    Load every PDF in `pdf_dir` into LangChain Document objects.
    Each page becomes its own Document, tagged with source + page metadata.
    """
    pdf_dir = Path(pdf_dir)
    pdf_paths = sorted(pdf_dir.glob("*.pdf"))

    if not pdf_paths:
        raise FileNotFoundError(
            f"No PDFs found in {pdf_dir}. Drop your source PDFs there first."
        )

    documents: List[Document] = []
    for path in pdf_paths:
        loader = PyPDFLoader(str(path))
        pages = loader.load()  # one Document per page
        for page in pages:
            page.metadata["source"] = path.name
        documents.extend(pages)

    print(f"Loaded {len(documents)} pages from {len(pdf_paths)} PDF(s).")
    return documents


def split_documents(
    documents: List[Document],
    chunk_size: int = config.CHUNK_SIZE,
    chunk_overlap: int = config.CHUNK_OVERLAP,
) -> List[Document]:
    """
    Split page-level Documents into smaller overlapping chunks that are
    better sized for embedding + retrieval precision.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=config.SEPARATORS,
    )
    chunks = splitter.split_documents(documents)

    # Give each chunk a stable id for traceability in evaluation/citations
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i

    print(f"Split into {len(chunks)} chunks "
          f"(chunk_size={chunk_size}, overlap={chunk_overlap}).")
    return chunks


def load_and_split(pdf_dir: Path = config.DATA_DIR) -> List[Document]:
    """Convenience wrapper: load PDFs and split them in one call."""
    docs = load_pdfs(pdf_dir)
    return split_documents(docs)


if __name__ == "__main__":
    chunks = load_and_split()
    print("\nSample chunk:\n")
    print(chunks[0].page_content[:300])
    print("\nMetadata:", chunks[0].metadata)
