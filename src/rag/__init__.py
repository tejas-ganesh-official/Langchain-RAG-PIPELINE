"""A small, modular RAG pipeline over any supported file type.

Each module owns one responsibility:
    document_loader -> read files into Documents (PDF/TXT/CSV/JSON/DOCX/XLSX/PPTX/…)
    text_splitter   -> chunk Documents
    embeddings      -> EmbeddingManager (SentenceTransformer)
    vector_store    -> VectorStore (ChromaDB)
    retriever       -> RAGRetriever
    llm             -> QwenLLM (Ollama)
    pipeline        -> AdvancedRAGPipeline
"""

from .document_loader import process_all_files
from .text_splitter import split_documents
from .embeddings import EmbeddingManager
from .vector_store import VectorStore
from .retriever import RAGRetriever
from .llm import QwenLLM, build_qwen_llm
from .pipeline import AdvancedRAGPipeline

__all__ = [
    "process_all_files",
    "split_documents",
    "EmbeddingManager",
    "VectorStore",
    "RAGRetriever",
    "QwenLLM",
    "build_qwen_llm",
    "AdvancedRAGPipeline",
]