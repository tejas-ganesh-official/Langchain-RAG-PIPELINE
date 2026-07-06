"""Retrieval: query the vector store for relevant document chunks."""

from typing import Any, Dict, List

from .embeddings import EmbeddingManager
from .vector_store import VectorStore


class RAGRetriever:
    """Handles query-based retrieval from the vector store."""

    def __init__(self, vector_store: VectorStore, embedding_manager: EmbeddingManager):
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        score_threshold: float = 0.0,
    ) -> List[Dict[str, Any]]:
        print(f"\n[Retriever] Query: '{query}'")
        print(f"[Retriever] Collection has {self.vector_store.collection.count()} chunks")

        query_embedding = self.embedding_manager.generate_embeddings([query])[0]

        try:
            n = min(top_k, self.vector_store.collection.count())
            if n == 0:
                print("[Retriever] Collection is empty — nothing to search.")
                return []

            results = self.vector_store.collection.query(
                query_embeddings=[query_embedding.tolist()],
                n_results=n,
            )

            retrieved_docs: List[Dict[str, Any]] = []

            if results["documents"] and results["documents"][0]:
                for i, (doc_id, document, metadata, distance) in enumerate(zip(
                    results["ids"][0],
                    results["documents"][0],
                    results["metadatas"][0],
                    results["distances"][0],
                )):
                    similarity_score = 1 - distance
                    print(f"  [{similarity_score:.3f}] {metadata.get('source_file', '?')} — {document[:80]!r}")

                    if similarity_score >= score_threshold:
                        retrieved_docs.append({
                            "id": doc_id,
                            "content": document,
                            "metadata": metadata,
                            "similarity_score": similarity_score,
                            "distance": distance,
                            "rank": i + 1,
                        })

                print(f"[Retriever] Passing {len(retrieved_docs)} chunk(s) to LLM")
            else:
                print("[Retriever] No documents returned by ChromaDB")

            return retrieved_docs

        except Exception as e:
            print(f"[Retriever] Error: {e}")
            return []