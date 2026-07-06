"""Advanced RAG pipeline: streaming, citations, history, summarization."""

import time
from typing import Any, Dict, Generator, List

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage


class AdvancedRAGPipeline:
    """Retrieval-augmented generation with conversation memory and streaming."""

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.history = []  # [{role, content}]

    def _build_prompt(self, question: str, context: str) -> str:
        return (
            f"Use the following context to answer the question thoroughly and clearly.\n"
            f"Draw on all relevant details from the context. Explain the answer well — break it down, "
            f"give examples if helpful, and structure it clearly. Do not pad, but do not cut important detail.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\nAnswer:"
        )

    def _get_sources(self, results) -> list:
        return [
            {
                "source": doc["metadata"].get(
                    "source_file", doc["metadata"].get("source", "unknown")
                ),
                "page": doc["metadata"].get("page", "unknown"),
                "score": doc["similarity_score"],
                "preview": doc["content"][:120] + "...",
            }
            for doc in results
        ]

    def _build_messages(self, final_message: str) -> list:
        """Build full message list: system + conversation history + current question."""
        messages = [
            SystemMessage(content=(
                "You are a knowledgeable and thorough assistant. "
                "When answering, explain your points clearly and in depth — break down complex topics, "
                "give examples where they help understanding, and walk the user through your reasoning. "
                "Do not pad with filler, but never truncate important detail either. "
                "If the answer has multiple parts or steps, structure them clearly using natural prose or lists. "
                "If working from provided context, use all relevant details — do not summarize so aggressively that meaning is lost. "
                "If the user refers to something from earlier in the conversation, use that context to answer correctly."
            ))
        ]
        for turn in self.history:
            messages.append(HumanMessage(content=turn["question"]))
            messages.append(AIMessage(content=turn["answer"]))
        messages.append(HumanMessage(content=final_message))
        return messages

    def stream_query(self, question: str, top_k: int = 3, min_score: float = 0.0) -> Generator[str, None, None]:
        """Stream the answer token by token.

        Yields:
            First chunk: __meta__:<from_documents>:<sources>
            Subsequent chunks: answer tokens
        """
        results = self.retriever.retrieve(question, top_k=top_k, score_threshold=min_score)

        if not results:
            from_documents = False
            sources = []
            # Pass conversation history + bare question
            messages = self._build_messages(question)
        else:
            from_documents = True
            sources = self._get_sources(results)
            context = "\n\n".join([doc["content"] for doc in results])
            # Pass conversation history + question grounded with context
            messages = self._build_messages(self._build_prompt(question, context))

        source_names = ",".join(s["source"] for s in sources)
        yield f"__meta__:{from_documents}:{source_names}\n"

        full_answer = ""
        for chunk in self.llm.stream(messages):
            token = chunk.content
            full_answer += token
            yield token

        # Save this turn to history
        self.history.append({
            "question": question,
            "answer": full_answer,
            "sources": sources,
            "from_documents": from_documents,
        })

    def query(self, question: str, top_k: int = 3, min_score: float = 0.0) -> Dict[str, Any]:
        """Non-streaming query, kept for CLI use."""
        results = self.retriever.retrieve(question, top_k=top_k, score_threshold=min_score)

        if not results:
            sources = []
            messages = self._build_messages(question)
            response = self.llm.invoke(messages)
            answer = response.content
        else:
            sources = self._get_sources(results)
            context = "\n\n".join([doc["content"] for doc in results])
            messages = self._build_messages(self._build_prompt(question, context))
            response = self.llm.invoke(messages)
            answer = "[From documents] " + response.content

        citations = [
            f"[{i+1}] {s['source']} (page {s['page']})"
            for i, s in enumerate(sources)
        ]
        answer_with_citations = (
            answer + "\n\nCitations:\n" + "\n".join(citations) if citations else answer
        )

        self.history.append({"question": question, "answer": answer, "sources": sources})

        return {
            "question": question,
            "answer": answer_with_citations,
            "sources": sources,
            "history": self.history,
        }