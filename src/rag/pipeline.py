"""Advanced RAG pipeline: streaming, citations, history, summarization."""

import time
from typing import Any, Dict

from langchain_core.messages import HumanMessage


class AdvancedRAGPipeline:
    """Retrieval-augmented generation with extra conveniences.

    Wraps a retriever + LLM and adds pseudo-streaming output, source
    citations, per-query history, and optional answer summarization.
    """

    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
        self.history = []  # Store query history

    def query(
        self,
        question: str,
        top_k: int = 5,
        min_score: float = 0.2,
        stream: bool = False,
        summarize: bool = False,
    ) -> Dict[str, Any]:
        # Retrieve relevant documents
        results = self.retriever.retrieve(
            question, top_k=top_k, score_threshold=min_score
        )
        if not results:
            print("No relevant context found — falling back to LLM knowledge.")
            sources = []
            context = ""
            response = self.llm.invoke([HumanMessage(content=question)])
            answer = response.content
        else:
            context = "\n\n".join([doc["content"] for doc in results])
            sources = [
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

            prompt = f"""Use the following context to answer the question concisely.
Context:
{context}

Question: {question}

Answer:"""

            if stream:
                print("Streaming answer:")
                for i in range(0, len(prompt), 80):
                    print(prompt[i : i + 80], end="", flush=True)
                    time.sleep(0.05)
                print()

            response = self.llm.invoke([HumanMessage(content=prompt)])
            answer = "[From documents] " + response.content

        # Add citations to answer
        citations = [
            f"[{i + 1}] {src['source']} (page {src['page']})"
            for i, src in enumerate(sources)
        ]
        answer_with_citations = (
            answer + "\n\nCitations:\n" + "\n".join(citations)
            if citations
            else answer
        )

        # Optionally summarize answer
        summary = None
        if summarize and answer:
            summary_prompt = f"Summarize the following answer in 2 sentences:\n{answer}"
            summary_resp = self.llm.invoke([HumanMessage(content=summary_prompt)])
            summary = summary_resp.content

        # Store query history
        self.history.append(
            {
                "question": question,
                "answer": answer,
                "sources": sources,
                "summary": summary,
            }
        )

        return {
            "question": question,
            "answer": answer_with_citations,
            "sources": sources,
            "summary": summary,
            "history": self.history,
        }