"""End-to-end RAG run: ingest files -> embed -> store -> retrieve -> answer.

Drop any supported file into ../data and it will be picked up automatically.
Supported: PDF, TXT, MD, CSV, JSON, DOCX, DOC, XLSX, XLS, PPTX, PPT.

Run:
    python main.py
"""

from pathlib import Path

from rag import (
    process_all_files,
    split_documents,
    EmbeddingManager,
    VectorStore,
    RAGRetriever,
    build_qwen_llm,
    AdvancedRAGPipeline,
)

# Always resolve data/ relative to this file, regardless of where you run from
DATA_DIR = Path(__file__).parent.parent / "data"


def main():
    # 1. Ingest all supported files
    all_documents = process_all_files(str(DATA_DIR))
    if not all_documents:
        print("No documents loaded — add files to the ../data folder and retry.")
        return

    # 2. Chunk
    chunks = split_documents(all_documents)

    # 3. Embed
    embedding_manager = EmbeddingManager()
    texts = [doc.page_content for doc in chunks]
    embeddings = embedding_manager.generate_embeddings(texts)

    # 4. Store
    vectorstore = VectorStore()
    vectorstore.add_documents(chunks, embeddings)

    # 5. Retriever
    retriever = RAGRetriever(vectorstore, embedding_manager)

    # 6. LLM
    qwen_llm = build_qwen_llm(model_name="qwen2.5:3b")
    if qwen_llm is None:
        return

    # 7. Pipeline
    adv_rag = AdvancedRAGPipeline(retriever, qwen_llm.llm)

    # 8. Interactive query loop
    print("\n" + "=" * 50)
    print("RAG is ready. Type your question, or 'exit' to quit.")
    print("=" * 50 + "\n")

    while True:
        try:
            query = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting.")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit", "q"}:
            print("Exiting.")
            break

        result = adv_rag.query(query, top_k=3, min_score=0.1)
        print(f"\nAnswer: {result['answer']}\n")


if __name__ == "__main__":
    main()