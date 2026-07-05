"""Text splitting: break documents into overlapping chunks for RAG."""

from typing import Any, List

from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(
    documents: List[Any],
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> List[Any]:
    """Split documents into smaller chunks for better RAG performance.

    Args:
        documents: LangChain documents to split.
        chunk_size: Target maximum characters per chunk.
        chunk_overlap: Characters shared between adjacent chunks.

    Returns:
        A list of chunked LangChain ``Document`` objects.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    split_docs = text_splitter.split_documents(documents)
    print(f"Split {len(documents)} documents into {len(split_docs)} chunks")

    # Show example of a chunk
    if split_docs:
        print("\nExample chunk:")
        print(f"Content: {split_docs[0].page_content[:200]}...")
        print(f"Metadata: {split_docs[0].metadata}")

    return split_docs
