"""Document loading: read files of any supported type into LangChain Documents."""

from pathlib import Path
from typing import Any, List

from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
    UnstructuredExcelLoader,
    UnstructuredPowerPointLoader,
    JSONLoader,
)

# Map file extensions to their loader class + any fixed kwargs
LOADER_MAP = {
    ".pdf":  (PyPDFLoader,                    {}),
    ".txt":  (TextLoader,                     {"encoding": "utf-8"}),
    ".md":   (TextLoader,                     {"encoding": "utf-8"}),
    ".csv":  (CSVLoader,                      {}),
    ".json": (JSONLoader,                     {"jq_schema": ".", "text_content": False}),
    ".docx": (UnstructuredWordDocumentLoader, {}),
    ".doc":  (UnstructuredWordDocumentLoader, {}),
    ".xlsx": (UnstructuredExcelLoader,        {}),
    ".xls":  (UnstructuredExcelLoader,        {}),
    ".pptx": (UnstructuredPowerPointLoader,   {}),
    ".ppt":  (UnstructuredPowerPointLoader,   {}),
}


def process_all_files(directory: str) -> List[Any]:
    """Recursively load every supported file in *directory*.

    Supported formats: PDF, TXT, MD, CSV, JSON, DOCX, DOC, XLSX, XLS, PPTX, PPT.
    Unsupported extensions are skipped with a warning.

    Args:
        directory: Root directory to search.

    Returns:
        A flat list of LangChain ``Document`` objects tagged with
        ``source_file`` and ``file_type`` metadata.
    """
    all_documents: List[Any] = []
    root = Path(directory)

    # Collect all files (skip hidden files and the vector_store folder)
    all_files = [
        f for f in root.rglob("*")
        if f.is_file()
        and not any(part.startswith(".") for part in f.parts)
        and "vector_store" not in f.parts
    ]

    supported   = [f for f in all_files if f.suffix.lower() in LOADER_MAP]
    unsupported = [f for f in all_files if f.suffix.lower() not in LOADER_MAP]

    print(f"Found {len(supported)} supported file(s) to process "
          f"({len(unsupported)} skipped).")
    if unsupported:
        print("  Skipped:", ", ".join(f.name for f in unsupported))

    for file in supported:
        ext = file.suffix.lower()
        loader_cls, loader_kwargs = LOADER_MAP[ext]
        print(f"\nProcessing: {file.name}  [{ext}]")
        try:
            loader = loader_cls(str(file), **loader_kwargs)
            documents = loader.load()

            for doc in documents:
                doc.metadata["source_file"] = file.name
                doc.metadata["file_type"]   = ext.lstrip(".")

            all_documents.extend(documents)
            print(f"  ✓ Loaded {len(documents)} document(s)")

        except Exception as e:
            print(f"  ✗ Error: {e}")

    print(f"\nTotal documents loaded: {len(all_documents)}")
    return all_documents