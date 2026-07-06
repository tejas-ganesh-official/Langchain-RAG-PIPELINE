"""FastAPI server exposing /upload and /chat/stream endpoints."""

import sys
import shutil
from pathlib import Path
from typing import List

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── resolve paths ──────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
SRC_DIR  = ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from rag import (
    process_all_files,
    split_documents,
    EmbeddingManager,
    VectorStore,
    RAGRetriever,
    build_qwen_llm,
    AdvancedRAGPipeline,
)

# ── boot the pipeline ──────────────────────────────────────────────────────────
print("Booting RAG pipeline…")
embedding_manager = EmbeddingManager()

# Fresh collection every boot — prevents duplicate accumulation
vectorstore = VectorStore(persist_directory=str(DATA_DIR / "vector_store"))
vectorstore.client.delete_collection(vectorstore.collection_name)
vectorstore.collection = vectorstore.client.get_or_create_collection(
    name=vectorstore.collection_name,
    metadata={
        "description": "PDF document embeddings for RAG",
        "hnsw:space": "cosine",
    },
)
print("Collection reset.")

# Ingest everything currently in data/
all_docs = process_all_files(str(DATA_DIR))
if all_docs:
    chunks     = split_documents(all_docs)
    texts      = [c.page_content for c in chunks]
    embeddings = embedding_manager.generate_embeddings(texts)
    vectorstore.add_documents(chunks, embeddings)
    print(f"Ingested {len(chunks)} chunks from {len(all_docs)} document(s).")
else:
    print("No files in data/ yet — drop files via the UI to get started.")

retriever        = RAGRetriever(vectorstore, embedding_manager)
qwen             = build_qwen_llm()
current_model    = "qwen2.5:3b"
pipeline         = AdvancedRAGPipeline(retriever, qwen.llm)
print("Pipeline ready.")

# ── app ────────────────────────────────────────────────────────────────────────
app = FastAPI(title="RAG Pipeline API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── helpers ────────────────────────────────────────────────────────────────────
LOADER_MAP_KEYS = {
    ".pdf", ".txt", ".md", ".csv", ".json",
    ".docx", ".doc", ".xlsx", ".xls", ".pptx", ".ppt",
}

def ingest_files(paths: List[Path]) -> int:
    """Embed and store a list of newly added files. Returns chunk count."""
    from langchain_community.document_loaders import (
        PyPDFLoader, TextLoader, CSVLoader,
        UnstructuredWordDocumentLoader, UnstructuredExcelLoader,
        UnstructuredPowerPointLoader, JSONLoader,
    )
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
    docs = []
    for f in paths:
        ext = f.suffix.lower()
        if ext not in LOADER_MAP:
            continue
        loader_cls, kwargs = LOADER_MAP[ext]
        try:
            loader = loader_cls(str(f), **kwargs)
            loaded = loader.load()
            for doc in loaded:
                doc.metadata["source_file"] = f.name
                doc.metadata["file_type"]   = ext.lstrip(".")
            docs.extend(loaded)
        except Exception as e:
            print(f"Error loading {f.name}: {e}")

    if not docs:
        return 0

    chunks     = split_documents(docs)
    texts      = [c.page_content for c in chunks]
    embeds     = embedding_manager.generate_embeddings(texts)
    vectorstore.add_documents(chunks, embeds)
    return len(chunks)


# ── endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload(files: List[UploadFile] = File(...)):
    """Save uploaded files to data/ and ingest them into ChromaDB."""
    saved: List[str] = []
    paths: List[Path] = []

    for file in files:
        dest = DATA_DIR / file.filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        with dest.open("wb") as f:
            shutil.copyfileobj(file.file, f)
        saved.append(file.filename)
        paths.append(dest)

    chunks_added = ingest_files(paths)
    return JSONResponse({
        "uploaded": saved,
        "chunks_added": chunks_added,
        "message": f"Ingested {chunks_added} chunks from {len(saved)} file(s).",
    })


@app.get("/files")
def list_files():
    """Return all files currently in data/ (excluding vector_store)."""
    files = [
        f.name for f in DATA_DIR.rglob("*")
        if f.is_file() and "vector_store" not in f.parts
    ]
    return {"files": files}


@app.get("/models")
def list_models():
    """Return all models available in the local Ollama instance."""
    import urllib.request, json as _json
    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=3) as r:
            data = _json.loads(r.read())
        models = [m["name"] for m in data.get("models", [])]
        return {"models": models, "current": current_model}
    except Exception as e:
        return {"models": [], "current": current_model, "error": str(e)}


class ModelRequest(BaseModel):
    model: str


@app.post("/model")
def set_model(req: ModelRequest):
    """Hot-swap the LLM model without restarting the server."""
    global pipeline, current_model, qwen
    try:
        new_qwen = build_qwen_llm(model_name=req.model)
        if new_qwen is None:
            raise ValueError(f"Could not load model: {req.model}")
        qwen          = new_qwen
        current_model = req.model
        pipeline      = AdvancedRAGPipeline(retriever, qwen.llm)
        print(f"Model swapped to: {current_model}")
        return {"status": "ok", "model": current_model}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})


class ChatRequest(BaseModel):
    message: str
    top_k: int = 3
    min_score: float = -1.0


@app.post("/chat/stream")
def chat_stream(req: ChatRequest):
    """Stream the RAG answer token by token using SSE."""
    def generate():
        for chunk in pipeline.stream_query(req.message, req.top_k, req.min_score):
            yield f"data: {chunk}\n\n"
        yield "data: __done__\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")


# ── serve frontend ─────────────────────────────────────────────────────────────
FRONTEND_DIR = ROOT / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")