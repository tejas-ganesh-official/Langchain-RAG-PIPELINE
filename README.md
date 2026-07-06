Install Ollama
ollama pull qwen2.5:7b
ollama pull qwen2.5:7b
ollama pull llama3.2:3b
ollama pull mistral
pip install -r requirements.txt
Drop files into data/
python src/main.py

***************
ragpicker/
  api/
    server.py           # FastAPI server — /upload, /files, /models, /model, /chat/stream
  frontend/
    index.html          # Single-file UI
  src/
    rag/
      __init__.py
      document_loader.py   # Loads all supported file types into LangChain Documents
      text_splitter.py     # Chunks documents with RecursiveCharacterTextSplitter
      embeddings.py        # SentenceTransformer embedding manager
      vector_store.py      # ChromaDB persistence layer
      retriever.py         # Semantic search over the vector store
      llm.py               # Ollama/Qwen wrapper
      pipeline.py          # RAG pipeline with conversation memory and streaming
    main.py              # CLI version (optional, runs without the web UI)
  data/
    vector_store/        # ChromaDB data (auto-created, gitignored)

***************

Clone the repo

bashgit clone https://github.com/yourusername/langchain-rag-pipeline.git
cd langchain-rag-pipeline


**************
Create a virtual environment and install dependencies

bashpython -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
***************
Run the server

bashpython -m uvicorn api.server:app --reload


