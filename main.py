from dotenv import load_dotenv
load_dotenv()

import os
import threading
import tempfile
import shutil
import chromadb
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
import anthropic
from ingest import ingest_pdf

app = FastAPI(title="RAG Doc Intelligence API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/health")
def health():
    return {"status": "ok", "ready": embedder is not None}

# These load in background AFTER server starts
embedder = None
chroma_client = None
claude = None

def load_resources():
    global embedder, chroma_client, claude
    from langchain_huggingface import HuggingFaceEmbeddings
    
    # Use EphemeralClient for Render free tier (no persistent filesystem)
    # For persistent storage, use environment variable ENABLE_PERSISTENT_CHROMA=true
    use_persistent = os.environ.get("ENABLE_PERSISTENT_CHROMA", "false").lower() == "true"
    
    if use_persistent:
        chroma_path = os.environ.get("CHROMA_PATH", "./chroma_db")
        try:
            # Ensure directory exists with proper permissions
            os.makedirs(chroma_path, exist_ok=True)
            chroma_client = chromadb.PersistentClient(path=chroma_path)
            print(f"Using persistent ChromaDB at {chroma_path}")
        except (OSError, PermissionError) as e:
            print(f"Failed to initialize persistent ChromaDB: {e}")
            print("Falling back to ephemeral (in-memory) ChromaDB")
            chroma_client = chromadb.EphemeralClient()
    else:
        # Use in-memory ephemeral client for free tier / testing
        chroma_client = chromadb.EphemeralClient()
        print("Using ephemeral (in-memory) ChromaDB")
    
    claude = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("All resources loaded!")

threading.Thread(target=load_resources, daemon=True).start()

# Give load_resources thread time to initialize
print("Ready!")


# --- Request/Response models ---
class QuestionRequest(BaseModel):
    question: str
    collection_name: str

class QuestionResponse(BaseModel):
    answer: str
    pages: list[int]
    collection_name: str


# --- Endpoints ---

@app.get("/")
def root():
    return {"status": "RAG Doc Intelligence API is running"}


@app.get("/collections")
def list_collections():
    """List all ingested documents"""
    collections = chroma_client.list_collections()
    return {"collections": [c.name for c in collections]}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and ingest a PDF"""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Save uploaded file to a temp location
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        if chroma_client is None:
            raise HTTPException(status_code=503, detail="ChromaDB not yet ready. Please try again in a moment.")
        
        collection_name = file.filename.replace(".pdf", "").replace(" ", "_").lower()
        ingest_pdf(tmp_path, collection_name=collection_name, chroma_client=chroma_client)
        return {
            "message": f"Successfully ingested '{file.filename}'",
            "collection_name": collection_name
        }
    finally:
        os.unlink(tmp_path)  # clean up temp file


@app.post("/ask", response_model=QuestionResponse)
def ask_question(request: QuestionRequest):
    """Ask a question about an ingested document"""
    try:
        collection = chroma_client.get_collection(request.collection_name)
    except Exception:
        raise HTTPException(
            status_code=404,
            detail=f"Collection '{request.collection_name}' not found. Please upload the document first."
        )

    # Retrieve relevant chunks
    query_embedding = embedder.embed_query(request.question)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    context_chunks = results["documents"][0]
    pages = [int(r.get("page", 0)) for r in results["metadatas"][0]]

    context = ""
    for i, (chunk, page) in enumerate(zip(context_chunks, pages)):
        context += f"\n[Chunk {i+1} - Page {page}]\n{chunk}\n"

    # Ask Claude
    message = claude.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"""Answer the question using ONLY the context provided below.
If the answer isn't in the context at all, say "I don't know based on the document."
You may make reasonable inferences from the context.
Always mention which page you found the answer on.

Context:
{context}

Question: {request.question}"""
        }]
    )

    return QuestionResponse(
        answer=message.content[0].text,
        pages=pages,
        collection_name=request.collection_name
    )


# Serve React frontend
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import os as _os

_frontend_build = _os.path.join(_os.path.dirname(__file__), "frontend", "build")
if _os.path.exists(_frontend_build):
    app.mount("/static", StaticFiles(directory=f"{_frontend_build}/static"), name="static")

    @app.get("/app")
    def serve_frontend():
        return FileResponse(f"{_frontend_build}/index.html")

    @app.get("/")
    def serve_root():
        return FileResponse(f"{_frontend_build}/index.html")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=port,
        timeout_keep_alive=120,
    )