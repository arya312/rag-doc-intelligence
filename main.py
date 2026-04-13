from dotenv import load_dotenv
load_dotenv()

import os
import chromadb
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import anthropic
from langchain_huggingface import HuggingFaceEmbeddings
from ingest import ingest_pdf
import tempfile
import shutil
from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="RAG Doc Intelligence API")

# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load shared resources once at startup (not on every request)
import threading

embedder = None

def load_models():
    global embedder
    print("Loading embedding model...")
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("Ready!")

threading.Thread(target=load_models, daemon=True).start()
chroma_client = chromadb.PersistentClient(path="./chroma_db")
claude = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
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
        collection_name = file.filename.replace(".pdf", "").replace(" ", "_").lower()
        ingest_pdf(tmp_path, collection_name=collection_name)
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