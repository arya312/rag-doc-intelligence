from dotenv import load_dotenv
load_dotenv()

import os
import threading
import tempfile
import shutil
import re
import chromadb
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import anthropic

app = FastAPI(title="RAG Doc Intelligence API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global resources — loaded in background
embedder = None
chroma_client = None
claude = None

def load_resources():
    global embedder, chroma_client, claude
    from langchain_huggingface import HuggingFaceEmbeddings
    chroma_client = chromadb.EphemeralClient()
    claude = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("All resources loaded!")

threading.Thread(target=load_resources, daemon=True).start()


# --- Models ---
class QuestionRequest(BaseModel):
    question: str
    collection_name: str

class QuestionResponse(BaseModel):
    answer: str
    pages: list[int]
    collection_name: str


# --- Endpoints ---

@app.get("/health")
def health():
    return {"status": "ok", "ready": embedder is not None}

@app.get("/collections")
def list_collections():
    if chroma_client is None:
        return {"collections": []}
    collections = chroma_client.list_collections()
    return {"collections": [c.name for c in collections]}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")
    if embedder is None or chroma_client is None:
        raise HTTPException(status_code=503, detail="System still loading, please try again in 30 seconds.")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        collection_name = re.sub(r'[^a-zA-Z0-9._-]', '', file.filename.replace(".pdf", "").replace(" ", "_")).lower()
        collection_name = re.sub(r'^[^a-zA-Z0-9]+', '', collection_name)
        collection_name = re.sub(r'[^a-zA-Z0-9]+$', '', collection_name) or "default"

        from ingest import ingest_pdf
        ingest_pdf(tmp_path, collection_name=collection_name, chroma_client=chroma_client, embedder=embedder)
        return {"message": f"Successfully ingested '{file.filename}'", "collection_name": collection_name}
    finally:
        os.unlink(tmp_path)

@app.post("/ask", response_model=QuestionResponse)
def ask_question(request: QuestionRequest):
    if embedder is None or chroma_client is None:
        raise HTTPException(status_code=503, detail="System still loading, please try again in 30 seconds.")
    try:
        collection = chroma_client.get_collection(request.collection_name)
    except Exception:
        raise HTTPException(status_code=404, detail=f"Collection '{request.collection_name}' not found.")

    query_embedding = embedder.embed_query(request.question)
    results = collection.query(query_embeddings=[query_embedding], n_results=3)
    context_chunks = results["documents"][0]
    pages = [int(r.get("page", 0)) for r in results["metadatas"][0]]

    context = ""
    for i, (chunk, page) in enumerate(zip(context_chunks, pages)):
        context += f"\n[Chunk {i+1} - Page {page}]\n{chunk}\n"

    message = claude.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{"role": "user", "content": f"""Answer using ONLY the context below.
If not in context, say "I don't know based on the document."
Always mention the page number.

Context:
{context}

Question: {request.question}"""}]
    )

    return QuestionResponse(answer=message.content[0].text, pages=pages, collection_name=request.collection_name)


# Serve React frontend
_frontend_build = os.path.join(os.path.dirname(__file__), "frontend", "build")
if os.path.exists(_frontend_build):
    app.mount("/static", StaticFiles(directory=f"{_frontend_build}/static"), name="static")

    @app.get("/app")
    def serve_frontend():
        return FileResponse(f"{_frontend_build}/index.html")

    @app.get("/")
    def serve_root():
        return FileResponse(f"{_frontend_build}/index.html")