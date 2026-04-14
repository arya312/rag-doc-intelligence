from dotenv import load_dotenv
load_dotenv()

import os
import sys
import time
import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

# Initialize default client for standalone CLI usage
_default_chroma_client = None

def _get_default_client():
    """Lazy initialize default client for CLI usage"""
    global _default_chroma_client
    if _default_chroma_client is None:
        use_persistent = os.environ.get("ENABLE_PERSISTENT_CHROMA", "false").lower() == "true"
        if use_persistent:
            try:
                os.makedirs("./chroma_db", exist_ok=True)
                _default_chroma_client = chromadb.PersistentClient(path="./chroma_db")
            except (OSError, PermissionError):
                print("Warning: Failed to initialize persistent ChromaDB, using ephemeral")
                _default_chroma_client = chromadb.EphemeralClient()
        else:
            _default_chroma_client = chromadb.EphemeralClient()
    return _default_chroma_client

def ingest_pdf(pdf_path: str, collection_name: str = None, chroma_client=None):
    if chroma_client is None:
        chroma_client = _get_default_client()
    
    if not os.path.exists(pdf_path):
        print(f"Error: file not found: {pdf_path}")
        sys.exit(1)

    if not pdf_path.endswith(".pdf"):
        print(f"Error: file must be a PDF")
        sys.exit(1)

    if collection_name is None:
        collection_name = os.path.basename(pdf_path).replace(".pdf", "").replace(" ", "_").lower()

    print(f"\n=== Ingesting: {pdf_path} ===")
    print(f"Collection name: {collection_name}")

    print("\n[1/4] Loading PDF...")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"      Loaded {len(pages)} pages")

    print("\n[2/4] Chunking text...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = splitter.split_documents(pages)
    print(f"      Created {len(chunks)} chunks")

    print("\n[3/4] Loading embedding model...")
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("      Model ready")

    print("\n[4/4] Embedding and storing in ChromaDB...")
    start = time.time()

    try:
        chroma_client.delete_collection(collection_name)
    except:
        pass

    collection = chroma_client.create_collection(collection_name)
    texts = [chunk.page_content for chunk in chunks]
    metadatas = [chunk.metadata for chunk in chunks]
    ids = [f"chunk_{i}" for i in range(len(chunks))]
    embeddings = embedder.embed_documents(texts)

    collection.add(
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )

    elapsed = round(time.time() - start, 1)
    print(f"      Stored {len(chunks)} chunks in {elapsed}s")
    print(f"\nDone! '{collection_name}' is ready to query.\n")
    return collection_name


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python ingest.py <path_to_pdf>")
        sys.exit(1)
    ingest_pdf(sys.argv[1])
