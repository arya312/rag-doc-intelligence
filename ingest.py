import os
import sys
import time
import chromadb
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings

def ingest_pdf(pdf_path: str, collection_name: str = None):
    """
    Ingest any PDF into ChromaDB.
    - pdf_path: path to the PDF file
    - collection_name: name to store it under (defaults to filename)
    """

    # Validate file exists
    if not os.path.exists(pdf_path):
        print(f"Error: file not found: {pdf_path}")
        sys.exit(1)

    if not pdf_path.endswith(".pdf"):
        print(f"Error: file must be a PDF")
        sys.exit(1)

    # Default collection name = filename without extension
    if collection_name is None:
        collection_name = os.path.basename(pdf_path).replace(".pdf", "").replace(" ", "_").lower()

    print(f"\n=== Ingesting: {pdf_path} ===")
    print(f"Collection name: {collection_name}")

    # Step 1: Load PDF
    print("\n[1/4] Loading PDF...")
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    print(f"      Loaded {len(pages)} pages")

    # Step 2: Chunk
    print("\n[2/4] Chunking text...")
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = splitter.split_documents(pages)
    print(f"      Created {len(chunks)} chunks")

    # Step 3: Load embedding model
    print("\n[3/4] Loading embedding model...")
    embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    print("      Model ready")

    # Step 4: Store in ChromaDB
    print("\n[4/4] Embedding and storing in ChromaDB...")
    start = time.time()

    client = chromadb.PersistentClient(path="./chroma_db")

    # Delete existing collection with same name so re-runs are safe
    try:
        client.delete_collection(collection_name)
        print(f"      Replaced existing collection '{collection_name}'")
    except:
        pass

    collection = client.create_collection(collection_name)

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
        print("Example: python ingest.py research_paper.pdf")
        sys.exit(1)

    pdf_path = sys.argv[1]
    ingest_pdf(pdf_path)
