from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
import chromadb

# Step 1: Load + chunk (same as before)
print("📄 Loading PDF...")
loader = PyPDFLoader("sample.pdf")
pages = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)
chunks = splitter.split_documents(pages)
print(f"✅ {len(chunks)} chunks ready")

# Step 2: Load embedding model (runs locally, no API key needed)
print("\n🧠 Loading embedding model (first time downloads ~90MB, be patient)...")
embedder = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"  # small, fast, good quality
)
print("✅ Embedding model loaded")

# Step 3: Set up ChromaDB
print("\n🗄️ Setting up ChromaDB...")
client = chromadb.PersistentClient(path="./chroma_db")

# Delete collection if it exists (so we can re-run safely)
try:
    client.delete_collection("attention_paper")
except:
    pass

collection = client.create_collection("attention_paper")
print("✅ Collection created")

# Step 4: Embed each chunk + store in ChromaDB
print("\n⚡ Embedding and storing chunks...")
texts = [chunk.page_content for chunk in chunks]
metadatas = [chunk.metadata for chunk in chunks]
ids = [f"chunk_{i}" for i in range(len(chunks))]

# Generate embeddings
embeddings = embedder.embed_documents(texts)
print(f"✅ Generated {len(embeddings)} embeddings")
print(f"   Each embedding is a vector of {len(embeddings[0])} numbers")

# Store everything in ChromaDB
collection.add(
    documents=texts,
    embeddings=embeddings,
    metadatas=metadatas,
    ids=ids
)
print(f"✅ Stored in ChromaDB!")

# Step 5: Quick sanity check — search for something
print("\n🔍 Testing retrieval — searching for 'attention mechanism'...")
query = "What is the attention mechanism?"
query_embedding = embedder.embed_query(query)

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=2
)

print("\n--- Most relevant chunk found ---")
print(results["documents"][0][0][:500])
print(f"\nFrom page: {results['metadatas'][0][0]['page']}")
