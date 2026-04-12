from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Step 1: Load PDF
loader = PyPDFLoader("sample.pdf")
pages = loader.load()
print(f"Pages loaded: {len(pages)}")

# Step 2: Chunk it
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # each chunk = max 1000 characters
    chunk_overlap=200,    # 200 characters repeated between chunks
    length_function=len,
)

chunks = splitter.split_documents(pages)

# Step 3: Inspect
print(f"Total chunks created: {len(chunks)}")
print(f"\n--- Chunk 1 ---")
print(chunks[0].page_content)
print(f"\n--- Chunk 2 (notice the overlap with chunk 1) ---")
print(chunks[1].page_content)
print(f"\n--- Chunk 1 length: {len(chunks[0].page_content)} chars ---")
