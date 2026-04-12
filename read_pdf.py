from langchain_community.document_loaders import PyPDFLoader

# Load the PDF
loader = PyPDFLoader("sample.pdf")
pages = loader.load()

# See what we got
print(f"Total pages loaded: {len(pages)}")
print(f"\n--- First 500 characters of page 1 ---")
print(pages[0].page_content[:500])
print(f"\n--- Metadata for page 1 ---")
print(pages[0].metadata)
