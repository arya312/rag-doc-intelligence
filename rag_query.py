import os
import chromadb
from langchain_community.embeddings import HuggingFaceEmbeddings
import anthropic

# Step 1: Load our stored ChromaDB
client = chromadb.PersistentClient(path="./chroma_db")
collection = client.get_collection("attention_paper")

# Step 2: Load the same embedding model
embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Step 3: Load Claude client
claude = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

def ask(question):
    print(f"\n❓ Question: {question}")
    
    # Convert question to embedding + find relevant chunks
    query_embedding = embedder.embed_query(question)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3  # get top 3 most relevant chunks
    )
    
    # Build context from retrieved chunks
    context_chunks = results["documents"][0]
    pages = [r["page"] for r in results["metadatas"][0]]
    
    context = ""
    for i, (chunk, page) in enumerate(zip(context_chunks, pages)):
        context += f"\n[Chunk {i+1} - Page {page}]\n{chunk}\n"
    
    print(f"📄 Retrieved {len(context_chunks)} chunks from pages: {pages}")
    
    # Send to Claude with the context
    message = claude.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"""Answer the question using ONLY the context provided below.
If the answer isn't in the context, say "I don't know based on the document."
Always mention which page you found the answer on.

Context:
{context}

Question: {question}"""
        }]
    )
    
    print(f"\n🤖 Claude's answer:")
    print(message.content[0].text)
    print("\n" + "="*60)

# Test with 3 different questions
ask("What is the attention mechanism?")
ask("How long did it take to train the model?")
ask("What BLEU score did the Transformer achieve?")
