import os
import sys
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
import anthropic

client = chromadb.PersistentClient(path="./chroma_db")
embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
claude = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

def ask(question: str, collection_name: str):
    print(f"\n❓ Question: {question}")

    collection = client.get_collection(collection_name)

    query_embedding = embedder.embed_query(question)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3
    )

    context_chunks = results["documents"][0]
    pages = [r.get("page", "?") for r in results["metadatas"][0]]

    context = ""
    for i, (chunk, page) in enumerate(zip(context_chunks, pages)):
        context += f"\n[Chunk {i+1} - Page {page}]\n{chunk}\n"

    print(f"📄 Retrieved 3 chunks from pages: {pages}")

    message = claude.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[{
            "role": "user",
            "content": f"""Answer the question using ONLY the context provided below.
If the answer isn't in the context at all, say "I don't know based on the document."
You may make reasonable inferences from the context (e.g. if it says "175B model" you can infer that means 175 billion parameters).
Always mention which page you found the answer on.

Context:
{context}

Question: {question}"""
        }]
    )

    print(f"\n🤖 Claude's answer:")
    print(message.content[0].text)
    print("\n" + "="*60)


if __name__ == "__main__":
    # Default to attention paper, or pass collection name as argument
    collection_name = sys.argv[1] if len(sys.argv) > 1 else "attention_is_all_you_need"

    print(f"Querying collection: '{collection_name}'")
    print("Type your question and press Enter. Type 'quit' to exit.\n")

    while True:
        question = input("Your question: ").strip()
        if question.lower() in ["quit", "exit", "q"]:
            print("Bye!")
            break
        if question:
            ask(question, collection_name)
