import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
import anthropic
import chromadb
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
load_dotenv()

# Load models once
print("Loading models...")
similarity_model = SentenceTransformer("all-MiniLM-L6-v2")
embedder = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
claude = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
chroma_client = chromadb.PersistentClient(path="./chroma_db")
print("Ready!\n")


def retrieve_chunks(question: str, collection_name: str, n: int = 3):
    """Retrieve top n chunks for a question"""
    collection = chroma_client.get_collection(collection_name)
    query_embedding = embedder.embed_query(question)
    results = collection.query(query_embeddings=[query_embedding], n_results=n)
    chunks = results["documents"][0]
    pages = [r.get("page", 0) for r in results["metadatas"][0]]
    return chunks, pages


def get_answer(question: str, chunks: list) -> str:
    """Get Claude's answer given retrieved chunks"""
    context = "\n".join([f"[Chunk {i+1}]\n{c}" for i, c in enumerate(chunks)])
    
    import time
    for attempt in range(3):  # retry up to 3 times
        try:
            message = claude.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=512,
                messages=[{
                    "role": "user",
                    "content": f"""Answer the question using ONLY the context below.
If the answer isn't in the context, say "I don't know based on the document."
You may make reasonable inferences from the context.

Context:
{context}

Question: {question}"""
                }]
            )
            return message.content[0].text
        except Exception as e:
            if "overloaded" in str(e).lower() and attempt < 2:
                print(f"      API busy, retrying in 10s... (attempt {attempt+1}/3)")
                time.sleep(10)
            else:
                raise


def score_hallucination(answer: str, chunks: list) -> dict:
    """
    Hallucination detection: measure how similar the answer is
    to the source chunks using semantic similarity.

    Score close to 1.0 = answer is grounded in the chunks
    Score close to 0.0 = answer may be hallucinated
    """
    # Encode answer and each chunk
    answer_embedding = similarity_model.encode(answer, convert_to_tensor=True)
    chunk_embeddings = similarity_model.encode(chunks, convert_to_tensor=True)

    # Compute cosine similarity between answer and each chunk
    similarities = util.cos_sim(answer_embedding, chunk_embeddings)[0]
    max_similarity = float(similarities.max())
    avg_similarity = float(similarities.mean())

    # Classify grounding level
    if max_similarity >= 0.5:
        verdict = "GROUNDED"
        color = "green"
    elif max_similarity >= 0.35:
        verdict = "PARTIAL"
        color = "yellow"
    else:
        verdict = "HALLUCINATED"
        color = "red"

    return {
        "max_similarity": round(max_similarity, 3),
        "avg_similarity": round(avg_similarity, 3),
        "verdict": verdict,
        "color": color
    }


def score_retrieval(question: str, chunks: list) -> dict:
    """
    Retrieval quality: measure how relevant the retrieved chunks
    are to the question.

    Score close to 1.0 = retrieved chunks are highly relevant
    Score close to 0.0 = retrieved chunks are off-topic
    """
    question_embedding = similarity_model.encode(question, convert_to_tensor=True)
    chunk_embeddings = similarity_model.encode(chunks, convert_to_tensor=True)

    similarities = util.cos_sim(question_embedding, chunk_embeddings)[0]
    avg_relevance = float(similarities.mean())
    max_relevance = float(similarities.max())

    if avg_relevance >= 0.4:
        quality = "GOOD"
    elif avg_relevance >= 0.25:
        quality = "FAIR"
    else:
        quality = "POOR"

    return {
        "avg_relevance": round(avg_relevance, 3),
        "max_relevance": round(max_relevance, 3),
        "quality": quality
    }


def evaluate_question(question: str, collection_name: str) -> dict:
    """Full evaluation pipeline for one question"""
    print(f"\n{'='*60}")
    print(f"Q: {question}")

    # Step 1: Retrieve
    chunks, pages = retrieve_chunks(question, collection_name)
    print(f"Retrieved from pages: {pages}")

    # Step 2: Answer
    answer = get_answer(question, chunks)
    print(f"\nAnswer: {answer[:200]}...")

    # Step 3: Score
    hallucination = score_hallucination(answer, chunks)
    retrieval = score_retrieval(question, chunks)

    print(f"\nHallucination check: {hallucination['verdict']} (similarity: {hallucination['max_similarity']})")
    print(f"Retrieval quality:   {retrieval['quality']} (relevance: {retrieval['avg_relevance']})")

    return {
        "question": question,
        "answer": answer,
        "pages": pages,
        "hallucination": hallucination,
        "retrieval": retrieval
    }


def run_evaluation(collection_name: str, questions: list) -> dict:
    """Run evaluation on a list of questions and produce a summary"""
    print(f"\nRunning evaluation on '{collection_name}' with {len(questions)} questions...")

    results = []
    for q in questions:
        result = evaluate_question(q, collection_name)
        results.append(result)

    # Summary stats
    hallucination_scores = [r["hallucination"]["max_similarity"] for r in results]
    retrieval_scores = [r["retrieval"]["avg_relevance"] for r in results]
    verdicts = [r["hallucination"]["verdict"] for r in results]

    summary = {
        "collection": collection_name,
        "total_questions": len(questions),
        "grounded": verdicts.count("GROUNDED"),
        "partial": verdicts.count("PARTIAL"),
        "hallucinated": verdicts.count("HALLUCINATED"),
        "avg_hallucination_score": round(float(np.mean(hallucination_scores)), 3),
        "avg_retrieval_score": round(float(np.mean(retrieval_scores)), 3),
        "results": results
    }

    print(f"\n{'='*60}")
    print(f"EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Total questions:     {summary['total_questions']}")
    print(f"Grounded answers:    {summary['grounded']}")
    print(f"Partial answers:     {summary['partial']}")
    print(f"Hallucinated:        {summary['hallucinated']}")
    print(f"Avg grounding score: {summary['avg_hallucination_score']}")
    print(f"Avg retrieval score: {summary['avg_retrieval_score']}")

    # Save results to file
    with open("eval_results.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved to eval_results.json")

    return summary


if __name__ == "__main__":
    # Test questions for the attention paper
    questions = [
        "What is the attention mechanism?",
        "How long did training take?",
        "What BLEU score did the Transformer achieve?",
        "What are the limitations of recurrent neural networks?",
        "What is multi-head attention?",
    ]

    run_evaluation("sample", questions)
