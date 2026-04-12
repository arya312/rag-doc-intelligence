# RAG Doc Intelligence

A Retrieval-Augmented Generation (RAG) system that lets you chat with your documents using Claude AI.

## What this project does
Upload PDFs → system chunks + embeds them → ask questions → get accurate answers grounded in your docs.

## Tech Stack
- Claude (Anthropic) — LLM for answering questions
- ChromaDB — vector database for storing embeddings
- LangChain — document loading and chunking
- FastAPI — backend API
- React — frontend UI

## Progress
- [x] Week 1: Environment setup, API connection, embeddings basics
- [ ] Week 2: PDF ingestion pipeline
- [ ] Week 3: FastAPI backend + React frontend
- [ ] Week 4: Evaluation + hallucination detection

## Setup
```bash
pip install anthropic chromadb langchain langchain-community pypdf sentence-transformers python-dotenv
export ANTHROPIC_API_KEY="your-api-key"
python test_embeddings.py
```
