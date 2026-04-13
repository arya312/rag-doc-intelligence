# RAG Doc Intelligence

A production-grade Retrieval-Augmented Generation (RAG) system that lets you upload any PDF and ask it questions — with grounded answers, page citations, and hallucination detection.

🚀 **Live demo:** https://rag-doc-intelligence.onrender.com  
📖 **API docs:** https://rag-doc-intelligence.onrender.com/docs

---

## What it does

Upload a PDF → the system chunks and embeds it → ask any question → get an accurate answer grounded in your document with page citations → evaluation layer detects if the answer is hallucinated or grounded.

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| LLM | Claude (Anthropic) |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Vector DB | ChromaDB |
| Document loading | LangChain + PyPDF |
| Backend API | FastAPI + Uvicorn |
| Frontend | React + TypeScript |
| Evaluation | Semantic similarity scoring |
| Hosting | Render |

---

## Features

- Upload any PDF via browser UI
- Automatic chunking with 200-character overlap to avoid boundary loss
- Semantic search — finds relevant chunks by meaning, not keyword matching
- Grounded answers with exact page citations
- Hallucination detection — scores every answer for faithfulness to source
- Retrieval quality scoring — measures if the right chunks were retrieved
- Interactive API docs at `/docs`
- Evaluation dashboard with per-question breakdown

---

## Evaluation results (Attention Is All You Need paper)

| Metric | Score |
|---|---|
| Grounded answers | 5/5 (100%) |
| Hallucinations detected | 0 |
| Avg grounding score | 0.75 |
| Avg retrieval score | 0.499 |

---

## Local setup

```bash
# Clone
git clone https://github.com/arya312/rag-doc-intelligence
cd rag-doc-intelligence

# Install dependencies
pip install -r requirements.txt

# Set API key
echo 'ANTHROPIC_API_KEY="your_key_here"' > .env

# Ingest a PDF
python ingest.py your_document.pdf

# Start the backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# In a second terminal, start the frontend
cd frontend && npm install && npm start
```

Visit `http://localhost:3000/app` for the UI or `http://localhost:8000/docs` for the API.

---

## Built by

Arya — actively looking for SDE-II / ML Engineer roles in London and across Europe.  
[GitHub](https://github.com/arya312) · [LinkedIn](https://linkedin.com/in/your-profile)
