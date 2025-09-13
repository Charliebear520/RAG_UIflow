# RAG Visualizer (Vite + React on Vercel, FastAPI backend)

An end-to-end, interactive UI to visualize each step of a RAG pipeline:

- Upload documents (text or PDF)
- Chunking with adjustable size and overlap
- Embedding (local TF-IDF or OpenAI embeddings)
- Retrieval (top-k chunks with scores and content)
- Generation (final answer + structured reasoning steps)

Frontend: Vite + React (deployable to Vercel). Backend: FastAPI (run locally or deploy anywhere).

## Quick start

Prereqs: Node 18+ and Python 3.10+ recommended.

### Backend (FastAPI)

1) Create and activate a virtual env, then install deps:

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2) (Optional) Enable OpenAI providers by copying `.env.example` to `.env` and setting your keys.

3) Run the API locally:

```bash
uvicorn app.main:app --reload --port 8000
```

The API will be at http://localhost:8000, docs at http://localhost:8000/docs

### Frontend (Vite + React)

1) Install deps and run dev:

```bash
cd ../frontend
npm install
npm run dev
```

Vite dev server runs at http://localhost:5173 and proxies /api to http://localhost:8000 for local dev.

### Deploy

- Frontend: Push `frontend` to a GitHub repo and import into Vercel. It will run `npm run build` and serve `dist/`.
- Backend: Deploy to your preferred host (Render, Fly.io, Railway, etc.). Set `VITE_API_BASE_URL` on Vercel to your backend URL.

## Architecture

- backend/
  - FastAPI app with endpoints for upload, chunk, embed, retrieve, generate
  - Local embedding via TF-IDF (no external model) or OpenAI (if configured)
  - In-memory store; single-process demo use (not production persistent)
- frontend/
  - React UI with panels for each step, wired to the backend
  - Minimal styling; easy to extend

## Notes

- Local embedding uses TF-IDF per document. For production-grade embeddings, use OpenAI or a SentenceTransformer provider and a vector DB.
- Generation can use OpenAI (if configured) or a simple extractive fallback. The backend returns a structured `steps` array rather than raw chain-of-thought.
