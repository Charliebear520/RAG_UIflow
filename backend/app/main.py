from __future__ import annotations

import io
import os
import uuid
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()


def get_env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.lower() in {"1", "true", "yes", "on"}


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
USE_OPENAI_EMBEDDING = get_env_bool("USE_OPENAI_EMBEDDING", False)
USE_OPENAI_COMPLETION = get_env_bool("USE_OPENAI_COMPLETION", False)

try:
    import httpx
except Exception:  # pragma: no cover
    httpx = None


# ---- Simple in-memory store (demo only) ----
@dataclass
class DocRecord:
    id: str
    filename: str
    text: str
    chunks: List[str]
    chunk_size: int
    overlap: int


class InMemoryStore:
    def __init__(self) -> None:
        self.docs: Dict[str, DocRecord] = {}
        self.tfidf: Optional[TfidfVectorizer] = None
        self.embeddings = None  # matrix for chunks
        self.chunk_doc_ids: List[str] = []
    self.chunks_flat = []

    def reset_embeddings(self):
        self.tfidf = None
        self.embeddings = None
        self.chunk_doc_ids = []
    self.chunks_flat = []


store = InMemoryStore()


app = FastAPI(title="RAG Visualizer API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # During dev; restrict in prod
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChunkRequest(BaseModel):
    doc_id: str
    chunk_size: int = 500
    overlap: int = 50


class EmbedRequest(BaseModel):
    doc_ids: Optional[List[str]] = None  # if None, embed all


class RetrieveRequest(BaseModel):
    query: str
    k: int = 5


class GenerateRequest(BaseModel):
    query: str
    top_k: int = 5


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/upload")
async def upload(file: UploadFile = File(...)):
    # For simplicity, treat all as text; for PDFs, a proper parser should be used
    content = await file.read()
    try:
        text = content.decode("utf-8", errors="ignore")
    except Exception:
        text = str(content)
    doc_id = str(uuid.uuid4())
    store.docs[doc_id] = DocRecord(
        id=doc_id,
        filename=file.filename,
        text=text,
        chunks=[],
        chunk_size=0,
        overlap=0,
    )
    # When uploading new docs, prior embeddings are invalid
    store.reset_embeddings()
    return {"doc_id": doc_id, "filename": file.filename, "num_chars": len(text)}


def sliding_window_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    if chunk_size <= 0:
        return [text]
    if overlap >= chunk_size:
        overlap = max(0, chunk_size - 1)
    chunks: List[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        chunks.append(text[start:end])
        if end == n:
            break
        start = end - overlap
    return chunks


@app.post("/chunk")
def chunk(req: ChunkRequest):
    doc = store.docs.get(req.doc_id)
    if not doc:
        return JSONResponse(status_code=404, content={"error": "doc not found"})
    chunks = sliding_window_chunks(doc.text, req.chunk_size, req.overlap)
    doc.chunks = chunks
    doc.chunk_size = req.chunk_size
    doc.overlap = req.overlap
    # invalidates embeddings for safety
    store.reset_embeddings()
    return {"doc_id": doc.id, "num_chunks": len(chunks), "chunk_size": req.chunk_size, "overlap": req.overlap, "sample": chunks[:3]}


async def embed_openai(texts: List[str]) -> List[List[float]]:
    if not httpx:
        raise RuntimeError("httpx not available")
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")
    # Use OpenAI embeddings API v1
    url = "https://api.openai.com/v1/embeddings"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    out: List[List[float]] = []
    async with httpx.AsyncClient(timeout=60) as client:
        # batch to avoid token limits; simple small batches
        batch_size = 64
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            payload = {"input": batch, "model": model}
            r = await client.post(url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
            out.extend([d["embedding"] for d in data["data"]])
    return out


@app.post("/embed")
async def embed(req: EmbedRequest):
    # gather chunks across selected docs
    selected = req.doc_ids or list(store.docs.keys())
    all_chunks: List[str] = []
    chunk_doc_ids: List[str] = []
    for d in selected:
        doc = store.docs.get(d)
        if doc and doc.chunks:
            all_chunks.extend(doc.chunks)
            chunk_doc_ids.extend([doc.id] * len(doc.chunks))

    if not all_chunks:
        return JSONResponse(status_code=400, content={"error": "no chunks to embed"})

    if USE_OPENAI_EMBEDDING:
        vectors = await embed_openai(all_chunks)
        # keep as numpy matrix-like list of lists; for cosine sim we'll rely on sklearn if available
        store.tfidf = None
        store.embeddings = vectors
        store.chunk_doc_ids = chunk_doc_ids
        store.chunks_flat = all_chunks
        return {"provider": "openai", "num_vectors": len(vectors)}
    else:
        # TF-IDF fallback (per-chunk bag-of-words)
        vectorizer = TfidfVectorizer(max_features=4096, stop_words="english")
        X = vectorizer.fit_transform(all_chunks)
        store.tfidf = vectorizer
        store.embeddings = X
        store.chunk_doc_ids = chunk_doc_ids
        store.chunks_flat = all_chunks
        return {"provider": "tfidf", "num_vectors": X.shape[0], "num_features": X.shape[1]}


def rank_with_tfidf(query: str, k: int):
    assert store.tfidf is not None
    q = store.tfidf.transform([query])
    sims = cosine_similarity(q, store.embeddings).ravel()  # type: ignore[arg-type]
    idxs = sims.argsort()[::-1][:k]
    return idxs, sims[idxs]


def rank_with_openai(query: str, k: int):
    # cosine similarity on dense vectors list
    import numpy as np
    vecs = np.array(store.embeddings, dtype=float)  # type: ignore[assignment]
    qvec = np.array(asyncio_run(embed_openai([query]))[0], dtype=float)
    # normalize
    vecs_norm = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-8)
    q_norm = qvec / (np.linalg.norm(qvec) + 1e-8)
    sims = vecs_norm @ q_norm
    idxs = np.argsort(-sims)[:k]
    return idxs.tolist(), sims[idxs].tolist()


def asyncio_run(coro):
    import asyncio
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = None
    if loop and loop.is_running():
        # create task and wait
        return asyncio.run_coroutine_threadsafe(coro, loop).result()
    return asyncio.run(coro)


@app.post("/retrieve")
def retrieve(req: RetrieveRequest):
    if store.embeddings is None:
        return JSONResponse(status_code=400, content={"error": "run /embed first"})
    if store.tfidf is not None:
        idxs, sims = rank_with_tfidf(req.query, req.k)
    else:
        idxs, sims = rank_with_openai(req.query, req.k)

    # Use the same order as built in /embed
    chunks_flat = store.chunks_flat
    mapping_doc_ids = store.chunk_doc_ids

    results = []
    for rank, (i, score) in enumerate(zip(idxs, sims), start=1):
        if i < 0 or i >= len(chunks_flat):
            continue
        results.append({
            "rank": rank,
            "score": float(score),
            "doc_id": mapping_doc_ids[i],
            "chunk_index": i,
            "content": chunks_flat[i][:2000],
        })
    return {"query": req.query, "k": req.k, "results": results}


async def openai_chat(messages: List[Dict[str, str]]) -> str:
    if not httpx:
        raise RuntimeError("httpx not available")
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY not set")
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"}
    model = os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini")
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.2,
    }
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(url, headers=headers, json=payload)
        r.raise_for_status()
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()


def simple_extractive_answer(query: str, contexts: List[str]) -> str:
    # very simple heuristic: return the highest-overlap sentences as an extractive summary
    import re
    from collections import Counter
    q_terms = [t.lower() for t in re.findall(r"\w+", query)]
    counts = Counter()
    sents: List[str] = []
    for ctx in contexts:
        sents.extend(re.split(r"(?<=[.!?])\s+", ctx))
    for s in sents:
        tokens = [t.lower() for t in re.findall(r"\w+", s)]
        overlap = len(set(tokens) & set(q_terms))
        if overlap:
            counts[s] = overlap
    best = [s for s, _ in counts.most_common(5)]
    return " \n".join(best) if best else "No relevant answer found in context."


@app.post("/generate")
def generate(req: GenerateRequest):
    # retrieve first
    r = retrieve(RetrieveRequest(query=req.query, k=req.top_k))
    if isinstance(r, JSONResponse):
        return r
    results = r["results"]
    contexts = [item["content"] for item in results]

    reasoning_steps = [
        {"type": "plan", "text": "Read query, identify entities and constraints."},
        {"type": "gather", "text": f"Collect top-{req.top_k} chunks as context."},
        {"type": "synthesize", "text": "Synthesize answer grounded in retrieved text."},
    ]

    if USE_OPENAI_COMPLETION:
        prompt = [
            {"role": "system", "content": "You are a helpful assistant. Answer using ONLY the provided context. If missing, say you don't know."},
            {"role": "user", "content": f"Query: {req.query}\n\nContext:\n" + "\n---\n".join(contexts)},
        ]
        try:
            answer = asyncio_run(openai_chat(prompt))
        except Exception as e:
            answer = f"OpenAI call failed: {e}. Falling back to extractive answer.\n" + simple_extractive_answer(req.query, contexts)
    else:
        answer = simple_extractive_answer(req.query, contexts)

    return {
        "query": req.query,
        "answer": answer,
        "contexts": results,
        "steps": reasoning_steps,
    }


@app.get("/docs/schema")
def schema():
    # Minimal shape for frontend wiring/testing
    return {
        "upload": {"POST": {"multipart": True}},
        "chunk": {"POST": {"json": {"doc_id": "str", "chunk_size": "int", "overlap": "int"}}},
        "embed": {"POST": {"json": {"doc_ids": "List[str]|None"}}},
        "retrieve": {"POST": {"json": {"query": "str", "k": "int"}}},
        "generate": {"POST": {"json": {"query": "str", "top_k": "int"}}},
    }
