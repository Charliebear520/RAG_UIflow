from __future__ import annotations

import io
import os
import uuid
from dataclasses import dataclass
import re
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv
from PyPDF2 import PdfReader

load_dotenv()


def get_env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.lower() in {"1", "true", "yes", "on"}


GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
USE_GEMINI_EMBEDDING = get_env_bool("USE_GEMINI_EMBEDDING", False)
USE_GEMINI_COMPLETION = get_env_bool("USE_GEMINI_COMPLETION", False)

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
        self.embeddings = None  # matrix for chunks (numpy array or list)
        self.chunk_doc_ids: List[str] = []
        self.chunks_flat: List[str] = []

    def reset_embeddings(self):
        """Clear vector/index state so embeddings can be recomputed."""
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


async def embed_gemini(texts: List[str]) -> List[List[float]]:
    """Call Google Generative API (Gemini) embeddings endpoint using API key.

    Note: This uses the REST endpoint pattern with the API key in query params.
    """
    if not httpx:
        raise RuntimeError("httpx not available")
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY not set")
    model = os.getenv("GOOGLE_EMBEDDING_MODEL", "embed-gecko-001")
    # endpoint pattern: models/{model}:embed
    url = f"https://generativelanguage.googleapis.com/v1beta2/models/{model}:embed?key={GOOGLE_API_KEY}"
    out: List[List[float]] = []
    async with httpx.AsyncClient(timeout=60) as client:
        batch_size = 64
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            payload = {"input": batch}
            r = await client.post(url, json=payload)
            r.raise_for_status()
            data = r.json()
            # expected shape: { data: [ { embedding: [...] }, ... ] }
            out.extend([d.get("embedding") for d in data.get("data", [])])
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

    if USE_GEMINI_EMBEDDING:
        vectors = await embed_gemini(all_chunks)
        # keep as numpy matrix-like list of lists; for cosine sim we'll rely on numpy if available
        store.tfidf = None
        store.embeddings = vectors
        store.chunk_doc_ids = chunk_doc_ids
        store.chunks_flat = all_chunks
        return {"provider": "gemini", "num_vectors": len(vectors)}
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


def rank_with_gemini(query: str, k: int):
    # cosine similarity on dense vectors list
    import numpy as np
    vecs = np.array(store.embeddings, dtype=float)  # type: ignore[assignment]
    qvec = np.array(asyncio_run(embed_gemini([query]))[0], dtype=float)
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
        idxs, sims = rank_with_gemini(req.query, req.k)

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


async def gemini_chat(messages: List[Dict[str, str]]) -> str:
    if not httpx:
        raise RuntimeError("httpx not available")
    if not GOOGLE_API_KEY:
        raise RuntimeError("GOOGLE_API_KEY not set")
    model = os.getenv("GOOGLE_CHAT_MODEL", "gemini-1.5")
    # Use Generative Language API: models/{model}:generateText
    url = f"https://generativelanguage.googleapis.com/v1beta2/models/{model}:generateText?key={GOOGLE_API_KEY}"
    # Flatten messages into a single prompt
    prompt = "".join([f"{m.get('role','user')}: {m.get('content','')}\n" for m in messages])
    payload = {"prompt": {"text": prompt}, "temperature": 0.2}
    async with httpx.AsyncClient(timeout=60) as client:
        r = await client.post(url, json=payload)
        r.raise_for_status()
        data = r.json()
        # Expect generatedText in response
        if "candidates" in data:
            return data["candidates"][0].get("output", "").strip()
        return data.get("output", "").strip()


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

    if USE_GEMINI_COMPLETION:
        prompt = [
            {"role": "system", "content": "You are a helpful assistant. Answer using ONLY the provided context. If missing, say you don't know."},
            {"role": "user", "content": f"Query: {req.query}\n\nContext:\n" + "\n---\n".join(contexts)},
        ]
        try:
            answer = asyncio_run(gemini_chat(prompt))
        except Exception as e:
            answer = f"Gemini call failed: {e}. Falling back to extractive answer.\n" + simple_extractive_answer(req.query, contexts)
    else:
        answer = simple_extractive_answer(req.query, contexts)

    return {
        "query": req.query,
        "answer": answer,
        "contexts": results,
        "steps": reasoning_steps,
    }


@app.post("/convert")
async def convert(file: UploadFile = File(...)):
    try:
        # Validate file type
        if not file.filename or not file.filename.lower().endswith('.pdf'):
            raise HTTPException(status_code=400, detail="只支持PDF文件格式")
        
        # Reset file pointer to beginning
        await file.seek(0)
        
        # Read PDF content safely; skip pages with no text
        try:
            reader = PdfReader(file.file)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"無法讀取PDF文件: {str(e)}")
        
        texts = []
        for page in reader.pages:
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            texts.append(t)
        
        if not any(texts):  # No text extracted
            raise HTTPException(status_code=400, detail="PDF文件中没有找到可提取的文本内容")
            
        full_text = "\n".join(texts)

        def normalize_digits(s: str) -> str:
            # Convert fullwidth digits to ASCII for simpler matching
            fw = "０１２３４５６７８９"
            hw = "0123456789"
            return s.translate(str.maketrans(fw, hw))

        # Determine law name: first non-empty line containing a legal keyword, else filename
        lines = [normalize_digits((ln or "").strip()) for ln in full_text.splitlines()]
        law_name = None
        for ln in lines:
            if not ln:
                continue
            if any(key in ln for key in ["法", "條例", "法規", "法律"]):
                law_name = ln
                break
        if not law_name:
            base = os.path.splitext(file.filename or "document")[0]
            law_name = base or "未命名法規"

        chapter_re = re.compile(r"^第\s*([一二三四五六七八九十百千0-9]+)\s*章[\u3000\s]*(.*)$")
        section_re = re.compile(r"^第\s*([一二三四五六七八九十百千0-9]+)\s*節[\u3000\s]*(.*)$")
        article_re = re.compile(r"^第\s*([一二三四五六七八九十百千0-9]+(?:之[一二三四五六七八九十0-9]+)?)\s*條[\u3000\s]*(.*)$")

        def parse_item_line(ln: str):
            # Match common item markers like 「一、」「1.」「（一）」「(1)」「1）」 etc.
            ln = ln.lstrip()
            # （一） or (1)
            m = re.match(r"^[（(]([0-9０-９一二三四五六七八九十]+)[）)]\s*(.*)$", ln)
            if m:
                return m.group(1), m.group(2), "parentheses"
            # 一、 二、 十、 style (Chinese numerals with punctuation)
            m = re.match(r"^([一二三四五六七八九十]+)[、．.）)]\s*(.*)$", ln)
            if m:
                return m.group(1), m.group(2), "chinese_with_punct"
            # 1. 1、 1) styles (Arabic numbers with punctuation)
            m = re.match(r"^([0-9０-９]+)[、．.）)]\s*(.*)$", ln)
            if m:
                return m.group(1), m.group(2), "arabic_with_punct"
            # 1 2 3 styles (Arabic numbers followed by space, common in ROC legal documents)
            m = re.match(r"^([0-9０-９]+)\s+(.*)$", ln)
            if m:
                return m.group(1), m.group(2), "arabic_space"
            # 一 二 三 styles (Chinese numerals followed by space, sub-items)
            m = re.match(r"^([一二三四五六七八九十]+)\s+(.*)$", ln)
            if m:
                return m.group(1), m.group(2), "chinese_space"
            return None, None, None

        structure: Dict[str, Any] = {"law_name": law_name, "chapters": []}
        current_chapter: Optional[Dict[str, Any]] = None
        current_section: Optional[Dict[str, Any]] = None
        current_article: Optional[Dict[str, Any]] = None
        current_item: Optional[Dict[str, Any]] = None
        current_sub_item: Optional[Dict[str, Any]] = None

        def ensure_chapter():
            nonlocal current_chapter
            if current_chapter is None:
                current_chapter = {"chapter": "未分類章", "sections": []}
                structure["chapters"].append(current_chapter)

        def ensure_section():
            nonlocal current_section
            ensure_chapter()
            if current_section is None:
                current_section = {"section": "未分類節", "articles": []}
                current_chapter["sections"].append(current_section)

        for raw in lines:
            ln = raw.strip()
            if not ln:
                continue

            # Headings
            m = chapter_re.match(ln)
            if m:
                title = f"第{m.group(1)}章" + (f" {m.group(2).strip()}" if m.group(2) else "")
                current_chapter = {"chapter": title, "sections": []}
                structure["chapters"].append(current_chapter)
                current_section = None
                current_article = None
                current_item = None
                current_sub_item = None
                continue

            m = section_re.match(ln)
            if m:
                ensure_chapter()
                title = f"第{m.group(1)}節" + (f" {m.group(2).strip()}" if m.group(2) else "")
                current_section = {"section": title, "articles": []}
                current_chapter["sections"].append(current_section)
                current_article = None
                current_item = None
                current_sub_item = None
                current_sub_item = None
                continue

            m = article_re.match(ln)
            if m:
                ensure_section()
                title = f"第{m.group(1)}條"
                rest = m.group(2).strip() if m.group(2) else ""
                current_article = {"article": title, "content": rest, "items": []}
                current_section["articles"].append(current_article)
                current_item = None
                current_sub_item = None
                current_sub_item = None
                continue

            # Items within article
            if current_article is not None:
                num, content, item_type = parse_item_line(ln)
                if num is not None:
                    num = normalize_digits(num)
                    
                    # Determine if this is a sub-item (Chinese numerals after Arabic numerals)
                    if (item_type == "chinese_space" or item_type == "chinese_with_punct") and current_item is not None:
                        # This is a sub-item of the current item
                        if "sub_items" not in current_item:
                            current_item["sub_items"] = []
                        sub_item = {"item": str(num), "content": content or "", "sub_items": []}
                        current_item["sub_items"].append(sub_item)
                        current_sub_item = sub_item
                    else:
                        # This is a main item
                        current_item = {"item": str(num), "content": content or "", "sub_items": []}
                        current_article["items"].append(current_item)
                        current_sub_item = None
                else:
                    # continuation line
                    if current_sub_item is not None:
                        # Append to current sub-item
                        sep = "\n" if current_sub_item["content"] else ""
                        current_sub_item["content"] = f"{current_sub_item['content']}{sep}{ln}"
                    elif current_item is not None:
                        # Check if we have sub-items and append to the last sub-item
                        if "sub_items" in current_item and current_item["sub_items"]:
                            last_sub_item = current_item["sub_items"][-1]
                            sep = "\n" if last_sub_item["content"] else ""
                            last_sub_item["content"] = f"{last_sub_item['content']}{sep}{ln}"
                        else:
                            sep = "\n" if current_item["content"] else ""
                            current_item["content"] = f"{current_item['content']}{sep}{ln}"
                    else:
                        # accumulate into article content
                        if "content" not in current_article or current_article["content"] is None:
                            current_article["content"] = ln
                        else:
                            current_article["content"] = (current_article["content"] + "\n" + ln).strip()
                continue

            # If no article yet, but we have text, place it under a default article
            if current_section is not None and current_article is None:
                current_article = {"article": "未標示條文", "content": ln, "items": []}
                current_section["articles"].append(current_article)
                current_item = None
                current_sub_item = None
            elif current_article is None:
                ensure_section()
                current_article = {"article": "未標示條文", "content": ln, "items": []}
                current_section["articles"].append(current_article)
                current_item = None
                current_sub_item = None
            else:
                # fallback append
                current_article["content"] = (current_article.get("content", "") + "\n" + ln).strip()

        return structure
    except HTTPException:
        # Re-raise HTTPExceptions with their original status codes
        raise
    except Exception as e:
        # For other unexpected errors, return 500
        raise HTTPException(status_code=500, detail=f"處理PDF時發生意外錯誤: {str(e)}")


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
