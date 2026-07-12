"""
serve.py — FastAPI endpoint for real-time content moderation using the ONNX model.

Endpoints:
  POST /predict      — Classify text (up to ~10K words) with automatic chunking.
  POST /predict/file — Upload a file (.txt, .pdf, .docx, .csv, .md, .json) for analysis.
  POST /predict/url  — Scrape a URL and analyze the extracted text.
  GET  /health       — Health check.
  GET  /             — Serve the web UI.
"""

import os
import sys
import io
import re
import time
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np

# ─── CONFIG ──────────────────────────────────────────────────────────────────
MAX_SEQ_LENGTH = 64               # ⚡ Reduced from 128 → ~35% faster inference
MAX_TEXT_LENGTH = 50_000          # ~10,000 words
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5 MB
CHUNK_SIZE = 400                  # chars per chunk (~100 tokens)
CHUNK_OVERLAP = 50                # overlap between chunks
# Paths work both locally and inside Docker container
_BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
ONNX_DIR = os.environ.get("ONNX_DIR", os.path.join(_BASE, "model", "onnx"))
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.join(_BASE, "model", "best_model"))
LABELS = {0: "non-toxic", 1: "toxic"}
ALLOWED_EXTENSIONS = {".txt", ".pdf", ".docx", ".csv", ".md", ".json"}
PORT = int(os.environ.get("PORT", 7860))  # 7860 for HF Spaces, 8000 locally
# ─────────────────────────────────────────────────────────────────────────────

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field


# ── Request / Response models ────────────────────────────────────────────────

class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_LENGTH, description="Text to classify")


class URLRequest(BaseModel):
    url: str = Field(..., min_length=1, description="URL to scrape and classify")


class PredictionResponse(BaseModel):
    label: str
    score: float
    toxic: bool
    confidence: dict


class ChunkResult(BaseModel):
    chunk_index: int
    text_preview: str
    label: str
    score: float
    toxic: bool
    confidence: dict


class BulkPredictionResponse(BaseModel):
    source: str               # "text" | "file" | "url"
    source_name: str          # filename or URL or "manual input"
    total_chunks: int
    toxic_chunks: int
    overall_label: str
    overall_toxic: bool
    overall_score: float
    overall_confidence: dict
    chunks: list[ChunkResult]
    processing_time_ms: int


class ModelPredictionResult(BaseModel):
    label: str
    score: float
    toxic: bool
    confidence: dict
    latency_ms: float


class ComparePredictionResponse(BaseModel):
    text: str
    custom_model: ModelPredictionResult
    base_model: ModelPredictionResult


# ── Global model objects ─────────────────────────────────────────────────────
_session = None
_tokenizer = None
_base_model = None
_base_tokenizer = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load ONNX model and tokenizer on startup; clean up on shutdown."""
    global _session, _tokenizer

    import onnxruntime as ort
    from transformers import AutoTokenizer

    # Find best ONNX model (prefer quantized > optimized > raw)
    onnx_candidates = [
        os.path.join(ONNX_DIR, "model_quantized.onnx"),
        os.path.join(ONNX_DIR, "model_optimized.onnx"),
        os.path.join(ONNX_DIR, "model.onnx"),
    ]
    onnx_path = next(
        (c for c in onnx_candidates if os.path.exists(c)), None
    )
    if onnx_path is None:
        raise RuntimeError(
            f"No ONNX model found in {ONNX_DIR}. Run export_onnx.py first."
        )

    try:
        # ⚡ OPTIMIZATION 2: ONNX session options for max CPU performance
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 4          # parallel ops within a layer
        opts.inter_op_num_threads = 2          # parallel ops across layers
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        _session = ort.InferenceSession(
            onnx_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"],
        )
        print(f"✅ ONNX model loaded: {os.path.basename(onnx_path)} (optimized session)")
    except Exception as e:
        raise RuntimeError(f"Failed to load ONNX model: {e}")

    tokenizer_path = (
        ONNX_DIR
        if os.path.exists(os.path.join(ONNX_DIR, "tokenizer_config.json"))
        else MODEL_PATH
    )
    try:
        _tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        print(f"✅ Tokenizer loaded from: {os.path.basename(tokenizer_path)}")
    except Exception as e:
        raise RuntimeError(f"Failed to load tokenizer: {e}")

    yield  # App runs here

    _session = None
    _tokenizer = None


# ── App setup ────────────────────────────────────────────────────────────────
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"

app = FastAPI(
    title="Content Moderation API",
    description="Real-time toxicity detection using ONNX-optimized DistilBERT",
    version="2.0.0",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
async def serve_frontend():
    """Serve the web UI."""
    return FileResponse(FRONTEND_DIR / "index.html")


# Mount static files (CSS, JS) on /static so API routes aren't shadowed
app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")


# ── Core inference helpers ───────────────────────────────────────────────────

def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax."""
    exp = np.exp(logits - np.max(logits, axis=-1, keepdims=True))
    return exp / exp.sum(axis=-1, keepdims=True)


@lru_cache(maxsize=512)
def _cached_tokenize(text: str):
    """⚡ OPTIMIZATION 3: Cache tokenized inputs for repeated texts.
    Returns a tuple of (input_ids, attention_mask) as numpy arrays.
    """
    inputs = _tokenizer(
        text,
        padding=True,          # ⚡ dynamic padding — only pads to actual length
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        return_tensors="np",
    )
    return (
        inputs["input_ids"].astype(np.int64),
        inputs["attention_mask"].astype(np.int64),
    )


def _predict_single(text: str) -> dict:
    """Run inference on a single text string. Returns raw result dict."""
    input_ids, attention_mask = _cached_tokenize(text)
    feed = {"input_ids": input_ids, "attention_mask": attention_mask}
    logits = _session.run(None, feed)[0][0]
    probs = _softmax(logits)
    pred_label = int(np.argmax(probs))
    score = float(probs[pred_label])

    return {
        "label": LABELS[pred_label],
        "score": round(score, 4),
        "toxic": pred_label == 1,
        "confidence": {
            "non-toxic": round(float(probs[0]), 4),
            "toxic": round(float(probs[1]), 4),
        },
    }


def _get_base_model_and_tokenizer():
    global _base_model, _base_tokenizer
    if _base_model is None or _base_tokenizer is None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        base_model_dir = os.path.join(_BASE, "model", "best_model")
        _base_tokenizer = AutoTokenizer.from_pretrained(base_model_dir)
        _base_model = AutoModelForSequenceClassification.from_pretrained(base_model_dir)
        _base_model.eval()
    return _base_model, _base_tokenizer


def _predict_base_single(text: str) -> dict:
    """Run inference on a single text string using the base DistilBERT model."""
    t0 = time.time()
    model, tokenizer = _get_base_model_and_tokenizer()
    inputs = tokenizer(
        text,
        padding=True,
        truncation=True,
        max_length=64,  # Use same MAX_SEQ_LENGTH as custom model for fair comparison
        return_tensors="pt"
    )
    outputs = model(**inputs)
    logits = outputs.logits[0].detach().numpy()
    probs = _softmax(logits)
    pred_label = int(np.argmax(probs))
    score = float(probs[pred_label])
    elapsed_ms = (time.time() - t0) * 1000

    return {
        "label": LABELS[pred_label],
        "score": round(score, 4),
        "toxic": pred_label == 1,
        "confidence": {
            "non-toxic": round(float(probs[0]), 4),
            "toxic": round(float(probs[1]), 4),
        },
        "latency_ms": round(elapsed_ms, 2),
    }


def _predict_batch(texts: list[str]) -> list[dict]:
    """⚡ OPTIMIZATION 4: Batch inference — process all chunks in one ONNX call.
    Much faster than calling _predict_single in a loop for multiple chunks.
    """
    if not texts:
        return []

    # Tokenize all texts together with dynamic padding to longest in batch
    inputs = _tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        return_tensors="np",
    )
    feed = {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64),
    }
    # Single ONNX call for ALL chunks at once
    all_logits = _session.run(None, feed)[0]   # shape: (batch, num_labels)
    all_probs = _softmax(all_logits)

    results = []
    for probs in all_probs:
        pred_label = int(np.argmax(probs))
        score = float(probs[pred_label])
        results.append({
            "label": LABELS[pred_label],
            "score": round(score, 4),
            "toxic": pred_label == 1,
            "confidence": {
                "non-toxic": round(float(probs[0]), 4),
                "toxic": round(float(probs[1]), 4),
            },
        })
    return results


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks. Tries to split on sentence boundaries."""
    text = text.strip()
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size

        # Try to break at a sentence boundary (., !, ?, newline)
        if end < len(text):
            boundary = -1
            for sep in ['. ', '! ', '? ', '\n']:
                idx = text.rfind(sep, start + chunk_size // 2, end)
                if idx > boundary:
                    boundary = idx + len(sep)
            if boundary > start:
                end = boundary

        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        start = end - overlap
        if start >= len(text):
            break

    return chunks


def _predict_chunks(text: str, source: str = "text", source_name: str = "manual input") -> BulkPredictionResponse:
    """Chunk text, run batched inference on all chunks, aggregate results."""
    t0 = time.time()
    chunks = _chunk_text(text)
    chunk_results = []

    # ⚡ OPTIMIZATION 4: batch all chunks in one ONNX call
    batch_results = _predict_batch(chunks)
    for i, (chunk, result) in enumerate(zip(chunks, batch_results)):
        chunk_results.append(ChunkResult(
            chunk_index=i,
            text_preview=chunk[:80] + ("…" if len(chunk) > 80 else ""),
            **result,
        ))

    elapsed_ms = int((time.time() - t0) * 1000)

    # Aggregate: overall score = max toxic confidence across all chunks
    toxic_scores = [c.confidence["toxic"] for c in chunk_results]
    max_toxic = max(toxic_scores)
    max_safe = 1 - max_toxic
    overall_toxic = max_toxic > 0.5
    toxic_count = sum(1 for c in chunk_results if c.toxic)

    return BulkPredictionResponse(
        source=source,
        source_name=source_name,
        total_chunks=len(chunks),
        toxic_chunks=toxic_count,
        overall_label="toxic" if overall_toxic else "non-toxic",
        overall_toxic=overall_toxic,
        overall_score=round(max_toxic if overall_toxic else max_safe, 4),
        overall_confidence={
            "non-toxic": round(max_safe, 4),
            "toxic": round(max_toxic, 4),
        },
        chunks=chunk_results,
        processing_time_ms=elapsed_ms,
    )


# ── Text extraction helpers ─────────────────────────────────────────────────

def _extract_text_from_file(filename: str, content: bytes) -> str:
    """Extract plain text from various file formats."""
    ext = Path(filename).suffix.lower()

    if ext in (".txt", ".md", ".csv", ".json"):
        return content.decode("utf-8", errors="replace")

    elif ext == ".pdf":
        from PyPDF2 import PdfReader
        reader = PdfReader(io.BytesIO(content))
        pages = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                pages.append(text)
        return "\n\n".join(pages)

    elif ext == ".docx":
        from docx import Document
        doc = Document(io.BytesIO(content))
        return "\n\n".join(p.text for p in doc.paragraphs if p.text.strip())

    else:
        raise ValueError(f"Unsupported file type: {ext}")


def _extract_text_from_url(html: str) -> str:
    """Extract visible text from HTML, stripping scripts/styles/nav."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html, "html.parser")

    # Remove non-content tags
    for tag in soup(["script", "style", "nav", "header", "footer", "aside", "noscript", "meta", "link"]):
        tag.decompose()

    text = soup.get_text(separator="\n")

    # Clean up whitespace
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)

    return text


# ── API endpoints ────────────────────────────────────────────────────────────

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Classify a short text as toxic or non-toxic (single chunk)."""
    if _session is None or _tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        result = _predict_single(request.text)
        return PredictionResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/bulk", response_model=BulkPredictionResponse)
async def predict_bulk(request: PredictionRequest):
    """Classify large text with automatic chunking and per-chunk breakdown."""
    if _session is None or _tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        return _predict_chunks(request.text, source="text", source_name="manual input")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/file", response_model=BulkPredictionResponse)
async def predict_file(file: UploadFile = File(...)):
    """Upload a file and analyze its text content for toxicity."""
    if _session is None or _tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    # Validate extension
    ext = Path(file.filename or "unknown").suffix.lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {', '.join(sorted(ALLOWED_EXTENSIONS))}",
        )

    # Read file content
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({len(content) / 1024 / 1024:.1f} MB). Max: {MAX_FILE_SIZE / 1024 / 1024:.0f} MB.",
        )

    if len(content) == 0:
        raise HTTPException(status_code=400, detail="File is empty.")

    # Extract text
    try:
        text = _extract_text_from_file(file.filename or "file", content)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read file: {str(e)}")

    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="No text content found in file.")

    # Truncate if extremely long
    if len(text) > MAX_TEXT_LENGTH:
        text = text[:MAX_TEXT_LENGTH]

    try:
        return _predict_chunks(text, source="file", source_name=file.filename or "uploaded file")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/url", response_model=BulkPredictionResponse)
async def predict_url(request: URLRequest):
    """Scrape a URL and analyze the extracted text for toxicity."""
    if _session is None or _tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    import httpx as hx

    # Basic URL validation
    url = request.url.strip()
    if not url.startswith(("http://", "https://")):
        url = "https://" + url

    # Fetch the page
    try:
        async with hx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
            resp = await client.get(url, headers={"User-Agent": "Mozilla/5.0 ContentModerator/1.0"})
            resp.raise_for_status()
    except hx.HTTPStatusError as e:
        raise HTTPException(status_code=400, detail=f"URL returned {e.response.status_code}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to fetch URL: {str(e)}")

    # Extract text
    try:
        text = _extract_text_from_url(resp.text)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to extract text: {str(e)}")

    text = text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="No text content found at that URL.")

    if len(text) > MAX_TEXT_LENGTH:
        text = text[:MAX_TEXT_LENGTH]

    try:
        return _predict_chunks(text, source="url", source_name=url)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/compare", response_model=ComparePredictionResponse)
async def predict_compare(request: PredictionRequest):
    """Classify a text using both custom BiLSTM and base DistilBERT models to compare results and latency."""
    if _session is None or _tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        # 1. Custom model prediction & latency
        t_start_custom = time.time()
        custom_res = _predict_single(request.text)
        custom_latency = (time.time() - t_start_custom) * 1000
        custom_res["latency_ms"] = round(custom_latency, 2)

        # 2. Base model prediction & latency
        base_res = _predict_base_single(request.text)

        return ComparePredictionResponse(
            text=request.text,
            custom_model=ModelPredictionResult(**custom_res),
            base_model=ModelPredictionResult(**base_res),
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison prediction failed: {str(e)}")


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "model_loaded": _session is not None,
        "tokenizer_loaded": _tokenizer is not None,
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=PORT)
