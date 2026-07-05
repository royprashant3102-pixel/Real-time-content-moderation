"""
serve.py — FastAPI endpoint for real-time content moderation using the ONNX model.

POST /predict — Takes text, returns label + confidence score.
"""

import os
import sys
import numpy as np

# ─── CONFIG ──────────────────────────────────────────────────────────────────
MAX_SEQ_LENGTH = 128
ONNX_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model", "onnx"))
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model", "best_model"))
LABELS = {0: "non-toxic", 1: "toxic"}
# ─────────────────────────────────────────────────────────────────────────────

from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Text to classify")


class PredictionResponse(BaseModel):
    label: str
    score: float
    toxic: bool
    confidence: dict


# Global model objects (loaded once at startup)
_session = None
_tokenizer = None


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
        _session = ort.InferenceSession(
            onnx_path, providers=["CPUExecutionProvider"]
        )
        print(f"✅ ONNX model loaded: {os.path.basename(onnx_path)}")
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
app = FastAPI(
    title="Content Moderation API",
    description="Real-time toxicity detection using ONNX-optimized DistilBERT",
    version="1.0.0",
    lifespan=lifespan,
)


def _softmax(logits):
    """Compute softmax probabilities."""
    exp = np.exp(logits - np.max(logits))
    return exp / exp.sum()


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Classify text as toxic or non-toxic."""
    if _session is None or _tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    try:
        # Tokenize
        inputs = _tokenizer(
            request.text,
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors="np",
        )

        # Run inference
        feed = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        }
        logits = _session.run(None, feed)[0][0]

        # Post-process
        probs = _softmax(logits)
        pred_label = int(np.argmax(probs))
        score = float(probs[pred_label])

        return PredictionResponse(
            label=LABELS[pred_label],
            score=round(score, 4),
            toxic=(pred_label == 1),
            confidence={
                "non-toxic": round(float(probs[0]), 4),
                "toxic": round(float(probs[1]), 4),
            },
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}",
        )


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
    uvicorn.run(app, host="0.0.0.0", port=8000)
