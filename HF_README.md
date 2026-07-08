---
title: Real-Time Content Moderation
emoji: 🛡️
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 7860
pinned: true
short_description: DistilBERT toxicity detector — 5.48ms avg latency, ONNX INT8
---

# 🛡️ Real-Time Content Moderation

A production-grade text classifier that detects **toxic / policy-violating content**, fine-tuned from **DistilBERT** and optimized for low-latency (**5.48 ms avg**) inference using **ONNX + INT8 quantization**, served via a **FastAPI** REST endpoint with a full web UI.

---

## 🚀 Live Demo

Open the app above — paste any text, upload a file, or enter a URL to scan for toxic content in real time.

---

## ⚡ Performance

| Model | Mean Latency | Speedup |
|---|---|---|
| PyTorch (CPU) | 10.81 ms | baseline |
| **ONNX INT8 Quantized** | **5.48 ms** | **1.97×** |

> **9× faster than the 50ms production target!**

---

## 🏗️ Architecture

```
raw text → tokenize → DistilBERT (fine-tuned) → ONNX INT8 quantized → FastAPI /predict
```

| Component | Technology |
|---|---|
| Base model | `distilbert-base-uncased` (66M params) |
| Dataset | `google/civil_comments` (10,000 samples) |
| Fine-tuning | HuggingFace `Trainer`, 2 epochs |
| Export | `torch.onnx.export` + ORT graph optimizer |
| Optimization | Dynamic INT8 quantization (74.9% smaller) |
| Serving | FastAPI + uvicorn + ONNX Runtime |

---

## 📊 Results

| Metric | Value |
|---|---|
| **Accuracy** | **95.27%** |
| **F1 Score** | **0.502** |
| Model size | **64 MB** (from 255 MB — 74.9% reduction) |
| Avg latency | **5.48 ms** |

---

## 🔌 API Endpoints

| Endpoint | Description |
|---|---|
| `POST /predict` | Classify a single text |
| `POST /predict/bulk` | Large text with chunk breakdown |
| `POST /predict/file` | Upload `.txt`, `.pdf`, `.docx`, `.csv` |
| `POST /predict/url` | Scrape and analyze any URL |
| `GET /health` | Health check |
| `GET /docs` | Swagger UI |

### Example
```bash
curl -X POST https://your-space.hf.space/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "You are so stupid, nobody likes you."}'
```

```json
{
  "label": "toxic",
  "score": 0.8932,
  "toxic": true,
  "confidence": {"non-toxic": 0.1068, "toxic": 0.8932}
}
```

---

## 🔗 Source Code

[GitHub Repository](https://github.com/royprashant3102-pixel/Real-time-content-moderation)
