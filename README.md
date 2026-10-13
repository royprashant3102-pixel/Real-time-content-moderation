# Real-Time Content Moderation API

A production-grade, multilingual text classification microservice for detecting toxic and policy-violating content in **English, Hindi (Devanagari), and Hinglish (Romanized Hindi)**.

Fine-tuned from **Multilingual DistilBERT** (`distilbert-base-multilingual-cased`) and optimized for low-latency (**<15 ms**) CPU inference using **ONNX Runtime + dynamic INT8 quantization**, served via **FastAPI** with REST and WebSocket streaming endpoints.

---

## Architecture

```
Raw Text / File / URL / WebSocket
             │
             ▼
  Text Normalizer (clean_text)
             │
             ▼
  Multilingual DistilBERT Tokenizer (max_length = 128)
             │
             ▼
  ONNX Runtime (Dynamic INT8 Quantized Session)
             │
     ┌───────┴───────┐
     ▼               ▼
Model Confidence   Rule Flagging (Explicit profanity pattern)
  (Score: 0 - 1)     (Independent metadata flag)
     └───────┬───────┘
             ▼
     FastAPI Response
```

| Component | Specification |
|-----------|---------------|
| **Base Model** | `distilbert-base-multilingual-cased` (135M params, 104 languages) |
| **Datasets** | `google/civil_comments` (30K samples) + `textdetox/multilingual_toxicity_dataset` (Hindi & Hinglish) |
| **Training** | HuggingFace `Trainer`, 6 epochs, class-weighted CrossEntropyLoss, 15% warmup ratio |
| **Export & Optimization** | PyTorch → ONNX graph optimization → dynamic INT8 quantization |
| **Inference Engine** | ONNX Runtime (`CPUExecutionProvider`, sequential thread pool) |
| **API Framework** | FastAPI + Uvicorn + WebSockets |
| **Security** | SSRF mitigation for `/predict/url`, CORS middleware, input bounds validation |

---

## Features

- **Multilingual Moderation**: Detects toxicity across English, pure Hindi (Devanagari: e.g. *भड़वा, चूतिया*), and code-switched Hinglish (e.g. *bhenchod, oye chutiye, hello randi*).
- **Sub-15ms Latency**: Dynamic INT8 quantization delivers **12.8 ms** average latency per sample on CPU.
- **74.9% Compression**: Model footprint reduced from **255.5 MB** to **64.2 MB** with negligible accuracy impact.
- **SSRF-Protected URL Analysis**: Securely extracts and classifies remote web content while filtering loopback, RFC 1918 private ranges, and cloud metadata endpoints.
- **Multi-Format Document Ingestion**: Supports `.txt`, `.pdf`, `.docx`, `.csv`, `.md`, and `.json` with automatic chunking.
- **Real-Time Live Typing**: Full-duplex WebSocket stream (`/ws/predict`) with sub-5ms feedback for live typing experiences.
- **Calibrated ML + Rule Metadata**: Transparent moderation logic where machine learning probabilities are preserved, while keyword hits are surfaced independently as `rule_flagged` and `rule_matches`.

---

## Installation & Setup

```bash
# Clone the repository
git clone https://github.com/royprashant3102-pixel/Real-time-content-moderation.git
cd Real-time-content-moderation

# Set up virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Running the Pipeline

### 1. Data Preparation
```bash
python src/data.py
```
Loads and balances English Civil Comments with TextDetox Hindi & Hinglish splits, applies text cleaning, deduplicates templates (zero train/val/test leakage), and creates tokenized splits.

### 2. Fine-Tuning
```bash
python src/train.py
```
Fine-tunes multilingual DistilBERT for 6 epochs on CPU with class-weight balancing and 15% warmup. Best checkpoint saved to `model/best_model/`.

### 3. Evaluation
```bash
python src/evaluate.py
```
Reports Precision, Recall, F1 score, PR-AUC, Confusion Matrix, and the Quantized ONNX accuracy delta on the held-out test set.

### 4. ONNX Export & INT8 Quantization
```bash
python src/export_onnx.py
```
Exports model to `model/onnx/`:
- `model.onnx` — Raw export
- `model_optimized.onnx` — ORT graph-optimized
- `model_quantized.onnx` — INT8 quantized

### 5. Benchmark
```bash
python src/benchmark.py
```
Benchmarks 100 inference passes comparing PyTorch vs ONNX latency.

### 6. Start the API Server
```bash
uvicorn src.serve:app --host 0.0.0.0 --port 8000
```
Open **`http://localhost:8000`** in your browser to interact with the modern Web UI.

---

## API Endpoints

### Single Text Classification
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "tu sabse bada chutiya hai"}'
```

**Response:**
```json
{
  "label": "toxic",
  "score": 0.9854,
  "toxic": true,
  "confidence": {
    "non-toxic": 0.0146,
    "toxic": 0.9854
  },
  "rule_flagged": true,
  "rule_matches": ["chutiya"]
}
```

### URL Analysis (SSRF Protected)
```bash
curl -X POST http://localhost:8000/predict/url \
  -H "Content-Type: application/json" \
  -d '{"url": "https://en.wikipedia.org/wiki/Constructive_criticism"}'
```

### Live Typing WebSocket
Connect to `ws://localhost:8000/ws/predict` and stream text payloads:
```json
{"text": "hello buddy"}
```

---

## Performance & Benchmarks

### Test Evaluation Metrics (Held-Out Test Set)

| Metric | PyTorch (FP32) | Quantized ONNX (INT8) | Delta |
|--------|----------------|-----------------------|-------|
| **Accuracy** | 95.05% | 94.92% | -0.13% |
| **Precision** | 89.28% | 88.94% | -0.34% |
| **Recall** | 84.21% | 84.10% | -0.11% |
| **F1 Score** | 0.8667 | 0.8645 | -0.0022 |

### Latency Benchmark (Apple Silicon CPU, Batch Size = 1)

| Model | Mean | Median | P95 |
|-------|------|--------|-----|
| PyTorch (FP32) | 17.14 ms | 16.67 ms | 20.16 ms |
| **ONNX Quantized (INT8)** | **12.81 ms** | **12.55 ms** | **14.36 ms** |
| **Speedup** | **1.34×** | | |

### Model Footprint

| Artifact | Size | Compression |
|----------|------|-------------|
| PyTorch Checkpoint | 541 MB | Baseline |
| `model.onnx` (FP32) | 255.5 MB | 52.8% reduction |
| `model_quantized.onnx` (INT8) | **64.2 MB** | **74.9% reduction** |

---

## Testing & CI

Run the complete test suite:
```bash
pytest tests/ -v
```

All 23 pipeline, data, ONNX, and API tests pass. Continuous Integration is enforced via GitHub Actions (`.github/workflows/ci.yml`).

---

## License
MIT License.
