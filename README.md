# Real-Time Content Moderation API

A production-grade, multilingual text classification microservice for detecting toxic and policy-violating content in **English, Hindi (Devanagari), and Hinglish (Romanized Hindi)**.

Fine-tuned from **Multilingual DistilBERT** (`distilbert-base-multilingual-cased`) and optimized for low-latency CPU inference using **ONNX Runtime + dynamic INT8 quantization**, served via **FastAPI** with REST and WebSocket streaming endpoints.

**Live demo:** [real-time-content-moderation-1.onrender.com](https://real-time-content-moderation-1.onrender.com)
*(hosted on Render's free tier — File Upload and URL Analysis are disabled on this deployment due to the 512MB RAM limit; both work locally. The free instance also sleeps after 15 min idle, so the first load may take 30-60s to wake up.)*

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
- **Low-Latency CPU Inference**: Dynamic INT8 quantization keeps single-request round-trip latency in the low single-digit milliseconds on CPU.
- **~75% Compression**: Model footprint reduced from **516 MB** (PyTorch FP32) to **129 MB** (ONNX INT8) with a modest, documented accuracy tradeoff (see Performance section).
- **SSRF-Protected URL Analysis**: Securely extracts and classifies remote web content while filtering loopback, RFC 1918 private ranges, and cloud metadata endpoints. *(Disabled on the free-tier live demo — see note above; fully functional locally.)*
- **Multi-Format Document Ingestion**: Supports `.txt`, `.pdf`, `.docx`, `.csv`, `.md`, and `.json` with automatic chunking. *(Disabled on the free-tier live demo — see note above; fully functional locally.)*
- **Real-Time Live Typing**: Full-duplex WebSocket stream (`/ws/predict`) with live per-keystroke feedback, debounced client-side and staleness-checked against out-of-order responses.
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
  "score": 0.9975,
  "toxic": true,
  "confidence": {
    "non-toxic": 0.0025,
    "toxic": 0.9975
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
| **Accuracy** | 92.08% | 88.65% | -3.43% |
| **Precision** | 82.94% | 78.91% | -4.03% |
| **Recall** | 94.81% | 88.16% | -6.65% |
| **F1 Score** | 0.8848 | 0.8328 | -0.0520 |
| **PR-AUC** | 0.9487 | 0.8585 | -0.0902 |

Quantization trades a few points of accuracy for a ~75% smaller, faster-loading model — a deliberate tradeoff for deployment on memory-constrained hosts. Run `python src/evaluate.py` to reproduce against your own retrain.

### Model Footprint

| Artifact | Size | Compression |
|----------|------|-------------|
| PyTorch Checkpoint | 516 MB | Baseline |
| `model_quantized.onnx` (INT8) | **129 MB** | **~75% reduction** |

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
