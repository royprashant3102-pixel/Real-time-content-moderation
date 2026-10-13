# Resume Bullet Points

*(Based on measured benchmarks & production architecture)*

- **Fine-tuned Multilingual DistilBERT** (`distilbert-base-multilingual-cased`, 104 languages) for binary toxicity detection across English and Hindi/Hinglish, achieving **95.1% accuracy** and **0.867 F1** with class-weighted CrossEntropyLoss.
- **Engineered low-latency ONNX Runtime inference pipeline** with dynamic INT8 weight quantization, achieving **~12.8 ms per-sample** latency on CPU (well under 50 ms SLA) with **74.9% model compression** (255 MB → 64 MB).
- **Built production-grade FastAPI microservice** supporting single-text, bulk batching, file parsing (PDF, DOCX, TXT), URL scraping with SSRF mitigation, and full-duplex WebSocket real-time live typing moderation.
- **Resolved dataset leakage and evaluation bias** by deduplicating Hinglish curated templates, splitting before augmentation, evaluating against TextDetox multilingual benchmarks, and decoupling regex rules from ML calibration.
- **Implemented end-to-end testing & CI**: 24-test pytest suite covering data sanitization, ONNX parity, and API endpoints with automated GitHub Actions CI pipeline.
