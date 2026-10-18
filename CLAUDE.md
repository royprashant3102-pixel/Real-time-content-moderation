# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Setup
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt          # full (training + serving)
pip install -r requirements-deploy.txt   # serving only, no torch — what Docker/Render actually installs

# Full pipeline, in order (each stage reads/writes model/ and data/)
python src/data.py            # downloads civil_comments + textdetox, builds train/val/test splits
python src/train.py           # fine-tunes distilbert-base-multilingual-cased -> model/best_model/
python src/evaluate.py        # PyTorch vs quantized-ONNX metrics on held-out test set
python src/export_onnx.py     # model/best_model/ -> model/onnx/{model,model_optimized,model_quantized}.onnx
python src/benchmark.py       # latency: PyTorch vs ONNX

# Run the server
uvicorn src.serve:app --host 0.0.0.0 --port 8000   # local
# Docker/Render use: uvicorn src.serve:app --host 0.0.0.0 --port 7860 (see Dockerfile)

# Tests
pytest tests/ -v                                   # full suite (23 tests)
pytest tests/ -v -k "not test_compare_models_script"  # what CI actually runs (this one shells out)
pytest tests/test_pipeline.py::TestAPI -v           # one class
pytest tests/test_pipeline.py::TestAPI::test_predict_toxic_text -v  # one test
```

Git LFS is required to clone/pull `model/onnx/model_quantized.onnx` (tracked per `.gitattributes`; `git lfs install` once per machine).

## Architecture

**Two parallel model lineages exist in `src/` — know which one you're touching.** The *production* pipeline is `data.py` -> `train.py` -> `evaluate.py` -> `export_onnx.py`, fine-tuning `distilbert-base-multilingual-cased` on English (`civil_comments`) + Hindi/Hinglish (`textdetox/multilingual_toxicity_dataset`) data, producing `model/onnx/model_quantized.onnx`. This is the only model `serve.py` loads for `/predict`, `/predict/bulk`, `/predict/file`, `/predict/url`, and `/ws/predict`. A separate, earlier experiment — `custom_model.py` / `custom_train.py` / `custom_export_onnx.py` (a BiLSTM) plus `data_multilingual.py` / `train_multilingual.py` / `export_multilingual_onnx.py` (an earlier multilingual-specific variant before it was folded into `data.py`) — is kept only for benchmarking via `compare_models.py` and `test_on_kaggle.py`, and is covered by `tests/test_pipeline.py`'s `TestMultilingualPipeline`/`TestCompareScript` classes. Don't assume code in one lineage reflects what's deployed; `serve.py` is the source of truth for what's live.

**The deployed container cannot run `/predict/compare`.** `Dockerfile` installs from `requirements-deploy.txt`, which has no PyTorch. But `serve.py`'s `_get_base_model_and_tokenizer()` (used by `/predict/compare`, which loads `model/best_model/` via `AutoModelForSequenceClassification` to compare against the ONNX model) does `import torch` lazily inside the function — this works locally (`requirements.txt` has torch) but will throw in the deployed container the first time that endpoint is hit. The frontend's Compare Models UI tab has already been removed for this reason; the backend endpoint itself is still there and still broken in prod.

**`serve.py` is a single-file FastAPI app** (~790 lines) — not split into routers/services. Key internal pieces, in load order:
- `lifespan()` loads the ONNX session once at startup, preferring `model_quantized.onnx` > `model_optimized.onnx` > `model.onnx` (first one found in `ONNX_DIR`), plus the tokenizer from the same directory.
- `_predict_single` / `_predict_batch` run inference; `_cached_tokenize` (`lru_cache(maxsize=128)`) memoizes tokenization for repeated short strings (used by the WebSocket path, where users retype similar prefixes).
- `_check_rules` is a regex-based profanity matcher (`EXPLICIT_PROFANITY_PATTERN`) that runs **independently** of the ML model and surfaces as `rule_flagged`/`rule_matches` metadata — it does not alter the model's score. A prior design (`_apply_toxicity_boost`, removed in commit `5af2240`) used to clamp/boost ML scores based on keyword presence; that was deliberately ripped out in favor of fixing the training data instead (see `data.py`'s short non-toxic examples), so don't reintroduce score-clamping as a quick fix for misclassifications — retrain/recalibrate instead.
- `_chunk_text` + `_predict_chunks` handle long input: splits on sentence boundaries near `CHUNK_SIZE` (400 chars, `CHUNK_OVERLAP` 50), batches all chunks through one ONNX call via `_predict_batch`, and the chunk with the single highest toxic score determines the overall verdict.
- `_extract_text_from_file` / `_extract_text_from_url` are memory-conscious on purpose: PyPDF2/python-docx objects are explicitly `del`ed and `gc.collect()`ed after use, and ONNX thread counts are overridable via `ORT_INTRA_THREADS`/`ORT_INTER_THREADS` env vars — this project has been OOM-killed before on Render's free 512MB tier when processing uploads, so don't casually raise these without re-testing against that constraint.
- `/ws/predict` echoes back the text each prediction was computed for (`echo_text`) specifically so the frontend can discard stale/out-of-order responses — the client used to render whichever WebSocket reply arrived last even if it was for an earlier keystroke; `frontend/script.js`'s `ws.onmessage` depends on this field matching the live textarea value before it will update the UI. Don't drop `echo_text` from a response without also removing that client-side check.
- `TOXIC_THRESHOLD` (0.65, in `serve.py`) must stay in sync with `MAX_SEQ_LENGTH` (128) in `data.py`/`export_onnx.py`/`serve.py` — these were explicitly unified in commit `c43803b` after drifting apart.

**`frontend/` is hand-rolled HTML/CSS/vanilla JS** (no build step, no framework) served directly by FastAPI's `StaticFiles` mount from `serve_frontend()`. `index.html` has four input modes as tabs (Text, File, URL — the latter two are gated behind `.is-disabled` + a tooltip on the hosted demo because of the OOM issue above, but still call working backend endpoints if re-enabled), each with its own result-rendering function in `script.js` keyed by DOM id naming convention (`gauge-fill-<mode>`, `verdict-badge-<mode>`, etc.) — when adding a new input mode, follow that id pattern since `script.js`'s render functions assume it.

**Model artifacts are intentionally partially git-ignored.** Only `model_quantized.onnx` is committed (via Git LFS); `model.onnx` and `model_optimized.onnx` are regenerable intermediates from `export_onnx.py` and stay out of git even though `.gitattributes` has LFS rules for them (in case someone wants to track them later). `model/best_model/` (the PyTorch checkpoint) is also git-ignored except tokenizer config files — if you retrain, the new `model_quantized.onnx` must be `git add`ed and pushed explicitly, since nothing does this automatically, and a stale committed model next to fresh code was once the direct cause of a production bug (deployed model predated data/calibration fixes already merged into `data.py`/`serve.py`).
