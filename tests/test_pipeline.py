"""
test_pipeline.py — Unit tests for the content moderation pipeline.

Tests each module: data, model, ONNX, and API.
Run with: pytest tests/ -v
"""

import os
import sys
import pytest
import numpy as np

# ── pytest-asyncio configuration ─────────────────────────────────────────────
# For pytest-asyncio >= 0.21: auto mode makes all async test functions
# automatically treated as async tests without needing @pytest.mark.asyncio
pytest_plugins = ("pytest_asyncio",)
import torch

# Ensure src is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

# Paths
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL_PATH = os.path.join(PROJECT_ROOT, "model", "best_custom_model.pth")
ONNX_DIR = os.path.join(PROJECT_ROOT, "model", "onnx")
TOKENIZER_DIR = os.path.join(PROJECT_ROOT, "data", "tokenizer")


# ═══════════════════════════════════════════════════════════════════════════
# DATA TESTS
# ═══════════════════════════════════════════════════════════════════════════

class TestData:
    def test_clean_text_removes_urls(self):
        from data import clean_text
        result = clean_text("Check http://example.com NOW!!!")
        assert "http" not in result
        assert result == result.lower()

    def test_clean_text_lowercase(self):
        from data import clean_text
        result = clean_text("Hello WORLD")
        assert result == "hello world"

    def test_synthetic_dataset_size(self):
        from data import _create_synthetic_dataset
        texts, labels = _create_synthetic_dataset(100)
        assert len(texts) == 100
        assert len(labels) == 100

    def test_synthetic_dataset_binary_labels(self):
        from data import _create_synthetic_dataset
        _, labels = _create_synthetic_dataset(200)
        assert set(labels).issubset({0, 1})
        assert 0 in labels
        assert 1 in labels

    def test_tokenize_data_columns(self):
        from data import tokenize_data, _create_synthetic_dataset
        texts, labels = _create_synthetic_dataset(50)
        dataset, tokenizer = tokenize_data(texts, labels)
        assert "input_ids" in dataset.column_names
        assert "attention_mask" in dataset.column_names
        assert "label" in dataset.column_names
        assert len(dataset) == 50

    def test_split_dataset_sizes(self):
        from data import tokenize_data, split_dataset, _create_synthetic_dataset
        texts, labels = _create_synthetic_dataset(200)
        dataset, _ = tokenize_data(texts, labels)
        splits = split_dataset(dataset)
        assert "train" in splits
        assert "val" in splits
        assert "test" in splits
        total = sum(len(splits[k]) for k in ("train", "val", "test"))
        assert total == 200


# ═══════════════════════════════════════════════════════════════════════════
# MODEL TESTS (require train.py to have been run)
# ═══════════════════════════════════════════════════════════════════════════

class TestModel:
    @pytest.fixture(autouse=True)
    def check_model_exists(self):
        if not os.path.exists(MODEL_PATH):
            pytest.skip("Custom model weights not found. Run custom_train.py first.")

    def test_model_loads(self):
        from custom_model import CustomBiLSTMClassifier
        from transformers import AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
        model = CustomBiLSTMClassifier(vocab_size=tokenizer.vocab_size)
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        
        assert model is not None
        assert model.embedding.num_embeddings == tokenizer.vocab_size

    def test_model_predicts_shape(self):
        import torch
        from custom_model import CustomBiLSTMClassifier
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
        model = CustomBiLSTMClassifier(vocab_size=tokenizer.vocab_size)
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        model.eval()

        inputs = tokenizer(
            "This is a test.",
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=64,
        )
        with torch.no_grad():
            outputs = model(inputs["input_ids"], inputs["attention_mask"])

        assert outputs.shape == (1, 2)


# ═══════════════════════════════════════════════════════════════════════════
# ONNX TESTS (require export_onnx.py to have been run)
# ═══════════════════════════════════════════════════════════════════════════

class TestONNX:
    @pytest.fixture(autouse=True)
    def check_onnx_exists(self):
        candidates = [
            os.path.join(ONNX_DIR, f)
            for f in ["model.onnx", "model_optimized.onnx", "model_quantized.onnx"]
        ]
        if not any(os.path.exists(f) for f in candidates):
            pytest.skip("ONNX model not found. Run export_onnx.py first.")

    def _get_onnx_path(self):
        for name in ("model_quantized.onnx", "model_optimized.onnx", "model.onnx"):
            p = os.path.join(ONNX_DIR, name)
            if os.path.exists(p):
                return p

    def _get_tokenizer_path(self):
        return (
            ONNX_DIR
            if os.path.exists(os.path.join(ONNX_DIR, "tokenizer_config.json"))
            else MODEL_PATH
        )

    def test_onnx_session_loads(self):
        import onnxruntime as ort
        session = ort.InferenceSession(
            self._get_onnx_path(), providers=["CPUExecutionProvider"]
        )
        assert session is not None

    def test_onnx_output_shape(self):
        import onnxruntime as ort
        from transformers import AutoTokenizer

        session = ort.InferenceSession(
            self._get_onnx_path(), providers=["CPUExecutionProvider"]
        )
        tokenizer = AutoTokenizer.from_pretrained(self._get_tokenizer_path())
        inputs = tokenizer(
            "Test input",
            return_tensors="np",
            padding="max_length",
            truncation=True,
            max_length=128,
        )
        feed = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        }
        result = session.run(None, feed)
        assert result[0].shape == (1, 2)

    def test_onnx_matches_pytorch(self):
        """Raw ONNX logits must be close to PyTorch logits."""
        import torch
        import onnxruntime as ort
        from custom_model import CustomBiLSTMClassifier
        from transformers import AutoTokenizer

        if not os.path.exists(MODEL_PATH):
            pytest.skip("PyTorch custom model not found")

        raw_onnx = os.path.join(ONNX_DIR, "model.onnx")
        if not os.path.exists(raw_onnx):
            pytest.skip("Raw (non-quantized) ONNX model not found")

        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
        model = CustomBiLSTMClassifier(vocab_size=tokenizer.vocab_size)
        model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
        model.eval()

        text = "A neutral sentence for comparison testing."
        inputs = tokenizer(
            text, return_tensors="pt", padding="max_length",
            truncation=True, max_length=64
        )
        with torch.no_grad():
            pt_logits = model(inputs["input_ids"], inputs["attention_mask"]).numpy()

        session = ort.InferenceSession(raw_onnx, providers=["CPUExecutionProvider"])
        feed = {
            "input_ids": inputs["input_ids"].numpy().astype(np.int64),
            "attention_mask": inputs["attention_mask"].numpy().astype(np.int64),
        }
        onnx_logits = session.run(None, feed)[0]
        np.testing.assert_allclose(pt_logits, onnx_logits, atol=1e-4)

    def test_quantized_model_size(self):
        """Quantized model must be significantly smaller than raw."""
        raw = os.path.join(ONNX_DIR, "model.onnx")
        quantized = os.path.join(ONNX_DIR, "model_quantized.onnx")
        if not (os.path.exists(raw) and os.path.exists(quantized)):
            pytest.skip("Both raw and quantized ONNX models needed.")
        raw_mb = os.path.getsize(raw) / 1e6
        quant_mb = os.path.getsize(quantized) / 1e6
        assert quant_mb < raw_mb * 0.5, (
            f"Quantized ({quant_mb:.1f} MB) should be <50% of raw ({raw_mb:.1f} MB)"
        )


# ═══════════════════════════════════════════════════════════════════════════
# API TESTS (require ONNX model to exist)
# ═══════════════════════════════════════════════════════════════════════════

@pytest.mark.asyncio(loop_scope="session")
class TestAPI:
    @pytest.fixture(autouse=True)
    def check_onnx_for_api(self):
        candidates = [
            os.path.join(ONNX_DIR, f)
            for f in ["model.onnx", "model_optimized.onnx", "model_quantized.onnx"]
        ]
        if not any(os.path.exists(f) for f in candidates):
            pytest.skip("ONNX model not found. Run export_onnx.py first.")

    async def test_health_check(self):
        from httpx import AsyncClient, ASGITransport
        from serve import app
        async with AsyncClient(
            transport=ASGITransport(app=app, raise_app_exceptions=True),
            base_url="http://test",
        ) as client:
            # Manually trigger lifespan
            async with app.router.lifespan_context(app):
                response = await client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"

    async def test_predict_returns_valid_response(self):
        from httpx import AsyncClient, ASGITransport
        from serve import app
        async with AsyncClient(
            transport=ASGITransport(app=app, raise_app_exceptions=True),
            base_url="http://test",
        ) as client:
            async with app.router.lifespan_context(app):
                response = await client.post(
                    "/predict", json={"text": "Thank you for this helpful article."}
                )
        assert response.status_code == 200
        data = response.json()
        assert data["label"] in ("toxic", "non-toxic")
        assert 0.0 <= data["score"] <= 1.0
        assert isinstance(data["toxic"], bool)
        assert abs(sum(data["confidence"].values()) - 1.0) < 1e-3

    async def test_predict_toxic_text(self):
        from httpx import AsyncClient, ASGITransport
        from serve import app
        async with AsyncClient(
            transport=ASGITransport(app=app, raise_app_exceptions=True),
            base_url="http://test",
        ) as client:
            async with app.router.lifespan_context(app):
                response = await client.post(
                    "/predict",
                    json={"text": "You are an idiot and should be banned."},
                )
        assert response.status_code == 200
        data = response.json()
        assert data["label"] in ("toxic", "non-toxic")

    async def test_predict_empty_text_rejected(self):
        from httpx import AsyncClient, ASGITransport
        from serve import app
        async with AsyncClient(
            transport=ASGITransport(app=app, raise_app_exceptions=False),
            base_url="http://test",
        ) as client:
            response = await client.post("/predict", json={"text": ""})
        assert response.status_code == 422  # Pydantic validation error

    async def test_predict_bulk_endpoint(self):
        from httpx import AsyncClient, ASGITransport
        from serve import app
        async with AsyncClient(
            transport=ASGITransport(app=app, raise_app_exceptions=True),
            base_url="http://test",
        ) as client:
            async with app.router.lifespan_context(app):
                response = await client.post(
                    "/predict/bulk",
                    json={"text": "This is sample line one. " * 30 + "You are so stupid and awful. " + "This is sample line two. " * 30}
                )
        assert response.status_code == 200
        data = response.json()
        assert "total_chunks" in data
        assert data["total_chunks"] > 1
        assert "chunks" in data
        assert "overall_label" in data
        assert data["overall_toxic"] is True

    async def test_predict_file_endpoint(self):
        from httpx import AsyncClient, ASGITransport
        from serve import app
        async with AsyncClient(
            transport=ASGITransport(app=app, raise_app_exceptions=True),
            base_url="http://test",
        ) as client:
            async with app.router.lifespan_context(app):
                files = {"file": ("test.txt", b"great article, very informative and well written", "text/plain")}
                response = await client.post("/predict/file", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["source"] == "file"
        assert data["source_name"] == "test.txt"
        assert data["overall_toxic"] is False

    async def test_predict_url_endpoint(self):
        from unittest.mock import patch, MagicMock
        from httpx import AsyncClient, ASGITransport, Response
        from serve import app
        
        # Mock response from httpx
        mock_response = MagicMock(spec=Response)
        mock_response.status_code = 200
        mock_response.text = "<html><body><h1>Hello World</h1><p>great article, very informative and well written</p></body></html>"
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.AsyncClient.get", return_value=mock_response):
            async with AsyncClient(
                transport=ASGITransport(app=app, raise_app_exceptions=True),
                base_url="http://test",
            ) as client:
                async with app.router.lifespan_context(app):
                    response = await client.post(
                        "/predict/url",
                        json={"url": "https://example.com/nice-page"}
                    )
        assert response.status_code == 200
        data = response.json()
        assert data["source"] == "url"
        assert data["source_name"] == "https://example.com/nice-page"
        assert data["overall_toxic"] is False

    async def test_predict_compare_endpoint(self):
        from httpx import AsyncClient, ASGITransport
        from serve import app
        async with AsyncClient(
            transport=ASGITransport(app=app, raise_app_exceptions=True),
            base_url="http://test",
        ) as client:
            async with app.router.lifespan_context(app):
                response = await client.post(
                    "/predict/compare",
                    json={"text": "You are an idiot and should be banned."}
                )
        assert response.status_code == 200
        data = response.json()
        assert "text" in data
        assert "custom_model" in data
        assert "base_model" in data
        
        # Verify custom model fields
        custom = data["custom_model"]
        assert "label" in custom
        assert "score" in custom
        assert "toxic" in custom
        assert "confidence" in custom
        assert "latency_ms" in custom
        
        # Verify base model fields
        base = data["base_model"]
        assert "label" in base
        assert "score" in base
        assert "toxic" in base
        assert "confidence" in base
        assert "latency_ms" in base


