"""
benchmark.py — Measure and compare inference latency: PyTorch vs ONNX.

Runs inference on 100 samples and reports average latency.
Target: ONNX should be faster and under ~50ms per sample.
"""

import os
import sys
import time
import numpy as np

# ─── CONFIG ──────────────────────────────────────────────────────────────────
NUM_SAMPLES = 100
MAX_SEQ_LENGTH = 64               # ⚡ Matches serve.py optimization
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model", "best_model"))
ONNX_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model", "onnx"))
WARMUP_RUNS = 10
# ─────────────────────────────────────────────────────────────────────────────


def generate_sample_inputs(tokenizer, n: int = NUM_SAMPLES):
    """Generate sample tokenized inputs for benchmarking."""
    sample_texts = [
        "This is a great article, very informative and well written.",
        "You are such an idiot and should be banned from this site.",
        "I think we can have a constructive discussion about this topic.",
        "Shut up you worthless fool, nobody cares about your opinion.",
        "Thank you for sharing your perspective on this important issue.",
    ]

    inputs_list = []
    for i in range(n):
        text = sample_texts[i % len(sample_texts)]
        inputs = tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt",
        )
        inputs_list.append(inputs)

    return inputs_list


def benchmark_pytorch(model, inputs_list):
    """Benchmark PyTorch model inference."""
    import torch

    model.eval()
    device = next(model.parameters()).device

    # Warmup
    with torch.no_grad():
        for i in range(min(WARMUP_RUNS, len(inputs_list))):
            inp = {k: v.to(device) for k, v in inputs_list[i].items()}
            _ = model(**inp)

    # Benchmark
    latencies = []
    with torch.no_grad():
        for inputs in inputs_list:
            inp = {k: v.to(device) for k, v in inputs.items()}
            start = time.perf_counter()
            _ = model(**inp)
            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms

    return latencies


def benchmark_onnx(session, inputs_list):
    """Benchmark ONNX Runtime inference."""

    # Warmup
    for i in range(min(WARMUP_RUNS, len(inputs_list))):
        inp = inputs_list[i]
        feed = {
            "input_ids": inp["input_ids"].numpy(),
            "attention_mask": inp["attention_mask"].numpy(),
        }
        _ = session.run(None, feed)

    # Benchmark
    latencies = []
    for inputs in inputs_list:
        feed = {
            "input_ids": inputs["input_ids"].numpy(),
            "attention_mask": inputs["attention_mask"].numpy(),
        }
        start = time.perf_counter()
        _ = session.run(None, feed)
        end = time.perf_counter()
        latencies.append((end - start) * 1000)  # ms

    return latencies


def print_results(name, latencies):
    """Pretty-print latency statistics."""
    arr = np.array(latencies)
    print(f"\n  {name}:")
    print(f"    Mean:   {arr.mean():.2f} ms")
    print(f"    Median: {np.median(arr):.2f} ms")
    print(f"    P95:    {np.percentile(arr, 95):.2f} ms")
    print(f"    P99:    {np.percentile(arr, 99):.2f} ms")
    print(f"    Min:    {arr.min():.2f} ms")
    print(f"    Max:    {arr.max():.2f} ms")
    return arr.mean()


def benchmark():
    """Run full benchmark: PyTorch vs ONNX."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    import onnxruntime as ort

    # ── Load models ──────────────────────────────────────────────────────────
    print("── Loading models ──")

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"PyTorch model not found at {MODEL_PATH}. Run train.py first."
        )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    pytorch_model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    pytorch_model.eval()
    print(f"✅ PyTorch model loaded")

    # Find best ONNX model (prefer quantized > optimized > raw)
    onnx_candidates = [
        os.path.join(ONNX_DIR, "model_quantized.onnx"),
        os.path.join(ONNX_DIR, "model_optimized.onnx"),
        os.path.join(ONNX_DIR, "model.onnx"),
    ]
    onnx_path = None
    for candidate in onnx_candidates:
        if os.path.exists(candidate):
            onnx_path = candidate
            break

    if onnx_path is None:
        raise FileNotFoundError(
            f"ONNX model not found in {ONNX_DIR}. Run export_onnx.py first."
        )

    session = ort.InferenceSession(
        onnx_path,
        # ⚡ OPTIMIZATION: ONNX session tuning for max CPU performance
        sess_options=(
            lambda o: (
                setattr(o, "intra_op_num_threads", 4),
                setattr(o, "inter_op_num_threads", 2),
                setattr(o, "graph_optimization_level", ort.GraphOptimizationLevel.ORT_ENABLE_ALL),
                o
            )[-1]
        )(ort.SessionOptions()),
        providers=["CPUExecutionProvider"],
    )
    print(f"✅ ONNX model loaded: {os.path.basename(onnx_path)}")

    # ── Generate inputs ──────────────────────────────────────────────────────
    print(f"\n── Generating {NUM_SAMPLES} sample inputs ──")
    inputs_list = generate_sample_inputs(tokenizer, NUM_SAMPLES)

    # ── Benchmark ────────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"⏱️  LATENCY BENCHMARK ({NUM_SAMPLES} samples)")
    print(f"{'='*50}")

    pt_latencies = benchmark_pytorch(pytorch_model, inputs_list)
    pt_mean = print_results("PyTorch (CPU)", pt_latencies)

    onnx_latencies = benchmark_onnx(session, inputs_list)
    onnx_mean = print_results(f"ONNX ({os.path.basename(onnx_path)})", onnx_latencies)

    # ── Summary ──────────────────────────────────────────────────────────────
    speedup = pt_mean / onnx_mean if onnx_mean > 0 else float("inf")
    under_50ms = onnx_mean < 50

    print(f"\n{'='*50}")
    print(f"📊 SUMMARY")
    print(f"{'='*50}")
    print(f"  PyTorch mean latency: {pt_mean:.2f} ms")
    print(f"  ONNX mean latency:    {onnx_mean:.2f} ms")
    print(f"  Speedup:              {speedup:.2f}x")
    print(f"  ONNX under 50ms:      {'✅ YES' if under_50ms else '❌ NO'} ({onnx_mean:.2f} ms)")

    return {
        "pytorch_mean_ms": pt_mean,
        "onnx_mean_ms": onnx_mean,
        "speedup": speedup,
        "under_50ms": under_50ms,
    }


if __name__ == "__main__":
    results = benchmark()
