"""
export_onnx.py — Convert the fine-tuned DistilBERT to ONNX, optimize, and quantize.

Produces three artifacts:
  1. model.onnx           — raw ONNX export
  2. model_optimized.onnx — graph-optimized
  3. model_quantized.onnx — dynamically quantized (INT8 weights)
"""

import os
import sys
import shutil

# ─── CONFIG ──────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model", "best_model"))
ONNX_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model", "onnx"))
MAX_SEQ_LENGTH = 128
# ─────────────────────────────────────────────────────────────────────────────


def export_to_onnx():
    """Export the PyTorch model to ONNX format."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run train.py first."
        )

    print("── Step 1: Exporting to ONNX ──")
    os.makedirs(ONNX_DIR, exist_ok=True)

    # Load model and tokenizer
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model.eval()

    # Create dummy input
    dummy_text = "This is a sample text for ONNX export."
    inputs = tokenizer(
        dummy_text,
        padding="max_length",
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        return_tensors="pt",
    )

    onnx_path = os.path.join(ONNX_DIR, "model.onnx")

    # Export
    torch.onnx.export(
        model,
        (inputs["input_ids"], inputs["attention_mask"]),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence"},
            "attention_mask": {0: "batch_size", 1: "sequence"},
            "logits": {0: "batch_size"},
        },
        opset_version=14,
        do_constant_folding=True,
    )

    # Verify
    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"✅ ONNX model exported and verified: {onnx_path}")

    # Also save the tokenizer alongside
    tokenizer.save_pretrained(ONNX_DIR)

    return onnx_path


def optimize_onnx(onnx_path: str):
    """Apply graph-level optimizations using onnxruntime's native optimizer.

    Uses GraphOptimizationLevel.ORT_ENABLE_BASIC which folds constants,
    eliminates redundant nodes, and performs layout optimizations — equivalent
    to what ORTOptimizer did but without requiring optimum's config.json.
    """
    import onnxruntime as ort

    print("\n── Step 2: Optimizing ONNX model (native ORT graph optimizer) ──")
    optimized_path = os.path.join(ONNX_DIR, "model_optimized.onnx")

    try:
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = (
            ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        )
        sess_options.optimized_model_filepath = optimized_path

        # Loading the session triggers optimization and writes the optimized model
        ort.InferenceSession(
            onnx_path,
            sess_options=sess_options,
            providers=["CPUExecutionProvider"],
        )

        if os.path.exists(optimized_path):
            raw_size = os.path.getsize(onnx_path) / (1024 * 1024)
            opt_size = os.path.getsize(optimized_path) / (1024 * 1024)
            print(f"✅ Optimized model saved: {optimized_path}")
            print(f"   Raw: {raw_size:.1f} MB → Optimized: {opt_size:.1f} MB")
            return optimized_path
        else:
            print("⚠️  Optimized model file not created. Using raw ONNX.")
            return onnx_path

    except Exception as e:
        print(f"⚠️  Optimization failed: {e}")
        print("   Continuing with un-optimized model.")
        return onnx_path


def quantize_onnx(onnx_path: str):
    """Apply dynamic quantization (INT8 weights) to reduce model size."""
    from onnxruntime.quantization import quantize_dynamic, QuantType

    print("\n── Step 3: Quantizing ONNX model (dynamic INT8) ──")

    quantized_path = os.path.join(ONNX_DIR, "model_quantized.onnx")

    try:
        quantize_dynamic(
            model_input=onnx_path,
            model_output=quantized_path,
            weight_type=QuantType.QInt8,
        )
        print(f"✅ Quantized model saved: {quantized_path}")

        # Compare sizes
        raw_size = os.path.getsize(onnx_path) / (1024 * 1024)
        quant_size = os.path.getsize(quantized_path) / (1024 * 1024)
        reduction = (1 - quant_size / raw_size) * 100

        print(f"\n📊 Model Size Comparison:")
        print(f"   Original:  {raw_size:.1f} MB")
        print(f"   Quantized: {quant_size:.1f} MB")
        print(f"   Reduction: {reduction:.1f}%")

        return quantized_path

    except Exception as e:
        print(f"⚠️  Quantization failed: {e}")
        print("   Continuing with un-quantized model.")
        return onnx_path


def export_optimize_quantize():
    """Full ONNX pipeline: export → optimize → quantize."""
    onnx_path = export_to_onnx()
    optimized_path = optimize_onnx(onnx_path)
    quantized_path = quantize_onnx(optimized_path)

    print(f"\n✅ ONNX pipeline complete!")
    print(f"   Raw:       {os.path.join(ONNX_DIR, 'model.onnx')}")
    print(f"   Optimized: {optimized_path}")
    print(f"   Quantized: {quantized_path}")

    return quantized_path


if __name__ == "__main__":
    export_optimize_quantize()
