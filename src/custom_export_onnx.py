import os
import sys
import torch
from transformers import AutoTokenizer

# ─── CONFIG ──────────────────────────────────────────────────────────────────
MODEL_WEIGHTS_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model", "best_custom_model.pth"))
ONNX_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model", "onnx"))
TOKENIZER_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "tokenizer"))
MAX_SEQ_LENGTH = 64
# ─────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, os.path.dirname(__file__))
from custom_model import CustomBiLSTMClassifier

def export_to_onnx():
    print("── Step 1: Exporting Custom BiLSTM Model to ONNX ──")
    os.makedirs(ONNX_DIR, exist_ok=True)

    # 1. Load the tokenizer to get vocab size
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    
    # 2. Instantiate and load custom model weights
    model = CustomBiLSTMClassifier(vocab_size=tokenizer.vocab_size)
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location="cpu"))
    model.eval()

    # 3. Create dummy inputs (batch_size=1, seq_len=64)
    dummy_input_ids = torch.ones((1, MAX_SEQ_LENGTH), dtype=torch.long)
    dummy_attention_mask = torch.ones((1, MAX_SEQ_LENGTH), dtype=torch.long)

    onnx_path = os.path.join(ONNX_DIR, "model.onnx")

    # 4. Export to ONNX
    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask),
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

    import onnx
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print(f"✅ Custom ONNX model exported and verified: {onnx_path}")

    # Save tokenizer alongside the model files
    tokenizer.save_pretrained(ONNX_DIR)
    return onnx_path

def optimize_onnx(onnx_path: str):
    import onnxruntime as ort
    print("\n── Step 2: Optimizing Custom ONNX Model ──")
    optimized_path = os.path.join(ONNX_DIR, "model_optimized.onnx")

    try:
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
        sess_options.optimized_model_filepath = optimized_path

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
            return onnx_path
    except Exception as e:
        print(f"⚠️ Optimization failed: {e}")
        return onnx_path

def quantize_onnx(onnx_path: str):
    from onnxruntime.quantization import quantize_dynamic, QuantType
    print("\n── Step 3: Quantizing Custom ONNX Model (Dynamic INT8) ──")
    quantized_path = os.path.join(ONNX_DIR, "model_quantized.onnx")

    try:
        quantize_dynamic(
            model_input=onnx_path,
            model_output=quantized_path,
            weight_type=QuantType.QInt8,
        )
        raw_size = os.path.getsize(onnx_path) / (1024 * 1024)
        quant_size = os.path.getsize(quantized_path) / (1024 * 1024)
        reduction = (1 - quant_size / raw_size) * 100

        print(f"✅ Quantized custom model saved: {quantized_path}")
        print(f"📊 Model Size Comparison:")
        print(f"   Original:  {raw_size:.2f} MB")
        print(f"   Quantized: {quant_size:.2f} MB")
        print(f"   Reduction: {reduction:.1f}%")
        return quantized_path
    except Exception as e:
        print(f"⚠️ Quantization failed: {e}")
        return onnx_path

def main():
    if not os.path.exists(MODEL_WEIGHTS_PATH):
        print(f"❌ Error: Model weights not found at {MODEL_WEIGHTS_PATH}. Run custom_train.py first.")
        sys.exit(1)
        
    onnx_path = export_to_onnx()
    opt_path = optimize_onnx(onnx_path)
    quantize_onnx(opt_path)
    print("\n🎉 Custom model ONNX pipeline finished successfully!")

if __name__ == "__main__":
    main()
