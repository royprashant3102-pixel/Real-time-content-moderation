"""
export_multilingual_onnx.py — Export the fine-tuned multilingual model to ONNX, optimize, and quantize.

Saves files to model/onnx/, replacing the monolingual model for deployment.
"""

import os
import sys

# ─── CONFIG ──────────────────────────────────────────────────────────────────
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model", "best_model_multilingual"))
ONNX_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model", "onnx"))
MAX_SEQ_LENGTH = 128
# ─────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, os.path.dirname(__file__))
from export_onnx import export_to_onnx, optimize_onnx, quantize_onnx

def main():
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Multilingual model not found at {MODEL_PATH}. Run train_multilingual.py first.")
        sys.exit(1)
        
    print("── Starting Multilingual ONNX Pipeline ──")
    
    # Temporarily monkeypatch export_onnx.MODEL_PATH and export_onnx.ONNX_DIR
    import export_onnx
    export_onnx.MODEL_PATH = MODEL_PATH
    export_onnx.ONNX_DIR = ONNX_DIR
    export_onnx.MAX_SEQ_LENGTH = MAX_SEQ_LENGTH
    
    raw_path = export_to_onnx()
    opt_path = optimize_onnx(raw_path)
    quant_path = quantize_onnx(opt_path)
    
    print("\n🎉 Multilingual model ONNX pipeline finished successfully!")
    print(f"   Deployed to: {ONNX_DIR}")

if __name__ == "__main__":
    main()
