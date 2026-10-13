"""
evaluate.py — Evaluate fine-tuned PyTorch model and Quantized ONNX model on the test set.

Reports:
  - Accuracy, Precision, Recall, F1 Score
  - PR-AUC (Precision-Recall Area Under Curve)
  - Confusion Matrix
  - Quantization Impact Delta (PyTorch vs ONNX INT8)
"""

import os
import sys
import numpy as np

# ─── CONFIG ──────────────────────────────────────────────────────────────────
MAX_SAMPLES = 10_000
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model", "best_model"))
ONNX_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model", "onnx"))
BATCH_SIZE = 32
TOXIC_THRESHOLD = 0.65
# ─────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, os.path.dirname(__file__))
from data import prepare_data


def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def evaluate():
    """Load saved PyTorch and ONNX models and evaluate on the test split."""
    import torch
    from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        precision_recall_fscore_support,
        accuracy_score,
        average_precision_score,
    )

    # ── 1. Prepare data ──────────────────────────────────────────────────────
    print("── Step 1: Loading test data ──")
    splits, tokenizer = prepare_data(max_samples=MAX_SAMPLES)
    test_dataset = splits["test"]

    # ── 2. Load PyTorch model ────────────────────────────────────────────────
    print("\n── Step 2: Loading PyTorch model ──")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run train.py first."
        )

    model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
    print(f"✅ PyTorch model loaded from: {MODEL_PATH}")

    # ── 3. Run predictions (PyTorch) ─────────────────────────────────────────
    print("\n── Step 3: Running PyTorch evaluation ──")
    training_args = TrainingArguments(
        output_dir="/tmp/eval_output",
        per_device_eval_batch_size=BATCH_SIZE,
        report_to="none",
        use_cpu=not torch.cuda.is_available(),
    )

    trainer = Trainer(model=model, args=training_args)
    predictions = trainer.predict(test_dataset)
    pt_logits = predictions.predictions
    pt_probs = softmax(pt_logits)[:, 1]
    labels = predictions.label_ids
    pt_preds = (pt_probs >= TOXIC_THRESHOLD).astype(int)

    # Compute PyTorch metrics
    pt_prec, pt_rec, pt_f1, _ = precision_recall_fscore_support(
        labels, pt_preds, average="binary", zero_division=0
    )
    pt_acc = accuracy_score(labels, pt_preds)
    pt_pr_auc = average_precision_score(labels, pt_probs)
    pt_cm = confusion_matrix(labels, pt_preds)

    print("\n" + "=" * 60)
    print("📊 PYTORCH MODEL EVALUATION (threshold = 0.65)")
    print("=" * 60)
    print(f"  Accuracy:  {pt_acc * 100:.2f}%")
    print(f"  Precision: {pt_prec * 100:.2f}%")
    print(f"  Recall:    {pt_rec * 100:.2f}%")
    print(f"  F1 Score:  {pt_f1:.4f}")
    print(f"  PR-AUC:    {pt_pr_auc:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"                  Predicted Non-Toxic  Predicted Toxic")
    print(f"  Actual Non-Toxic  {pt_cm[0][0]:>18}  {pt_cm[0][1]:>15}")
    print(f"  Actual Toxic      {pt_cm[1][0]:>18}  {pt_cm[1][1]:>15}")

    # ── 4. Evaluate Quantized ONNX Model (if exists) ─────────────────────────
    quantized_onnx = os.path.join(ONNX_DIR, "model_quantized.onnx")
    onnx_results = None

    if os.path.exists(quantized_onnx):
        import onnxruntime as ort
        print("\n" + "=" * 60)
        print("⚡ QUANTIZED ONNX INT8 EVALUATION")
        print("=" * 60)

        session = ort.InferenceSession(quantized_onnx, providers=["CPUExecutionProvider"])
        onnx_probs = []

        # Run in batches
        for i in range(0, len(test_dataset), BATCH_SIZE):
            batch = test_dataset[i:i + BATCH_SIZE]
            feed = {
                "input_ids": np.array(batch["input_ids"], dtype=np.int64),
                "attention_mask": np.array(batch["attention_mask"], dtype=np.int64),
            }
            logits = session.run(None, feed)[0]
            batch_probs = softmax(logits)[:, 1]
            onnx_probs.extend(batch_probs)

        onnx_probs = np.array(onnx_probs)
        onnx_preds = (onnx_probs >= TOXIC_THRESHOLD).astype(int)

        ox_prec, ox_rec, ox_f1, _ = precision_recall_fscore_support(
            labels, onnx_preds, average="binary", zero_division=0
        )
        ox_acc = accuracy_score(labels, onnx_preds)
        ox_pr_auc = average_precision_score(labels, onnx_probs)
        ox_cm = confusion_matrix(labels, onnx_preds)

        print(f"  Accuracy:  {ox_acc * 100:.2f}% (delta: {(ox_acc - pt_acc) * 100:+.2f}%)")
        print(f"  Precision: {ox_prec * 100:.2f}% (delta: {(ox_prec - pt_prec) * 100:+.2f}%)")
        print(f"  Recall:    {ox_rec * 100:.2f}% (delta: {(ox_rec - pt_rec) * 100:+.2f}%)")
        print(f"  F1 Score:  {ox_f1:.4f} (delta: {ox_f1 - pt_f1:+.4f})")
        print(f"  PR-AUC:    {ox_pr_auc:.4f} (delta: {ox_pr_auc - pt_pr_auc:+.4f})")

        onnx_results = {
            "accuracy": ox_acc,
            "precision": ox_prec,
            "recall": ox_rec,
            "f1": ox_f1,
            "pr_auc": ox_pr_auc,
            "confusion_matrix": ox_cm.tolist(),
        }

    results = {
        "pytorch": {
            "accuracy": pt_acc,
            "precision": pt_prec,
            "recall": pt_rec,
            "f1": pt_f1,
            "pr_auc": pt_pr_auc,
            "confusion_matrix": pt_cm.tolist(),
        },
        "quantized_onnx": onnx_results,
    }

    print("\n✅ Evaluation complete!")
    return results


if __name__ == "__main__":
    results = evaluate()
