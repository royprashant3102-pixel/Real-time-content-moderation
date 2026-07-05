"""
evaluate.py — Evaluate the fine-tuned model on the test set.

Reports precision, recall, F1, accuracy, and confusion matrix.
"""

import os
import sys
import numpy as np

# ─── CONFIG ──────────────────────────────────────────────────────────────────
MAX_SAMPLES = 10_000
MODEL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model", "best_model"))
BATCH_SIZE = 32
# ─────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, os.path.dirname(__file__))
from data import prepare_data


def evaluate():
    """Load the saved model and evaluate on the test split."""
    import torch
    from transformers import AutoModelForSequenceClassification, Trainer, TrainingArguments
    from sklearn.metrics import (
        classification_report,
        confusion_matrix,
        precision_recall_fscore_support,
        accuracy_score,
    )

    # ── 1. Prepare data ──────────────────────────────────────────────────────
    print("── Loading data ──")
    splits, tokenizer = prepare_data(max_samples=MAX_SAMPLES)
    test_dataset = splits["test"]

    # ── 2. Load saved model ──────────────────────────────────────────────────
    print("\n── Loading saved model ──")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Run train.py first."
        )

    try:
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        print(f"✅ Model loaded from: {MODEL_PATH}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise

    # ── 3. Run predictions ───────────────────────────────────────────────────
    print("\n── Running predictions on test set ──")
    training_args = TrainingArguments(
        output_dir="/tmp/eval_output",
        per_device_eval_batch_size=BATCH_SIZE,
        report_to="none",
        use_cpu=not torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
    )

    predictions = trainer.predict(test_dataset)
    preds = np.argmax(predictions.predictions, axis=-1)
    labels = predictions.label_ids

    # ── 4. Compute metrics ───────────────────────────────────────────────────
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    accuracy = accuracy_score(labels, preds)

    print("\n" + "=" * 50)
    print("📊 TEST SET EVALUATION RESULTS")
    print("=" * 50)

    print(f"\n  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    print(f"  F1 Score:  {f1:.4f}")

    print(f"\n── Classification Report ──")
    print(classification_report(
        labels, preds,
        target_names=["Non-Toxic", "Toxic"],
        zero_division=0,
    ))

    print(f"── Confusion Matrix ──")
    cm = confusion_matrix(labels, preds)
    print(f"                  Predicted")
    print(f"                  Non-Toxic  Toxic")
    print(f"  Actual Non-Toxic  {cm[0][0]:>6}  {cm[0][1]:>6}")
    print(f"  Actual Toxic      {cm[1][0]:>6}  {cm[1][1]:>6}")

    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "confusion_matrix": cm.tolist(),
    }

    print("\n✅ Evaluation complete!")
    return results


if __name__ == "__main__":
    results = evaluate()
