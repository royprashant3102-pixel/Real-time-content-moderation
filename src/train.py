"""
train.py — Fine-tune DistilBERT for binary toxicity classification.

Designed to run on CPU-only machines with small defaults.
Config variables at top for easy tuning.
"""

import os
import sys
import torch

# ─── CONFIG ──────────────────────────────────────────────────────────────────
NUM_EPOCHS = 2
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 50
MAX_SAMPLES = 10_000          # Passed to data.prepare_data()
MODEL_NAME = "distilbert-base-uncased"
NUM_LABELS = 2
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model"))
LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))
# ─────────────────────────────────────────────────────────────────────────────

# Add parent to path so we can import sibling modules
sys.path.insert(0, os.path.dirname(__file__))
from data import prepare_data


def get_device():
    """Select best available device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        # MPS can be unstable for training; use CPU for reliability
        return torch.device("cpu")
    return torch.device("cpu")


def compute_metrics(eval_pred):
    """Compute accuracy, precision, recall, F1 for the Trainer."""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support
    import numpy as np

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    acc = accuracy_score(labels, preds)

    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def train():
    """Fine-tune DistilBERT and save the model."""
    from transformers import (
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
    )

    device = get_device()
    print(f"🖥️  Using device: {device}")

    # ── 1. Prepare data ──────────────────────────────────────────────────────
    print("\n── Step 1: Preparing data ──")
    splits, tokenizer = prepare_data(max_samples=MAX_SAMPLES)

    # ── 2. Load pre-trained model ────────────────────────────────────────────
    print("\n── Step 2: Loading pre-trained DistilBERT ──")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=NUM_LABELS,
        )
        print(f"✅ Model loaded: {MODEL_NAME}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise

    # ── 3. Configure training ────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_steps=WARMUP_STEPS,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_dir=LOG_DIR,
        logging_steps=50,
        save_total_limit=2,
        report_to="none",           # No wandb/tensorboard needed
        fp16=False,                  # Disable for CPU compatibility
        use_cpu=not torch.cuda.is_available(),
        seed=42,
    )

    # ── 4. Train ─────────────────────────────────────────────────────────────
    print("\n── Step 3: Training ──")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=splits["train"],
        eval_dataset=splits["val"],
        compute_metrics=compute_metrics,
    )

    try:
        train_result = trainer.train()
        print(f"\n✅ Training complete!")
        print(f"   Train loss: {train_result.metrics.get('train_loss', 'N/A'):.4f}")
    except Exception as e:
        print(f"❌ Training failed: {e}")
        raise

    # ── 5. Save best model ───────────────────────────────────────────────────
    print("\n── Step 4: Saving model ──")
    final_model_path = os.path.join(OUTPUT_DIR, "best_model")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"💾 Model saved to: {final_model_path}")

    # ── 6. Quick validation metrics ──────────────────────────────────────────
    print("\n── Step 5: Validation metrics ──")
    val_metrics = trainer.evaluate(splits["val"])
    for k, v in val_metrics.items():
        if isinstance(v, float):
            print(f"   {k}: {v:.4f}")
        else:
            print(f"   {k}: {v}")

    return trainer, splits


if __name__ == "__main__":
    trainer, splits = train()
    print("\n✅ train.py completed successfully!")
