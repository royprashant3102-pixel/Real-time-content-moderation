"""
train.py — Fine-tune multilingual DistilBERT for binary toxicity classification.

Improvements over v1:
  - Multilingual model: distilbert-base-multilingual-cased (104 languages)
  - 30K+ samples (English + Hinglish/Hindi)
  - 4 epochs for better convergence
  - Class-weight balancing via weighted CrossEntropyLoss
  - Better warmup scheduling for larger dataset

Designed to run on CPU-only machines.
Config variables at top for easy tuning.
"""

import os
import sys
import torch
import numpy as np

# ─── CONFIG ──────────────────────────────────────────────────────────────────
NUM_EPOCHS = 4
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1            # 10% of total steps for warmup (better than fixed steps)
MAX_SAMPLES = 30_000           # Passed to data.prepare_data()
MODEL_NAME = "distilbert-base-multilingual-cased"   # Multilingual (104 languages)
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


def compute_class_weights(train_dataset):
    """Compute inverse-frequency class weights from the training set labels.
    
    This ensures the model pays more attention to the minority class (toxic),
    which is critical for imbalanced datasets like civil_comments (~10% toxic).
    
    Returns: torch.FloatTensor of shape (num_classes,)
    """
    labels = train_dataset["label"]
    if hasattr(labels, "tolist"):
        labels = labels.tolist()
    
    labels_array = np.array(labels)
    class_counts = np.bincount(labels_array, minlength=NUM_LABELS)
    total = len(labels_array)
    
    # Inverse frequency weighting: weight = total / (num_classes * count)
    weights = total / (NUM_LABELS * class_counts.astype(np.float64))
    
    print(f"\n⚖️  Class Weight Balancing:")
    print(f"   Class 0 (non-toxic): {class_counts[0]:>6} samples → weight = {weights[0]:.4f}")
    print(f"   Class 1 (toxic):     {class_counts[1]:>6} samples → weight = {weights[1]:.4f}")
    print(f"   Toxic class gets {weights[1]/weights[0]:.1f}x more weight in the loss")
    
    return torch.FloatTensor(weights)


class WeightedTrainer(torch.nn.Module):
    """Custom Trainer subclass that uses weighted CrossEntropyLoss
    to handle class imbalance. The minority class (toxic) gets a higher
    weight so the model is penalized more for missing toxic content.
    """
    pass  # We'll use the HuggingFace Trainer's compute_loss override


def train():
    """Fine-tune multilingual DistilBERT with class-weight balancing and save the model."""
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

    # ── 2. Compute class weights ─────────────────────────────────────────────
    class_weights = compute_class_weights(splits["train"])

    # ── 3. Load pre-trained model ────────────────────────────────────────────
    print(f"\n── Step 2: Loading pre-trained {MODEL_NAME} ──")
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=NUM_LABELS,
        )
        print(f"✅ Model loaded: {MODEL_NAME}")
        param_count = sum(p.numel() for p in model.parameters())
        print(f"   Parameters: {param_count:,}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        raise

    # ── 4. Create custom Trainer with weighted loss ──────────────────────────
    class WeightedLossTrainer(Trainer):
        """Trainer that overrides compute_loss to use class-weighted CrossEntropyLoss."""
        
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            
            # Move class weights to same device as logits
            weight = class_weights.to(logits.device)
            loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
            loss = loss_fn(logits, labels)
            
            return (loss, outputs) if return_outputs else loss

    # ── 5. Configure training ────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE * 2,
        learning_rate=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
        warmup_ratio=WARMUP_RATIO,         # 10% warmup instead of fixed steps
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

    # ── 6. Train ─────────────────────────────────────────────────────────────
    print(f"\n── Step 3: Training ({NUM_EPOCHS} epochs, {len(splits['train'])} samples) ──")
    print(f"   Estimated time: ~25-40 min on Apple Silicon CPU")
    
    trainer = WeightedLossTrainer(
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

    # ── 7. Save best model ───────────────────────────────────────────────────
    print("\n── Step 4: Saving model ──")
    final_model_path = os.path.join(OUTPUT_DIR, "best_model")
    trainer.save_model(final_model_path)
    tokenizer.save_pretrained(final_model_path)
    print(f"💾 Model saved to: {final_model_path}")

    # ── 8. Quick validation metrics ──────────────────────────────────────────
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
