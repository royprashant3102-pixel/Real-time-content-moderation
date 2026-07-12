"""
train_multilingual.py — Fine-tune distilbert-base-multilingual-cased on English and Hindi dataset.
"""

import os
import sys
import torch

# ─── CONFIG ──────────────────────────────────────────────────────────────────
NUM_EPOCHS = 1                # Keep small for fast CPU fine-tuning demonstration
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_STEPS = 50
MAX_SAMPLES_PER_LANG = 400   # Set small for reasonable runtimes on CPU
MODEL_NAME = "distilbert-base-multilingual-cased"
NUM_LABELS = 2
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model", "best_model_multilingual"))
LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))
# ─────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, os.path.dirname(__file__))
from data_multilingual import prepare_multilingual_data
from train import compute_metrics, get_device

def train_multilingual(max_samples_per_lang: int = MAX_SAMPLES_PER_LANG):
    """Fine-tune the multilingual model and save it."""
    from transformers import (
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
    )
    
    device = get_device()
    print(f"🖥️ Using device: {device}")
    
    # 1. Load splits
    splits, tokenizer = prepare_multilingual_data(max_samples_per_lang=max_samples_per_lang)
    
    # 2. Load model
    print(f"\n── Loading pre-trained model: {MODEL_NAME} ──")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS
    )
    
    # 3. Configure training arguments
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
        logging_steps=20,
        save_total_limit=1,
        report_to="none",
        fp16=False,
        use_cpu=not torch.cuda.is_available(),
        seed=42,
    )
    
    # 4. Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=splits["train"],
        eval_dataset=splits["val"],
        compute_metrics=compute_metrics,
    )
    
    # 5. Train
    print("\n── Fine-tuning Multilingual Model ──")
    trainer.train()
    print("✅ Training complete!")
    
    # 6. Save model and tokenizer
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"💾 Saved best multilingual model and tokenizer to: {OUTPUT_DIR}")
    
    return OUTPUT_DIR

if __name__ == "__main__":
    train_multilingual()
