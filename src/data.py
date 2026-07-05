"""
data.py — Load, clean, tokenize, and split the dataset for content moderation.

Uses the 'civil_comments' dataset from Hugging Face. Falls back to a synthetic
sample if download fails.

Config variables at top for easy tuning.
"""

import os
import sys

# ─── CONFIG ──────────────────────────────────────────────────────────────────
MAX_SAMPLES = 10_000          # Total samples to use (keep small for fast runs)
MAX_SEQ_LENGTH = 128          # DistilBERT max tokens per input
TOXICITY_THRESHOLD = 0.5      # civil_comments score >= this → toxic (label=1)
TEST_SIZE = 0.15
VAL_SIZE = 0.15               # of the remaining after test split
RANDOM_SEED = 42
MODEL_NAME = "distilbert-base-uncased"
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
# ─────────────────────────────────────────────────────────────────────────────

import re
import numpy as np

def clean_text(text: str) -> str:
    """Basic text cleaning: lowercase, strip URLs, extra whitespace."""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", "", text)        # remove URLs
    text = re.sub(r"[^a-z0-9\s.,!?'\"-]", " ", text)    # keep basic punctuation
    text = re.sub(r"\s+", " ", text).strip()
    return text


def load_dataset_safe(max_samples: int = MAX_SAMPLES):
    """
    Try to load 'google/civil_comments' from Hugging Face.
    Falls back to a synthetic dataset if download fails.

    Returns: (texts, labels) as lists.
    """
    try:
        from datasets import load_dataset
        print("⏳ Loading civil_comments dataset from Hugging Face...")
        ds = load_dataset(
            "google/civil_comments",
            split=f"train[:{max_samples}]",
            trust_remote_code=True,
        )
        texts = [clean_text(row["text"]) for row in ds]
        labels = [1 if row["toxicity"] >= TOXICITY_THRESHOLD else 0 for row in ds]
        print(f"✅ Loaded {len(texts)} samples from civil_comments.")
        return texts, labels

    except Exception as e:
        print(f"⚠️  Failed to load civil_comments: {e}")
        print("   Falling back to synthetic dataset...")
        return _create_synthetic_dataset(max_samples)


def _create_synthetic_dataset(n: int = 1000):
    """Generate a small synthetic toxic/non-toxic dataset for testing."""
    import random
    random.seed(RANDOM_SEED)

    toxic_templates = [
        "you are such an idiot and should be banned",
        "i hate people like you, go away",
        "this is the stupidest thing i have ever read",
        "what a moron, you know nothing",
        "shut up you worthless fool",
        "you disgust me, you are terrible",
        "people like you are ruining everything",
        "you are a complete loser and nobody likes you",
        "this is garbage and you should be ashamed",
        "i wish you would just disappear forever",
    ]
    non_toxic_templates = [
        "thank you for sharing your perspective on this topic",
        "i think we can have a constructive discussion about this",
        "that is an interesting point, i appreciate your input",
        "i respectfully disagree with your analysis here",
        "great article, very informative and well written",
        "could you provide more details on this subject",
        "i learned something new today, thanks for posting",
        "this is a thoughtful and balanced view of the issue",
        "well said, i completely agree with your assessment",
        "let us work together to find a good solution",
    ]

    texts, labels = [], []
    for _ in range(n):
        if random.random() < 0.3:  # ~30% toxic to simulate class imbalance
            texts.append(random.choice(toxic_templates))
            labels.append(1)
        else:
            texts.append(random.choice(non_toxic_templates))
            labels.append(0)

    print(f"✅ Created synthetic dataset with {n} samples.")
    return texts, labels


def tokenize_data(texts, labels):
    """Tokenize texts using the DistilBERT tokenizer. Returns a HF Dataset."""
    from transformers import AutoTokenizer
    from datasets import Dataset

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    dataset = Dataset.from_dict({"text": texts, "label": labels})

    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            padding="max_length",
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
        )

    dataset = dataset.map(tokenize_fn, batched=True, batch_size=256)
    dataset = dataset.remove_columns(["text"])
    dataset.set_format("torch")
    return dataset, tokenizer


def split_dataset(dataset):
    """Split dataset into train / val / test."""
    # First split: separate out test
    split1 = dataset.train_test_split(test_size=TEST_SIZE, seed=RANDOM_SEED)
    # Second split: separate val from remaining train
    split2 = split1["train"].train_test_split(
        test_size=VAL_SIZE / (1 - TEST_SIZE), seed=RANDOM_SEED
    )

    return {
        "train": split2["train"],
        "val": split2["test"],
        "test": split1["test"],
    }


def report_class_distribution(splits: dict):
    """Print class distribution for each split."""
    print("\n📊 Class Distribution:")
    print(f"{'Split':<10} {'Total':>7} {'Non-Toxic (0)':>15} {'Toxic (1)':>12} {'% Toxic':>10}")
    print("-" * 58)
    for name, ds in splits.items():
        labels = ds["label"].tolist() if hasattr(ds["label"], "tolist") else ds["label"]
        total = len(labels)
        toxic = sum(labels)
        non_toxic = total - toxic
        pct = 100.0 * toxic / total if total > 0 else 0
        print(f"{name:<10} {total:>7} {non_toxic:>15} {toxic:>12} {pct:>9.1f}%")
    print()


def prepare_data(max_samples: int = MAX_SAMPLES):
    """
    Full pipeline: load → clean → tokenize → split → report.
    Returns: (splits_dict, tokenizer)
    """
    texts, labels = load_dataset_safe(max_samples)
    dataset, tokenizer = tokenize_data(texts, labels)
    splits = split_dataset(dataset)
    report_class_distribution(splits)

    # Save tokenizer for later use
    os.makedirs(DATA_DIR, exist_ok=True)
    tokenizer.save_pretrained(os.path.join(DATA_DIR, "tokenizer"))
    print(f"💾 Tokenizer saved to {os.path.join(DATA_DIR, 'tokenizer')}")

    return splits, tokenizer


if __name__ == "__main__":
    splits, tokenizer = prepare_data()
    print("✅ data.py completed successfully!")
    print(f"   Train: {len(splits['train'])} | Val: {len(splits['val'])} | Test: {len(splits['test'])}")
