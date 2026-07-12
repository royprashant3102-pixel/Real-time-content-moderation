"""
data_multilingual.py — Load, clean, tokenize, and split a mixed English-Hindi toxicity dataset.

Uses:
  - English: 'civil_comments' (or synthetic fallback)
  - Hindi: 'textdetox/multilingual_toxicity_dataset' (split 'hi')
"""

import os
import sys
import re
import numpy as np
from datasets import Dataset

# ─── CONFIG ──────────────────────────────────────────────────────────────────
MAX_SAMPLES_PER_LANG = 2500  # Default samples per language for training
MAX_SEQ_LENGTH = 128
TOXICITY_THRESHOLD = 0.5
TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_SEED = 42
MODEL_NAME = "distilbert-base-multilingual-cased"
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
# ─────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, os.path.dirname(__file__))
from data import clean_text, load_dataset_safe

def load_hindi_dataset(max_samples: int):
    """Load Hindi toxicity dataset from Hugging Face."""
    from datasets import load_dataset
    try:
        print("⏳ Loading multilingual toxicity dataset (Hindi split) from Hugging Face...")
        # Since 'hi' is a split name in this dataset
        ds = load_dataset(
            "textdetox/multilingual_toxicity_dataset",
            split=f"hi[:{max_samples}]",
            trust_remote_code=True
        )
        texts = [clean_text(row["text"]) for row in ds]
        labels = [int(row["toxic"]) for row in ds]
        print(f"✅ Loaded {len(texts)} Hindi samples.")
        return texts, labels
    except Exception as e:
        print(f"⚠️ Failed to load Hindi dataset: {e}")
        # Synthetic Hindi fallback
        return _create_synthetic_hindi_dataset(max_samples)

def _create_synthetic_hindi_dataset(n: int = 100):
    """Fallback synthetic Hindi data creator."""
    import random
    random.seed(RANDOM_SEED)
    
    toxic_templates = [
        "तुम पागल हो और तुम्हें प्रतिबंधित किया जाना चाहिए",
        "मुझे तुमसे नफरत है, दूर जाओ",
        "यह सबसे मूर्खतापूर्ण बात है जो मैंने कभी पढ़ी है",
        "तुम बिल्कुल मूर्ख हो, तुम्हें कुछ नहीं पता",
        "अपना मुंह बंद रखो मूर्ख",
    ]
    non_toxic_templates = [
        "इस विषय पर अपने दृष्टिकोण को साझा करने के लिए धन्यवाद",
        "मुझे लगता है कि हम इस बारे में रचनात्मक चर्चा कर सकते हैं",
        "यह एक दिलचस्प बिंदु है, मैं आपके योगदान की सराहना करता हूं",
        "मैं यहां आपके विश्लेषण से सम्मानपूर्वक असहमत हूं",
        "महान लेख, बहुत जानकारीपूर्ण और अच्छी तरह से लिखा गया",
    ]
    
    texts, labels = [], []
    for _ in range(n):
        if random.random() < 0.3:
            texts.append(random.choice(toxic_templates))
            labels.append(1)
        else:
            texts.append(random.choice(non_toxic_templates))
            labels.append(0)
            
    print(f"✅ Created synthetic Hindi dataset with {n} samples.")
    return texts, labels

def prepare_multilingual_data(max_samples_per_lang: int = MAX_SAMPLES_PER_LANG):
    """Load, clean, merge, tokenize and split English and Hindi data."""
    from transformers import AutoTokenizer
    
    # 1. Load English data
    print("\n── Ingesting English Data ──")
    en_texts, en_labels = load_dataset_safe(max_samples=max_samples_per_lang)
    
    # 2. Load Hindi data
    print("\n── Ingesting Hindi Data ──")
    hi_texts, hi_labels = load_hindi_dataset(max_samples=max_samples_per_lang)
    
    # 3. Merge and Shuffle
    merged_texts = en_texts + hi_texts
    merged_labels = en_labels + hi_labels
    
    # Shuffle indices
    np.random.seed(RANDOM_SEED)
    shuffled_idx = np.random.permutation(len(merged_texts))
    texts = [merged_texts[i] for i in shuffled_idx]
    labels = [merged_labels[i] for i in shuffled_idx]
    
    # 4. Tokenize
    print("\n── Tokenizing Multilingual Data ──")
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
    
    # 5. Splits
    split1 = dataset.train_test_split(test_size=TEST_SIZE, seed=RANDOM_SEED)
    split2 = split1["train"].train_test_split(
        test_size=VAL_SIZE / (1 - TEST_SIZE), seed=RANDOM_SEED
    )
    
    splits = {
        "train": split2["train"],
        "val": split2["test"],
        "test": split1["test"],
    }
    
    # Report stats
    print(f"\n📊 Multilingual Split Distribution:")
    for name, ds in splits.items():
        lbls = ds["label"].tolist()
        total = len(lbls)
        toxic = sum(lbls)
        print(f"  {name:<10} split: {total:>5} total | {toxic:>4} toxic ({100.0*toxic/total:.1f}%)")
        
    # Save tokenizer for deploying
    output_tok_dir = os.path.join(DATA_DIR, "tokenizer_multilingual")
    os.makedirs(output_tok_dir, exist_ok=True)
    tokenizer.save_pretrained(output_tok_dir)
    print(f"\n💾 Multilingual tokenizer saved to {output_tok_dir}")
    
    return splits, tokenizer

if __name__ == "__main__":
    prepare_multilingual_data(max_samples_per_lang=200)
    print("✅ data_multilingual.py execution complete.")
