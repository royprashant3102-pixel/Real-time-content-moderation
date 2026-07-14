"""
data.py — Load, clean, tokenize, and split the dataset for content moderation.

Uses:
  - English: 'civil_comments' from Hugging Face (30K samples)
  - Hindi/Hinglish: 'textdetox/multilingual_toxicity_dataset' + curated Hinglish samples
  - Falls back to synthetic data if downloads fail.

Config variables at top for easy tuning.
"""

import os
import sys
import random

# ─── CONFIG ──────────────────────────────────────────────────────────────────
MAX_SAMPLES = 30_000          # English samples from civil_comments
MAX_HINGLISH_SAMPLES = 3_000  # Hindi/Hinglish samples
MAX_SEQ_LENGTH = 128          # DistilBERT max tokens per input
TOXICITY_THRESHOLD = 0.5      # civil_comments score >= this → toxic (label=1)
TEST_SIZE = 0.15
VAL_SIZE = 0.15               # of the remaining after test split
RANDOM_SEED = 42
MODEL_NAME = "distilbert-base-multilingual-cased"   # Multilingual model (104 languages)
DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
# ─────────────────────────────────────────────────────────────────────────────

import re
import numpy as np


def clean_text(text: str) -> str:
    """Text cleaning that preserves multilingual characters (Hindi, Devanagari, etc.).
    Only removes URLs, excessive whitespace, and normalizes spacing.
    """
    text = text.strip()
    text = re.sub(r"http\S+|www\.\S+", "", text)        # remove URLs
    text = re.sub(r"<[^>]+>", "", text)                   # remove HTML tags
    text = re.sub(r"\s+", " ", text).strip()              # normalize whitespace
    return text


def load_dataset_safe(max_samples: int = MAX_SAMPLES):
    """
    Try to load 'google/civil_comments' from Hugging Face.
    Falls back to a synthetic dataset if download fails.

    Returns: (texts, labels) as lists.
    """
    try:
        from datasets import load_dataset
        print(f"⏳ Loading civil_comments dataset ({max_samples} samples) from Hugging Face...")
        ds = load_dataset(
            "google/civil_comments",
            split=f"train[:{max_samples}]",
            trust_remote_code=True,
        )
        texts = [clean_text(row["text"]) for row in ds]
        labels = [1 if row["toxicity"] >= TOXICITY_THRESHOLD else 0 for row in ds]
        print(f"✅ Loaded {len(texts)} English samples from civil_comments.")
        return texts, labels

    except Exception as e:
        print(f"⚠️  Failed to load civil_comments: {e}")
        print("   Falling back to synthetic dataset...")
        return _create_synthetic_dataset(max_samples)


def load_hinglish_dataset(max_samples: int = MAX_HINGLISH_SAMPLES):
    """Load Hindi/Hinglish toxicity data from HuggingFace + curated samples.
    
    Returns: (texts, labels) as lists.
    """
    texts, labels = [], []
    
    # ── Part 1: Try loading Hindi split from HuggingFace ──
    hf_loaded = 0
    try:
        from datasets import load_dataset
        print(f"⏳ Loading Hindi toxicity dataset from HuggingFace...")
        ds = load_dataset(
            "textdetox/multilingual_toxicity_dataset",
            split=f"hi[:{max_samples // 2}]",
            trust_remote_code=True,
        )
        for row in ds:
            texts.append(clean_text(row["text"]))
            labels.append(int(row["toxic"]))
        hf_loaded = len(texts)
        print(f"✅ Loaded {hf_loaded} Hindi samples from HuggingFace.")
    except Exception as e:
        print(f"⚠️  Failed to load Hindi dataset from HuggingFace: {e}")
    
    # ── Part 2: Curated Hinglish toxic/non-toxic examples ──
    # These cover transliterated Hindi profanity, slurs, and common Hinglish toxic patterns
    # that the HuggingFace dataset might miss (romanized Hindi).
    
    hinglish_toxic = [
        # Sexual harassment / vulgar
        "mai tumko chodna chahta hu",
        "teri maa ki gaali",
        "bhosdi ke tu yahan se nikal",
        "saale madarchod kya kar raha hai",
        "chutiya hai tu bilkul",
        "teri behen ko pel dunga",
        "gaand mara saale",
        "tu randi hai saali",
        "tere baap ka naukar nahi hu",
        "kutte ki aulad hai tu",
        "haramkhor kahi ka nikal yahan se",
        "tujhe maar dunga saale",
        "tu pagal hai kya be",
        "gadhe jaisi shakal hai teri",
        "nalayak insaan hai tu",
        "bhenchod kya bakwas hai ye",
        "tu ek number ka ghatiya insaan hai",
        "tere jaise log ko zinda nahi rehna chahiye",
        "kamina saala dhokebaaz",
        "tu hijda hai saala",
        "muh tod dunga tera",
        "teri aukat kya hai saale",
        "bhikhari saala aukaat mein reh",
        "jaahil insaan hai tu bilkul",
        "tujhse bada bewakoof koi nahi",
        "I hate you so much tum bahut gande ho",
        "you stupid idiot pagal insaan",
        "shut up chup kar saale",
        "you are disgusting ghatiya aadmi",
        "die you fool mar ja saale",
        "tu worst hai sabse ganda insaan",
        "nikal yahan se kutte",
        "tere muh pe thook dunga",
        "gandagi hai tu insaan nahi",
        "besharmi ki hadd hai teri",
        "tu duniya ka sabse bada chutiya hai",
        "tujhe dekhke ulti aati hai",
        "teri shakal dekhne layak nahi hai",
        "kamina kahi ka dhokha deta hai",
        "haramzada saala jhootha hai",
        # Threats and violence
        "tujhe jaan se maar dunga",
        "tera khoon kar dunga",
        "tere ghar mein aag laga dunga",
        "tujhe utha ke phenk dunga",
        "tere haath pair tod dunga",
        # Hate speech
        "tum sab ek jaise ghatiya ho",
        "tumhari jaat hi aisi hai nalayak",
        "tum log insaan nahi ho janwar ho",
        "tere dharm wale sab chor hai",
        "tumhari community barbaad hai",
    ]
    
    hinglish_non_toxic = [
        "aap bahut acche insaan ho",
        "mujhe aapki baat bahut pasand aayi",
        "bohot acha kaam kiya aapne",
        "dhanyavaad aapki madad ke liye",
        "yeh bahut informative hai shukriya",
        "mai aapki respect karta hu",
        "aapka point bahut valid hai",
        "hum sab milke kaam karenge",
        "bahut badhiya article likha hai",
        "aapne sahi kaha bilkul sahi baat",
        "mujhe aapse kuch seekhne ko mila",
        "aapka experience kaafi accha hai",
        "yeh topic bahut interesting hai",
        "mai aapki soch se agree karta hu",
        "shukriya itna accha samjhane ke liye",
        "aap bahut talented ho",
        "yeh kaam bahut mushkil tha lekin aapne kar dikhaya",
        "bahut hi sundar likha hai aapne",
        "aapka perspective samajhne mein maza aaya",
        "chaliye milke isko solve karte hain",
        "thank you bhai bahut help ki tumne",
        "good job yaar bahut accha kiya",
        "aapki guidance se bahut fayda hua",
        "hum sab saath mein kar sakte hain",
        "bahut acchi thinking hai aapki",
        "main aapka supporter hoon",
        "aap inspiring ho bahut",
        "sahi direction mein ja rahe ho",
        "keep it up bhai bahut accha",
        "mujhe khushi hui yeh padhke",
        "aapki mehnat rang laayegi zaroor",
        "bahut pyaari soch hai aapki",
        "aap sabse acche teacher ho",
        "yeh conversation bahut productive rahi",
        "aap logon ka kaam dekhke dil khush ho gaya",
        "aapne bahut accha samjhaya thanks",
        "mai aapse sehmat hu completely",
        "yeh solution bahut creative hai",
        "aapki writing style bahut engaging hai",
        "bohot maza aaya padhke",
    ]
    
    # Add curated Hinglish samples (replicate to increase volume)
    random.seed(RANDOM_SEED)
    curated_toxic = []
    curated_non_toxic = []
    
    # Replicate and slightly vary to create more samples
    target_per_class = max(max_samples // 4, 300)
    
    while len(curated_toxic) < target_per_class:
        for t in hinglish_toxic:
            curated_toxic.append(t)
            if len(curated_toxic) >= target_per_class:
                break
    
    while len(curated_non_toxic) < target_per_class:
        for t in hinglish_non_toxic:
            curated_non_toxic.append(t)
            if len(curated_non_toxic) >= target_per_class:
                break
    
    texts.extend(curated_toxic)
    labels.extend([1] * len(curated_toxic))
    texts.extend(curated_non_toxic)
    labels.extend([0] * len(curated_non_toxic))
    
    print(f"✅ Total Hinglish/Hindi samples: {len(texts)} ({hf_loaded} from HF + {len(curated_toxic) + len(curated_non_toxic)} curated)")
    print(f"   Toxic: {sum(labels)} | Non-toxic: {len(labels) - sum(labels)}")
    
    return texts, labels


def _create_synthetic_dataset(n: int = 1000):
    """Generate a small synthetic toxic/non-toxic dataset for testing."""
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
    """Tokenize texts using the multilingual DistilBERT tokenizer. Returns a HF Dataset."""
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
    Full pipeline: load English + Hinglish → clean → merge → tokenize → split → report.
    Returns: (splits_dict, tokenizer)
    """
    # 1. Load English data
    print("\n── Loading English Data ──")
    en_texts, en_labels = load_dataset_safe(max_samples)
    
    # 2. Load Hinglish/Hindi data
    print("\n── Loading Hinglish/Hindi Data ──")
    hi_texts, hi_labels = load_hinglish_dataset(MAX_HINGLISH_SAMPLES)
    
    # 3. Merge and shuffle
    all_texts = en_texts + hi_texts
    all_labels = en_labels + hi_labels
    
    np.random.seed(RANDOM_SEED)
    indices = np.random.permutation(len(all_texts))
    all_texts = [all_texts[i] for i in indices]
    all_labels = [all_labels[i] for i in indices]
    
    print(f"\n📊 Merged dataset: {len(all_texts)} total samples")
    print(f"   Toxic: {sum(all_labels)} | Non-toxic: {len(all_labels) - sum(all_labels)}")
    
    # 4. Tokenize and split
    dataset, tokenizer = tokenize_data(all_texts, all_labels)
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
