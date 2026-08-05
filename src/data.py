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
MAX_HINGLISH_SAMPLES = 8_000  # Hindi/Hinglish samples (expanded for better Hindi coverage)
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
    # EXPANDED: 200+ templates for robust Hindi/Hinglish coverage.
    
    hinglish_toxic = [
        # ── Standalone slurs (critical — model must learn these words are toxic alone) ──
        "randi",
        "chutiya",
        "bhenchod",
        "madarchod",
        "gaandu",
        "harami",
        "kutti",
        "saala",
        "haramkhor",
        "kamina",
        "bhosdi",
        "lauda",
        "lodu",
        "chinal",
        "hijda",
        "chakka",
        "dalla",
        "bhadwa",
        "raand",
        "suar",
        "kutta",
        "gadha",
        "ullu",
        
        # ── Devanagari standalone slurs ──
        "रांडी",
        "चूतिया",
        "भड़वा",
        "मादरचोद",
        "भेंचोद",
        "गांडू",
        "हरामी",
        "कमीना",
        "कुत्ती",
        "लौड़ा",
        "छिनाल",
        "हिजड़ा",
        "साला",
        "हरामखोर",
        "दल्ला",
        "सुअर",
        "गधा",
        
        # ── Short greeting + slur combinations (exactly like "hello randi") ──
        "hello randi",
        "hi randi",
        "hey randi",
        "oye randi",
        "abe randi",
        "hello chutiye",
        "hi chutiye",
        "oye chutiye",
        "abe chutiye",
        "hello gaandu",
        "oye gaandu",
        "hello kamini",
        "oye kamina",
        "abe saale",
        "oye bhenchod",
        "hello harami",
        "hey madarchod",
        "hi bhadwe",
        "oye hijde",
        "abe lodu",
        "hello kutte",
        "oye suar",
        "hey dalla",
        
        # ── Sexual harassment / vulgar ──
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
        "tu duniya ka sabse bada chutiya hai",
        "tujhe dekhke ulti aati hai",
        "teri shakal dekhne layak nahi hai",
        "kamina kahi ka dhokha deta hai",
        "haramzada saala jhootha hai",
        "tu raand hai saali kameeni",
        "tera muh kala hona chahiye",
        "gandagi hai tu insaan nahi",
        "besharmi ki hadd hai teri",
        "nikal yahan se kutte",
        "tere muh pe thook dunga",
        "tu laude jaisa insaan hai",
        "teri maa ko gaali du kya",
        "bhosdi ke tera kaam hi ganda hai",
        "chutmarike sun idhar",
        "randi ke bacche",
        "randi ka bacha hai tu",
        "madarchod nikl yahan se",
        "bhenchod band kar ye bakwas",
        "gaandu saala kuch kaam kar",
        "tu bhadwa hai saala",
        "dalla saala log ko bechta hai",
        
        # ── Mixed English-Hindi (code-switching) ──
        "I hate you so much tum bahut gande ho",
        "you stupid idiot pagal insaan",
        "shut up chup kar saale",
        "you are disgusting ghatiya aadmi",
        "die you fool mar ja saale",
        "tu worst hai sabse ganda insaan",
        "you are such a chutiya honestly",
        "what a randi you are",
        "you are nothing but a gaandu",
        "stop being such a kamina",
        "everyone knows you are harami",
        "you are the biggest madarchod ever",
        "just die already mar ja bhosdike",
        "you are a randi everyone knows",
        "such a bhenchod move by you",
        "this is so chutiya behavior",
        "you sound like a complete lodu",
        "behave yourself you randi",
        "you are acting like a dalla",
        "stop being a bhadwa",
        "you disgusting hijda",
        "absolute kutti behavior from you",
        
        # ── Devanagari script abuse (full sentences) ──
        "तू रांडी है साली",
        "भड़वा कहीं का निकल यहां से",
        "चूतिया है तू बिल्कुल",
        "मादरचोद क्या कर रहा है",
        "गांडू साला कुछ काम कर",
        "कमीना साला धोकेबाज़",
        "हरामखोर कहीं का",
        "तेरी औकात क्या है साले",
        "तुझे मार डालूंगा",
        "तेरा मुंह तोड़ दूंगा",
        "कुत्ते की औलाद है तू",
        "तू लौड़े जैसा इंसान है",
        "रांडी के बच्चे",
        "भेंचोद बंद कर ये बकवास",
        "तू छिनाल है साली",
        "दल्ला साला लोगों को बेचता है",
        "सुअर कहीं का गंदा इंसान",
        "तू हिजड़ा है साला",
        "तेरी बहन को पेल दूंगा",
        "गांड मरा साले",
        
        # ── Threats and violence ──
        "tujhe jaan se maar dunga",
        "tera khoon kar dunga",
        "tere ghar mein aag laga dunga",
        "tujhe utha ke phenk dunga",
        "tere haath pair tod dunga",
        "tujhe zinda nahi chodunga",
        "tujhe khatam kar dunga",
        "teri haddi tod dunga",
        "tujhe hospital pahuncha dunga",
        "tere ghar waalon ko dekh lunga",
        "tujhe gaad dunga zameen mein",
        "tujhe jala dunga saale",
        "tera band baja dunga",
        "tujhe nanga karke ghuma dunga",
        
        # ── Hate speech (caste, religion, community) ──
        "tum sab ek jaise ghatiya ho",
        "tumhari jaat hi aisi hai nalayak",
        "tum log insaan nahi ho janwar ho",
        "tere dharm wale sab chor hai",
        "tumhari community barbaad hai",
        "tum neech log ho",
        "tumhari jaat ko koi izzat nahi deta",
        "tere dharm ke log gande hai",
        "tum dalit ho isliye ghatiya ho",
        "tumhare community ke sab chor hai",
        
        # ── Misspellings / leetspeak variations (evasion attempts) ──
        "r@ndi",
        "rndii",
        "ch*tiya",
        "chutiya",
        "chootiya",
        "m@darchod",
        "bh3nchod",
        "g@@ndu",
        "gaandu saala",
        "k@mina",
        "har@mi",
        "bhosadike",
        "bhosdike",
        "l0du",
        "rand1",
        "bh@dwa",
    ]
    
    hinglish_non_toxic = [
        # ── Polite conversation ──
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
        
        # ── Greetings and casual ──
        "hello bhai kaise ho",
        "hi dost sab theek hai",
        "namaste aapka din shubh ho",
        "hey yaar kya haal hai",
        "hello ji aap kaise hain",
        "hi friend aaj kaisa raha din",
        "namaste ji bahut accha laga milke",
        "hey bhai long time no see",
        "hello didi kaise ho aap",
        "hi sir kya chal raha hai",
        "aaj mausam bahut accha hai",
        "kya plan hai aaj ka",
        "chai peene chalein",
        "aaj ka khana bahut accha tha",
        "film bahut acchi thi yaar",
        
        # ── Professional / educational ──
        "yeh project bahut accha ban raha hai",
        "aapki presentation bahut informative thi",
        "meeting mein bahut acche points raise kiye aapne",
        "aapka code review bahut helpful tha",
        "yeh feature bahut zaruri hai users ke liye",
        "deadline se pehle complete kar lenge",
        "team ne bahut accha kaam kiya",
        "aapki leadership se project successful hua",
        "yeh bug fix bahut important tha thanks",
        "documentation bahut clear likhi hai aapne",
        "aapki training session se bahut seekha",
        "yeh course bahut helpful hai",
        "exam ki taiyari chal rahi hai",
        "padhai mein dhyan lagao acha karogay",
        "teacher ne bahut accha padhaya aaj",
        
        # ── Devanagari non-toxic ──
        "आपका बहुत बहुत धन्यवाद",
        "आपने बहुत अच्छा काम किया",
        "मुझे आपसे मिलकर खुशी हुई",
        "यह बहुत अच्छा विचार है",
        "आपकी मदद से सब ठीक हो गया",
        "नमस्ते आपका दिन शुभ हो",
        "बहुत बढ़िया काम कर रहे हो",
        "आपकी सोच बहुत सही है",
        "मिलकर काम करेंगे तो सब हो जाएगा",
        "आज का दिन बहुत अच्छा रहा",
        "बहुत सुंदर लिखा है आपने",
        "आपका अनुभव हम सबके लिए प्रेरणादायक है",
        "यह जानकारी बहुत उपयोगी है",
        "शुक्रिया इतना अच्छा समझाने के लिए",
        "आप बहुत मेहनती इंसान हैं",
        
        # ── Mixed English-Hindi non-toxic ──
        "this is really accha kaam by the team",
        "I think ye idea bahut creative hai",
        "great work yaar keep it up",
        "very nice article bahut informative",
        "let's discuss ye topic tomorrow",
        "your presentation was bahut helpful",
        "thank you for helping meri madad ke liye",
        "I appreciate aapki dedication towards work",
        "this solution is bahut smart hai",
        "we should collaborate milke kaam karein",
        "amazing effort by sabhi team members",
        "really proud of aapki achievement",
        "well done bhai bahut accha hua",
        "looking forward to agle project ki taraf",
        "this feedback is bahut constructive hai",
    ]
    
    # ── Data augmentation for diversity ──
    # Instead of just duplicating templates, create variations
    augmented_toxic = []
    augmented_non_toxic = []
    
    # Augmentation 1: Case variations for romanized text
    for t in hinglish_toxic:
        augmented_toxic.append(t)
        # Only augment ASCII text (not Devanagari)
        if t.isascii():
            augmented_toxic.append(t.upper())
            augmented_toxic.append(t.capitalize())
    
    for t in hinglish_non_toxic:
        augmented_non_toxic.append(t)
        if t.isascii():
            augmented_non_toxic.append(t.capitalize())
    
    # Augmentation 2: Add common prefixes/suffixes to standalone slurs
    slur_prefixes = ["tu ", "ye ", "kya ", "abe ", "oye ", "sale ", "tera ", ""]
    slur_suffixes = [" hai", " saala", " insaan", " log", " kahi ka", ""]
    standalone_slurs = [
        "randi", "chutiya", "bhenchod", "madarchod", "gaandu",
        "harami", "kamina", "bhadwa", "dalla", "lodu", "kutti",
    ]
    for slur in standalone_slurs:
        for prefix in slur_prefixes:
            for suffix in slur_suffixes:
                combo = f"{prefix}{slur}{suffix}".strip()
                if combo not in augmented_toxic and len(combo) > len(slur):
                    augmented_toxic.append(combo)
    
    # Deduplicate
    augmented_toxic = list(dict.fromkeys(augmented_toxic))
    augmented_non_toxic = list(dict.fromkeys(augmented_non_toxic))
    
    # Add curated samples (replicate if needed to fill target)
    target_per_class = max(max_samples // 3, 500)
    
    curated_toxic = []
    curated_non_toxic = []
    
    while len(curated_toxic) < target_per_class:
        for t in augmented_toxic:
            curated_toxic.append(t)
            if len(curated_toxic) >= target_per_class:
                break
    
    while len(curated_non_toxic) < target_per_class:
        for t in augmented_non_toxic:
            curated_non_toxic.append(t)
            if len(curated_non_toxic) >= target_per_class:
                break
    
    texts.extend(curated_toxic)
    labels.extend([1] * len(curated_toxic))
    texts.extend(curated_non_toxic)
    labels.extend([0] * len(curated_non_toxic))
    
    print(f"✅ Total Hinglish/Hindi samples: {len(texts)} ({hf_loaded} from HF + {len(curated_toxic) + len(curated_non_toxic)} curated)")
    print(f"   Toxic: {sum(labels)} | Non-toxic: {len(labels) - sum(labels)}")
    print(f"   Unique toxic templates: {len(augmented_toxic)} | Unique non-toxic templates: {len(augmented_non_toxic)}")
    
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
