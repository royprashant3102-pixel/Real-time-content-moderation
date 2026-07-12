"""
test_on_kaggle.py — Evaluate fine-tuned models on the Kaggle Jigsaw Toxic Comment Classification Challenge dataset.

Loads:
  1. Fine-tuned DistilBERT (PyTorch)
  2. Custom BiLSTM (ONNX INT8 Quantized)

Calculates performance metrics (Accuracy, Precision, Recall, F1 Score, Latency).
Saves results to kaggle_test_results.md.
"""

import os
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset as TorchDataset
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from datasets import load_dataset

# Ensure src directory is in path
sys.path.insert(0, os.path.dirname(__file__))
from data import clean_text
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# ─── CONFIG ──────────────────────────────────────────────────────────────────
MAX_SAMPLES = 2000  # Number of samples from Kaggle dataset for evaluation
BATCH_SIZE = 32
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# ─────────────────────────────────────────────────────────────────────────────

class JigsawDataset(TorchDataset):
    def __init__(self, texts, labels, tokenizer, max_len=64):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        inputs = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )
        
        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long)
        }

def evaluate_pytorch_model(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    latencies = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            t0 = time.perf_counter()
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            dt = (time.perf_counter() - t0) * 1000 / len(labels)
            latencies.append(dt)
            
            preds = torch.argmax(logits, dim=-1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    acc = accuracy_score(all_labels, all_preds)
    mean_latency = np.mean(latencies)
    
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "latency_ms": mean_latency
    }

def evaluate_onnx_model(onnx_path, texts, labels, tokenizer, max_len=64):
    import onnxruntime as ort
    
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])
    
    all_preds = []
    all_labels = []
    latencies = []
    
    for i in range(0, len(texts), BATCH_SIZE):
        batch_texts = texts[i:i+BATCH_SIZE]
        batch_labels = labels[i:i+BATCH_SIZE]
        
        inputs = tokenizer(
            batch_texts,
            padding="max_length",
            truncation=True,
            max_length=max_len,
            return_tensors="np"
        )
        
        t0 = time.perf_counter()
        feed = {
            "input_ids": inputs["input_ids"].astype(np.int64),
            "attention_mask": inputs["attention_mask"].astype(np.int64),
        }
        logits = session.run(None, feed)[0]
        dt = (time.perf_counter() - t0) * 1000 / len(batch_labels)
        latencies.append(dt)
        
        preds = np.argmax(logits, axis=-1)
        all_preds.extend(preds)
        all_labels.extend(batch_labels)
        
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    acc = accuracy_score(all_labels, all_preds)
    mean_latency = np.mean(latencies)
    
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "latency_ms": mean_latency
    }

def main():
    device = torch.device("cpu")  # Force CPU for fair benchmark latency comparison
    print(f"🖥️ Using device for benchmark: {device}")
    
    print("\n── Step 1: Loading Kaggle Toxic Comment Dataset from Hugging Face ──")
    try:
        # Load Jigsaw dataset from HF hub
        ds = load_dataset(
            "thesofakillers/jigsaw-toxic-comment-classification-challenge",
            split="train",
            trust_remote_code=True
        )
        
        # Take a slice from the end of the training set for testing
        test_slice = ds.select(range(len(ds) - MAX_SAMPLES, len(ds)))
        
        texts = [clean_text(row["comment_text"]) for row in test_slice]
        # Any of the toxicity columns set to 1 means toxic
        labels = [
            1 if (row["toxic"] == 1 or row["severe_toxic"] == 1 or row["obscene"] == 1 or
                  row["threat"] == 1 or row["insult"] == 1 or row["identity_hate"] == 1) else 0
            for row in test_slice
        ]
        print(f"✅ Loaded {len(texts)} samples from Jigsaw dataset.")
    except Exception as e:
        print(f"❌ Failed to load Kaggle dataset: {e}")
        sys.exit(1)
        
    # Load tokenizers
    tokenizer_dir = os.path.join(PROJECT_ROOT, "data", "tokenizer")
    if not os.path.exists(tokenizer_dir):
        tokenizer_dir = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    
    results = {}
    
    # 1. Fine-tuned DistilBERT
    print("\n── Evaluating Fine-tuned DistilBERT on Jigsaw ──")
    ft_path = os.path.join(PROJECT_ROOT, "model", "best_model")
    if os.path.exists(ft_path):
        try:
            ft_distilbert = AutoModelForSequenceClassification.from_pretrained(ft_path).to(device)
            torch_dataset = JigsawDataset(texts, labels, tokenizer, max_len=64)
            dataloader = DataLoader(torch_dataset, batch_size=BATCH_SIZE, shuffle=False)
            results["Fine-tuned DistilBERT"] = evaluate_pytorch_model(
                ft_distilbert, dataloader, device
            )
            print("✅ Fine-tuned DistilBERT evaluation complete.")
        except Exception as e:
            print(f"⚠️ Could not evaluate Fine-tuned DistilBERT: {e}")
    else:
        print(f"⚠️ Fine-tuned DistilBERT path not found at {ft_path}. Skipping.")
        
    # 2. Custom BiLSTM ONNX Quantized
    print("\n── Evaluating Custom BiLSTM Classifier (ONNX Quantized) on Jigsaw ──")
    onnx_path = os.path.join(PROJECT_ROOT, "model", "onnx", "model_quantized.onnx")
    if os.path.exists(onnx_path):
        try:
            results["Custom BiLSTM (ONNX INT8)"] = evaluate_onnx_model(
                onnx_path, texts, labels, tokenizer, max_len=64
            )
            print("✅ Custom BiLSTM (ONNX INT8) evaluation complete.")
        except Exception as e:
            print(f"⚠️ Could not evaluate Custom BiLSTM ONNX: {e}")
    else:
        print(f"⚠️ Custom ONNX model not found at {onnx_path}. Skipping.")
        
    if not results:
        print("❌ No models were evaluated.")
        return
        
    markdown_table = (
        "| Model Name | Accuracy | Precision | Recall | F1 Score | Latency per sample (ms) |\n"
        "| :--- | :---: | :---: | :---: | :---: | :---: |\n"
    )
    for model_name, metrics in results.items():
        markdown_table += (
            f"| {model_name} | {metrics['accuracy']:.4f} | {metrics['precision']:.4f} | "
            f"{metrics['recall']:.4f} | {metrics['f1']:.4f} | {metrics['latency_ms']:.3f} ms |\n"
        )
        
    print("\n📊 KAGGLE TEST RESULTS:")
    print("-" * 80)
    print(markdown_table)
    print("-" * 80)
    
    output_file = os.path.join(PROJECT_ROOT, "kaggle_test_results.md")
    with open(output_file, "w") as f:
        f.write("# Kaggle Toxic Comment Challenge Test Results\n\n")
        f.write(f"Evaluated on {len(texts)} samples from the Jigsaw Toxic Comment Classification Challenge dataset.\n")
        f.write("All latencies measured on CPU execution to ensure comparability.\n\n")
        f.write(markdown_table)
        
    print(f"\n💾 Saved results comparison to {output_file}")

if __name__ == "__main__":
    main()
