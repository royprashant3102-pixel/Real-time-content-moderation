"""
compare_models.py — Evaluate and compare the performance of different models on the test set.

Loads:
  1. Base Pre-trained DistilBERT (un-fine-tuned)
  2. Fine-tuned DistilBERT
  3. Custom BiLSTM PyTorch
  4. Custom BiLSTM ONNX Quantized

Computes Accuracy, Precision, Recall, F1 Score, and Mean Latency per sample.
Saves comparison table to model_comparison_results.md.
"""

import os
import sys
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

# Ensure src directory is in path
sys.path.insert(0, os.path.dirname(__file__))
from data import prepare_data
from custom_model import CustomBiLSTMClassifier

# ─── CONFIG ──────────────────────────────────────────────────────────────────
MAX_SAMPLES = 2000  # Number of samples for comparison evaluation
BATCH_SIZE = 32
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_pytorch_model(model, dataloader, device, is_bilstm=False):
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
            if is_bilstm:
                logits = model(input_ids, attention_mask)
            else:
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

def evaluate_onnx_model(onnx_path, dataset):
    import onnxruntime as ort
    
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    session = ort.InferenceSession(onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])
    
    all_preds = []
    all_labels = []
    latencies = []
    
    input_ids_all = dataset["input_ids"].numpy()
    attention_mask_all = dataset["attention_mask"].numpy()
    labels_all = dataset["label"].numpy()
    
    for i in range(0, len(labels_all), BATCH_SIZE):
        input_ids = input_ids_all[i:i+BATCH_SIZE]
        attention_mask = attention_mask_all[i:i+BATCH_SIZE]
        labels = labels_all[i:i+BATCH_SIZE]
        
        t0 = time.perf_counter()
        feed = {
            "input_ids": input_ids.astype(np.int64),
            "attention_mask": attention_mask.astype(np.int64),
        }
        logits = session.run(None, feed)[0]
        dt = (time.perf_counter() - t0) * 1000 / len(labels)
        latencies.append(dt)
        
        preds = np.argmax(logits, axis=-1)
        all_preds.extend(preds)
        all_labels.extend(labels)
        
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
    
    print("\n── Step 1: Loading test dataset ──")
    splits, tokenizer = prepare_data(max_samples=MAX_SAMPLES)
    test_dataset = splits["test"]
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print(f"📊 Evaluated on {len(test_dataset)} test samples.")
    
    results = {}
    
    # 1. Base Pre-trained DistilBERT
    print("\n── Evaluating Base Pre-trained DistilBERT ──")
    try:
        from transformers import AutoModelForSequenceClassification
        base_distilbert = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", 
            num_labels=2
        ).to(device)
        results["Base DistilBERT (Pre-trained)"] = evaluate_pytorch_model(
            base_distilbert, test_loader, device, is_bilstm=False
        )
        print("✅ Base DistilBERT evaluation complete.")
    except Exception as e:
        print(f"⚠️ Could not evaluate Base DistilBERT: {e}")
        
    # 2. Fine-tuned DistilBERT
    print("\n── Evaluating Fine-tuned DistilBERT ──")
    ft_path = os.path.join(PROJECT_ROOT, "model", "best_model")
    if os.path.exists(ft_path):
        try:
            from transformers import AutoModelForSequenceClassification
            ft_distilbert = AutoModelForSequenceClassification.from_pretrained(ft_path).to(device)
            results["Fine-tuned DistilBERT"] = evaluate_pytorch_model(
                ft_distilbert, test_loader, device, is_bilstm=False
            )
            print("✅ Fine-tuned DistilBERT evaluation complete.")
        except Exception as e:
            print(f"⚠️ Could not evaluate Fine-tuned DistilBERT: {e}")
    else:
        print(f"⚠️ Fine-tuned DistilBERT path not found at {ft_path}. Skipping.")
        
    # 3. Custom BiLSTM Classifier (PyTorch)
    print("\n── Evaluating Custom BiLSTM Classifier (PyTorch) ──")
    bilstm_path = os.path.join(PROJECT_ROOT, "model", "best_custom_model.pth")
    if os.path.exists(bilstm_path):
        try:
            custom_model = CustomBiLSTMClassifier(vocab_size=tokenizer.vocab_size).to(device)
            custom_model.load_state_dict(torch.load(bilstm_path, map_location=device))
            results["Custom BiLSTM (PyTorch)"] = evaluate_pytorch_model(
                custom_model, test_loader, device, is_bilstm=True
            )
            print("✅ Custom BiLSTM (PyTorch) evaluation complete.")
        except Exception as e:
            print(f"⚠️ Could not evaluate Custom BiLSTM PyTorch: {e}")
    else:
        print(f"⚠️ Custom BiLSTM weights not found at {bilstm_path}. Skipping.")
        
    # 4. Custom BiLSTM Classifier (ONNX Quantized)
    print("\n── Evaluating Custom BiLSTM Classifier (ONNX Quantized) ──")
    onnx_path = os.path.join(PROJECT_ROOT, "model", "onnx", "model_quantized.onnx")
    if os.path.exists(onnx_path):
        try:
            results["Custom BiLSTM (ONNX INT8)"] = evaluate_onnx_model(
                onnx_path, test_dataset
            )
            print("✅ Custom BiLSTM (ONNX INT8) evaluation complete.")
        except Exception as e:
            print(f"⚠️ Could not evaluate Custom BiLSTM ONNX: {e}")
    else:
        print(f"⚠️ Custom ONNX model not found at {onnx_path}. Skipping.")
        
    # ── Step 3: Print and Save Results ──
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
        
    print("\n📊 MODEL COMPARISON RESULTS:")
    print("-" * 80)
    print(markdown_table)
    print("-" * 80)
    
    output_file = os.path.join(PROJECT_ROOT, "model_comparison_results.md")
    with open(output_file, "w") as f:
        f.write("# Model Comparison Results\n\n")
        f.write(f"Evaluated on a subset of the test split containing {len(test_dataset)} samples.\n")
        f.write("All latencies measured on CPU execution to ensure comparability.\n\n")
        f.write(markdown_table)
        
    print(f"\n💾 Saved results comparison to {output_file}")

if __name__ == "__main__":
    main()
