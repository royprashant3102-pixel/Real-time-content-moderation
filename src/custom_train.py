import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

# ─── CONFIG ──────────────────────────────────────────────────────────────────
NUM_EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
MAX_GRAD_NORM = 1.0  # gradient clipping for training stability
MAX_SAMPLES = 10_000
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "model"))
# ─────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, os.path.dirname(__file__))
from data import prepare_data
from custom_model import CustomBiLSTMClassifier

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        
        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM)
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(dataloader)

def evaluate(model, dataloader, device):
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device)
            
            logits = model(input_ids, attention_mask)
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average="binary", zero_division=0
    )
    acc = accuracy_score(all_labels, all_preds)
    
    return {
        "accuracy": acc,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }

def main():
    device = get_device()
    print(f"🖥️ Using device: {device}")
    
    # 1. Load splits using our custom dataset builder
    print("\n── Step 1: Loading and Preprocessing Data ──")
    splits, tokenizer = prepare_data(max_samples=MAX_SAMPLES)
    
    # 2. Setup Dataloaders
    train_loader = DataLoader(splits["train"], batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(splits["val"], batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(splits["test"], batch_size=BATCH_SIZE, shuffle=False)
    
    # 3. Initialize Custom Model
    print("\n── Step 2: Initializing Custom BiLSTM Classifier ──")
    model = CustomBiLSTMClassifier(vocab_size=tokenizer.vocab_size).to(device)
    
    # Calculate class weights for CrossEntropyLoss due to class imbalance
    labels = splits["train"]["label"].numpy()
    neg_count = np.sum(labels == 0)
    pos_count = np.sum(labels == 1)
    
    # Set positive class weight slightly higher to address class imbalance
    weight = torch.tensor([1.0, neg_count / max(pos_count, 1)], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=weight)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # 4. Training loop
    print("\n── Step 3: Training Loop ──")
    best_f1 = -1
    best_model_path = os.path.join(OUTPUT_DIR, "best_custom_model.pth")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    for epoch in range(1, NUM_EPOCHS + 1):
        loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        metrics = evaluate(model, val_loader, device)
        
        print(f"Epoch {epoch}/{NUM_EPOCHS} - Loss: {loss:.4f} | Val Acc: {metrics['accuracy']:.4f} | Val F1: {metrics['f1']:.4f}")
        
        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            torch.save(model.state_dict(), best_model_path)
            print(f"💾 Saved new best model to: {best_model_path}")
            
    # 5. Evaluate on test set
    print("\n── Step 4: Final Test Evaluation ──")
    # Load best model weights
    model.load_state_dict(torch.load(best_model_path))
    test_metrics = evaluate(model, test_loader, device)
    print(f"Test Accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Test Precision: {test_metrics['precision']:.4f}")
    print(f"Test Recall:    {test_metrics['recall']:.4f}")
    print(f"Test F1:        {test_metrics['f1']:.4f}")

if __name__ == "__main__":
    main()
