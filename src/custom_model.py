import torch
import torch.nn as nn

class CustomBiLSTMClassifier(nn.Module):
    """
    A custom PyTorch text classification model using a Bidirectional LSTM.
    Takes token IDs, runs them through an Embedding layer, a BiLSTM, and a linear head.
    """
    def __init__(self, vocab_size: int = 30522, embedding_dim: int = 128, hidden_dim: int = 128, num_classes: int = 2):
        super().__init__()
        # 1. Word Embedding Layer
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim, padding_idx=0)
        
        # 2. Bidirectional LSTM Layer
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.3
        )
        
        # 3. Fully Connected Classification Head
        # Bidirectional means output dimension of LSTM is hidden_dim * 2
        self.fc = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features=hidden_dim * 2, out_features=64),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(in_features=64, out_features=num_classes)
        )

    def forward(self, input_ids, attention_mask=None):
        # input_ids shape: (batch_size, seq_len)
        batch_size = input_ids.size(0)
        embedded = self.embedding(input_ids) # (batch_size, seq_len, embedding_dim)
        
        # Initialize hidden and cell states dynamically to support any batch size in ONNX
        h0 = torch.zeros(self.lstm.num_layers * 2, batch_size, self.lstm.hidden_size, dtype=embedded.dtype, device=input_ids.device)
        c0 = torch.zeros(self.lstm.num_layers * 2, batch_size, self.lstm.hidden_size, dtype=embedded.dtype, device=input_ids.device)
        
        # LSTM forward pass
        # outputs shape: (batch_size, seq_len, hidden_dim * 2)
        outputs, (hidden, cell) = self.lstm(embedded, (h0, c0))
        
        # We pool the outputs by taking the mean across the sequence length, 
        # but ignoring padding if attention_mask is provided.
        if attention_mask is not None:
            # Mask shape: (batch_size, seq_len, 1)
            mask = attention_mask.unsqueeze(-1).float()
            # Element-wise multiply to zero out padded tokens
            masked_outputs = outputs * mask
            # Sum and divide by sequence length of actual tokens
            sum_outputs = torch.sum(masked_outputs, dim=1)
            lengths = torch.clamp(torch.sum(mask, dim=1), min=1.0)
            pooled = sum_outputs / lengths
        else:
            pooled = torch.mean(outputs, dim=1) # (batch_size, hidden_dim * 2)

        # Classification logits
        logits = self.fc(pooled) # (batch_size, num_classes)
        return logits
