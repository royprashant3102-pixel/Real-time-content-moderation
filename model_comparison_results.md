# Model Comparison Results

Evaluated on a subset of the test split containing 300 samples.
All latencies measured on CPU execution to ensure comparability.

| Model Name | Accuracy | Precision | Recall | F1 Score | Latency per sample (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Base DistilBERT (Pre-trained) | 0.0467 | 0.0467 | 1.0000 | 0.0892 | 14.928 ms |
| Fine-tuned DistilBERT | 0.9700 | 0.7273 | 0.5714 | 0.6400 | 15.039 ms |
| Custom BiLSTM (PyTorch) | 0.9467 | 0.4500 | 0.6429 | 0.5294 | 1.058 ms |
| Custom BiLSTM (ONNX INT8) | 0.9467 | 0.4500 | 0.6429 | 0.5294 | 0.742 ms |
