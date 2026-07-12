# Model Comparison Results

Evaluated on a subset of the test split containing 2 samples.
All latencies measured on CPU execution to ensure comparability.

| Model Name | Accuracy | Precision | Recall | F1 Score | Latency per sample (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Base DistilBERT (Pre-trained) | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 213.538 ms |
| Fine-tuned DistilBERT | 0.5000 | 0.0000 | 0.0000 | 0.0000 | 22.441 ms |
| Custom BiLSTM (PyTorch) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 4.061 ms |
| Custom BiLSTM (ONNX INT8) | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.905 ms |
