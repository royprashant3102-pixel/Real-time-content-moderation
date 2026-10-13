# Kaggle Toxic Comment Challenge Test Results

Evaluated on 10 samples from the Jigsaw Toxic Comment Classification Challenge dataset.
All latencies measured on CPU execution to ensure comparability.

| Model Name | Accuracy | Precision | Recall | F1 Score | Latency per sample (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Fine-tuned DistilBERT | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 30.825 ms |
| Custom BiLSTM (ONNX INT8) | 0.9000 | 0.0000 | 0.0000 | 0.0000 | 22.059 ms |
