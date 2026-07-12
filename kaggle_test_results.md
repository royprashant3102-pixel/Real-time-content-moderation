# Kaggle Toxic Comment Challenge Test Results

Evaluated on 10 samples from the Jigsaw Toxic Comment Classification Challenge dataset.
All latencies measured on CPU execution to ensure comparability.

| Model Name | Accuracy | Precision | Recall | F1 Score | Latency per sample (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Fine-tuned DistilBERT | 0.9000 | 0.0000 | 0.0000 | 0.0000 | 11.064 ms |
| Custom BiLSTM (ONNX INT8) | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 9.025 ms |
