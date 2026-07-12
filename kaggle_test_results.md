# Kaggle Toxic Comment Challenge Test Results

Evaluated on 2000 samples from the Jigsaw Toxic Comment Classification Challenge dataset.
All latencies measured on CPU execution to ensure comparability.

| Model Name | Accuracy | Precision | Recall | F1 Score | Latency per sample (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Fine-tuned DistilBERT | 0.9505 | 0.8344 | 0.6300 | 0.7179 | 6.734 ms |
| Custom BiLSTM (ONNX INT8) | 0.8890 | 0.4415 | 0.4150 | 0.4278 | 0.349 ms |
