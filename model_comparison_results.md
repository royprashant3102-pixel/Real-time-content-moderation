# Model Comparison Results

Evaluated on a subset of the test split containing 1402 samples.
All latencies measured on CPU execution to ensure comparability.

| Model Name | Accuracy | Precision | Recall | F1 Score | Latency per sample (ms) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Fine-tuned DistilBERT | 0.9608 | 0.9359 | 0.9961 | 0.9650 | 38.162 ms |
| Custom BiLSTM (ONNX INT8) | 0.9529 | 0.9223 | 0.9974 | 0.9584 | 33.076 ms |
