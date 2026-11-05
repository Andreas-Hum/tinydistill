| Model | Runs on | Test accuracy | Latency per text | Size | Speed-up |
|---|---|---|---|---|---|
| Teacher: `Qwen/Qwen2.5-7B-Instruct`, 4-bit, 16 most similar examples in the prompt | RTX 4090 | 0.851 | 157 ms | 5.4 GB | 1x |
| Student: MiniLM-L6, int8 ONNX | CPU, 2 threads | 0.855 | 1.0 ms | 23 MB | 154x |
