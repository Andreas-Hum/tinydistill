| Model | Runs on | Test accuracy | Latency per text | Size | Speed-up |
|---|---|---|---|---|---|
| Teacher: `Qwen/Qwen2.5-7B-Instruct`, 4-bit, 16 most similar examples in the prompt, plus a linear probe | RTX 4090 | 0.859 | 155 ms | 5.4 GB | 1x |
| Student: MiniLM-L6, int8 ONNX | CPU, 2 threads | 0.872 | 1.0 ms | 23 MB | 153x |
