| Model | Runs on | Test accuracy | Latency per text | Size | Speed-up |
|---|---|---|---|---|---|
| Teacher: `Qwen/Qwen2.5-7B-Instruct`, 4-bit, 8 examples per class in the prompt | RTX 4090 | 0.783 | 264 ms | 5.4 GB | 1x |
| Student: MiniLM-L6, int8 ONNX | CPU, 2 threads | 0.814 | 1.2 ms | 23 MB | 214x |
| Student: static embeddings, int8 ONNX | CPU, 2 threads | 0.802 | 0.1 ms | 8 MB | 3244x |
