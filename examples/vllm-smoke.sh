#!/usr/bin/env bash
# Checks the vLLM teacher against the local one: Qwen2.5-1.5B scores 300 financial tweets both ways, and the
# largest difference between the two sets of label scores is saved to examples/vllm-smoke.json.
# Needs vLLM in its own environment: VLLM=/path/to/venv-with-vllm/bin/vllm.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs
model=Qwen/Qwen2.5-1.5B-Instruct
data=hf:zeroshot/twitter-financial-news-sentiment:validation
# Without a CUDA toolkit (nvcc): no torch.compile or CUDA graphs (--enforce-eager), and the PyTorch sampler
# instead of FlashInfer's, which is compiled on first use. Scoring only reads prompt log-probabilities.
VLLM_USE_FLASHINFER_SAMPLER=0 "${VLLM:-vllm}" serve "$model" --enable-prefix-caching --gpu-memory-utilization 0.3 --max-model-len 4096 \
  --enforce-eager --host 127.0.0.1 --port 8000 > runs/vllm-server.log 2>&1 &
server=$!
trap 'kill $server 2>/dev/null || true' EXIT
until curl -sf --max-time 5 http://127.0.0.1:8000/health > /dev/null; do
  kill -0 $server 2>/dev/null || { echo "vLLM server stopped; see runs/vllm-server.log" >&2; exit 1; }
  sleep 5
done

for where in local vllm; do
  args=(examples/financial-tweets/task.yaml --data "$data" --limit 300 --teacher "$model" --device cuda
        --out "runs/vllm-smoke-$where")
  [ "$where" = vllm ] && args+=(--vllm-url http://127.0.0.1:8000/v1)
  tinydistill label "${args[@]}"
done
python - <<'PY'
import json
rows = {w: [json.loads(line) for line in open(f"runs/vllm-smoke-{w}/train.jsonl")] for w in ("local", "vllm")}
diffs = [abs(a - b) for x, y in zip(rows["local"], rows["vllm"]) for a, b in zip(x["scores"], y["scores"])]
same = sum(max(range(3), key=x["scores"].__getitem__) == max(range(3), key=y["scores"].__getitem__)
           for x, y in zip(rows["local"], rows["vllm"]))
result = {"texts": len(rows["local"]), "max_abs_diff": max(diffs), "mean_abs_diff": sum(diffs) / len(diffs),
          "same_top_label": same / len(rows["local"])}
json.dump(result, open("examples/vllm-smoke.json", "w"), indent=2)
open("examples/vllm-smoke.md", "w").write(
    "| Texts | Same top label | Largest difference in a label score | Mean difference |\n|---|---|---|---|\n"
    f"| {result['texts']} | {result['same_top_label']:.1%} | {result['max_abs_diff']:.3f} | {result['mean_abs_diff']:.4f} |\n"
    "\nQwen2.5-1.5B-Instruct scoring 300 financial tweets locally (bf16, tinydistill) and through vLLM "
    "(prompt log-probabilities); label scores are log-probabilities, so small differences are numerical.\n"
)
print(result)
PY
