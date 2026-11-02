#!/usr/bin/env bash
# Checks the vLLM teacher against the local one: Qwen2.5-1.5B scores 300 financial tweets both ways, and the
# largest difference between the two sets of label scores is saved to examples/vllm-smoke.json.
# Needs vLLM in its own environment: VLLM=/path/to/venv-with-vllm/bin/vllm.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p runs
model=Qwen/Qwen2.5-1.5B-Instruct
data=hf:zeroshot/twitter-financial-news-sentiment:validation
"${VLLM:-vllm}" serve "$model" --enable-prefix-caching --gpu-memory-utilization 0.3 --max-model-len 4096 \
  --port 8000 > runs/vllm-server.log 2>&1 &
server=$!
trap 'kill $server' EXIT
until curl -sf http://127.0.0.1:8000/health > /dev/null; do sleep 5; done

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
print(result)
PY
