"""A static web page that runs an exported model entirely in the browser (transformers.js + onnxruntime-web)."""

from __future__ import annotations

import html
import json
import shutil
from pathlib import Path

TRANSFORMERS_JS = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1"

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root { --bg: #fbfbfa; --fg: #1d1d1b; --muted: #5f5e5a; --line: #dcdbd6; --accent: #2a78d6; --bar: #cde2fb; }
  @media (prefers-color-scheme: dark) {
    :root { --bg: #161615; --fg: #efeee9; --muted: #b3b2aa; --line: #393935; --accent: #5598e7; --bar: #1c3a63; }
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--fg); font: 16px/1.5 system-ui, sans-serif; }
  main { max-width: 680px; margin: 0 auto; padding: 32px 16px; }
  h1 { font-size: 1.5rem; margin: 0 0 4px; }
  p.sub { color: var(--muted); margin: 0 0 24px; }
  label { font-weight: 600; display: block; margin-bottom: 6px; }
  textarea { width: 100%; min-height: 96px; padding: 10px; font: inherit; color: inherit; background: transparent;
             border: 1px solid var(--line); border-radius: 8px; resize: vertical; }
  .row { display: flex; gap: 8px; flex-wrap: wrap; margin: 12px 0 24px; align-items: center; }
  button { font: inherit; padding: 8px 16px; border-radius: 8px; border: 1px solid var(--accent); cursor: pointer;
           background: var(--accent); color: #fff; }
  button.ghost { background: transparent; color: var(--fg); border-color: var(--line); }
  button:disabled { opacity: .5; cursor: progress; }
  ol { list-style: none; padding: 0; margin: 0; }
  li { position: relative; padding: 8px 12px; margin-bottom: 6px; border-radius: 6px; overflow: hidden; }
  li .fill { position: absolute; inset: 0 auto 0 0; background: var(--bar); z-index: 0; }
  li span { position: relative; z-index: 1; }
  li .p { float: right; font-variant-numeric: tabular-nums; color: var(--muted); }
  #status { color: var(--muted); }
  footer { margin-top: 40px; color: var(--muted); font-size: .9rem; }
  a { color: var(--accent); }
</style>
</head>
<body>
<main>
  <h1>__TITLE__</h1>
  <p class="sub">A __SIZE__ MB model distilled from an LLM with tinydistill. It runs in your browser; nothing you type
    leaves this page.</p>
  <label for="text">Text</label>
  <textarea id="text" placeholder="Type a message">__FIRST__</textarea>
  <div class="row">
    <button id="go" disabled>Classify</button>
    __SAMPLES__
    <span id="status" aria-live="polite">Loading the model...</span>
  </div>
  <ol id="out" aria-label="Top labels"></ol>
  <footer>Made with <a href="https://github.com/Andreas-Hum/tinydistill">tinydistill</a> and
    <a href="https://github.com/huggingface/transformers.js">transformers.js</a>.</footer>
</main>
<script type="module">
  import { pipeline, env } from "__TRANSFORMERS_JS__";
  env.allowLocalModels = true;  // the model files sit next to this page
  env.allowRemoteModels = false;
  env.localModelPath = "./";
  const text = document.getElementById("text"), go = document.getElementById("go");
  const out = document.getElementById("out"), status = document.getElementById("status");
  const clf = await pipeline("text-classification", "model", { dtype: "__DTYPE__" });
  async function run() {
    if (!text.value.trim()) return;
    const t0 = performance.now();
    const top = await clf(text.value, { top_k: 5 });
    status.textContent = `${(performance.now() - t0).toFixed(0)} ms`;
    out.replaceChildren(...top.map(({ label, score }) => {
      const li = document.createElement("li");
      li.innerHTML = `<div class="fill" style="width:${(score * 100).toFixed(1)}%"></div>`;
      const name = document.createElement("span"); name.textContent = label;
      const p = document.createElement("span"); p.className = "p"; p.textContent = score.toFixed(3);
      li.append(name, p); return li;
    }));
  }
  go.onclick = run;
  for (const b of document.querySelectorAll("button.sample")) b.onclick = () => { text.value = b.dataset.text; run(); };
  go.disabled = false; status.textContent = "Ready";
  run();
</script>
</body>
</html>
"""


def build_site(onnx_dir: Path, out: Path, title: str | None = None, samples: int = 3) -> Path:
    """Write ``out/index.html`` and ``out/model/`` (config, tokenizer, ONNX) for transformers.js."""
    if not (onnx_dir / "config.json").exists():
        raise SystemExit(f"{onnx_dir} has no config.json: export it again with `tinydistill export` (0.3.0+)")
    int8 = (onnx_dir / "model.int8.onnx").exists()
    model_dir = out / "model"
    (model_dir / "onnx").mkdir(parents=True, exist_ok=True)
    for f in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
    ):
        if (onnx_dir / f).exists():
            shutil.copy(onnx_dir / f, model_dir / f)
    src = onnx_dir / ("model.int8.onnx" if int8 else "model.onnx")
    shutil.copy(src, model_dir / "onnx" / ("model_quantized.onnx" if int8 else "model.onnx"))

    run = onnx_dir.parent
    task = json.loads((run / "meta.json").read_text()).get("task") if (run / "meta.json").exists() else None
    texts = []
    if (run / "test.jsonl").exists():
        with (run / "test.jsonl").open(encoding="utf-8") as f:
            texts = [json.loads(line)["text"] for _, line in zip(range(samples), f, strict=False)]
    name = title or (task or run.name).replace("-", " ").replace("_", " ").title()
    buttons = "\n    ".join(
        f'<button class="ghost sample" data-text="{html.escape(t)}">Example {i + 1}</button>'
        for i, t in enumerate(texts)
    )
    page = (
        PAGE.replace("__TITLE__", html.escape(name))
        .replace("__SIZE__", f"{src.stat().st_size / 1e6:.0f}")
        .replace("__FIRST__", html.escape(texts[0]) if texts else "")
        .replace("__SAMPLES__", buttons)
        .replace("__DTYPE__", "q8" if int8 else "fp32")
        .replace("__TRANSFORMERS_JS__", TRANSFORMERS_JS)
    )
    (out / "index.html").write_text(page, encoding="utf-8")
    return out / "index.html"
