"""A static web page that runs an exported model entirely in the browser (transformers.js + onnxruntime-web)
and shows what happens inside it: the tokens, what each layer would answer, where the classifier looks, the
probabilities over every label, which words matter, the teacher's soft labels and the cascade."""

from __future__ import annotations

import hashlib
import html
import json
import math
import re
import shutil
import tempfile
from pathlib import Path

TRANSFORMERS_JS = "https://cdn.jsdelivr.net/npm/@huggingface/transformers@3.8.1"

PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
  :root {
    --bg: #fbfbfa; --card: #ffffff; --fg: #1d1d1b; --muted: #5f5e5a; --line: #dcdbd6; --soft: #f1f0ec;
    --accent: #2a78d6; --bar: #cde2fb; --teacher: #b45309; --teacher-bar: #fde3c3;
    --pos: 217, 98, 43; --neg: 42, 120, 214; --att: 42, 120, 214;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --bg: #161615; --card: #1e1e1c; --fg: #efeee9; --muted: #b3b2aa; --line: #393935; --soft: #262624;
      --accent: #5598e7; --bar: #1c3a63; --teacher: #f0a35e; --teacher-bar: #4d2f10;
      --pos: 240, 140, 90; --neg: 100, 160, 240; --att: 100, 160, 240;
    }
  }
  * { box-sizing: border-box; }
  body { margin: 0; background: var(--bg); color: var(--fg); font: 16px/1.5 system-ui, sans-serif; }
  main { max-width: 780px; margin: 0 auto; padding: 32px 16px 48px; }
  h1 { font-size: 1.6rem; margin: 0 0 4px; }
  h2 { font-size: 1.15rem; margin: 40px 0 4px; }
  h3 { font-size: 1rem; margin: 0; }
  p { margin: 0 0 12px; }
  .sub, .note { color: var(--muted); }
  .note { font-size: .92rem; }
  label { font-weight: 600; display: block; margin-bottom: 6px; }
  textarea { width: 100%; min-height: 84px; padding: 10px; font: inherit; color: inherit; background: var(--card);
             border: 1px solid var(--line); border-radius: 8px; resize: vertical; }
  .row { display: flex; gap: 8px; flex-wrap: wrap; margin: 12px 0 0; align-items: center; }
  button { font: inherit; padding: 7px 14px; border-radius: 8px; border: 1px solid var(--accent); cursor: pointer;
           background: var(--accent); color: #fff; }
  button.ghost { background: transparent; color: var(--fg); border-color: var(--line); }
  button.ghost[aria-pressed="true"] { border-color: var(--accent); color: var(--accent); font-weight: 600; }
  button:disabled { opacity: .5; cursor: progress; }
  #status { color: var(--muted); font-size: .92rem; }
  .answer { margin: 24px 0 0; padding: 16px; border: 1px solid var(--line); border-radius: 10px; background: var(--card); }
  .answer .big { font-size: 1.45rem; font-weight: 650; }
  .answer .meta { color: var(--muted); font-variant-numeric: tabular-nums; }
  .flow { display: flex; gap: 6px; flex-wrap: wrap; margin: 14px 0 0; font-size: .85rem; }
  .flow span { padding: 3px 9px; border-radius: 999px; border: 1px solid var(--line); color: var(--muted);
               transition: background .25s, color .25s, border-color .25s; }
  .flow span.on { background: var(--accent); border-color: var(--accent); color: #fff; }
  ol.steps { list-style: none; padding: 0; margin: 16px 0 0; counter-reset: step; }
  ol.steps > li { position: relative; padding: 0 0 26px 44px; counter-increment: step;
                  opacity: .25; transition: opacity .35s; }
  ol.steps > li.done { opacity: 1; }
  ol.steps > li::before { content: counter(step); position: absolute; left: 0; top: 0; width: 28px; height: 28px;
    border-radius: 50%; display: grid; place-items: center; font-weight: 650; font-size: .9rem;
    background: var(--soft); border: 1px solid var(--line); }
  ol.steps > li.done::before { background: var(--accent); border-color: var(--accent); color: #fff; }
  ol.steps > li::after { content: ""; position: absolute; left: 13.5px; top: 32px; bottom: 4px; width: 1px;
                         background: var(--line); }
  ol.steps > li:last-child::after { display: none; }
  .step-head { display: flex; justify-content: space-between; gap: 12px; align-items: baseline; margin-bottom: 4px; }
  .ms { color: var(--muted); font-size: .85rem; font-variant-numeric: tabular-nums; white-space: nowrap; }
  .box { margin-top: 10px; padding: 12px; border: 1px solid var(--line); border-radius: 8px; background: var(--card);
         overflow-x: auto; }
  .chips { display: flex; flex-wrap: wrap; gap: 4px; }
  .chip { padding: 2px 7px; border-radius: 5px; border: 1px solid var(--line); font: 14px/1.6 ui-monospace, monospace;
          background: var(--soft); }
  .chip.special { color: var(--muted); font-style: italic; }
  .chip.cont { border-left: 3px solid var(--accent); }
  .chip small { color: var(--muted); font-size: .72rem; margin-left: 4px; }
  table { border-collapse: collapse; width: 100%; font-size: .93rem; font-variant-numeric: tabular-nums; }
  td { padding: 4px 6px; vertical-align: middle; }
  td.n { color: var(--muted); white-space: nowrap; width: 1%; }
  td.l { width: 42%; }
  .barcell { position: relative; height: 20px; background: var(--soft); border-radius: 4px; overflow: hidden; min-width: 80px; }
  .barcell div { position: absolute; inset: 0 auto 0 0; background: var(--bar); }
  .barcell span { position: relative; padding-left: 6px; font-size: .82rem; line-height: 20px; }
  .bars { list-style: none; padding: 0; margin: 0; }
  .bars li { position: relative; padding: 5px 10px; margin-bottom: 4px; border-radius: 6px; overflow: hidden;
             background: var(--soft); }
  .bars li .fill { position: absolute; inset: 0 auto 0 0; background: var(--bar); }
  .bars.teacher li .fill { background: var(--teacher-bar); }
  .bars li span { position: relative; }
  .bars li .p { float: right; color: var(--muted); font-variant-numeric: tabular-nums; }
  .bars li.gold span:first-of-type::after { content: "  gold"; color: var(--teacher); font-size: .78rem; font-weight: 650; }
  .spectrum { display: flex; align-items: flex-end; gap: 1px; height: 64px; margin-top: 12px; }
  .spectrum div { flex: 1; background: var(--accent); min-height: 1px; border-radius: 1px 1px 0 0; }
  .words { line-height: 2.1; font-size: 1.02rem; }
  .words span { padding: 2px 3px; border-radius: 4px; }
  .legend { display: flex; gap: 14px; flex-wrap: wrap; font-size: .85rem; color: var(--muted); margin-top: 8px; }
  .legend i { display: inline-block; width: 12px; height: 12px; border-radius: 3px; vertical-align: -1px; margin-right: 4px; }
  .cols { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; }
  .cols h3 { font-size: .92rem; margin-bottom: 6px; }
  .cols h3.t { color: var(--teacher); }
  @media (max-width: 600px) { .cols { grid-template-columns: 1fr; } }
  input[type=range] { width: 100%; accent-color: var(--accent); }
  .verdict { font-weight: 650; }
  footer { margin-top: 40px; color: var(--muted); font-size: .9rem; }
  a { color: var(--accent); }
  @media (prefers-reduced-motion: reduce) { * { transition: none !important; } }
</style>
</head>
<body>
<main>
  <h1>__TITLE__</h1>
  <p class="sub">A __SIZE__ MB model distilled from a large language model with tinydistill. It runs in your
    browser, and nothing you type leaves this page. __SUMMARY__</p>
  <label for="text">Text</label>
  <textarea id="text" placeholder="Type a message">__FIRST__</textarea>
  <div class="row">
    <button id="go" disabled>Classify</button>
    __SAMPLES__
    <button class="ghost" id="slow" aria-pressed="false" title="Step through the model one stage at a time">Slow motion</button>
  </div>
  <div class="answer" aria-live="polite">
    <div class="big" id="answer">Loading the model...</div>
    <div class="meta" id="status"></div>
    <div class="flow" aria-hidden="true">
      <span>text</span><span>tokens</span><span>layers</span><span>classifier</span><span>probabilities</span>
    </div>
  </div>

  <h2>Inside the model</h2>
  <p class="note">Every step below is computed by the model in this page as it classifies your text. Change the
    text or pick an example and watch the steps fill in.</p>
  <ol class="steps" id="steps">
    <li id="s-tok"><div class="step-head"><h3>The tokenizer splits the text</h3><span class="ms"></span></div>
      <p class="note">The model never sees letters or words, only pieces from a fixed vocabulary of __VOCAB__.
        Rare words are split into several pieces (marked with a blue edge); hover a piece for its number.
        <i>[CLS]</i> is added at the start: the classifier reads the model's output at that position.</p>
      <div class="box"><div class="chips" id="tokens"></div></div></li>
    <li id="s-layers"><div class="step-head"><h3>Layer by layer</h3><span class="ms"></span></div>
      <p class="note">The pieces pass through __LAYERS__ transformer layers. Each layer lets every piece look at the
        others and update itself. If we stop after a layer and apply the final classifier, we see what the model
        would answer at that depth: early layers usually do not know yet, and the answer forms on the way up.</p>
      <div class="box"><table id="layers"></table></div></li>
    <li id="s-att"><div class="step-head"><h3>Where the classifier looks</h3><span class="ms"></span></div>
      <p class="note">Attention: how strongly the <i>[CLS]</i> position draws on each piece, averaged over the
        layer's attention heads. Darker means more. Pick a layer. <i>[CLS]</i> and <i>[SEP]</i> often collect
        much of the attention as a resting place, so compare the word pieces with each other.</p>
      <div class="row" id="att-layers" style="margin-top:0"></div>
      <div class="box"><div class="chips" id="attention"></div></div></li>
    <li id="s-out"><div class="step-head"><h3>From scores to probabilities</h3><span class="ms"></span></div>
      <p class="note">The classifier gives one score per label (__NLABELS__ labels) and a __ACTIVATION__ turns them
        into probabilities. The top five are below; the strip shows all of them, largest first. The small
        probabilities on other labels are what distillation passes on: the teacher's soft labels tell the
        student which labels are similar, not just which one is right.</p>
      <div class="box"><ol class="bars" id="top"></ol><div class="spectrum" id="spectrum"
        role="img" aria-label="Probabilities of all labels, largest first"></div></div></li>
    <li id="s-why"><div class="step-head"><h3>Which words matter</h3><span class="ms"></span></div>
      <p class="note">Each word is removed in turn and the model runs again on what is left. Words whose removal
        lowers the probability of <b id="why-label"></b> support it; words whose removal raises it count against
        it. Hover a word for the change.</p>
      <div class="box"><div class="words" id="words"></div>
        <div class="legend"><span><i style="background:rgba(var(--pos),.7)"></i>supports the answer</span>
          <span><i style="background:rgba(var(--neg),.7)"></i>counts against it</span></div></div></li>
    <li id="s-teacher"><div class="step-head"><h3>Teacher and student</h3></div>
      <p class="note">The student learned to imitate the soft labels of __TEACHER__. The teacher is far too large to
        run here, but its saved soft labels for the examples are in this page.</p>
      <div class="box" id="teacher"></div></li>
    <li id="s-cascade"><div class="step-head"><h3>When to ask the teacher</h3></div>
      <p class="note">In a cascade the student answers and passes the texts it is least sure about to the teacher.
        The slider sets how many; accuracy and the confidence threshold are measured on the held-out test set.</p>
      <div class="box" id="cascade"></div></li>
  </ol>
  <footer>Made with <a href="https://github.com/Andreas-Hum/tinydistill">tinydistill</a> and
    <a href="https://github.com/huggingface/transformers.js">transformers.js</a>. __NOTE__</footer>
</main>
<script type="application/json" id="data">__DATA__</script>
<script type="module">
  import { AutoTokenizer, AutoModel, env } from "__TRANSFORMERS_JS__";
  env.allowLocalModels = true;  // the model files sit next to this page
  env.allowRemoteModels = false;
  env.localModelPath = "./";
  const D = JSON.parse(document.getElementById("data").textContent);
  const $ = (id) => document.getElementById(id);
  const el = (tag, props = {}, ...kids) => {
    const e = Object.assign(document.createElement(tag), props);
    e.append(...kids);
    return e;
  };
  const name = (i) => D.labels[i].replaceAll("_", " ");
  const pct = (x) => (100 * x).toFixed(x >= 0.995 || x < 0.001 ? 0 : 1) + "%";

  // ---- core: the model and what it computes (no page elements here) ----
  const tok = await AutoTokenizer.from_pretrained(D.model);
  const model = await AutoModel.from_pretrained(D.model, { dtype: D.dtype });
  const softmax = (z) => { const m = Math.max(...z), e = z.map((x) => Math.exp(x - m)), s = e.reduce((a, b) => a + b);
                           return e.map((x) => x / s); };
  const toProbs = (z) => D.multi ? z.map((x) => 1 / (1 + Math.exp(-x))) : softmax(z);
  const rowsOf = (t) => { const [n, ...rest] = t.dims, k = rest.reduce((a, b) => a * b, 1);
                          return Array.from({ length: n }, (_, i) => Array.from(t.data.slice(i * k, (i + 1) * k))); };
  const split = (row, k) => Array.from({ length: row.length / k }, (_, i) => row.slice(i * k, (i + 1) * k));
  const argmax = (xs) => xs.reduce((b, x, i) => (x > xs[b] ? i : b), 0);
  const encode = (texts) => tok(texts, { padding: true, truncation: true, max_length: 128 });

  async function analyse(text) {
    const t0 = performance.now();
    const enc = encode([text]);
    const ids = Array.from(enc.input_ids.data, Number);
    const tokens = tok.model.convert_ids_to_tokens(ids);
    const t1 = performance.now();
    const out = await model(enc);
    const t2 = performance.now();
    const p = toProbs(rowsOf(out.logits)[0]);
    const top = argmax(p);
    const layers = out.layer_logits ? split(rowsOf(out.layer_logits)[0], D.labels.length).map(toProbs) : null;
    const attention = out.cls_attention ? split(rowsOf(out.cls_attention)[0], ids.length) : null;
    const words = text.trim().split(/\\s+/).filter(Boolean).slice(0, 60);
    let effect = [];
    if (words.length > 1) {  // remove each word in turn: one batch of texts
      const o = await model(encode(words.map((_, i) => words.filter((_, j) => j !== i).join(" "))));
      effect = rowsOf(o.logits).map((z) => p[top] - toProbs(z)[top]);
    }
    const t3 = performance.now();
    return { tokens, ids, p, top, layers, attention, words, effect,
             ms: { tokenize: t1 - t0, model: t2 - t1, words: t3 - t2 } };
  }
  // ---- end of core ----

  const text = $("text"), go = $("go"), status = $("status");
  let current = null, attLayer = null, runId = 0;

  function bars(list, probs, teacher = false, gold = null) {
    const ol = el("ol", { className: "bars" + (teacher ? " teacher" : "") });
    for (const i of list) {
      const li = el("li", { className: i === gold ? "gold" : "" },
        el("div", { className: "fill", style: `width:${(probs[i] * 100).toFixed(1)}%` }),
        el("span", { textContent: name(i) }), el("span", { className: "p", textContent: pct(probs[i]) }));
      ol.append(li);
    }
    return ol;
  }
  const topk = (p, k) => p.map((x, i) => [x, i]).sort((a, b) => b[0] - a[0]).slice(0, k).map(([, i]) => i);

  function showTokens(r) {
    $("tokens").replaceChildren(...r.tokens.map((t, i) => {
      const special = /^\\[.*\\]$|^<.*>$/.test(t), cont = t.startsWith("##");
      return el("span", { className: "chip" + (special ? " special" : "") + (cont ? " cont" : ""),
                          title: `token ${r.ids[i]}`, textContent: cont ? t.slice(2) : t });
    }));
    $("s-tok").querySelector(".ms").textContent = `${r.tokens.length} pieces, ${r.ms.tokenize.toFixed(1)} ms`;
  }

  function showLayers(r) {
    const table = $("layers");
    if (!r.layers) { table.replaceChildren(el("tr", {}, el("td", { textContent: "No layer outputs: this model was exported without them, or the browser kept an older copy (reload)." }))); return; }
    const head = el("tr", {}, el("td", { className: "n" }), el("td", { className: "note", textContent: "would answer" }),
      el("td", { className: "note", textContent: `chance of "${name(r.top)}"` }));
    table.replaceChildren(head, ...r.layers.map((lp, l) => {
      const best = argmax(lp);
      return el("tr", {},
        el("td", { className: "n", textContent: `layer ${l + 1}` }),
        el("td", { className: "l", textContent: name(best),
                   style: best === r.top ? "font-weight:600" : "color:var(--muted)" }),
        el("td", {}, el("div", { className: "barcell", title: `probability of "${name(r.top)}" after layer ${l + 1}` },
          el("div", { style: `width:${(lp[r.top] * 100).toFixed(1)}%` }),
          el("span", { textContent: pct(lp[r.top]) }))));
    }));
    $("s-layers").querySelector(".ms").textContent = `${r.ms.model.toFixed(1)} ms for the whole model`;
  }

  function showAttention(r) {
    const box = $("att-layers");
    if (!r.attention) { box.replaceChildren(); $("attention").textContent = "No attention outputs: this model was exported without them, or the browser kept an older copy (reload)."; return; }
    const L = r.attention.length;
    if (attLayer === null || attLayer >= L) attLayer = L - 1;
    box.replaceChildren(...r.attention.map((_, l) => {
      const b = el("button", { className: "ghost", textContent: `layer ${l + 1}` });
      b.setAttribute("aria-pressed", String(l === attLayer));
      b.onclick = () => { attLayer = l; showAttention(current); };
      return b;
    }));
    const a = r.attention[attLayer], m = Math.max(...a.slice(1)) || 1;  // scale without [CLS] itself
    $("attention").replaceChildren(...r.tokens.map((t, i) => {
      const w = Math.min(1, a[i] / m);
      return el("span", { className: "chip", title: `${pct(a[i])} of the attention`,
                          textContent: t.startsWith("##") ? t.slice(2) : t,
                          style: `background:rgba(var(--att),${(0.08 + 0.8 * w).toFixed(2)});` +
                                 (w > 0.55 ? "color:#fff;border-color:transparent" : "") });
    }));
  }

  function showOutput(r) {
    $("top").replaceChildren(...bars(topk(r.p, 5), r.p).children);
    const sorted = [...r.p].sort((a, b) => b - a), m = sorted[0] || 1;
    $("spectrum").replaceChildren(...sorted.map((x) =>
      el("div", { style: `height:${Math.max(1, Math.sqrt(x / m) * 100).toFixed(1)}%`, title: pct(x) })));
    $("s-out").querySelector(".ms").textContent = D.multi ? "sigmoid per label" : "softmax";
  }

  function showWords(r) {
    $("why-label").textContent = name(r.top);
    const m = Math.max(1e-9, ...r.effect.map(Math.abs));
    $("words").replaceChildren(...r.words.flatMap((w, i) => {
      const e = r.effect[i] ?? 0, a = Math.abs(e) / m;
      const s = el("span", { textContent: w, title: `removing it: ${e >= 0 ? "-" : "+"}${pct(Math.abs(e))}`,
                             style: `background:rgba(var(${e >= 0 ? "--pos" : "--neg"}),${(0.75 * a).toFixed(2)})` });
      return [s, " "];
    }));
    $("s-why").querySelector(".ms").textContent = r.words.length > 1
      ? `${r.words.length} more runs, ${r.ms.words.toFixed(0)} ms` : "needs two or more words";
  }

  function showTeacher(r) {
    const box = $("teacher"), sample = D.samples.find((s) => s.text === text.value);
    if (!sample || !sample.teacher) {
      box.replaceChildren(el("p", { className: "note", style: "margin:0",
        textContent: "Pick one of the examples to compare the student with the teacher's soft labels for it." }));
      return;
    }
    const tp = Array(D.labels.length).fill(0);
    for (const [i, x] of sample.teacher) tp[i] = x;
    const agree = argmax(tp) === r.top;
    box.replaceChildren(
      el("div", { className: "cols" },
        el("div", {}, el("h3", { className: "t", textContent: "Teacher (LLM)" }), bars(topk(tp, 5), tp, true, sample.gold)),
        el("div", {}, el("h3", { textContent: "Student (this page)" }), bars(topk(r.p, 5), r.p, false, sample.gold))),
      el("p", { className: "note", style: "margin:10px 0 0", textContent:
        (agree ? "They agree on the top label." : "They disagree on the top label.") +
        (sample.gold !== null ? ` The correct label is "${name(sample.gold)}".` : "") }));
  }

  function showCascade(r) {
    const box = $("cascade");
    if (!D.cascade || D.multi) { $("s-cascade").hidden = true; return; }
    const pts = D.cascade, slider = el("input", { type: "range", min: 0, max: pts.length - 1, step: 1,
      value: Math.min(2, pts.length - 1) });
    slider.setAttribute("aria-label", "Share of texts sent to the teacher");
    const line = el("p", { style: "margin:8px 0 0" }), mine = el("p", { className: "verdict", style: "margin:8px 0 0" });
    const update = () => {
      const c = pts[slider.value], conf = r.p[r.top];
      line.textContent = `Send the ${pct(c.share_to_teacher)} least confident texts to the teacher: every text below ` +
        `${pct(c.threshold)} confidence goes, and test accuracy is ${pct(c.accuracy)}.`;
      mine.textContent = conf < c.threshold && c.share_to_teacher > 0
        ? `This text (${pct(conf)} confident) would go to the teacher.`
        : `The student answers this text itself (${pct(conf)} confident).`;
    };
    slider.oninput = update; update();
    box.replaceChildren(slider, line, mine);
  }

  const steps = ["s-tok", "s-layers", "s-att", "s-out", "s-why", "s-teacher", "s-cascade"];
  const flow = [...document.querySelectorAll(".flow span")];
  const sleep = (ms) => new Promise((ok) => setTimeout(ok, ms));
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;

  async function run() {
    if (!text.value.trim()) return;
    const id = ++runId;
    go.disabled = true;
    for (const s of steps) $(s).classList.remove("done");
    for (const f of flow) f.classList.remove("on");
    const r = await analyse(text.value);
    if (id !== runId) return;
    current = r;
    const conf = r.p[r.top];
    $("answer").textContent = name(r.top);
    status.textContent = `${pct(conf)} confident, ${(r.ms.tokenize + r.ms.model).toFixed(0)} ms in your browser`;
    // replay the steps in order, so the page shows how the input flows through the model
    const show = [showTokens, showLayers, showAttention, showOutput, showWords, showTeacher, showCascade];
    for (let i = 0; i < steps.length; i++) {
      show[i](r);
      $(steps[i]).classList.add("done");
      if (flow[i]) flow[i].classList.add("on");
      if (i === 3) flow[4].classList.add("on");
      if (!reduced || slow) await sleep(slow ? 900 : 140);
      if (id !== runId) return;
    }
    go.disabled = false;
  }

  let slow = false;
  $("slow").onclick = () => { slow = !slow; $("slow").setAttribute("aria-pressed", String(slow)); if (slow) run(); };
  go.onclick = run;
  text.addEventListener("keydown", (e) => { if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) run(); });
  for (const b of document.querySelectorAll("button.sample")) b.onclick = () => { text.value = b.dataset.text; run(); };
  go.disabled = false;
  run();
</script>
</body>
</html>
"""


def _samples(rows: list[dict], n: int) -> list[dict]:
    """Up to n test rows with different gold labels, taken from across the file (test files are often sorted
    by label, so the first rows would all show the same one)."""
    picked, labels = [], set()
    step = max(1, len(rows) // (4 * n or 1))
    for r in rows[::step] + rows:
        key = json.dumps(r.get("gold"))
        if key not in labels and len(picked) < n:
            labels.add(key)
            picked.append(r)
    return picked


def _teacher_probs(scores: list[float], multi: bool, k: int = 10) -> list[list]:
    """The teacher's top-k label probabilities as [index, probability] pairs (scores are log-probabilities,
    or log-odds for multi-label)."""
    if multi:
        p = [1 / (1 + math.exp(-s)) for s in scores]
    else:
        m = max(scores)
        e = [math.exp(s - m) for s in scores]
        p = [x / sum(e) for x in e]
    return [[i, round(p[i], 4)] for i in sorted(range(len(p)), key=lambda i: -p[i])[:k]]


def _summary(res: dict | None, deployed: str) -> str:
    """One sentence on quality, from the run's results.json (never typed by hand)."""
    if not res:
        return ""
    row = next((m for m in res["models"] if m["role"] == "student" and deployed in m["name"]), None)
    teacher = next((m for m in res["models"] if m["role"] == "teacher"), None)
    if not row or row.get("accuracy") is None:
        return ""
    return (
        f"Test accuracy {row['accuracy']:.3f} on {res['data']['n_test']} held-out texts. "
        f"Distilled from {res.get('teacher')}"
        + (f" ({teacher['name'].removeprefix('teacher, ')})" if teacher else "")
        + f" into {res.get('student')}."
    )


def build_site(
    onnx_dir: Path, out: Path, title: str | None = None, samples: int = 6, note: str | None = None
) -> Path:
    """Write ``out/index.html`` and ``out/model-<hash>/`` (config, tokenizer, ONNX) for transformers.js. When the run's
    PyTorch student is next to the export, the page's model also returns each layer's answer and the
    attention of the first token, for the "Inside the model" view. ``note`` is an extra line for the footer,
    such as the dataset's licence."""
    from .export import export_explain
    from .static import is_static

    if not (onnx_dir / "config.json").exists():
        raise SystemExit(f"{onnx_dir} has no config.json: export it again with `tinydistill export` (0.3.0+)")
    if is_static(onnx_dir):
        raise SystemExit("the browser page needs a transformer student; static students run with `serve`")
    int8 = (onnx_dir / "model.int8.onnx").exists()
    run = onnx_dir.parent
    with tempfile.TemporaryDirectory() as tmp:
        onnx_file = (
            export_explain(run / "student", Path(tmp), int8)
            if (run / "student" / "config.json").exists()
            else None
        )
        onnx_file = onnx_file or onnx_dir / ("model.int8.onnx" if int8 else "model.onnx")
        # Browsers cache model files by URL, so the folder is named after the model's contents: a new model
        # gets a new URL and no visitor keeps an old copy.
        digest = hashlib.sha256(onnx_file.read_bytes()).hexdigest()[:10]
        model_dir = out / f"model-{digest}"
        for old in out.glob("model*"):  # this function's earlier output
            if old.is_dir() and re.fullmatch(r"model(-[0-9a-f]{10})?", old.name):
                shutil.rmtree(old)
        (model_dir / "onnx").mkdir(parents=True)
        target = model_dir / "onnx" / ("model_quantized.onnx" if int8 else "model.onnx")
        shutil.copy(onnx_file, target)
    for f in (
        "config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "vocab.txt",
    ):
        if (onnx_dir / f).exists():
            shutil.copy(onnx_dir / f, model_dir / f)

    config = json.loads((onnx_dir / "config.json").read_text(encoding="utf-8"))
    labels = json.loads((onnx_dir / "labels.json").read_text(encoding="utf-8"))
    multi = config.get("problem_type") == "multi_label_classification"
    meta = json.loads((run / "meta.json").read_text(encoding="utf-8")) if (run / "meta.json").exists() else {}
    res = (
        json.loads((run / "results.json").read_text(encoding="utf-8"))
        if (run / "results.json").exists()
        else None
    )
    rows = []
    if (run / "test.jsonl").exists():
        with (run / "test.jsonl").open(encoding="utf-8") as f:
            rows = _samples([json.loads(line) for line in f if line.strip()], samples)
    data = {
        "labels": labels,
        "multi": multi,
        "dtype": "q8" if int8 else "fp32",
        "model": model_dir.name,
        "samples": [
            {
                "text": r["text"],
                "gold": r.get("gold") if not multi else None,
                "teacher": _teacher_probs(r["scores"], multi) if r.get("scores") else None,
            }
            for r in rows
        ],
        "cascade": (res or {}).get("cascade"),
    }
    vocab = json.loads((onnx_dir / "tokenizer.json").read_text(encoding="utf-8"))["model"].get("vocab", {})
    teacher_row = next((m for m in (res or {}).get("models", []) if m["role"] == "teacher"), None)
    teacher = f"the teacher, {meta.get('teacher') or 'a large language model'}" + (
        f" ({teacher_row['name'].removeprefix('teacher, ')})" if teacher_row else ""
    )
    name = title or (meta.get("task") or run.name).replace("-", " ").replace("_", " ").title()
    buttons = "\n    ".join(
        f'<button class="ghost sample" data-text="{html.escape(r["text"])}">Example {i + 1}</button>'
        for i, r in enumerate(rows)
    )
    page = (
        PAGE.replace("__TITLE__", html.escape(name))
        .replace("__SIZE__", f"{target.stat().st_size / 1e6:.0f}")
        .replace("__FIRST__", html.escape(rows[0]["text"]) if rows else "")
        .replace("__SAMPLES__", buttons)
        .replace("__VOCAB__", f"{len(vocab):,} pieces" if vocab else "pieces")
        .replace("__LAYERS__", str(config.get("num_hidden_layers", "several")))
        .replace("__NLABELS__", str(len(labels)))
        .replace("__ACTIVATION__", "sigmoid per label" if multi else "softmax")
        .replace("__TEACHER__", html.escape(teacher))
        .replace("__TRANSFORMERS_JS__", TRANSFORMERS_JS)
        .replace("__SUMMARY__", html.escape(_summary(res, "int8" if int8 else "fp32")))
        .replace("__NOTE__", html.escape(note or ""))
        .replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
    )
    (out / "index.html").write_text(page, encoding="utf-8")
    return out / "index.html"
