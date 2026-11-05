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
KATEX = "https://cdn.jsdelivr.net/npm/katex@0.16.22"  # loaded only when a visitor turns on "Show the math"
BUNDLE = 200  # test texts in the page, for "Random test text" and "Check it yourself"
BUNDLE_CHARS = 150_000  # ...as long as their texts stay under this many characters in all

PAGE = r"""<!doctype html>
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
  button.sample { max-width: 100%; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  button:disabled { opacity: .5; cursor: progress; }
  :focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  #status { color: var(--muted); font-size: .92rem; }
  .card { margin: 18px 0 22px; padding: 10px 12px; border: 1px solid var(--line); border-radius: 10px;
          background: var(--card); }
  .card table { font-size: .95rem; }
  .card th { text-align: left; font-weight: 650; padding: 4px 6px; }
  .card th.t, .card td.t { color: var(--teacher); }
  .card td { overflow-wrap: anywhere; }
  .card td:first-child { color: var(--muted); width: 28%; }
  .card .note { margin: 6px 6px 0; font-size: .85rem; }
  .answer { margin: 24px 0 0; padding: 16px; border: 1px solid var(--line); border-radius: 10px; background: var(--card); }
  .answer .big { font-size: 1.45rem; font-weight: 650; }
  .answer .meta { color: var(--muted); font-variant-numeric: tabular-nums; }
  .answer .bars { margin-top: 10px; }
  #unsure { margin: 8px 0 0; color: var(--teacher); }
  progress { width: 100%; accent-color: var(--accent); }
  ul.verdict { list-style: none; padding: 0; margin: 12px 0 0; }
  ul.verdict li { margin-bottom: 2px; }
  .flow { display: flex; gap: 6px; flex-wrap: wrap; margin: 14px 0 0; font-size: .85rem; }
  .flow span { padding: 3px 9px; border-radius: 999px; border: 1px solid var(--line); color: var(--muted);
               transition: background .25s, color .25s, border-color .25s; }
  .flow span.on { background: var(--accent); border-color: var(--accent); color: #fff; }
  details.fold { margin: 32px 0 0; }
  details.fold > summary { font-size: 1.15rem; font-weight: 650; cursor: pointer; }
  input[type=search] { width: 100%; padding: 7px 10px; margin: 10px 0 8px; font: inherit; color: inherit;
                       background: var(--card); border: 1px solid var(--line); border-radius: 8px; }
  ul.labels { columns: 2 220px; margin: 0; padding-left: 20px; font-size: .93rem; }
  .math { margin-top: 10px; overflow-x: auto; }
  .math .katex-display { margin: 6px 0; overflow-x: auto; overflow-y: hidden; }
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
  .bars li { display: flex; align-items: baseline; gap: 10px; }  /* the percentage never sits on a long label */
  .bars li span { position: relative; min-width: 0; overflow-wrap: anywhere; }
  .bars li .p { flex: none; margin-left: auto; color: var(--muted); font-variant-numeric: tabular-nums; }
  .bars li.gold span:first-of-type::after { content: "  gold"; color: var(--teacher); font-size: .78rem; font-weight: 650; }
  .spectrum { display: flex; align-items: flex-end; gap: 1px; height: 64px; margin-top: 12px; }
  .spectrum div { flex: 1; background: var(--accent); min-height: 1px; border-radius: 1px 1px 0 0; }
  .words { line-height: 2.1; font-size: 1.02rem; }
  .words span, .words mark { padding: 2px 3px; border-radius: 4px; color: inherit; }
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
  nav.pages { display: inline-flex; gap: 2px; padding: 3px; margin: 0 0 20px; border: 1px solid var(--line);
              border-radius: 999px; background: var(--card); font-size: .92rem; }
  nav.pages a { padding: 3px 14px; border-radius: 999px; color: var(--muted); text-decoration: none; }
  nav.pages a:hover { color: var(--fg); background: var(--soft); }
  nav.pages a[aria-current="page"] { background: var(--accent); color: #fff; font-weight: 600; }
  .deeper { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 16px; margin: 14px 0 0; padding: 14px 16px;
            border: 1px solid var(--accent); border-radius: 10px; background: var(--card); }
  .deeper p { flex: 1 1 260px; margin: 0; color: var(--muted); font-size: .92rem; }
  .deeper p b { display: block; color: var(--fg); font-size: 1.02rem; }
  .deeper a { white-space: nowrap; }
  .deeper a.go { padding: 7px 14px; border-radius: 8px; background: var(--accent); color: #fff; text-decoration: none;
                 font-weight: 600; }
  .deeper a.go:hover { filter: brightness(1.1); }
  @media (prefers-reduced-motion: reduce) { * { transition: none !important; scroll-behavior: auto !important; } }
  .top { display: flex; flex-wrap: wrap; align-items: flex-start; gap: 8px 12px; margin: 0 0 20px; }
  .top nav.pages { margin: 0; }
  __PICKER_CSS__
</style>
</head>
<body>
<main>
  <div class="top">__TABS__ __PICKER__</div>
  <p class="note" id="no-inside" hidden>This model has no "Inside the model" view, so it opens here. That tab
    shows the page's own model.</p>
  <h1 data-t="title">__TITLE__</h1>
  <p class="sub">A <span data-t="size">__SIZE__</span> MB model distilled from a large language model with
    tinydistill. It runs in your browser, and nothing you type leaves this page.
    <span data-t="summary">__SUMMARY__</span></p>
  __CARD__
  <label for="text">Text</label>
  <textarea id="text" placeholder="Type a message">__FIRST__</textarea>
  <p class="note" style="margin:4px 0 0">The answer updates as you type.</p>
  <div class="row" id="samples">
    __SAMPLES__
  </div>
  <div class="row">
    <button class="ghost" id="random">Random test text</button>
    <button class="ghost" id="replay">Show the steps</button>
  </div>
  <div class="answer">
    <div id="loading"><p id="load-text" style="margin:0 0 6px">Loading the model...</p>
      <progress id="progress" aria-label="Model download"></progress></div>
    <div class="big" id="answer" aria-live="polite"></div>
    <div class="meta" id="status"></div>
    <p id="unsure" hidden></p>
    <ol class="bars" id="top3"></ol>
    <p class="note" id="reason-note" style="margin:12px 0 2px"></p>
    <div class="words" id="reason"></div>
    <div id="verdict"></div>
    <div class="flow" aria-hidden="true">
      <span>text</span><span>tokens</span><span>layers</span><span>classifier</span><span>probabilities</span>
    </div>
  </div>
  __INSIDE__

  <section id="check-it">
    <h2>Check it yourself</h2>
    <p class="note" id="check-note"></p>
    <button id="check">Run them</button>
    <p id="check-out" aria-live="polite" style="margin-top:10px"></p>
  </section>

  <details class="fold" id="all-labels"><summary>All <span data-t="nlabels">__NLABELS__</span> labels</summary>
    <input type="search" id="label-q" placeholder="Search the labels" aria-label="Search the labels">
    <ul class="labels" id="label-list"></ul></details>

  <details class="fold" id="inside"><summary>How does it work?</summary>
  <p class="note" style="margin-top:8px">Every step below is computed by the model in this page as it classifies
    your text. "Show the steps" plays them in order; slow motion plays them slowly every time.</p>
  <div class="row" style="margin-top:0">
    <button class="ghost" id="slow" aria-pressed="false" title="Step through the model one stage at a time">Slow motion</button>
    <button class="ghost" id="math" aria-pressed="false">Show the math</button>
  </div>
  <ol class="steps" id="steps">
    <li id="s-tok"><div class="step-head"><h3>The tokenizer splits the text</h3><span class="ms"></span></div>
      <p class="note">The model never sees letters or words, only pieces from a fixed vocabulary of <span data-t="vocab">__VOCAB__</span>.
        Rare words are split into several pieces (marked with a blue edge); hover a piece for its number.
        <i>[CLS]</i> is added at the start: the classifier reads the model's output at that position.</p>
      <div class="box"><div class="chips" id="tokens"></div></div>
      <div class="math" id="m-tok" hidden></div></li>
    <li id="s-layers"><div class="step-head"><h3>Layer by layer</h3><span class="ms"></span></div>
      <p class="note">The pieces pass through <span data-t="layers">__LAYERS__</span> transformer layers. Each layer lets every piece look at the
        others and update itself. If we stop after a layer and apply the final classifier, we see what the model
        would answer at that depth: early layers usually do not know yet, and the answer forms on the way up.</p>
      <div class="box"><table id="layers"></table></div></li>
    <li id="s-att"><div class="step-head"><h3>Where the classifier looks</h3><span class="ms"></span></div>
      <p class="note">Attention: how strongly the <i>[CLS]</i> position draws on each piece, averaged over the
        layer's attention heads. Darker means more. Pick a layer. <i>[CLS]</i> and <i>[SEP]</i> often collect
        much of the attention as a resting place, so compare the word pieces with each other.</p>
      <div class="row" id="att-layers" style="margin-top:0"></div>
      <div class="box"><div class="chips" id="attention"></div></div>
      <div class="math" id="m-att" hidden></div></li>
    <li id="s-out"><div class="step-head"><h3>From scores to probabilities</h3><span class="ms"></span></div>
      <p class="note">The classifier gives one score per label (<span data-t="nlabels">__NLABELS__</span> labels) and a
        <span data-t="activation">__ACTIVATION__</span> turns them
        into probabilities. The top five are below; the strip shows all of them, largest first. The small
        probabilities on other labels are what distillation passes on: the teacher's soft labels tell the
        student which labels are similar, not just which one is right.</p>
      <div class="box"><ol class="bars" id="top"></ol><div class="spectrum" id="spectrum"
        role="img" aria-label="Probabilities of all labels, largest first"></div></div>
      <div class="math" id="m-out" hidden></div></li>
    <li id="s-why"><div class="step-head"><h3>Which words matter</h3><span class="ms"></span></div>
      <p class="note">Each word is removed in turn and the model runs again on what is left. Words whose removal
        lowers the probability of <b id="why-label"></b> support it; words whose removal raises it count against
        it. Hover a word for the change.</p>
      <div class="box"><div class="words" id="words"></div>
        <div class="legend"><span><i style="background:rgba(var(--pos),.7)"></i>supports the answer</span>
          <span><i style="background:rgba(var(--neg),.7)"></i>counts against it</span></div></div></li>
    <li id="s-teacher"><div class="step-head"><h3>Teacher and student</h3></div>
      <p class="note">The student learned to imitate the soft labels of <span data-t="teacher">__TEACHER__</span>. The teacher is far too large to
        run here, but its saved soft labels for the test texts are in this page.</p>
      <div class="box" id="teacher"></div>
      <div class="math" id="m-teacher" hidden></div></li>
    <li id="s-cascade"><div class="step-head"><h3>When to ask the teacher</h3></div>
      <p class="note">In a cascade the student answers and passes the texts it is least sure about to the teacher.
        The slider sets how many; accuracy and the confidence threshold are measured on the held-out test set.</p>
      <div class="box" id="cascade"></div></li>
  </ol>
  </details>
  <footer>Made with <a href="https://github.com/Andreas-Hum/tinydistill">tinydistill</a> and
    <a href="https://github.com/huggingface/transformers.js">transformers.js</a>. __NOTE__</footer>
</main>
<script type="application/json" id="data">__DATA__</script>
<script type="module">
  import { AutoTokenizer, AutoModel, env } from "__TRANSFORMERS_JS__";
  env.localModelPath = "./";  // the model files sit next to this page, unless it loads them from the Hub
  const KATEX = "__KATEX__";
  const LOW = 0.5;  // under this top probability the page says the text may fit none of the labels
  let D = JSON.parse(document.getElementById("data").textContent);
  const $ = (id) => document.getElementById(id);
  const el = (tag, props = {}, ...kids) => {
    const e = Object.assign(document.createElement(tag), props);
    e.append(...kids);
    return e;
  };
__PICKER_JS__
  const name = (i) => D.labels[i].replaceAll("_", " ");
  // an answer in words; the task's `outside` label (D.outside) means the text fits none of the others
  const said = (i) => (i === D.outside ? "None of the labels fits" : name(i));
  const names = (list) => (list.length ? list.map(said).join(", ") : "none of the labels");
  const pct = (x) => (100 * x).toFixed(x >= 0.995 || x < 0.001 ? 0 : 1) + "%";
  const sleep = (ms) => new Promise((ok) => setTimeout(ok, ms));
  const reduced = matchMedia("(prefers-reduced-motion: reduce)").matches;

  // ---- what works before the model is loaded ----
  // ?text= opens the page on a text; the links to the explainer carry the current one
  const asked = new URLSearchParams(location.search).get("text");
  if (asked?.trim()) $("text").value = asked;
  const withText = () => {  // the links to the explainer carry the text, and the model when it has that page
    for (const a of document.querySelectorAll("a[data-base]")) {
      const u = new URL(a.dataset.base, location.href), q = $("text").value.trim();
      if (q) u.searchParams.set("text", q);
      if (pickNow.explainer) withModel(u);
      a.href = u.href;
    }
  };
  $("text").addEventListener("input", withText);
  const labelList = $("label-list");
  $("label-q").oninput = (e) => {
    const q = e.target.value.trim().toLowerCase();
    for (const li of labelList.children) li.hidden = !li.textContent.toLowerCase().includes(q);
  };
  const short = (s) => (s.length <= 40 ? s : s.slice(0, 37).trimEnd() + "...");
  let N = 0;
  function fill() {  // everything that comes from the model's data: at the start, and when another model is picked
    document.title = D.text.title;
    for (const e of document.querySelectorAll("[data-t]")) e.textContent = D.text[e.dataset.t];
    showCard();
    $("samples").replaceChildren(...D.samples.slice(0, D.buttons).map((s) => el("button", { className: "ghost sample",
      textContent: short(s.text), title: s.text, onclick: () => { text.value = s.text; classify(); } })));
    labelList.replaceChildren(...D.labels.map((_, i) => el("li", { textContent: name(i) })));
    N = D.samples.length;
    $("check-it").hidden = $("random").hidden = !N;
    $("check").textContent = `Run the ${N} test texts`;
    $("check-out").textContent = "";
    $("check-note").textContent = `This page holds ${N} texts from the test set` +
      (D.n_test ? ` (of ${D.n_test})` : "") + ", picked to cover every label, so they are a sample, not the full " +
      "test set. The button runs each one through the model in your browser and compares its answers with the " +
      "correct labels and with the teacher's.";
    $("unsure").textContent = `The model is not sure: no label reaches ${pct(LOW)}, so this text may fit none ` +
      `of the ${D.labels.length} labels.`;
    for (const e of document.querySelectorAll(".deeper")) e.style.display = pickNow.explainer ? "" : "none";
    $("no-inside").hidden = pickNow.explainer || !document.querySelector("nav.pages");
    withText();
  }
  function showCard() {  // teacher against student, from the data's rows (the same table as _card in web.py)
    const td = (t, className = "") => el("td", { className, textContent: t });
    $("card").hidden = !D.card.length;
    $("card-rows").replaceChildren(el("tr", {}, el("th"), el("th", { className: "t", textContent: "Teacher (LLM)" }),
      el("th", { textContent: "Student (this page)" })), ...D.card.map(([h, t, s]) => el("tr", {}, td(h), td(t, "t"),
      s === null ? el("td", {}, el("span", { id: "live-ms", textContent: "measured as you type" })) : td(s))));
  }
  if (pickNow !== MODELS[0]) {  // ?model= asks for one of the other models
    try {
      D = await pageData(pickNow, "simple");
      if (!asked?.trim() && D.samples.length) $("text").value = D.samples[0].text;
    } catch (e) { fallBack(e); }
  }
  fill();

  // ---- core: the model and what it computes (no page elements here) ----
  const progress = (a, b) => {  // download progress of every file, summed
    $("progress").max = b; $("progress").value = a;
    $("load-text").textContent = `Downloading the model: ${(a / 1e6).toFixed(1)} of ${(b / 1e6).toFixed(1)} MB`;
  };
  let tok, model;
  try {
    [tok, model] = await loadModel(D, progress);
  } catch (e) {  // the model ?model= asked for did not load: this page's own instead
    if (D === D0) { $("load-text").textContent = `The model did not load: ${e.message}`; throw e; }
    fallBack(e); D = D0; fill();
    [tok, model] = await loadModel(D, progress);
  }
  $("loading").hidden = true;
  picker(MODELS, (m) => switchTo(m, "simple", async (d, t, mod) => {
    D = d; tok = t; model = mod; times.length = 0; attLayer = null;
    fill();
    await classify();
  }));
  let queue = Promise.resolve();
  const call = (enc) => {  // one model call at a time; resolves to [outputs, milliseconds]
    const p = queue.then(async () => { const t = performance.now(), o = await model(enc); return [o, performance.now() - t]; });
    queue = p.catch(() => {});
    return p;
  };
  const softmax = (z) => { const m = Math.max(...z), e = z.map((x) => Math.exp(x - m)), s = e.reduce((a, b) => a + b);
                           return e.map((x) => x / s); };
  const toProbs = (z) => D.multi ? z.map((x) => 1 / (1 + Math.exp(-x))) : softmax(z);
  const rowsOf = (t) => { const [n, ...rest] = t.dims, k = rest.reduce((a, b) => a * b, 1);
                          return Array.from({ length: n }, (_, i) => Array.from(t.data.slice(i * k, (i + 1) * k))); };
  const split = (row, k) => Array.from({ length: row.length / k }, (_, i) => row.slice(i * k, (i + 1) * k));
  const argmax = (xs) => xs.reduce((b, x, i) => (x > xs[b] ? i : b), 0);
  const encode = (texts) => tok(texts, { padding: true, truncation: true, max_length: D.max_length });
  // the answer as a list of labels: the top one, or (multi-label) every label over 50%
  const chosen = (p) => D.multi ? p.flatMap((x, i) => (x > 0.5 ? [i] : [])) : [argmax(p)];
  const times = [];  // milliseconds per text in this browser

  async function analyse(text) {
    const t0 = performance.now();
    const enc = encode([text]);
    const ids = Array.from(enc.input_ids.data, Number);
    const tokens = tok.model.convert_ids_to_tokens(ids);
    const t1 = performance.now();
    const [out, ms] = await call(enc);
    const z = rowsOf(out.logits)[0], p = toProbs(z), top = argmax(p);
    const layers = out.layer_logits ? split(rowsOf(out.layer_logits)[0], D.labels.length).map(toProbs) : null;
    const attention = out.cls_attention ? split(rowsOf(out.cls_attention)[0], ids.length) : null;
    times.push(t1 - t0 + ms);
    return { text, tokens, ids, z, p, top, pred: chosen(p), layers, attention, words: null, effect: null,
             ms: { tokenize: t1 - t0, model: ms } };
  }

  async function effects(r) {  // remove each word in turn: one batch of texts
    const words = r.text.trim().split(/\s+/).filter(Boolean).slice(0, 60);
    const t = performance.now();
    let effect = [];
    if (words.length > 1) {
      const [o] = await call(encode(words.map((_, i) => words.filter((_, j) => j !== i).join(" "))));
      effect = rowsOf(o.logits).map((z) => r.p[r.top] - toProbs(z)[r.top]);
    }
    Object.assign(r, { words, effect });
    r.ms.words = performance.now() - t;
  }
  // ---- end of core ----

  const sampleOf = (t) => D.samples.find((s) => s.text === t);
  const dense = (pairs) => { const p = Array(D.labels.length).fill(0); for (const [i, x] of pairs) p[i] = x; return p; };
  const goldOf = (s) => (s && s.gold !== null && s.gold !== undefined ? [s.gold].flat() : null);
  const same = (a, b) => a.length === b.length && a.every((x) => b.includes(x));
  const median = (xs) => { const s = [...xs].sort((a, b) => a - b); return s[Math.floor(s.length / 2)]; };
  const text = $("text");
  let current = null, attLayer = null, runId = 0, slow = false, math = false, katex = null;

  function bars(list, probs, teacher = false, gold = []) {
    const ol = el("ol", { className: "bars" + (teacher ? " teacher" : "") });
    for (const [n, i] of list.entries()) {
      if (n && probs[i] < 0.0005 && !gold.includes(i)) continue;  // labels at 0% would only list in label order
      const li = el("li", { className: gold.includes(i) ? "gold" : "" },
        el("div", { className: "fill", style: `width:${(probs[i] * 100).toFixed(1)}%` }),
        el("span", { textContent: name(i) }), el("span", { className: "p", textContent: pct(probs[i]) }));
      ol.append(li);
    }
    return ol;
  }
  const topk = (p, k) => p.map((x, i) => [x, i]).sort((a, b) => b[0] - a[0]).slice(0, k).map(([, i]) => i);

  function showSpeed() {
    const live = $("live-ms");
    if (live && times.length) live.textContent = `${median(times).toFixed(1)} ms in this browser`;
  }

  function showAnswer(r) {
    $("answer").textContent = names(r.pred);
    $("status").textContent = `${pct(r.p[r.top])} confident, ${(r.ms.tokenize + r.ms.model).toFixed(0)} ms in your browser`;
    $("unsure").hidden = r.p[r.top] >= LOW;
    $("top3").replaceChildren(...bars(topk(r.p, 3), r.p).children);
    showSpeed();
    showReason(r);
    showVerdict(r);
  }

  function showReason(r) {  // the text again, with the words that support the answer highlighted
    const note = $("reason-note"), box = $("reason");
    if (!r.effect) { note.textContent = "Finding the words that support the answer..."; box.textContent = r.text; return; }
    if (r.words.length < 2) { note.textContent = "Type two or more words to see which ones support the answer."; box.textContent = r.text; return; }
    note.textContent = `Why: the highlighted words support "${name(r.top)}" (darker means more).`;
    const m = Math.max(1e-9, ...r.effect);
    box.replaceChildren(...r.words.flatMap((w, i) => {
      const e = r.effect[i] ?? 0;
      return [e > 0 ? el("mark", { textContent: w, title: `removing it: -${pct(e)}`,
                                   style: `background:rgba(var(--pos),${(0.15 + 0.6 * e / m).toFixed(2)})` }) : w, " "];
    }));
  }

  function showVerdict(r) {  // for a test text: the student, the teacher and the gold label
    const s = sampleOf(r.text), gold = goldOf(s), box = $("verdict");
    if (!gold) { box.replaceChildren(); return; }
    const mark = (pred) => (same(pred, gold) ? "correct" : "wrong");
    const item = (who, what) => el("li", {}, el("b", { textContent: who + ": " }), what);
    box.replaceChildren(el("ul", { className: "verdict" },
      item("Student (this page)", `${names(r.pred)}, ${mark(r.pred)}`),
      s.teacher ? item("Teacher", `${names(chosen(dense(s.teacher)))}, ${mark(chosen(dense(s.teacher)))}`) : "",
      item("Gold label", names(gold))));
  }

  function showTokens(r) {
    $("tokens").replaceChildren(...r.tokens.map((t, i) => {
      const special = /^\[.*\]$|^<.*>$/.test(t), cont = t.startsWith("##");
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
      b.onclick = () => { attLayer = l; showAttention(current); showMath(current); };
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
    if (!r.effect) { $("words").textContent = "Computed once you stop typing."; $("s-why").querySelector(".ms").textContent = ""; return; }
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
    const box = $("teacher"), sample = sampleOf(r.text);
    if (!sample || !sample.teacher) {
      box.replaceChildren(el("p", { className: "note", style: "margin:0",
        textContent: "Pick an example or a random test text to compare the student with the teacher's soft labels for it." }));
      return;
    }
    const tp = dense(sample.teacher), gold = goldOf(sample) ?? [];
    const agree = same(chosen(tp), r.pred);
    box.replaceChildren(
      el("div", { className: "cols" },
        el("div", {}, el("h3", { className: "t", textContent: "Teacher (LLM)" }), bars(topk(tp, 5), tp, true, gold)),
        el("div", {}, el("h3", { textContent: "Student (this page)" }), bars(topk(r.p, 5), r.p, false, gold))),
      el("p", { className: "note", style: "margin:10px 0 0", textContent:
        (agree ? "They agree on the answer." : "They disagree on the answer.") +
        (gold.length ? ` The correct answer is "${names(gold)}".` : "") }));
  }

  function showCascade(r) {
    const box = $("cascade");
    $("s-cascade").hidden = !D.cascade || D.multi;
    if ($("s-cascade").hidden) return;
    const pts = D.cascade, first = pts[0], last = pts[pts.length - 1];
    const slider = el("input", { type: "range", min: 0, max: pts.length - 1, step: 1, value: Math.min(2, pts.length - 1) });
    slider.setAttribute("aria-label", "Share of texts sent to the teacher");
    // the first point is the student alone, the last the teacher alone
    const alone = first.share_to_teacher === 0 && last.share_to_teacher === 1 && first.accuracy >= last.accuracy
      ? el("p", { style: "margin:0 0 8px", textContent: `Here the student alone (${pct(first.accuracy)} accurate) ` +
          `is at least as accurate as the teacher alone (${pct(last.accuracy)}), so sending texts to the teacher does not help.` })
      : "";
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
    box.replaceChildren(alone, slider, line, mine);
  }

  // ---- "Show the math": each step's formula with this text's numbers (KaTeX, loaded on first use) ----
  const ESC = { "\\": "\\textbackslash{}", "^": "\\textasciicircum{}", "~": "\\textasciitilde{}" };
  const T = (s) => "\\text{" + s.replace(/[\\{}$&#^_%~]/g, (c) => ESC[c] ?? "\\" + c) + "}";
  const f = (x, d = 3) => (x === 0 || (Math.abs(x) >= 1e-3 && Math.abs(x) < 1e6) ? x.toFixed(d) : x.toExponential(2))
    .replace(/e\+?(-?\d+)$/, "\\times 10^{$1}");
  function put(id, items) {  // TeX strings become formulas; {note} becomes a line of text
    $(id).replaceChildren(...items.map((x) => {
      if (typeof x !== "string") return el("p", { className: "note", textContent: x.note });
      const d = el("div");
      katex.render(x, d, { displayMode: true, throwOnError: false });
      return d;
    }));
  }

  function showMath(r) {
    if (!math || !katex || !r) return;
    const C = D.config, d = C.hidden_size, H = C.num_attention_heads, K = D.labels.length, n = r.ids.length;
    const type = C.model_type ?? "";
    // embeddings: RoBERTa-style models count positions from the padding id + 1; ModernBERT has none (rotary)
    const off = /roberta|camembert|mpnet/.test(type) ? (C.pad_token_id ?? 1) + 1 : 0;
    const pos = type !== "modernbert", seg = (C.type_vocab_size ?? 0) > 0;
    const emb = (id, i) => `E_{${id}}` + (pos ? ` + P_{${i}}` : "") + (seg ? " + T_{0}" : "");
    const first = r.ids.slice(0, 4).map((id, i) => String.raw`x_{${i}} &= \mathrm{LayerNorm}\big(${emb(id, i + off)}\big) && ${T(r.tokens[i])}`);
    put("m-tok", [
      String.raw`\mathrm{ids} = (${r.ids.slice(0, 16).join(",\\ ")}${n > 16 ? String.raw`,\ \dots` : ""})`,
      String.raw`x_i = \mathrm{LayerNorm}\big(${emb(String.raw`\mathrm{id}_i`, off ? `i+${off}` : "i")}\big),\quad E \in \mathbb{R}^{${C.vocab_size ?? "V"} \times ${d ?? "d"}}`,
      String.raw`\begin{aligned}${first.join(String.raw`\\`)}${n > 4 ? String.raw`\\ &\ \vdots` : ""}\end{aligned}`,
      { note: "E is the embedding table, one row per vocabulary piece: each id picks its row" +
        (pos ? "; P adds the position" : "; the position enters inside attention (rotary embeddings)") +
        (seg ? " and T the segment" : "") + "." },
    ]);

    const dk = d && H ? d / H : null;
    const att = [String.raw`\mathrm{head}_h(x_0) = \operatorname{softmax}\!\Big(\frac{q^{h}_{0}\,(K^{h})^{\top}}{\sqrt{d_k}}${type === "mpnet" ? " + b^{h}_{0}" : ""}\Big)\,V^{h},\quad h = 1,\dots,${H ?? "H"},\quad d_k = \frac{d}{H}` + (dk ? ` = \\frac{${d}}{${H}} = ${dk}` : "")];
    if (r.attention) {
      const a = r.attention[attLayer ?? r.attention.length - 1];
      att.push(String.raw`\bar{\alpha}^{(${(attLayer ?? r.attention.length - 1) + 1})}_{0} = \frac{1}{${H ?? "H"}}\sum_{h} \operatorname{softmax}(\cdot) = \big(` +
        a.slice(0, 10).map((x, i) => String.raw`\underset{${T(r.tokens[i].replace(/^##/, ""))}}{${f(x)}}`).join(",\\ ") +
        (a.length > 10 ? String.raw`,\ \dots` : "") + String.raw`\big)`);
      att.push({ note: "q₀ is the [CLS] row of Q = xW_Q; K = xW_K and V = xW_V. The weights shown are the heads' softmax rows for [CLS], averaged, in the layer picked above." });
    }
    put("m-att", att);

    const tops = topk(r.p, 3), tanh = /^(bert|roberta|xlm-roberta|camembert|mpnet)$/.test(type);
    const out = [
      String.raw`z = W\,${tanh ? String.raw`\tanh\big(W_p\, h_{0} + b_p\big)` : "g(h)"} + b,\quad W \in \mathbb{R}^{${K} \times ${d ?? "d"}}`,
      String.raw`\begin{aligned}` + tops.map((i) => String.raw`z_{${T(name(i))}} &= ${f(r.z[i])}`).join(String.raw`\\`) + String.raw`\end{aligned}`,
      { note: tanh ? "h₀ is the last layer's output at [CLS]; W_p and b_p are the pooling layer, W and b the classifier."
                   : "g is the model's pooling head; W and b are the classifier." },
    ];
    if (D.multi) {
      out.push(String.raw`p_i = \sigma(z_i) = \frac{1}{1 + e^{-z_i}}`,
        String.raw`\begin{aligned}` + tops.map((i) => String.raw`p_{${T(name(i))}} &= \frac{1}{1 + e^{-(${f(r.z[i])})}} = ${f(r.p[i])}`).join(String.raw`\\`) + String.raw`\end{aligned}`,
        { note: "Each label gets its own probability; every label over 0.5 is part of the answer." });
    } else {
      const S = r.z.reduce((a, x) => a + Math.exp(x), 0);
      out.push(String.raw`p_i = \frac{e^{z_i}}{\sum_{j=1}^{${K}} e^{z_j}},\qquad \sum_{j=1}^{${K}} e^{z_j} = ${f(S)}`,
        String.raw`\begin{aligned}` + tops.map((i) => String.raw`p_{${T(name(i))}} &= \frac{e^{${f(r.z[i])}}}{${f(S)}} = \frac{${f(Math.exp(r.z[i]))}}{${f(S)}} = ${f(r.p[i])}`).join(String.raw`\\`) + String.raw`\end{aligned}`);
    }
    put("m-out", out);

    const lam = D.ce_weight, s = sampleOf(r.text), gold = goldOf(s);
    const dist = D.multi
      ? [String.raw`\mathcal{L} = \frac{1}{${K}}\sum_{i=1}^{${K}} \mathrm{BCE}(p^{T}_i, q_i) + \lambda\,\frac{1}{${K}}\sum_{i=1}^{${K}} \mathrm{BCE}(y_i, q_i),\quad q_i = \sigma(z_i)`,
         String.raw`\mathrm{BCE}(t, q) = -\big(t \log q + (1 - t)\log(1 - q)\big)` + (lam !== null ? String.raw`,\quad \lambda = ${lam}` : "")]
      : [String.raw`\mathcal{L} = \underbrace{\sum_{i=1}^{${K}} p^{T}_i \log\frac{p^{T}_i}{q_i}}_{\mathrm{KL}(p^{T}\,\|\,q)} + \lambda\,\big(-\log q_{y}\big)` + (lam !== null ? String.raw`,\quad \lambda = ${lam}` : "")];
    dist.push({ note: "This is the loss the student was trained with. pᵀ is the teacher's probabilities (its soft labels), q the student's, y the gold label. " +
      "Both terms are averaged over a batch of training texts; the second counts only texts with a gold label" +
      (lam === 0 ? ", and its weight λ is 0 in this run, so the student learned from the teacher alone." : ".") });
    if (s && s.teacher) {
      const terms = s.teacher.filter(([, x]) => x > 0);
      const mass = terms.reduce((a, [, x]) => a + x, 0);
      if (D.multi) {
        const bce = (t, q) => -(t * Math.log(Math.max(q, 1e-12)) + (1 - t) * Math.log(Math.max(1 - q, 1e-12)));
        dist.push(String.raw`\begin{aligned}` + terms.slice(0, 3).map(([i, x]) =>
          String.raw`\mathrm{BCE}(p^{T}_{${T(name(i))}}, q) &= \mathrm{BCE}(${f(x)}, ${f(r.p[i])}) = ${f(bce(x, r.p[i]))}`).join(String.raw`\\`) + String.raw`\end{aligned}`,
          { note: "For this test text, three of the labels; the page stores the teacher's probabilities of its top labels only, so the sum over all labels is not shown." });
      } else {
        const kl = terms.map(([i, x]) => x * Math.log(x / Math.max(r.p[i], 1e-300)));
        dist.push(String.raw`\mathrm{KL} \approx ` + terms.slice(0, 3).map(([i, x]) => String.raw`${f(x)} \log\frac{${f(x)}}{${f(r.p[i])}}`).join(" + ") +
          (terms.length > 3 ? String.raw` + \cdots` : "") + ` = ${f(kl.reduce((a, b) => a + b, 0))}`,
          { note: `For this test text: the sum over the ${terms.length} labels the page stores the teacher's probabilities for ` +
            `(${pct(mass)} of them), with q the student's answer above.` });
        if (gold) dist.push(String.raw`-\log q_{y} = -\log ${f(r.p[gold[0]])} = ${f(-Math.log(Math.max(r.p[gold[0]], 1e-300)))}`);
      }
    } else dist.push({ note: "Pick an example or a random test text to see the KL term for it, computed here." });
    put("m-teacher", dist);
  }

  async function loadKatex() {
    document.head.append(el("link", { rel: "stylesheet", href: `${KATEX}/dist/katex.min.css` }));
    return (await import(`${KATEX}/dist/katex.mjs`)).default;
  }

  // ---- running it ----
  const steps = ["s-tok", "s-layers", "s-att", "s-out", "s-why", "s-teacher", "s-cascade"];
  const show = [showTokens, showLayers, showAttention, showOutput, showWords, showTeacher, showCascade];
  const flow = [...document.querySelectorAll(".flow span")];

  async function showSteps(r, animate, id) {  // all at once, or one after another to show the flow
    for (const s of steps) $(s).classList.toggle("done", !animate);
    for (const x of flow) x.classList.toggle("on", !animate);
    for (let i = 0; i < steps.length; i++) {
      show[i](r);
      if (!animate) continue;
      $(steps[i]).classList.add("done");
      if (flow[i]) flow[i].classList.add("on");
      if (i === 3) flow[4].classList.add("on");
      if (!reduced || slow) await sleep(slow ? 900 : 140);
      if (id !== runId) return;
    }
    showMath(r);
  }

  async function classify(animate = false) {
    const value = text.value;
    withText();
    if (!value.trim()) return;
    const id = ++runId;
    const r = await analyse(value);
    if (id !== runId) return;
    current = r;
    showAnswer(r);
    await showSteps(r, animate || slow, id);
    await sleep(400);  // which words matter takes one more run per word: wait until typing has settled
    if (id !== runId) return;
    await effects(r);
    if (id !== runId) return;
    showReason(r);
    showWords(r);
  }

  let timer = null;
  text.addEventListener("input", () => { clearTimeout(timer); timer = setTimeout(() => classify(), 150); });
  $("random").onclick = () => { text.value = D.samples[Math.floor(Math.random() * N)].text; classify(); };
  $("replay").onclick = () => {
    $("inside").open = true;
    $("inside").scrollIntoView({ behavior: reduced ? "auto" : "smooth" });
    classify(true);
  };
  $("slow").onclick = () => { slow = !slow; $("slow").setAttribute("aria-pressed", String(slow)); if (slow) classify(true); };
  $("math").onclick = async () => {
    math = !math;
    $("math").setAttribute("aria-pressed", String(math));
    for (const m of document.querySelectorAll(".math")) m.hidden = !math;
    if (math && !katex) {
      try { katex = await loadKatex(); } catch (e) { $("m-tok").textContent = `Could not load KaTeX: ${e}`; return; }
    }
    showMath(current);
  };

  $("check").onclick = async () => {  // every bundled test text through the model, with running scores
    const btn = $("check"), out = $("check-out");
    btn.disabled = true;
    const preds = [], golds = [];
    let agree = 0, compared = 0, ms = 0;
    for (const [k, s] of D.samples.entries()) {
      const r = await analyse(s.text), gold = goldOf(s);
      ms += r.ms.tokenize + r.ms.model;
      if (gold) { preds.push(r.pred); golds.push(gold); }
      if (s.teacher) { compared++; agree += same(r.pred, chosen(dense(s.teacher))); }
      const right = preds.filter((p, j) => same(p, golds[j])).length;
      out.textContent = `${k + 1 < N ? "So far" : "Done"}, ${k + 1} of ${N} test texts: ` + [
        golds.length && !D.multi ? `accuracy ${pct(right / golds.length)} (${right} of ${golds.length} correct)` : "",
        golds.length && D.multi ? `macro-F1 ${macroF1(preds, golds).toFixed(3)}` : "",
        compared ? `agreement with the teacher ${pct(agree / compared)}` : "",
        `${(ms / (k + 1)).toFixed(1)} ms per text in this browser`,
      ].filter(Boolean).join(", ") + ".";
      showSpeed();
    }
    btn.disabled = false;
  };
  function macroF1(preds, golds) {  // as tinydistill's multilabel_metrics: mean F1 over the labels that occur
    const c = D.labels.map(() => [0, 0, 0]);  // true positives, false positives, false negatives
    preds.forEach((p, k) => {
      for (const i of p) c[i][golds[k].includes(i) ? 0 : 1]++;
      for (const i of golds[k]) if (!p.includes(i)) c[i][2]++;
    });
    const f1 = c.filter(([t, a, b]) => t + a + b).map(([t, a, b]) => (t ? (2 * t) / (2 * t + a + b) : 0));
    return f1.length ? f1.reduce((a, b) => a + b) / f1.length : NaN;
  }

  classify();
</script>
</body>
</html>
"""


def _samples(rows: list[dict], n: int) -> list[dict]:
    """Up to n test rows with different gold labels, taken from across the file (test files are often sorted
    by label, so the first rows would all show the same one)."""
    picked: list[dict] = []
    labels: set[str] = set()
    step = max(1, len(rows) // (4 * n or 1))
    for r in rows[::step] + rows:
        key = json.dumps(r.get("gold"))
        if key not in labels and len(picked) < n:
            labels.add(key)
            picked.append(r)
    return picked


def _bundle(rows: list[dict], n: int = BUNDLE, chars: int = BUNDLE_CHARS) -> tuple[list[dict], int]:
    """Up to n test rows for the page: first one per gold label (``_samples``), then rows from across the
    file, while their texts stay under ``chars`` characters in all. Also returns how many of the first kind
    there are."""
    first = _samples(rows, n)
    picked: list[dict] = []
    seen: set[int] = set()
    size = 0
    for r in first + rows[:: max(1, len(rows) // n)] + rows:
        if len(picked) < n and id(r) not in seen and size + len(r["text"]) <= chars:
            picked.append(r)
            seen.add(id(r))
            size += len(r["text"])
    return picked, sum(id(r) in seen for r in first)


def _outside(config: dict, labels: list[str]) -> int | None:
    """The index of the model's `outside` label (texts that fit none of the others), or None."""
    lid = config.get("tinydistill_outside")
    return labels.index(lid) if lid in labels else None


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


def _rows(res: dict | None, deployed: str) -> tuple[dict | None, dict | None]:
    """The deployed student's row of results.json and the teacher's (its GPU row when there is one)."""
    models = (res or {}).get("models", [])
    student = next((m for m in models if m["role"] == "student" and deployed in m["name"]), None)
    teachers = [m for m in models if m["role"] == "teacher"]
    gpu = [m for m in teachers if str(m.get("device", "")).startswith("cuda") and m.get("latency_ms_b1")]
    return student, (gpu + teachers or [None])[0]


def _summary(res: dict | None, deployed: str) -> str:
    """One sentence on quality, from the run's results.json (never typed by hand)."""
    row, teacher = _rows(res, deployed)
    if not res or not row or row.get("accuracy") is None:
        return ""
    return (
        f"Test accuracy {row['accuracy']:.3f} on {res['data']['n_test']} held-out texts. "
        f"Distilled from {res.get('teacher')}"
        + (f" ({teacher['name'].removeprefix('teacher, ')})" if teacher else "")
        + f" into {res.get('student')}."
    )


def _card_rows(res: dict | None, deployed: str, multi: bool) -> list[list]:
    """Teacher against student, side by side: model, size, time per text and test score, all from the run's
    results.json, as [heading, teacher, student] rows; the student's time (None) is measured in the visitor's
    browser. A missing number leaves its cell empty, and a row with no numbers is left out."""
    student, teacher = _rows(res, deployed)
    if not res or not student:
        return []
    teacher = teacher or {}
    key, metric = ("macro_f1", "Test macro-F1") if multi else ("accuracy", "Test accuracy")

    def size(row: dict) -> str:
        mb = row.get("size_mb")
        return (
            "" if mb is None else f"{mb / 1000:.1f} GB" if mb >= 1000 else f"{mb:.{0 if mb >= 10 else 1}f} MB"
        )

    def score(row: dict) -> str:
        return "" if row.get(key) is None else f"{row[key]:.1%}"

    ms = teacher.get("latency_ms_b1") if str(teacher.get("device", "")).startswith("cuda") else None
    how = teacher.get("name", "").removeprefix("teacher, ")
    rows = [
        ["Model", f"{res.get('teacher') or ''}" + (f" ({how})" if how else ""), res.get("student") or ""],
        ["Size", size(teacher), size(student)],
        ["Time per text", "" if ms is None else f"{ms:.0f} ms on a GPU", None],
        [metric, score(teacher), score(student)],
    ]
    return [r for r in rows if r[1] or r[2] != ""]


def _card(rows: list[list]) -> str:
    """The card's HTML; the page draws the same table from the rows when another model is picked."""
    body = "".join(
        f"<tr><td>{html.escape(h)}</td><td class='t'>{html.escape(t)}</td><td>"
        + (html.escape(s) if s is not None else '<span id="live-ms">measured as you type</span>')
        + "</td></tr>"
        for h, t, s in rows
    )
    return (
        f'<section class="card" id="card" aria-label="Teacher and student"{"" if rows else " hidden"}>'
        '<table id="card-rows"><tr><th></th><th class="t">Teacher (LLM)</th><th>Student (this page)</th></tr>'
        f"{body}</table><p class='note'>From the run's results on its held-out test texts; the student's time "
        "is measured in your browser.</p></section>"
    )


def _score(res: dict | None, deployed: str, multi: bool) -> str | None:
    """The deployed student's test score in words, for the model picker."""
    row, _ = _rows(res, deployed)
    key, what = ("macro_f1", "macro-F1") if multi else ("accuracy", "accuracy")
    return f"test {what} {row[key]:.3f}" if row and row.get(key) is not None else None


# the tabs between the two pages and the way into the explainer, when `web` wrote one: the explainer is the
# main page (one folder up) and this page is the simple one
TABS = (
    '<nav class="pages" aria-label="Pages"><a href="../" data-base="../">Inside the model</a>'
    '<a href="./" aria-current="page">Simple</a></nav>'
)
INSIDE = """<div class="deeper">
    <p><b>See inside the model on this text</b>Every layer, every attention head and the math, as the model
      reads it, in your browser.</p>
    <a href="../?tour=1" data-base="../?tour=1">Guided tour</a>
    <a class="go" href="../" data-base="../">See inside</a>
  </div>"""


def _short(text: str, n: int = 40) -> str:
    return text if len(text) <= n else text[: n - 3].rstrip() + "..."


# The model picker of both pages (`tinydistill web --models`): a menu of the page's own model and the Hub models
# it lists. The page's data has the menu's entries as `models` (see catalogue); picking one fetches its
# tinydistill-page.json and loads its ONNX model from the Hub with transformers.js, and ?model=user/name in the
# URL opens the page on it. The pages define `D` (let), `$`, `el` and import `env`, AutoTokenizer and AutoModel.
PICKER_HTML = """<div class="picker" id="picker" hidden>
    <details id="pick"><summary>Switch model</summary><ul id="pick-list" aria-label="Models"></ul></details>
    <p id="pick-msg" role="status" hidden><span id="pick-text"></span><progress id="pick-bar" hidden></progress></p>
  </div>"""
PICKER_CSS = """.picker { position: relative; flex: none; max-width: 100%; }
  .picker details { width: fit-content; }
  .picker summary { list-style: none; cursor: pointer; padding: 2px 12px; border: 1px solid var(--line);
                    border-radius: 999px; background: var(--card); white-space: nowrap; font-size: .9em; }
  .picker summary::-webkit-details-marker { display: none; }
  .picker summary::after { content: " \\25BE"; color: var(--muted); }
  .picker details[open] summary { border-color: var(--accent); color: var(--accent); }
  .picker ul { position: absolute; z-index: 50; top: calc(100% + 6px); left: 0; width: min(380px, calc(100vw - 32px));
               max-height: 60vh; overflow-y: auto; margin: 0; padding: 4px; list-style: none; background: var(--card);
               border: 1px solid var(--line); border-radius: 10px; box-shadow: 0 8px 28px rgba(0, 0, 0, .2); }
  .picker ul button { display: block; width: 100%; text-align: left; padding: 7px 10px; margin: 0; border: 0;
                      border-radius: 7px; background: transparent; color: var(--fg); font: inherit; font-size: .9em; }
  .picker ul button:hover { background: var(--soft); }
  .picker ul button[aria-current="true"] { background: var(--soft); box-shadow: inset 3px 0 var(--accent); }
  .picker ul button[aria-current="true"] b::after { content: "  (on show)"; color: var(--muted); font-weight: 400; }
  .picker ul b { display: block; }
  .picker ul span { display: block; color: var(--muted); font-size: .9em; overflow-wrap: anywhere; }
  #pick-msg { margin: 6px 0 0; color: var(--muted); font-size: .85em; max-width: 34em; }
  #pick-msg.err { color: var(--teacher); }
  #pick-bar { display: block; width: 100%; height: 6px; margin-top: 4px; accent-color: var(--accent); }
  #pick-bar[hidden] { display: none; }
  @media (max-width: 600px) {  /* fixed, so the menu never runs off the screen; its top stays under the button */
    .picker ul { position: fixed; top: auto; left: 16px; right: 16px; width: auto; margin-top: 6px; }
  }"""
PICKER_JS = r"""  // ---- the model picker: `web --models` lists other models, which load from the Hugging Face Hub ----
  const HUB = window.TINYDISTILL_HUB ?? "https://huggingface.co/";  // the browser tests serve a copy of a repo
  const D0 = D, MODELS = D.models;  // the menu: this page's own model first
  const fromURL = new URLSearchParams(location.search).get("model");  // ?model=user/name: one in the menu
  let pickNow = MODELS.find((m) => m.repo && m.repo === fromURL) ?? MODELS[0], pickSeq = 0;
  async function pageData(m, key) {  // a model's data for this page: its own, or from its tinydistill-page.json
    if (m === MODELS[0]) return D0;
    const r = await fetch(`${HUB}${m.repo}/resolve/main/tinydistill-page.json`);
    if (!r.ok) throw new Error(`its tinydistill-page.json gave HTTP ${r.status}`);
    const d = (await r.json())[key];
    if (!d) throw new Error("it has no data for this page");
    return { ...d, repo: m.repo };
  }
  function loadModel(d, progress) {  // [tokenizer, model], from next to the page or from the Hub
    env.allowLocalModels = !d.repo; env.allowRemoteModels = !!d.repo; env.remoteHost = HUB;
    const files = {};
    const progress_callback = (e) => {  // download progress of every file, summed
      if (e.status !== "progress" || !e.total) return;
      files[e.file] = [e.loaded, e.total];
      let a = 0, b = 0;
      for (const [x, y] of Object.values(files)) { a += x; b += y; }
      progress(a, b);
    };
    const id = d.repo ?? d.model;
    return Promise.all([AutoTokenizer.from_pretrained(id, { progress_callback }),
      AutoModel.from_pretrained(id, { dtype: d.dtype, model_file_name: d.file, progress_callback })]);
  }
  const withModel = (u) => { if (pickNow !== MODELS[0]) u.searchParams.set("model", pickNow.repo); };
  const setModelURL = () => {
    const u = new URL(location.href);
    u.searchParams.delete("model");
    withModel(u);
    history.replaceState(null, "", u);
  };
  const say = (msg, err = false) => {
    $("pick-msg").hidden = !msg; $("pick-msg").classList.toggle("err", err);
    $("pick-text").textContent = msg; $("pick-bar").hidden = true;
  };
  const markPick = () => {
    for (const b of document.querySelectorAll("#pick-list button")) {
      b.setAttribute("aria-current", String(b.dataset.repo === (pickNow.repo ?? "")));
    }
  };
  const failed = (m, e) => {  // the message when m did not load; pickNow is the model the page keeps
    console.warn(e);
    const why = e instanceof TypeError ? "the Hub could not be reached (are you offline?)" : e.message
      .replace(/^Could not locate file: "(?:.*\/)?([^/"]+)"/, "its $1 is missing on the Hub").replace(/\.+$/, "");
    say(`${m.title} did not load: ${why}. The page still runs ${pickNow.title}.`, true);
  };
  const fallBack = (e) => {  // the model ?model= asked for did not load: this page's own instead
    const m = pickNow;
    pickNow = MODELS[0];
    setModelURL(); markPick(); failed(m, e);
  };
  function picker(entries, choose) {  // the menu, a button per model
    if (entries.length < 2) return;
    $("picker").hidden = false;
    $("pick-list").replaceChildren(...entries.map((m) => {
      const b = el("button", { type: "button", onclick: () => { $("pick").open = false; if (m !== pickNow) choose(m); } },
        el("b", { textContent: m.title }), el("span", { textContent: [m.student, m.size, m.score].filter(Boolean).join(", ") }));
      b.dataset.repo = m.repo ?? "";
      return el("li", {}, b);
    }));
    markPick();
    $("pick").addEventListener("keydown", (e) => {
      if (e.key === "Escape" && $("pick").open) { $("pick").open = false; $("pick").querySelector("summary").focus(); }
    });
    document.addEventListener("click", (e) => { if (!$("pick").contains(e.target)) $("pick").open = false; });
  }
  async function switchTo(m, key, apply) {  // load m, then apply(data, tokenizer, model); on failure keep the old one
    const seq = ++pickSeq;
    say(`Loading ${m.title}...`);
    try {
      const d = await pageData(m, key);
      const [t, model] = await loadModel(d, (a, b) => {
        if (seq !== pickSeq) return;
        say(`Downloading ${m.title}: ${(a / 1e6).toFixed(1)} of ${(b / 1e6).toFixed(1)} MB`);
        Object.assign($("pick-bar"), { hidden: false, max: b, value: a });
      });
      if (seq !== pickSeq) return;
      pickNow = m;
      setModelURL(); markPick(); say("");
      await apply(d, t, model);
    } catch (e) {
      if (seq === pickSeq) failed(m, e);
    }
  }
"""


def catalogue(own: dict, explainer: bool, hub: dict[str, dict], key: str) -> list[dict]:
    """The model picker's entries: the page's own model (its data, ``explainer``: whether it has the explainer
    page), then each Hub model (repo -> its tinydistill-page.json), with what the menu shows. ``key``: the page's
    part of tinydistill-page.json, "simple" or "explainer"."""

    def entry(repo: str | None, page: dict) -> dict:
        d = page.get(key) or page["simple"]  # a model without the explainer shows its simple page's numbers
        mb = d.get("size_mb") or 0
        return {
            "repo": repo,
            "title": d["text"]["title"],
            "student": d.get("student"),
            "size": f"{mb:.{0 if mb >= 10 else 1}f} MB",
            "score": d.get("score"),
            "explainer": bool(page.get("explainer")),
        }

    mine = own.get("repo")
    return [entry(mine, {"simple": own, "explainer": own if explainer else None})] + [
        entry(repo, page) for repo, page in hub.items() if repo != mine
    ]


def site_model(onnx_dir: Path, tmp: Path) -> Path:
    """The ONNX file the simple page runs: with each layer's answer and the first token's attention when the
    run's PyTorch student is next to the export (export.export_explain, written to ``tmp``), else the export."""
    from .export import export_explain

    run, int8 = onnx_dir.parent, (onnx_dir / "model.int8.onnx").exists()
    found = export_explain(run / "student", tmp, int8) if (run / "student" / "config.json").exists() else None
    return found or onnx_dir / ("model.int8.onnx" if int8 else "model.onnx")


TOKENIZER_FILES = (
    "config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.txt",
)


def clear_models(out: Path) -> None:
    """Remove the model folders an earlier build wrote to ``out``."""
    for old in out.glob("model*"):
        if old.is_dir() and re.fullmatch(r"model(-[0-9a-f]{10})?", old.name):
            shutil.rmtree(old)


def copy_model(onnx_dir: Path, onnx_file: Path, out: Path) -> str:
    """Copy a page's model to ``out/model-<hash>/`` in the transformers.js layout (config, tokenizer and
    ``onnx/model_quantized.onnx`` for int8, else ``onnx/model.onnx``), replacing earlier copies; returns the
    folder's name. Browsers cache model files by URL, so the folder is named after the model's contents: a new
    model gets a new URL and no visitor keeps an old copy."""
    model_dir = out / f"model-{hashlib.sha256(onnx_file.read_bytes()).hexdigest()[:10]}"
    clear_models(out)
    (model_dir / "onnx").mkdir(parents=True)
    int8 = (onnx_dir / "model.int8.onnx").exists()
    shutil.copy(onnx_file, model_dir / "onnx" / ("model_quantized.onnx" if int8 else "model.onnx"))
    for f in TOKENIZER_FILES:
        if (onnx_dir / f).exists():
            shutil.copy(onnx_dir / f, model_dir / f)
    return model_dir.name


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def site_data(onnx_dir: Path, size_mb: float, title: str | None = None, samples: int = 6) -> dict:
    """Everything the simple page shows for a model, as the JSON it embeds, apart from where the model is
    (build_site adds the folder, ``model``; a page that loads it from the Hub adds ``repo``). ``size_mb``: the
    size of the ONNX file it runs; ``samples``: example buttons (one per gold label). `tinydistill push`
    uploads it in tinydistill-page.json."""
    run = onnx_dir.parent
    config, meta, res = _read(onnx_dir / "config.json"), _read(run / "meta.json"), _read(run / "results.json")
    labels = json.loads((onnx_dir / "labels.json").read_text(encoding="utf-8"))
    multi = config.get("problem_type") == "multi_label_classification"
    int8 = (onnx_dir / "model.int8.onnx").exists()
    rows: list[dict] = []
    n_first = 0
    if (run / "test.jsonl").exists():
        with (run / "test.jsonl").open(encoding="utf-8") as fh:
            rows, n_first = _bundle([json.loads(line) for line in fh if line.strip()])
    ce_weight = (res.get("training") or {}).get("ce_weight")
    deployed = "int8" if int8 else "fp32"
    vocab = json.loads((onnx_dir / "tokenizer.json").read_text(encoding="utf-8"))["model"].get("vocab", {})
    teacher_row = next((m for m in res.get("models", []) if m["role"] == "teacher"), None)
    teacher = f"the teacher, {meta.get('teacher') or 'a large language model'}" + (
        f" ({teacher_row['name'].removeprefix('teacher, ')})" if teacher_row else ""
    )
    return {
        "labels": labels,
        "multi": multi,
        "dtype": "q8" if int8 else "fp32",
        "max_length": config.get("tinydistill_max_length") or 128,
        "outside": _outside(config, labels),
        "samples": [
            {
                "text": r["text"],
                "gold": r.get("gold"),
                "teacher": _teacher_probs(r["scores"], multi) if r.get("scores") else None,
            }
            for r in rows
        ],
        "n_test": res.get("data", {}).get("n_test"),
        "cascade": res.get("cascade"),
        "ce_weight": meta.get("ce_weight") if ce_weight is None else ce_weight,
        "config": {
            k: config.get(k)
            for k in (
                "model_type",
                "hidden_size",
                "num_attention_heads",
                "vocab_size",
                "type_vocab_size",
                "pad_token_id",
            )
        },
        "buttons": min(samples, n_first),
        "card": _card_rows(res or None, deployed, multi),
        "size_mb": round(size_mb, 2),
        "student": res.get("student") or meta.get("student"),
        "score": _score(res or None, deployed, multi),
        # the page's text that depends on the model, by its data-t attribute
        "text": {
            "title": title or (meta.get("task") or run.name).replace("-", " ").replace("_", " ").title(),
            "size": f"{size_mb:.0f}",
            "summary": _summary(res or None, deployed),
            "vocab": f"{len(vocab):,} pieces" if vocab else "pieces",
            "layers": str(config.get("num_hidden_layers", "several")),
            "nlabels": str(len(labels)),
            "activation": "sigmoid per label" if multi else "softmax",
            "teacher": teacher,
        },
    }


def render_site(
    data: dict,
    out: Path,
    explainer: bool = False,
    note: str | None = None,
    hub: dict[str, dict] | None = None,
) -> Path:
    """Write ``out/index.html`` from a model's page data (site_data plus ``model`` or ``repo``). ``explainer``:
    the explainer page is one folder up, so this page gets tabs to it. ``hub``: other models for the picker,
    repo -> tinydistill-page.json."""
    text, rows = data["text"], data["samples"]
    data = data | {"models": catalogue(data, explainer, hub or {}, "simple")}
    buttons = "\n    ".join(
        f'<button class="ghost sample" data-text="{html.escape(r["text"])}" '
        f'title="{html.escape(r["text"])}">{html.escape(_short(r["text"]))}</button>'
        for r in rows[: data["buttons"]]
    )
    page = (
        PAGE.replace("__TABS__", TABS if explainer else "")
        .replace("__INSIDE__", INSIDE if explainer else "")
        .replace("__PICKER_CSS__", PICKER_CSS)
        .replace("__PICKER__", PICKER_HTML)
        .replace("__PICKER_JS__", PICKER_JS)
        .replace("__CARD__", _card(data["card"]))
        .replace("__FIRST__", html.escape(rows[0]["text"]) if rows else "")
        .replace("__SAMPLES__", buttons)
        .replace("__TRANSFORMERS_JS__", TRANSFORMERS_JS)
        .replace("__KATEX__", KATEX)
        .replace("__NOTE__", html.escape(note or ""))
    )
    for k, v in text.items():
        page = page.replace(f"__{k.upper()}__", html.escape(v))
    page = page.replace("__DATA__", json.dumps(data).replace("</", "<\\/"))
    out.mkdir(parents=True, exist_ok=True)
    (out / "index.html").write_text(page, encoding="utf-8")
    return out / "index.html"


def build_site(
    onnx_dir: Path,
    out: Path,
    title: str | None = None,
    samples: int = 6,
    note: str | None = None,
    explainer: bool = False,
    hub: dict[str, dict] | None = None,
) -> Path:
    """Write ``out/index.html`` and ``out/model-<hash>/`` (config, tokenizer, ONNX) for transformers.js. When the run's
    PyTorch student is next to the export, the page's model also returns each layer's answer and the
    attention of the first token, for the "Inside the model" view. ``samples`` example buttons (one per gold
    label); the page also holds up to ``BUNDLE`` test texts with their gold and teacher labels. ``note`` is an
    extra line for the footer, such as the dataset's licence. ``explainer``: the explainer page
    (``explainer.build_explainer``) is the main page one folder up (``out/..``), as `tinydistill web` writes it,
    so this page gets tabs to it and a link that opens it on the current text. ``hub``: other models for the
    model picker, Hub repo -> its tinydistill-page.json (see `tinydistill push`)."""
    from .static import is_static

    if not (onnx_dir / "config.json").exists():
        raise SystemExit(f"{onnx_dir} has no config.json: export it again with `tinydistill export` (0.3.0+)")
    if is_static(onnx_dir):
        raise SystemExit("the browser page needs a transformer student; static students run with `serve`")
    with tempfile.TemporaryDirectory() as tmp:
        onnx_file = site_model(onnx_dir, Path(tmp))
        size = onnx_file.stat().st_size / 1e6
        folder = copy_model(onnx_dir, onnx_file, out)
    data = site_data(onnx_dir, size, title, samples) | {"model": folder}
    return render_site(data, out, explainer, note, hub)
