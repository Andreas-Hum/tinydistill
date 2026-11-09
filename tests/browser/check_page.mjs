// Opens a page written by `tinydistill web` in Chromium at phone width and checks that the model runs in the
// browser, classifies as you type, checks itself on the bundled test texts, and that every step of the
// "Inside the model" view (with the math) fills in, with no errors. When the folder has an explainer page, the page
// has tabs to it and a link that opens it on the current text, and its "Simple" tab brings the text back.
//   node tests/browser/check_page.mjs http://127.0.0.1:8000/ [screenshot.png]
// NO_EXPLAIN=1 for a page whose model has no layer or attention outputs. HUB=http://.../hub/ loads Hub models
// from a local copy of the Hub (build_tiny_site.py writes one), and PICKER=1 also checks the model picker
// against the repositories in it.
import { chromium } from "playwright";

const [url, screenshot] = process.argv.slice(2);
const explain = !process.env.NO_EXPLAIN;
const browser = await chromium.launch();
const newPage = async (viewport) => {
  const p = await browser.newPage({ viewport });
  if (process.env.HUB) await p.addInitScript((hub) => { window.TINYDISTILL_HUB = hub; }, process.env.HUB);
  return p;
};
const page = await newPage({ width: 390, height: 900 });
const problems = [];
page.on("pageerror", (e) => problems.push(`page error: ${e}`));
page.on("console", (m) => m.type() === "error" && problems.push(`console error: ${m.text()}`));
const wait = (fn, arg) => page.waitForFunction(fn, arg, { timeout: 180000 });
const text = (sel) => page.locator(sel).first().textContent();

await page.goto(url);
// the first text: answer, every step, and which words matter once it settles
await wait(() => document.getElementById("loading").hidden && document.querySelectorAll("#steps > li.done").length >= 6
  && document.querySelectorAll("#words span").length >= 2);
const seen = await page.evaluate(() => ({
  answer: document.getElementById("answer").textContent,
  top3: document.querySelectorAll("#top3 li").length,
  tokens: document.querySelectorAll("#tokens .chip").length,
  layerRows: document.querySelectorAll("#layers tr").length,
  attention: document.querySelectorAll("#attention .chip").length,
  labels: document.querySelectorAll("#top li").length,
  words: document.querySelectorAll("#words span").length,
  teacher: document.querySelectorAll("#teacher li").length,
  card: document.querySelectorAll(".card tr").length,
  liveMs: document.getElementById("live-ms")?.textContent ?? "",
}));
console.log(JSON.stringify(seen));
if (!seen.answer) problems.push("the model gave no answer");
if (seen.top3 < 2) problems.push("no top labels under the answer");
if (seen.tokens < 3) problems.push("no tokens shown");
if (explain && seen.layerRows < 3) problems.push("no layer outputs");
if (explain && seen.attention < 3) problems.push("no attention outputs");
if (seen.labels < 2) problems.push("no label probabilities");
if (seen.words < 2) problems.push("no word importance");
if (seen.teacher < 2) problems.push("no teacher soft labels for the first example");
if (seen.card < 4) problems.push("no teacher and student card");
if (!/ms in this browser/.test(seen.liveMs)) problems.push(`the card has no measured speed: "${seen.liveMs}"`);
if (!/correct|wrong/.test(await text("#verdict"))) problems.push("no correct/wrong for the first test text");

// classify as you type: no button, the answer and the highlighted copy follow the text
await page.fill("#text", "something else entirely, typed by hand");
await wait(() => document.getElementById("reason").textContent.includes("typed by hand")
  && document.getElementById("reason-note").textContent.startsWith("Why"));
if ((await text("#verdict")) !== "") problems.push("a typed text shows a gold label");

// a random test text shows the student, the teacher and the gold label, marked in words
await page.click("#random");
await wait(() => /Gold label/.test(document.getElementById("verdict").textContent));

// check it yourself on every bundled test text
await page.click("#check");
await wait(() => document.getElementById("check-out").textContent.startsWith("Done"));
const check = await text("#check-out");
console.log(check);
if (!/(accuracy|macro-F1) .*agreement with the teacher .* ms per text/.test(check)) problems.push(`check: ${check}`);

// the searchable list of labels
await page.click("#all-labels > summary");
await page.fill("#label-q", (await text("#label-list li")).slice(0, 5));
if ((await page.locator("#label-list li:visible").count()) !== 1) problems.push("the label search does not filter");

// "How does it work?" is folded; show the steps, another layer's attention, and the math
await page.click("#replay");
await wait(() => document.getElementById("inside").open && document.querySelectorAll("#steps > li.done").length >= 6);
if (explain) await page.click("#att-layers button >> nth=0");
await page.click("#math");
await wait(() => document.querySelectorAll(".math .katex").length >= 6);
const math = await page.evaluate(() => [...document.querySelectorAll(".math .katex-error")].map((e) => e.title));
if (math.length) problems.push(`KaTeX errors: ${math.join("; ")}`);

const width = await page.evaluate(() => document.documentElement.scrollWidth);
if (width > 390) problems.push(`the page is ${width}px wide on a 390px screen`);
if (screenshot) await page.screenshot({ path: screenshot, fullPage: true });

// the tabs, only when `web` wrote the explainer, and the text going there and back
// `web` writes the explainer as the main page and this page to simple/, one folder down
const up = await page.request.get(new URL("../", url).href);
const explainer = up.ok() && (await up.text()).includes('aria-current="page">Inside the model');
const tabs = await page.locator("nav.pages a").count();
if (tabs !== (explainer ? 2 : 0)) problems.push(`${tabs} page tabs, with${explainer ? "" : "out"} an explainer page`);
if (!explainer && (await page.locator(".deeper").count())) problems.push("a link to the explainer, which does not exist");
if (explainer) {
  if ((await text('nav.pages a[aria-current="page"]')) !== "Simple") problems.push("the tabs do not mark this page");
  const mine = "a text of my own, there and back";
  await page.fill("#text", mine);
  for (const sel of [".deeper a.go", "nav.pages a:not([aria-current])"]) {
    const href = new URL(await page.getAttribute(sel, "href"));
    if (href.searchParams.get("text") !== mine) problems.push(`${sel} does not carry the text: ${href}`);
  }
  await page.click(".deeper a.go");
  await page.waitForFunction(() => document.body.dataset.ready === "1", null, { timeout: 180000 });
  if ((await page.inputValue("#text")) !== mine) problems.push("the explainer did not open on the text");
  if ((await text('nav.pages a[aria-current="page"]')) !== "Inside the model") problems.push("the explainer's tabs do not mark it");
  await page.click("nav.pages a:not([aria-current])");
  await wait((t) => document.getElementById("loading").hidden && document.getElementById("reason").textContent.includes(t), mine);
  if ((await page.inputValue("#text")) !== mine) problems.push("the \"Simple\" tab did not bring the text back");
}
// a long label never runs into its percentage (both at this width and on a phone)
for (const width of [1280, 390]) {
  await page.setViewportSize({ width, height: 800 });
  const clash = await page.evaluate(() => [...document.querySelectorAll(".bars li")].filter((li) => {
    const [label, p] = [li.querySelector("span"), li.querySelector(".p")];
    if (!label || !p || !li.offsetParent) return false;
    const a = label.getBoundingClientRect(), b = p.getBoundingClientRect();
    return a.right > b.left + 0.5 && a.bottom > b.top + 0.5 && b.bottom > a.top + 0.5;
  }).map((li) => li.textContent));
  if (clash.length) problems.push(`label over its percentage at ${width}px: ${clash.join(" | ")}`);
}
if (process.env.PICKER) await picker();
await browser.close();
if (problems.length) {
  console.error(problems.join("\n"));
  process.exit(1);
}
console.log("page ok");

// the model picker of `web --models`, with the local Hub of build_tiny_site.py: test/multi (three labels), test/plain
// (no "Inside the model" view) and test/broken (its model files are missing)
async function picker() {
  const p = await newPage({ width: 1280, height: 900 });
  // the 404s of test/broken's model files and the failed loads while offline are on purpose; told apart by URL
  // and error, since they can be logged after the page has already shown its message
  p.on("pageerror", (e) => problems.push(`picker: page error: ${e}`));
  p.on("console", (m) => m.type() === "error" && !/\/test\/broken\//.test(m.location().url)
    && !/ERR_INTERNET_DISCONNECTED/.test(m.text()) && problems.push(`picker: console error: ${m.text()}`));
  const until = (fn, arg) => p.waitForFunction(fn, arg, { timeout: 60000 });
  const labels = () => p.$$eval("#label-list li", (li) => li.map((x) => x.textContent));
  const model = () => new URL(p.url()).searchParams.get("model");
  const inside = async () => new URL(await p.getAttribute("nav.pages a:not([aria-current])", "href")).searchParams.get("model");
  const pick = async (repo) => { await p.click("#pick > summary"); await p.click(`#pick-list button[data-repo="${repo}"]`); };
  await p.goto(url);
  await until(() => document.getElementById("loading").hidden && document.getElementById("answer").textContent);
  const n = await p.locator("#pick-list button").count();
  if (n !== 4) problems.push(`picker: ${n} models in the menu, not 4`);
  const mine = "a text of my own, kept when the model changes";
  await p.fill("#text", mine);
  await until((t) => document.getElementById("reason").textContent.includes(t), mine);
  const card = await p.textContent("#card");

  // another model: its labels, card, answer and title, the same text, and ?model= in the URL and the tabs
  await pick("test/multi");
  await until(() => [...document.querySelectorAll("#label-list li")].some((li) => li.textContent === "sports")
    && /sports|politics|money|none of the labels/.test(document.getElementById("answer").textContent));
  if (model() !== "test/multi") problems.push(`picker: the URL says ?model=${model()}`);
  if ((await labels()).join() !== "sports,politics,money") problems.push(`picker: labels ${await labels()}`);
  if ((await p.textContent("#card")) === card || !(await p.textContent("#card")).includes("a second student")) problems.push("picker: the card did not change");
  if ((await p.textContent("h1")) !== "News Topics") problems.push(`picker: the title is ${await p.textContent("h1")}`);
  if ((await p.inputValue("#text")) !== mine) problems.push("picker: the text changed with the model");
  if ((await inside()) !== "test/multi") problems.push("picker: the \"Inside the model\" tab does not carry the model");
  if (!(await p.getAttribute('#pick-list button[data-repo="test/multi"]', "aria-current") === "true")) problems.push("picker: the menu does not mark the model on show");

  // a model that does not load: a message, and the page keeps the one it had
  await pick("test/broken");
  await until(() => document.getElementById("pick-msg").classList.contains("err") && !document.getElementById("pick-msg").hidden)
    .catch(() => problems.push("picker: no message for a model that does not load"));
  const msg = await p.textContent("#pick-msg");
  console.log("picker, broken model:", msg);
  if (!/did not load.*still runs News Topics/.test(msg)) problems.push(`picker: the message says "${msg}"`);
  if (model() !== "test/multi" || (await labels()).join() !== "sports,politics,money") problems.push("picker: a failed model did not keep the old one");
  await p.fill("#text", "the club signed a new player");
  await until(() => document.getElementById("reason").textContent.includes("new player"))
    .catch(() => problems.push("picker: the old model stopped answering after a failed one"));
  await p.context().setOffline(true);  // offline: the Hub cannot be reached
  await pick("test/plain");
  await until(() => /could not be reached/.test(document.getElementById("pick-msg").textContent))
    .catch(async () => problems.push(`picker: offline, the message says "${await p.textContent("#pick-msg")}"`));
  await p.context().setOffline(false);
  if (model() !== "test/multi") problems.push("picker: going offline lost the model");

  // ?model= opens the page on a model; one without the explainer says so and its tab goes to the page's own model
  await p.goto(`${url}?model=test/plain`);
  await until(() => document.getElementById("loading").hidden && document.getElementById("answer").textContent);
  if ((await labels()).join() !== "spam,ham") problems.push(`picker: ?model=test/plain shows labels ${await labels()}`);
  if (await p.isHidden("#no-inside")) problems.push("picker: no note that test/plain has no \"Inside the model\" view");
  if ((await inside()) !== null) problems.push("picker: the tab carries a model that has no explainer");

  // the tabs keep the model: over to the explainer with ?model=test/multi
  await p.goto(`${url}?model=test/multi`);
  await until(() => document.getElementById("loading").hidden && document.getElementById("answer").textContent);
  await p.click("nav.pages a:not([aria-current])");
  await until(() => document.body.dataset.ready === "1" && document.querySelectorAll("#probs .prow:not(.out)").length === 3)
    .catch(() => problems.push("picker: the explainer did not open on test/multi"));
  if (model() !== "test/multi") problems.push(`picker: the explainer's URL says ?model=${model()}`);
  await p.close();
  console.log("picker ok");
}
