// Opens a page written by `tinydistill web` in Chromium at phone width and checks that the model runs in the
// browser, classifies as you type, checks itself on the bundled test texts, and that every step of the
// "Inside the model" view (with the math) fills in, with no errors. When the folder has an explainer page, the page
// has tabs to it and a link that opens it on the current text, and its "Simple" tab brings the text back.
//   node tests/browser/check_page.mjs http://127.0.0.1:8000/ [screenshot.png]
// NO_EXPLAIN=1 for a page whose model has no layer or attention outputs.
import { chromium } from "playwright";

const [url, screenshot] = process.argv.slice(2);
const explain = !process.env.NO_EXPLAIN;
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 390, height: 900 } });
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
await page.fill("#label-q", "posit");
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
await browser.close();
if (problems.length) {
  console.error(problems.join("\n"));
  process.exit(1);
}
console.log("page ok");
