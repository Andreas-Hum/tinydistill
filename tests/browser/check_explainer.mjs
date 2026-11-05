// Opens explainer pages written by `tinydistill web` (DIR/, the main page) in Chromium, at phone width in light mode
// and at desktop width in dark mode. It checks that the network, the heads, the output and every detail tab draw
// something, that hovering a token highlights it everywhere, that typing reruns the model, that the formulas
// render, that the page never scrolls sideways (and fits one screen on the desktop) and that nothing logs an error.
// Then it opens the page with ?tour=1&text=... and runs the guided tour through every step (each spotlight on
// screen and the page driven as the step says) and out again, with the tabs and the text they carry.
//   node tests/browser/check_explainer.mjs http://127.0.0.1:8000/ [more URLs] [--shot explainer.png]
// HUB=http://.../hub/ loads Hub models from a local copy of the Hub (build_tiny_site.py writes one), and PICKER=1
// also checks the model picker of the first URL against the repositories in it.
import { chromium } from "playwright";

const args = process.argv.slice(2), shotAt = args.indexOf("--shot");
const shot = shotAt >= 0 ? args.splice(shotAt, 2)[1] : null;
const tabs = {  // per detail tab, something it must draw
  matrix: "#viz svg.mat image",
  embed: "#viz canvas.strip",
  map: "#viz .pt",
  cls: "#viz .contrib rect",
};
const browser = await chromium.launch();
const newPage = async (options) => {
  const p = await browser.newPage(options);
  if (process.env.HUB) await p.addInitScript((hub) => { window.TINYDISTILL_HUB = hub; }, process.env.HUB);
  return p;
};
const problems = [];
if (process.env.PICKER) await picker(args[0]);
for (const url of args) {
  for (const [width, scheme, reducedMotion] of [[390, "light", "reduce"], [1280, "dark", "no-preference"]]) {
    const where = `${url} at ${width}px, ${scheme}`;
    const page = await newPage({ viewport: { width, height: 820 }, colorScheme: scheme, reducedMotion });
    page.on("pageerror", (e) => problems.push(`${where}: page error: ${e}`));
    page.on("console", (m) => m.type() === "error" && problems.push(`${where}: console error: ${m.text()}`));
    await page.goto(url);
    try {
      await page.waitForFunction(() => document.body.dataset.ready === "1", null, { timeout: 180000 });
    } catch {
      problems.push(`${where}: the page never got ready; it says: ${await page.textContent("#status")}`);
      await page.close();
      continue;
    }
    const count = (s) => page.locator(s).count();
    const need = async (what, selector, n = 1) => {
      seen[what] = await count(selector);
      if (seen[what] < n) problems.push(`${where}: ${what} has ${seen[what]} (${selector})`);
    };
    const formulas = async (what, box) => {
      seen[`${what} formulas`] = await count(`${box} .katex`);
      if (!seen[`${what} formulas`]) problems.push(`${where}: ${what} has no rendered formula`);
      if (await count(`${box} .katex-error`)) problems.push(`${where}: a formula in ${what} did not parse`);
    };
    const seen = {};
    const T = await count("#tokens .tok"), cols = await count("#netbox .colhead");
    if (cols < 2) problems.push(`${where}: the network has ${cols} layer columns`);
    await need("tokens", "#tokens .tok");
    await need("network nodes", "#net .node", T * cols);
    await need("attention lines", "#net .edge");
    await need("heads", "#heads .head", 2);
    await need("probabilities", "#probs .prow:not(.out) polyline");
    if (!(await page.textContent("#answer")).trim() || (await page.textContent("#answer")) === "...") problems.push(`${where}: no answer`);
    // a long label never runs into its percentage
    const clash = await page.evaluate(() => [...document.querySelectorAll("#probs .prow:not(.out)")].filter((row) => {
      const a = row.querySelector(".nm").getBoundingClientRect(), b = row.querySelector(".pv").getBoundingClientRect();
      return a.right > b.left + 0.5;
    }).length);
    if (clash) problems.push(`${where}: ${clash} probability labels run into their percentage`);

    // hover is the main interaction: a token lights up in every column, its lines and its row in the matrix
    await page.locator("#tokens .tok").nth(1).hover();
    seen["highlighted nodes"] = await count("#net .node.hl");
    if (seen["highlighted nodes"] !== cols) problems.push(`${where}: hovering a token highlights ${seen["highlighted nodes"]} nodes, not ${cols}`);
    if (!(await count("#net .edge.hl"))) problems.push(`${where}: hovering a token highlights no attention line`);
    if (!/reads most from/.test(await page.textContent("#viz .readout"))) problems.push(`${where}: the matrix does not follow the hovered token`);
    // a layer shows its heads; clicking one pins it
    await page.locator("#netbox .colhead").nth(2).hover();
    if (!(await page.textContent("#strip-cap")).includes("Layer 2")) problems.push(`${where}: hovering layer 2 does not show its heads`);
    await page.locator("#heads .head").nth(2).click();
    if ((await page.locator("#heads .head").nth(2).getAttribute("aria-pressed")) !== "true") problems.push(`${where}: a head does not pin`);
    if (!(await page.textContent("#sel")).includes("head 2")) problems.push(`${where}: the network does not show the pinned head`);

    await page.click("#math-toggle");
    for (const [tab, selector] of Object.entries(tabs)) {
      await page.click(`#tab-${tab}`);
      if (tab === "map") await page.$eval("#map-layer", (s) => { s.value = "0"; s.dispatchEvent(new Event("input")); });
      await need(`tab "${tab}"`, selector);
      await formulas(`tab "${tab}"`, "#math");
    }
    await page.click("#tab-map");
    await page.locator("#tokens .tok").nth(1).hover();
    if (!(await count("#viz .pt.hl"))) problems.push(`${where}: the token map does not follow the hovered token`);
    for (const id of ["i-answer", "i-trace", "i-loss", "i-tokens"]) {
      await page.click(`[aria-controls="${id}"]`);
      await formulas(id, `#${id}`);
    }

    // teacher against student for a bundled example, and typing reruns the model
    if (await count(".chip")) {
      await page.locator(".chip").first().click();
      await page.waitForFunction(() => document.querySelector(".chip").getAttribute("aria-pressed") === "true");
      await need("teacher and student", "#teacher .pair li", 1);
      await need("loss terms", "#loss dd");
    }
    const before = await page.textContent("#status");
    await page.fill("#text", "a short text, typed live into the box");
    await page.waitForFunction((b) => document.querySelector("#status").textContent !== b, before, { timeout: 30000 })
      .catch(() => problems.push(`${where}: typing did not rerun the model`));
    await need("nodes after typing", "#net .node", (await count("#tokens .tok")) * cols);

    seen.width = await page.evaluate(() => document.documentElement.scrollWidth);
    seen.height = await page.evaluate(() => document.documentElement.scrollHeight);
    console.log(where, JSON.stringify(seen));
    if (seen.width > width) problems.push(`${where}: the page is ${seen.width}px wide`);
    if (width >= 1000 && seen.height > 820) problems.push(`${where}: the dashboard is ${seen.height}px tall, more than the screen`);
    if (shot && url === args[0] && width === 1280) {
      await page.locator(".chip").first().click();
      await page.click("#tab-matrix");
      await page.locator("#net .node").nth(3).hover();
      await page.waitForTimeout(500);
      await page.screenshot({ path: shot });
    }
    await tour(page, url, where);
    await page.close();
  }
}
await browser.close();
if (problems.length) {
  console.error(problems.join("\n"));
  process.exit(1);
}
console.log("explainer ok");

// the guided tour, started from the URL on a text of our own, then once more from its button
async function tour(page, url, where) {
  const mine = "a text of my own, for the tour";
  await page.goto(`${url}?tour=1&text=${encodeURIComponent(mine)}`);
  await page.waitForFunction(() => document.body.dataset.ready === "1" && !document.getElementById("tour").hidden, null, { timeout: 180000 });
  if ((await page.inputValue("#text")) !== mine) problems.push(`${where}: ?text= did not fill the text box`);
  if ((await page.textContent('nav.pages a[aria-current="page"]')) !== "Inside the model") problems.push(`${where}: the tabs do not mark this page`);
  const back = new URL(await page.getAttribute("nav.pages a:not([aria-current])", "href"));
  if (back.searchParams.get("text") !== mine) problems.push(`${where}: the "Simple" tab does not carry the text (${back})`);
  const cols = await page.locator("#netbox .colhead").count(), n = Number((await page.textContent("#tour-n")).split(" of ")[1]);
  if (!(n >= 8 && n <= 11)) problems.push(`${where}: the tour has ${n} steps`);
  const titles = [];
  for (let k = 1; k <= n; k++) {
    await page.waitForFunction((k) => document.getElementById("tour-n").textContent.startsWith(`Step ${k} of`)
      && document.getElementById("tour-text").textContent.length > 40, k, { timeout: 30000 })
      .catch(() => problems.push(`${where}: tour step ${k} never got its text`));
    const title = await page.textContent("#tour-title");
    titles.push(title);
    // the spotlight is on screen and above the card (on a phone the card docks at the bottom)
    const seen = await page.waitForFunction(() => {
      const s = document.getElementById("tour-spot").getBoundingClientRect(), c = document.getElementById("tour-card").getBoundingClientRect();
      const bottom = c.left < 10 && c.right > innerWidth - 10 ? c.top : innerHeight;  // docked: above the card
      const h = Math.min(s.bottom, bottom) - Math.max(s.top, 0);
      return !document.getElementById("tour-spot").hidden && s.width > 10 && h >= Math.min(60, s.height * 0.5) && s.left >= -10 && s.right <= innerWidth + 10;
    }, null, { timeout: 5000 }).then(() => true, () => false);
    if (!seen) problems.push(`${where}: tour step ${k} (${title}): the spotlight is not on screen`);
    if (!(await page.evaluate(() => document.getElementById("tour-card").contains(document.activeElement)))) problems.push(`${where}: tour step ${k}: the focus is not in the card`);
    const tab = async (t) => { if ((await page.getAttribute(`#tab-${t}`, "aria-selected")) !== "true") problems.push(`${where}: tour step "${title}" did not open the ${t} tab`); };
    if (/one token/.test(title)) {
      if ((await page.locator("#net .node.pin").count()) !== cols) problems.push(`${where}: "${title}" selected no token`);
      if (!(await page.locator("#net .edge.hl").count())) problems.push(`${where}: "${title}" lights up no attention line`);
    }
    if (/heads/.test(title) && !(await page.locator('#heads .head[aria-pressed="true"]:not([data-h="-1"])').count())) problems.push(`${where}: "${title}" pinned no head`);
    if (/matrix/.test(title)) await tab("matrix");
    if (/map/.test(title)) await tab("map");
    if (/classifier/.test(title)) await tab("cls");
    if (/teacher/i.test(title) && !(await page.locator("#teacher .pair li").count())) problems.push(`${where}: "${title}" shows no teacher`);
    if (k === 3) {  // back and forth with the arrow keys
      await page.keyboard.press("ArrowLeft");
      await page.waitForFunction(() => document.getElementById("tour-n").textContent.startsWith("Step 2 of"));
      await page.keyboard.press("ArrowRight");
      await page.waitForFunction(() => document.getElementById("tour-n").textContent.startsWith("Step 3 of"));
    }
    if (k < n) await page.keyboard.press("ArrowRight");
  }
  console.log(where, "tour:", titles.join(" / "));
  // the last step's button ends it, and the page is as it was: the visitor's text, nothing pinned, the first tab
  await page.click("#tour-next");
  await page.waitForFunction((t) => document.getElementById("tour").hidden && document.getElementById("text").value === t
    && [...document.querySelectorAll(".chip")].every((b) => b.getAttribute("aria-pressed") === "false"), mine, { timeout: 30000 })
    .catch(() => problems.push(`${where}: ending the tour did not bring back the visitor's text`));
  if (await page.locator("#net .node.pin").count()) problems.push(`${where}: a token stays selected after the tour`);
  if ((await page.getAttribute('#heads .head[data-h="-1"]', "aria-pressed")) !== "true") problems.push(`${where}: a head stays pinned after the tour`);
  if ((await page.getAttribute("#tab-matrix", "aria-selected")) !== "true") problems.push(`${where}: the tour left another tab open`);
  // the header button starts it again; Escape leaves it
  await page.click("#tour-start");
  await page.waitForFunction(() => !document.getElementById("tour").hidden && document.getElementById("tour-n").textContent.startsWith("Step 1 of"));
  await page.keyboard.press("Escape");
  if (!(await page.isHidden("#tour"))) problems.push(`${where}: Escape does not end the tour`);
}

// the model picker of `web --models`, with the local Hub of build_tiny_site.py: test/multi (three labels), test/plain
// (no explainer, so not in this page's menu) and test/broken (its model files are missing)
async function picker(url) {
  const p = await newPage({ viewport: { width: 1280, height: 820 } });
  let broken = false;  // the 404s of test/broken's model files are on purpose
  p.on("pageerror", (e) => problems.push(`picker: page error: ${e}`));
  p.on("console", (m) => m.type() === "error" && !(broken && /404|Failed to load resource/.test(m.text()))
    && problems.push(`picker: console error: ${m.text()}`));
  const until = (fn, arg) => p.waitForFunction(fn, arg, { timeout: 60000 });
  const rows = () => p.locator("#probs .prow:not(.out)").count();
  const model = () => new URL(p.url()).searchParams.get("model");
  const pick = async (repo) => { await p.click("#pick > summary"); await p.click(`#pick-list button[data-repo="${repo}"]`); };
  await p.goto(url);
  await until(() => document.body.dataset.ready === "1");
  const repos = await p.$$eval("#pick-list button", (b) => b.map((x) => x.dataset.repo));
  if (repos.join() !== ",test/multi,test/broken") problems.push(`picker: the menu has ${repos.join()}`);
  const mine = "a text of my own, kept when the model changes";
  await p.fill("#text", mine);
  await until(() => document.getElementById("status").textContent.includes("ms in your browser"));

  await pick("test/multi");
  await until(() => document.querySelectorAll("#probs .prow:not(.out)").length === 3 && document.querySelector("h1").textContent === "News Topics")
    .catch(async () => problems.push(`picker: test/multi did not load: ${await rows()} label rows, title ${await p.textContent("h1")}`));
  if (model() !== "test/multi") problems.push(`picker: the URL says ?model=${model()}`);
  if ((await p.inputValue("#text")) !== mine) problems.push("picker: the text changed with the model");
  const tab = new URL(await p.getAttribute("nav.pages a:not([aria-current])", "href"));
  if (tab.searchParams.get("model") !== "test/multi") problems.push(`picker: the "Simple" tab does not carry the model (${tab})`);
  if (!(await p.textContent("#samples")).includes("minister")) problems.push("picker: the examples did not change");
  await p.locator("#tokens .tok").nth(1).hover();  // the network is drawn for the new model, and hovering works
  if (!(await p.locator("#net .node.hl").count())) problems.push("picker: hovering a token after the switch highlights nothing");

  broken = true;
  await pick("test/broken");
  await until(() => document.getElementById("pick-msg").classList.contains("err") && !document.getElementById("pick-msg").hidden)
    .catch(() => problems.push("picker: no message for a model that does not load"));
  console.log("picker, broken model:", await p.textContent("#pick-msg"));
  if (model() !== "test/multi" || (await rows()) !== 3) problems.push("picker: a failed model did not keep the old one");
  broken = false;

  // a model without the explainer opens on the simple page, which says why
  await p.goto(`${url}?model=test/plain&text=hello`);
  await until(() => location.pathname.endsWith("/simple/") && document.getElementById("loading").hidden)
    .catch(() => problems.push(`picker: ?model=test/plain stayed at ${p.url()}`));
  if (await p.isHidden("#no-inside")) problems.push("picker: no note that test/plain has no explainer");
  if ((await p.inputValue("#text")) !== "hello") problems.push("picker: the text did not come along to the simple page");
  await p.close();
  console.log("picker ok");
}
