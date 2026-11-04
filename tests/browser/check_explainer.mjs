// Opens explainer pages written by `tinydistill web` (DIR/explainer/) in Chromium, at phone width in light mode
// and at desktop width in dark mode. It checks that the network, the heads, the output and every detail tab draw
// something, that hovering a token highlights it everywhere, that typing reruns the model, that the formulas
// render, that the page never scrolls sideways (and fits one screen on the desktop) and that nothing logs an error.
//   node tests/browser/check_explainer.mjs http://127.0.0.1:8000/explainer/ [more URLs] [--shot explainer.png]
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
const problems = [];
for (const url of args) {
  for (const [width, scheme] of [[390, "light"], [1280, "dark"]]) {
    const where = `${url} at ${width}px, ${scheme}`;
    const page = await browser.newPage({ viewport: { width, height: 820 }, colorScheme: scheme });
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
    await page.close();
  }
}
await browser.close();
if (problems.length) {
  console.error(problems.join("\n"));
  process.exit(1);
}
console.log("explainer ok");
