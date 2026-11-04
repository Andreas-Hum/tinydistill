// Opens explainer pages written by `tinydistill web` (DIR/explainer/) in Chromium, at phone width in light mode
// and at desktop width in dark mode, walks through every step and checks that each draws what it should, that
// the formulas render, and that nothing logs an error.
//   node tests/browser/check_explainer.mjs http://127.0.0.1:8000/explainer/ [more URLs] [--shot explainer.png]
import { chromium } from "playwright";

const args = process.argv.slice(2), shotAt = args.indexOf("--shot");
const shot = shotAt >= 0 ? args.splice(shotAt, 2)[1] : null;
const want = {  // per step, something it must draw
  tokens: "#stage-body .tok",
  embed: "#stage-body svg image",
  layers: "#stage-body .head",
  trace: "#stage-body polyline",
  map: "#stage-body .pt",
  cls: "#stage-body .contrib rect",
  output: "#stage-body .bars li",
  teacher: "#stage-body .bars li",
};
const browser = await chromium.launch();
const problems = [];
for (const url of args) {
  for (const [width, scheme] of [[390, "light"], [1280, "dark"]]) {
    const where = `${url} at ${width}px, ${scheme}`;
    const page = await browser.newPage({ viewport: { width, height: 900 }, colorScheme: scheme });
    page.on("pageerror", (e) => problems.push(`${where}: page error: ${e}`));
    page.on("console", (m) => m.type() === "error" && problems.push(`${where}: console error: ${m.text()}`));
    await page.goto(url);
    await page.waitForFunction(() => document.body.dataset.ready === "1", null, { timeout: 180000 });
    const stages = await page.locator("#overview .stage").count();
    if (stages < 8) problems.push(`${where}: the overview has ${stages} stages`);
    await page.locator("#overview .stage").nth(4).click();  // a layer: zooms into it
    if (!(await page.textContent("#stage-title")).includes("layer")) problems.push(`${where}: the overview does not zoom`);
    const seen = {};
    for (const [i, [step, selector]] of Object.entries(want).entries()) {
      await page.locator("#walk button").nth(i).click();
      if (step === "layers") {
        await page.locator("#stage-body .head").nth(1).click();
        await page.focus("#stage-body svg[data-key=matrix]");
        await page.keyboard.press("ArrowDown");
        await page.keyboard.press("ArrowRight");
        seen["attention lines"] = await page.locator("#stage-body .lines line").count();
        if (!seen["attention lines"]) problems.push(`${where}: no attention lines`);
      }
      if (step === "map") {
        await page.$eval("#map-layer", (s) => { s.value = "0"; s.dispatchEvent(new Event("input")); });
      }
      seen[step] = await page.locator(selector).count();
      seen[`${step} formulas`] = await page.locator("#math .katex").count();
      if (!seen[step]) problems.push(`${where}: step "${step}" drew nothing (${selector})`);
      if (!seen[`${step} formulas`]) problems.push(`${where}: step "${step}" has no rendered formula`);
      if (await page.locator("#math .katex-error").count()) problems.push(`${where}: a formula in "${step}" did not parse`);
    }
    seen.width = await page.evaluate(() => document.documentElement.scrollWidth);
    console.log(where, JSON.stringify(seen));
    if (seen.width > width) problems.push(`${where}: the page is ${seen.width}px wide`);
    if (shot && url === args[0] && width === 390) {
      await page.locator("#walk button").nth(2).click();
      await page.screenshot({ path: shot, fullPage: true });
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
