// Opens a page written by `tinydistill web` in Chromium at phone width and checks that the model runs in the
// browser and every step of the "Inside the model" view fills in, with no errors.
//   node tests/browser/check_page.mjs http://127.0.0.1:8000/ [screenshot.png]
import { chromium } from "playwright";

const [url, screenshot] = process.argv.slice(2);
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 390, height: 900 } });
const problems = [];
page.on("pageerror", (e) => problems.push(`page error: ${e}`));
page.on("console", (m) => m.type() === "error" && problems.push(`console error: ${m.text()}`));
await page.goto(url);
await page.waitForFunction(
  () => !document.getElementById("go").disabled && document.querySelectorAll("#steps > li.done").length >= 6,
  null,
  { timeout: 180000 },
);
await page.click("#att-layers button >> nth=0");  // another layer's attention
const seen = await page.evaluate(() => ({
  answer: document.getElementById("answer").textContent,
  tokens: document.querySelectorAll("#tokens .chip").length,
  layerRows: document.querySelectorAll("#layers tr").length,
  attention: document.querySelectorAll("#attention .chip").length,
  labels: document.querySelectorAll("#top li").length,
  words: document.querySelectorAll("#words span").length,
  teacher: document.querySelectorAll("#teacher li").length,
  width: document.documentElement.scrollWidth,
}));
console.log(JSON.stringify(seen));
if (seen.answer.startsWith("Loading")) problems.push("the model gave no answer");
if (seen.tokens < 3) problems.push("no tokens shown");
if (seen.layerRows < 3) problems.push("no layer outputs");
if (seen.attention < 3) problems.push("no attention outputs");
if (seen.labels < 2) problems.push("no label probabilities");
if (seen.words < 2) problems.push("no word importance");
if (seen.teacher < 2) problems.push("no teacher soft labels for the first example");
if (seen.width > 390) problems.push(`the page is ${seen.width}px wide on a 390px screen`);
if (screenshot) await page.screenshot({ path: screenshot, fullPage: true });
await browser.close();
if (problems.length) {
  console.error(problems.join("\n"));
  process.exit(1);
}
console.log("page ok");
