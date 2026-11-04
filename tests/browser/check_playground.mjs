// Opens the distillation playground (demo/playground/) in Chromium on blobs, lets it train for a few seconds and
// checks that the teacher and all three students reach well above chance on the test set, that every plane and
// the network view are drawn, that the URL reproduces a run, that hovering and clicking work, and that nothing
// logs an error; at desktop width in light mode and at phone width (no horizontal scroll) in dark mode.
//   node tests/browser/check_playground.mjs http://127.0.0.1:8000/playground/ [screenshot.png]
import { chromium } from "playwright";

const [url, screenshot] = process.argv.slice(2);
const k = 3;
const browser = await chromium.launch();
const problems = [];
for (const [width, scheme] of [[1280, "light"], [390, "dark"]]) {
  const where = `${width}px, ${scheme}`;
  const page = await browser.newPage({ viewport: { width, height: 860 }, colorScheme: scheme });
  page.on("pageerror", (e) => problems.push(`${where}: page error: ${e}`));
  page.on("console", (m) => m.type() === "error" && problems.push(`${where}: console error: ${m.text()}`));
  await page.goto(`${url}?ds=blobs&k=${k}&seed=7`);
  await page.waitForFunction(() => document.body.dataset.ready === "1", null, { timeout: 60000 });
  await page.waitForTimeout(3000);
  const seen = await page.evaluate(() => {
    const acc = (id) => parseFloat(document.getElementById(id).dataset.acc);
    const drawn = (c) => {  // distinct colours in a coarse sample of the canvas
      const ctx = c.getContext("2d"), { data } = ctx.getImageData(0, 0, c.width, c.height), seen = new Set();
      for (let i = 0; i < data.length; i += 4 * 97) seen.add(data.slice(i, i + 4).join());
      return seen.size;
    };
    return {
      teacher: acc("acc-teacher"),
      students: [0, 1, 2].map((i) => acc(`acc-${i}`)),
      epoch: parseInt(document.getElementById("epoch").textContent.replace(/,/g, ""), 10),
      rate: parseFloat(document.getElementById("speed").dataset.rate),
      colours: [...document.querySelectorAll("canvas.plane, #net-canvas")].map(drawn),
      losses: [0, 1, 2].map((i) => document.getElementById(`loss-${i}`).textContent),
      search: location.search,
    };
  });
  console.log(where, JSON.stringify(seen));
  const chance = 1 / k;
  if (!(seen.teacher > chance + 0.3)) problems.push(`${where}: teacher accuracy ${seen.teacher}`);
  seen.students.forEach((a, i) => {
    if (!(a > chance + 0.3)) problems.push(`${where}: student ${i + 1} accuracy ${a} is not well above chance`);
  });
  if (!(seen.epoch > 50)) problems.push(`${where}: only ${seen.epoch} epochs in 3 seconds`);
  seen.colours.forEach((n, i) => { if (n < 4) problems.push(`${where}: canvas ${i} looks blank (${n} colours)`); });
  if (seen.losses.some((l) => !(parseFloat(l) >= 0))) problems.push(`${where}: loss values ${seen.losses}`);
  if (!/seed=7/.test(seen.search) || !/ds=blobs/.test(seen.search)) problems.push(`${where}: URL ${seen.search}`);

  // the URL reproduces a run: the same 30 steps after a reload give the same numbers
  const thirty = async () => {
    await page.waitForFunction(() => document.body.dataset.ready === "1");
    if ((await page.textContent("#play")) === "Pause") await page.click("#play");
    await page.click("#reset");
    for (let i = 0; i < 30; i++) await page.click("#step");
    await page.waitForFunction(() => document.getElementById("epoch").textContent === "30");  // drawn next frame
    return page.evaluate(() => [0, 1, 2].map((i) => document.getElementById(`loss-${i}`).textContent
      + " " + document.getElementById(`acc-${i}`).dataset.acc).join(", ") + " at " + document.getElementById("epoch").textContent);
  };
  const before = await thirty();
  await page.reload();
  const after = await thirty();
  if (before !== after || !before.endsWith(" at 30")) problems.push(`${where}: not reproducible: ${before} / ${after}`);
  await page.click("#play");

  // hover shows the teacher and the three students; a click adds a gold point and keeps it in the URL
  const plane = page.locator("#plane-2");
  await plane.scrollIntoViewIfNeeded();
  await plane.hover({ position: { x: 30, y: 30 } });
  await page.waitForFunction(() => document.querySelectorAll("#readout .bar").length === 4);
  await plane.click({ position: { x: 40, y: 40 } });
  await page.waitForFunction(() => /pts=/.test(location.search) && document.body.dataset.ready === "1");

  const scroll = await page.evaluate(() => document.documentElement.scrollWidth);
  if (scroll > width) problems.push(`${where}: the page is ${scroll}px wide`);
  if (screenshot && width === 1280) await page.screenshot({ path: screenshot });
  await page.close();
}
await browser.close();
if (problems.length) {
  console.error(problems.join("\n"));
  process.exit(1);
}
console.log("playground ok");
