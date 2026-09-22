/**
 * Drives web/app.html in a real browser and reports what breaks.
 *
 * Not a unit test: it clicks, types, drags and resizes the way a person would,
 * including the inputs a person would get wrong. Every check emits a structured
 * finding so the agent reading this can hand the chat an actionable instruction
 * rather than "something looked off".
 *
 *   node tools/uitest/uitest.mjs [--json findings.json] [--headful]
 */

import puppeteer from "puppeteer-core";
import { pathToFileURL } from "node:url";
import { existsSync, writeFileSync, readFileSync, mkdtempSync, copyFileSync } from "node:fs";
import { createHash } from "node:crypto";
import { tmpdir } from "node:os";
import path from "node:path";

/**
 * web/app.html is authored as a FRAGMENT: the artifact runtime wraps it in its
 * own doctype, charset, viewport (viewport-fit=cover) and a small reset, and the
 * publish contract forbids the page from carrying those tags itself. Testing the
 * bare file over file:// therefore lands in quirks mode with no viewport meta and
 * reports layout bugs that do not exist in the published page. So we reproduce
 * the wrapper here and test what is actually served.
 */
const ARTIFACT_SKELETON_HEAD = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<style>
  :root { color-scheme: light; padding-top: env(safe-area-inset-top, 0px);
          padding-bottom: env(safe-area-inset-bottom, 0px); }
  body { margin: 0; font: 14px system-ui, sans-serif; background: #fafaf9; }
  img { max-width: 100%; }
  [hidden] { display: none !important; }
</style>
</head>
<body>
`;

function wrapLikeArtifact(srcHtml, destDir) {
  const body = readFileSync(srcHtml, "utf8");
  const dest = path.join(destDir, "index.html");
  writeFileSync(dest, ARTIFACT_SKELETON_HEAD + body + "\n</body>\n</html>\n");
  // model.js is referenced relatively, so it has to sit beside the page.
  copyFileSync(path.join(path.dirname(srcHtml), "model.js"), path.join(destDir, "model.js"));
  return dest;
}

const CHROME_CANDIDATES = [
  "C:/Program Files/Google/Chrome/Application/chrome.exe",
  "C:/Program Files (x86)/Google/Chrome/Application/chrome.exe",
  "C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe",
  "C:/Program Files/Microsoft/Edge/Application/msedge.exe",
  "/usr/bin/google-chrome",
  "/usr/bin/chromium",
];

const findings = [];
let checksRun = 0;

function report(severity, area, summary, detail, fix) {
  findings.push({ severity, area, summary, detail, fix });
}
function ok(area, note) {
  checksRun += 1;
  if (process.env.UITEST_VERBOSE) console.error(`  ok   ${area}: ${note}`);
}
function fail(severity, area, summary, detail, fix) {
  checksRun += 1;
  report(severity, area, summary, detail, fix);
  console.error(`  ${severity.toUpperCase()} ${area}: ${summary}`);
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

function findBrowser() {
  for (const p of CHROME_CANDIDATES) if (existsSync(p)) return p;
  throw new Error("no Chrome or Edge found; set CHROME_PATH");
}

/* ------------------------------------------------------------------ */

async function newPage(browser, {
  width = 1280, height = 900, reducedMotion = null,
  isMobile = false, hasTouch = false, deviceScaleFactor = 1,
} = {}) {
  const page = await browser.newPage();
  await page.setViewport({ width, height, isMobile, hasTouch, deviceScaleFactor });
  if (reducedMotion) await page.emulateMediaFeatures([
    { name: "prefers-reduced-motion", value: reducedMotion },
  ]);
  const errors = [];
  // Keep the stack. The message alone turned a one-line clock bug into an
  // unexplained flake the first time this harness ran.
  page.on("pageerror", (e) => errors.push(
    String(e.message || e) + (e.stack ? "\n    " + String(e.stack).split("\n").slice(1, 3).join("\n    ") : "")
  ));
  page.on("console", (m) => { if (m.type() === "error") errors.push("console: " + m.text()); });
  page.on("requestfailed", (r) => errors.push(`request failed: ${r.url()} (${r.failure()?.errorText})`));
  page.__errors = errors;
  return page;
}

async function load(page, url) {
  await page.goto(url, { waitUntil: "networkidle2", timeout: 60000 });
  await page.waitForFunction(() => document.querySelectorAll("#els input").length > 0,
    { timeout: 20000 }).catch(() => {});
  await sleep(900);
}

const readResults = (page) => page.evaluate(() => {
  const txt = (id) => (document.getElementById(id)?.textContent || "").trim();
  const num = (id) => {
    const t = txt(id).replace(/[^\d.,-]/g, "").replace(/,/g, "");
    const v = parseFloat(t);
    return Number.isFinite(v) ? v : null;
  };
  return {
    yieldForce: num("r-yf"), breakForce: num("r-bf"),
    stretch: num("r-ext"), finalLength: num("r-len"),
    ys: num("s-ys"), uts: num("s-uts"),
    rawYf: txt("r-yf"), rawBf: txt("r-bf"),
    phase: txt("phase"),
    guards: [...document.querySelectorAll("#guard .flagline")].map((n) => n.textContent.trim()),
    runDisabled: document.getElementById("run")?.disabled,
    pending: document.getElementById("results")?.classList.contains("pending"),
  };
});

async function setComposition(page, comp) {
  await page.evaluate((c) => {
    document.querySelectorAll("#els input").forEach((input) => {
      const el = input.id.replace("el-", "");
      input.value = c[el] === undefined ? "" : String(c[el]);
      input.dispatchEvent(new Event("input", { bubbles: true }));
    });
  }, comp);
  await sleep(160);
}

/* ------------------------------------------------------------------ */

async function checkLoad(page) {
  const s = await page.evaluate(() => ({
    three: typeof THREE !== "undefined",
    model: typeof window.SP_MODEL !== "undefined",
    inputs: document.querySelectorAll("#els input").length,
    presets: document.querySelectorAll("#presets .chip").length,
    canvasPx: (() => { const c = document.getElementById("viewport"); return c ? c.width * c.height : 0; })(),
    curvePx: (() => { const c = document.getElementById("curve"); return c ? c.width * c.height : 0; })(),
    bodyBg: getComputedStyle(document.body).backgroundColor,
    title: document.title,
  }));
  if (!s.three) fail("critical", "load", "THREE failed to load", s, "The CDN script tag did not define THREE. Check the cdnjs URL and the CSP allowlist.");
  else ok("load", "THREE present");
  if (!s.model) fail("critical", "load", "SP_MODEL failed to load", s, "model.js did not define window.SP_MODEL. Confirm it is published as a supporting file alongside the page.");
  else ok("load", "model present");
  if (s.inputs !== 9) fail("critical", "load", `expected 9 element inputs, found ${s.inputs}`, s, "The input grid did not render; the init IIFE probably threw before wiring.");
  else ok("load", "9 inputs rendered");
  if (s.canvasPx === 0) fail("high", "load", "3D canvas has zero backing pixels", s, "renderer.setSize never ran with a non-zero client size.");
  else ok("load", "3D canvas sized");
  if (s.curvePx === 0) fail("high", "load", "curve canvas has zero backing pixels", s, "drawCurve ran before layout; call it after a resize observer or rAF.");
  else ok("load", "curve canvas sized");

  const r = await readResults(page);
  if (r.yieldForce === null || r.breakForce === null) {
    fail("high", "load", "results are empty at rest", r, "The page should open in a working state with the default preset already computed.");
  } else ok("load", "results populated at rest");
  if (page.__errors.length) {
    fail("high", "load", `${page.__errors.length} console/page error(s) on load`, page.__errors.slice(0, 6), "Fix the thrown errors; they can leave the UI half-wired.");
  } else ok("load", "no console errors");
}

async function checkPresets(page) {
  const names = await page.$$eval("#presets .chip", (n) => n.map((x) => x.textContent.trim()));
  for (const name of names) {
    await page.evaluate((n) => {
      [...document.querySelectorAll("#presets .chip")].find((c) => c.textContent.trim() === n)?.click();
    }, name);
    await sleep(220);
    const r = await readResults(page);
    if (r.ys === null || r.uts === null) {
      fail("high", "presets", `preset ${name} produced no numbers`, r, `Clicking ${name} left the readout blank; check apply()/recompute().`);
      continue;
    }
    if (!(r.uts > r.ys)) {
      fail("critical", "presets", `preset ${name} violates UTS > YS`, r, "The parameterisation guarantees this; a violation means the assembly in the page diverged from targets.py.");
      continue;
    }
    if (r.yieldForce <= 0 || r.breakForce <= 0 || r.stretch <= 0) {
      fail("high", "presets", `preset ${name} produced a non-positive result`, r, "Check the engineering() conversion.");
      continue;
    }
    ok("presets", `${name}: ${r.ys} / ${r.uts} MPa`);
  }
}

async function checkPull(page) {
  const before = await readResults(page);
  const t0 = Date.now();
  await page.click("#run");
  await sleep(140);
  const during = await readResults(page);
  if (!during.runDisabled) {
    fail("medium", "pull", "run button is not disabled during the pull", during, "Guard against re-entrancy by disabling the control while the animation runs.");
  } else ok("pull", "button disabled during run");
  if (!during.pending) {
    fail("low", "pull", "results were not withheld during the pull", during, "The brief says the reading is taken at the break; results should grey out until fracture.");
  } else ok("pull", "results withheld during run");

  let settled = false;
  for (let i = 0; i < 60; i += 1) {
    const r = await readResults(page);
    if (r.runDisabled === false) { settled = true; break; }
    await sleep(500);
  }
  const elapsed = Date.now() - t0;
  if (!settled) {
    fail("critical", "pull", `the pull never finished (button still disabled after ${(elapsed / 1000).toFixed(0)}s)`,
      { elapsed }, "The animation is stuck or crawling. The watchdog should force completion; verify it fires and that finish() always runs.");
  } else if (elapsed > 9000) {
    fail("high", "pull", `the pull took ${(elapsed / 1000).toFixed(1)}s (expected about 3s)`,
      { elapsed }, "Frame cost is too high. Check for per-frame allocation in build()/drawCurve().");
  } else {
    ok("pull", `completed in ${(elapsed / 1000).toFixed(1)}s`);
  }

  const after = await readResults(page);
  if (settled && after.pending) {
    fail("high", "pull", "results stayed greyed out after fracture", after, "showResults() must clear the pending class when the break happens.");
  } else if (settled) ok("pull", "results revealed at fracture");
  if (settled && after.phase && !/fract/i.test(after.phase)) {
    fail("low", "pull", `phase reads "${after.phase}" after the run`, after, "The phase label should end at Fractured.");
  }
  if (settled && before.breakForce !== null && after.breakForce !== before.breakForce) {
    fail("medium", "pull", "the break force changed across a run with no input change", { before: before.breakForce, after: after.breakForce }, "Running the test must not alter the prediction; it only replays it.");
  }
}

async function checkAwkwardInputs(page) {
  const cases = [
    { name: "all fields empty", comp: {}, expect: "must not crash; some answer or a clear prompt" },
    { name: "carbon only", comp: { C: 0.4 } },
    { name: "every element zero", comp: { C: 0, Mn: 0, Si: 0, Cr: 0, Ni: 0, Mo: 0, V: 0, Cu: 0, Al: 0 } },
    { name: "negative carbon", comp: { C: -1, Mn: 0.8 }, awkward: true },
    { name: "absurd carbon (99 wt%)", comp: { C: 99 }, awkward: true },
    { name: "elements summing past 100 wt%", comp: { C: 50, Mn: 40, Cr: 30 }, awkward: true },
    { name: "tiny carbon (1e-6)", comp: { C: 0.000001, Mn: 0.5 } },
    { name: "many decimals", comp: { C: 0.123456789012345, Mn: 0.987654321 } },
    { name: "out-of-range carbon (2.5)", comp: { C: 2.5, Mn: 0.8 }, wantGuard: true },
    { name: "high-alloy stainless", comp: { C: 0.02, Cr: 25, Ni: 20, Mo: 6 } },
  ];

  for (const c of cases) {
    await setComposition(page, c.comp);
    const r = await readResults(page);
    const errs = page.__errors.length;

    if (errs) {
      fail("high", "inputs", `"${c.name}" threw`, page.__errors.slice(-3),
        "An input a user can physically type must never throw. Validate or clamp before prediction.");
      page.__errors.length = 0;
      continue;
    }
    const finite = [r.yieldForce, r.breakForce, r.stretch, r.finalLength, r.ys, r.uts];
    if (finite.some((v) => v === null)) {
      fail("medium", "inputs", `"${c.name}" left part of the readout blank`, r,
        "Show something explicit rather than an empty field: either a value or a stated reason there is none.");
      continue;
    }
    if (finite.some((v) => !Number.isFinite(v) || v < 0)) {
      fail("high", "inputs", `"${c.name}" produced a negative or non-finite number`, r,
        "Clamp or reject impossible inputs before they reach the model.");
      continue;
    }
    if (r.uts !== null && r.ys !== null && !(r.uts > r.ys)) {
      fail("critical", "inputs", `"${c.name}" violates UTS > YS`, r,
        "The invariant is supposed to hold by construction for every input.");
      continue;
    }
    if (c.wantGuard && r.guards.length === 0) {
      fail("high", "inputs", `"${c.name}" did not trigger the range guard`, r,
        "Carbon at 2.5 wt% is outside the trained 0.01-2.05 range and must be flagged as extrapolation.");
      continue;
    }
    if (c.awkward && r.guards.length === 0) {
      fail("medium", "inputs", `"${c.name}" was accepted silently`, r,
        "Physically impossible chemistry should be flagged, not answered with a confident number.");
      continue;
    }
    ok("inputs", `${c.name} -> ${r.ys}/${r.uts} MPa, ${r.guards.length} guard(s)`);
  }
}

async function checkInvariantSweep(page) {
  const res = await page.evaluate(() => {
    const inputs = [...document.querySelectorAll("#els input")];
    const out = { n: 0, violations: 0, nonFinite: 0, worst: null };
    const rnd = (a, b) => a + Math.random() * (b - a);
    for (let i = 0; i < 150; i += 1) {
      inputs.forEach((inp) => {
        inp.value = Math.random() < 0.25 ? "" : rnd(0, 30).toFixed(3);
      });
      inputs[0].dispatchEvent(new Event("input", { bubbles: true }));
      const num = (id) => parseFloat((document.getElementById(id).textContent || "")
        .replace(/[^\d.,-]/g, "").replace(/,/g, ""));
      const ys = num("s-ys"), uts = num("s-uts");
      out.n += 1;
      if (!Number.isFinite(ys) || !Number.isFinite(uts)) out.nonFinite += 1;
      else if (!(uts > ys)) { out.violations += 1; out.worst = { ys, uts }; }
    }
    return out;
  });
  if (res.violations > 0) {
    fail("critical", "invariant", `${res.violations}/${res.n} random compositions broke UTS > YS`, res,
      "The page's assembly has diverged from targets.py, where the inequality is algebraic.");
  } else if (res.nonFinite > 0) {
    fail("high", "invariant", `${res.nonFinite}/${res.n} random compositions produced a non-finite readout`, res,
      "Guard the display against NaN even if the model cannot produce one.");
  } else {
    ok("invariant", `UTS > YS held over ${res.n} random compositions`);
  }
}

async function checkOrbit(page) {
  const box = await page.$eval("#viewport", (c) => {
    const r = c.getBoundingClientRect();
    return { x: r.x + r.width / 2, y: r.y + r.height / 2 };
  });
  // Measured by compositor screenshot, NOT canvas.toDataURL(). The renderer is
  // created without preserveDrawingBuffer, so the drawing buffer is cleared
  // after compositing and a canvas readback returns a blank image of constant
  // size -- which made this check report a false failure. Adding
  // preserveDrawingBuffer to the page to satisfy a test would be the wrong fix;
  // it costs real performance.
  const clip = await page.$eval("#viewport", (c) => {
    const r = c.getBoundingClientRect();
    return { x: Math.round(r.x), y: Math.round(r.y), width: Math.round(r.width), height: Math.round(r.height) };
  });
  const shot = async () => createHash("md5").update(await page.screenshot({ clip })).digest("hex");

  const before = await shot();
  await page.mouse.move(box.x, box.y);
  await page.mouse.down();
  await page.mouse.move(box.x + 180, box.y - 70, { steps: 14 });
  await page.mouse.up();
  await sleep(320);
  const after = await shot();
  if (page.__errors.length) {
    fail("high", "orbit", "dragging the 3D view threw", page.__errors.slice(-3), "Check the pointer handlers and setPointerCapture.");
    page.__errors.length = 0;
  } else if (before === after && before !== 0) {
    fail("medium", "orbit", "dragging did not change the rendered view", { before, after },
      "The orbit handler may not be re-rendering; call render() after updating the camera.");
  } else {
    ok("orbit", "drag rotates the view");
  }
}

async function checkResponsive(page, url) {
  // isMobile matters: without it Chrome keeps the desktop layout viewport and
  // the check passes on a layout no phone will ever render.
  const phone = await newPage(page.browser(), {
    width: 390, height: 780, isMobile: true, hasTouch: true, deviceScaleFactor: 3,
  });
  await load(phone, url);
  const m = await phone.evaluate(() => ({
    scrollW: document.documentElement.scrollWidth,
    clientW: document.documentElement.clientWidth,
    runVisible: (() => { const r = document.getElementById("run")?.getBoundingClientRect();
      return !!r && r.width > 40; })(),
    canvasW: document.getElementById("viewport")?.getBoundingClientRect().width,
  }));
  if (m.scrollW > m.clientW + 1) {
    fail("high", "responsive", `horizontal overflow at 390px (${m.scrollW} > ${m.clientW})`, m,
      "Something is wider than the viewport; find the offending element and let it wrap or scroll in its own container.");
  } else ok("responsive", "no horizontal overflow at 390px");
  if (!m.runVisible) fail("medium", "responsive", "run button not usable at phone width", m, "Ensure the control panel is reachable on a phone.");
  else ok("responsive", "controls usable at phone width");
  if (phone.__errors.length) fail("medium", "responsive", "errors at phone width", phone.__errors.slice(0, 4), "Fix errors that only appear at small viewports.");
  await phone.close();
}

async function checkThemes(page, url) {
  for (const scheme of ["dark", "light"]) {
    const p = await newPage(page.browser());
    await p.emulateMediaFeatures([{ name: "prefers-color-scheme", value: scheme }]);
    await load(p, url);
    const c = await p.evaluate(() => {
      const cs = getComputedStyle(document.body);
      const h1 = document.querySelector("h1");
      return { bg: cs.backgroundColor, fg: cs.color, h1: h1 ? getComputedStyle(h1).color : null };
    });
    const parse = (s) => (s.match(/[\d.]+/g) || []).slice(0, 3).map(Number);
    const lum = (rgb) => 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2];
    const bg = parse(c.bg), fg = parse(c.fg);
    if (bg.length < 3 || c.bg === "rgba(0, 0, 0, 0)") {
      fail("high", "theme", `body has no explicit background in ${scheme}`, c,
        "A transparent body borrows the host's ground and can render one theme's text on the other's background.");
    } else if (Math.abs(lum(bg) - lum(fg)) < 60) {
      fail("high", "theme", `body text and background are too close in ${scheme}`, c,
        "Raise the contrast between --ink and --ground for this theme.");
    } else {
      ok("theme", `${scheme}: bg ${c.bg} / fg ${c.fg}`);
    }
    if (p.__errors.length) fail("low", "theme", `errors in ${scheme}`, p.__errors.slice(0, 3), "Investigate theme-specific errors.");
    await p.close();
  }
}

async function checkReducedMotion(page, url) {
  const p = await newPage(page.browser(), { reducedMotion: "reduce" });
  await load(p, url);
  const t0 = Date.now();
  await p.click("#run");
  let settled = false;
  for (let i = 0; i < 24; i += 1) {
    const r = await readResults(p);
    if (r.runDisabled === false) { settled = true; break; }
    await sleep(250);
  }
  const dt = Date.now() - t0;
  if (!settled) {
    fail("high", "reduced-motion", "run never completes with prefers-reduced-motion", { dt },
      "The reduced-motion path should jump straight to the fractured state and re-enable the control.");
  } else {
    const r = await readResults(p);
    if (r.pending) fail("medium", "reduced-motion", "results stayed hidden with reduced motion", r, "Reveal results on the reduced-motion path too.");
    else ok("reduced-motion", `settled in ${dt}ms`);
  }
  if (p.__errors.length) fail("medium", "reduced-motion", "errors with reduced motion", p.__errors.slice(0, 3), "Fix them.");
  await p.close();
}

async function checkReentrancy(page) {
  page.__errors.length = 0;
  await page.click("#run");
  await sleep(90);
  await page.click("#run").catch(() => {});
  await setComposition(page, { C: 0.6, Mn: 1.1 });
  await sleep(120);
  await page.evaluate(() => {
    [...document.querySelectorAll("#presets .chip")][0]?.click();
  });
  let settled = false;
  for (let i = 0; i < 40; i += 1) {
    const r = await readResults(page);
    if (r.runDisabled === false) { settled = true; break; }
    await sleep(400);
  }
  if (!settled) {
    fail("critical", "re-entrancy", "the UI is stuck after interacting during a run", {},
      "Changing inputs or clicking mid-animation left the button permanently disabled. Always route through finish().");
  } else if (page.__errors.length) {
    fail("high", "re-entrancy", "errors when interacting during a run", page.__errors.slice(0, 4),
      "Guard handlers that touch animation state while a run is in flight.");
  } else ok("re-entrancy", "survives input changes mid-run");
}

async function checkFirstClickOnSlowCpu(browser, url) {
  /* The failure a user actually reported: click, and nothing happens. It was
     intermittent at full speed (~1 in 4) and certain under throttling, because
     the bug was a race between performance.now() and the rAF timestamp. Throttle
     deliberately -- testing only the fastest case is how this got shipped. */
  const page = await newPage(browser);
  const cdp = await page.target().createCDPSession();
  await cdp.send("Emulation.setCPUThrottlingRate", { rate: 6 });
  await load(page, url);

  await page.click("#run");
  await sleep(4000);
  const r = await readResults(page);
  if (r.pending || /at rest/i.test(r.phase || "")) {
    fail("critical", "first-click", "the first click did nothing on a throttled CPU",
      { phase: r.phase, pending: r.pending, errors: page.__errors.slice(0, 3) },
      "The run never started. Seed the animation clock from the first rAF timestamp "
      + "(`if (t0 === null) t0 = now;`) rather than performance.now() at click time, "
      + "and clamp k to >= 0.");
  } else {
    let settled = false;
    for (let i = 0; i < 40; i += 1) {
      if ((await readResults(page)).runDisabled === false) { settled = true; break; }
      await sleep(500);
    }
    if (!settled) {
      fail("high", "first-click", "throttled run never settled", {},
        "The watchdog should force completion on slow hardware.");
    } else ok("first-click", "starts and completes under 6x CPU throttling");
  }
  await cdp.send("Emulation.setCPUThrottlingRate", { rate: 1 });
  await page.close();
}

async function checkDegradesWithoutThree(browser, url) {
  /* Corporate networks that block cdnjs, content blockers, offline use and
     blacklisted GPUs all land here. The chemistry prediction needs neither
     three.js nor WebGL, so losing it would be gratuitous. */
  const page = await newPage(browser);
  await page.setRequestInterception(true);
  page.on("request", (req) => (/three/i.test(req.url()) ? req.abort() : req.continue()));
  await load(page, url);

  const r = await readResults(page);
  const filled = await page.$$eval("#els input", (n) => n.filter((i) => i.value !== "").length);
  if (r.yieldForce === null || r.breakForce === null || filled === 0) {
    fail("high", "no-3d", "the page is inert when three.js cannot load",
      { results: r, fieldsFilled: filled },
      "Seed the inputs and results BEFORE initThree(), wrap initThree() in try/catch, "
      + "and make build() a no-op when there is no renderer.");
  } else {
    ok("no-3d", `still predicts without three.js (${r.yieldForce} kN at yield)`);
  }
  await page.close();
}

async function checkLeak(page) {
  const mem = async () => page.evaluate(() => (performance.memory ? performance.memory.usedJSHeapSize : 0));
  const start = await mem();
  for (let i = 0; i < 3; i += 1) {
    await page.click("#run");
    for (let j = 0; j < 40; j += 1) {
      const r = await readResults(page);
      if (r.runDisabled === false) break;
      await sleep(300);
    }
  }
  const end = await mem();
  if (start > 0 && end > start * 2.2 && end - start > 40e6) {
    fail("medium", "memory", `JS heap grew ${(((end - start) / 1e6)).toFixed(0)}MB over 3 runs`,
      { start, end }, "Something allocated per frame is not being released; check geometry and canvas handling.");
  } else ok("memory", `heap ${(start / 1e6).toFixed(0)}MB -> ${(end / 1e6).toFixed(0)}MB over 3 runs`);
}

async function checkA11y(page) {
  const a = await page.evaluate(() => {
    const unlabelled = [...document.querySelectorAll("input")]
      .filter((i) => !i.labels?.length && !i.getAttribute("aria-label")).map((i) => i.id);
    const run = document.getElementById("run");
    return {
      unlabelled,
      runName: run ? run.textContent.trim() : null,
      canvasHasLabel: !!document.getElementById("viewport")?.getAttribute("aria-label"),
      lang: document.documentElement.lang || null,
    };
  });
  if (a.unlabelled.length) {
    fail("low", "a11y", `${a.unlabelled.length} input(s) without a label`, a.unlabelled,
      "Associate a <label for> with every field.");
  } else ok("a11y", "all inputs labelled");
  if (!a.canvasHasLabel) {
    fail("low", "a11y", "the 3D canvas has no accessible name", a,
      "Add aria-label describing the specimen view, since the visual carries meaning.");
  }
}

/* ------------------------------------------------------------------ */

async function main() {
  const args = process.argv.slice(2);
  const jsonAt = args.indexOf("--json");
  const outPath = jsonAt >= 0 ? args[jsonAt + 1] : null;
  const staged = mkdtempSync(path.join(tmpdir(), "sp-uitest-"));
  const url = pathToFileURL(wrapLikeArtifact(path.resolve("web/app.html"), staged)).href;

  const browser = await puppeteer.launch({
    executablePath: process.env.CHROME_PATH || findBrowser(),
    headless: args.includes("--headful") ? false : "new",
    args: ["--allow-file-access-from-files", "--use-angle=swiftshader",
           "--enable-unsafe-swiftshader", "--no-sandbox", "--js-flags=--expose-gc"],
  });

  try {
    const page = await newPage(browser);
    await load(page, url);

    await checkLoad(page);
    await checkPresets(page);
    await checkPull(page);
    page.__errors.length = 0;
    await checkAwkwardInputs(page);
    await checkInvariantSweep(page);
    await checkOrbit(page);
    await checkReentrancy(page);
    await checkLeak(page);
    await checkA11y(page);
    await checkResponsive(page, url);
    await checkThemes(page, url);
    await checkReducedMotion(page, url);
    await checkFirstClickOnSlowCpu(browser, url);
    await checkDegradesWithoutThree(browser, url);
  } finally {
    await browser.close();
  }

  const order = { critical: 0, high: 1, medium: 2, low: 3 };
  findings.sort((a, b) => order[a.severity] - order[b.severity]);
  const result = {
    page: url,
    checks_run: checksRun,
    findings_count: findings.length,
    by_severity: findings.reduce((a, f) => { a[f.severity] = (a[f.severity] || 0) + 1; return a; }, {}),
    findings,
  };
  const text = JSON.stringify(result, null, 2);
  if (outPath) { writeFileSync(outPath, text); console.error(`\nwrote ${outPath}`); }
  console.log(text);
  process.exit(findings.some((f) => f.severity === "critical") ? 1 : 0);
}

main().catch((e) => { console.error("harness failed:", e); process.exit(2); });
