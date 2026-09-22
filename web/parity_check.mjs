/**
 * Runs the browser inference over a batch of compositions so Python can diff it
 * against the real pipeline. Having the same maths in two languages is only safe
 * because this check exists; see tests/test_web_parity.py.
 *
 *   node web/parity_check.mjs <model.json> <inputs.json> <outputs.json>
 *
 * inputs.json:  [{"C": 0.4, "Mn": 0.8, ...}, ...]   (absent element = not reported)
 * outputs.json: [{"yield_strength":…, "tensile_strength":…, "elongation":…}, ...]
 */

import { readFileSync, writeFileSync } from "node:fs";
import { predictLandmarks } from "./stresspredict.js";

const [modelPath, inputPath, outputPath] = process.argv.slice(2);
if (!modelPath || !inputPath || !outputPath) {
  console.error("usage: node web/parity_check.mjs <model.json> <inputs.json> <outputs.json>");
  process.exit(2);
}

const model = JSON.parse(readFileSync(modelPath, "utf8"));
const inputs = JSON.parse(readFileSync(inputPath, "utf8"));

const results = inputs.map((composition) => predictLandmarks(model, composition));

writeFileSync(outputPath, JSON.stringify(results));
console.log(`predicted ${results.length} compositions`);
