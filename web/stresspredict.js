/**
 * Browser-side inference for the stresspredict model.
 *
 * This is not a re-implementation of the idea -- it evaluates the exact trees
 * the Python pipeline fitted, so the page and `stresspredict.predict` return the
 * same numbers. `tests/test_web_parity.py` asserts that to 1e-9 on every
 * training row by running this file under Node, which is the only reason it is
 * safe to have the same maths written twice.
 *
 * Feature construction mirrors stresspredict/features.py exactly: nine element
 * weight fractions with missing meaning "not deliberately added" (filled 0.0),
 * eight derived metallurgical terms computed AFTER that fill, then one missing
 * indicator per element that had any missing value during training.
 */

export const ELEMENT_FILL = 0.0;

/** Build the 23-column feature row. Order must match model.feature_names. */
export function buildFeatures(model, composition) {
  const els = model.elements;
  const raw = els.map((el) => {
    const v = composition[el];
    return v === undefined || v === null || Number.isNaN(v) ? NaN : Number(v);
  });
  const filled = raw.map((v) => (Number.isNaN(v) ? ELEMENT_FILL : v));

  const g = (name) => filled[els.indexOf(name)];
  const C = g("C"), Mn = g("Mn"), Si = g("Si"), Cr = g("Cr");
  const Ni = g("Ni"), Mo = g("Mo"), V = g("V"), Cu = g("Cu");

  const totalAlloy = filled.reduce((a, b) => a + b, 0);
  const derived = {
    ce_iiw: C + Mn / 6 + (Cr + Mo + V) / 5 + (Ni + Cu) / 15,
    pcm: C + Si / 30 + Mn / 20 + Cu / 20 + Cr / 20 + Ni / 60 + Mo / 15 + V / 10,
    total_alloy: totalAlloy,
    fe_balance: 100.0 - totalAlloy,
    ss_proxy: 83 * Si + 32 * Mn + 33 * Ni + 11 * Cr + 11 * Mo + 39 * Cu,
    carbide_former: Cr + Mo + V,
    cr_eq_schaeffler: Cr + Mo + 1.5 * Si,
    ni_eq_schaeffler: Ni + 30 * C + 0.5 * Mn,
  };

  const byName = new Map();
  els.forEach((el, i) => byName.set(el, filled[i]));
  for (const [k, v] of Object.entries(derived)) byName.set(k, v);
  model.indicator_elements.forEach((el) => {
    byName.set(`missing__${el}`, Number.isNaN(raw[els.indexOf(el)]) ? 1.0 : 0.0);
  });

  return model.feature_names.map((name) => {
    if (!byName.has(name)) throw new Error(`feature ${name} was not built`);
    return byName.get(name);
  });
}

/** Walk one tree to its leaf. Mirrors sklearn: go left when value <= threshold. */
function treeValue(tree, x) {
  let node = 0;
  // Bounded to avoid spinning forever on a malformed export.
  for (let step = 0; step < 4096; step += 1) {
    if (tree.lf[node]) return tree.v[node];
    const value = x[tree.f[node]];
    node = Number.isNaN(value)
      ? (tree.m[node] ? tree.l[node] : tree.r[node])
      : (value <= tree.th[node] ? tree.l[node] : tree.r[node]);
  }
  throw new Error("tree traversal did not reach a leaf");
}

/** Raw (link-space) prediction: baseline plus one leaf per tree. */
export function rawPrediction(component, x) {
  let total = component.baseline;
  for (const tree of component.trees) total += treeValue(tree, x);
  return total;
}

function inverseLink(link, z) {
  if (link === "log") return Math.exp(z);
  if (link === "logit") return 1 / (1 + Math.exp(-z));
  throw new Error(`unknown link: ${link}`);
}

/** Predict a component in its own units (MPa, a ratio, or %). */
export function predictComponent(model, name, x) {
  const component = model.components[name];
  if (!component) throw new Error(`unknown component: ${name}`);
  return inverseLink(component.link, rawPrediction(component, x));
}

/**
 * Landmarks in physical units. UTS > YS holds by construction: the model
 * predicts the yield RATIO through a logistic link, which is strictly below 1.
 * The guard below only matters where the exponential saturates in float64, and
 * moves the value by one representable step rather than clipping a prediction.
 */
export function predictLandmarks(model, composition) {
  const x = buildFeatures(model, composition);
  if (model.parameterisation !== "ratio") {
    throw new Error(`the web demo only implements the ratio parameterisation`);
  }
  const uts = predictComponent(model, "tensile_strength", x);
  const ratio = predictComponent(model, "yield_ratio", x);
  const elongation = predictComponent(model, "elongation", x);

  let ys = uts * ratio;
  if (!(ys < uts)) ys = uts * (1 - Number.EPSILON);

  return { yield_strength: ys, tensile_strength: uts, elongation };
}

/**
 * Which supplied elements sit outside the range the model was trained on.
 * Naming the element and its trained range is far more actionable than a
 * distance score, and engineers trust it more.
 */
export function rangeGuard(model, composition) {
  const flags = [];
  for (const [el, value] of Object.entries(composition)) {
    if (value === undefined || value === null || Number.isNaN(value)) continue;
    const r = model.element_ranges[el];
    if (!r) {
      flags.push({ element: el, value, issue: "never reported in training" });
    } else if (value < r.min || value > r.max) {
      flags.push({
        element: el, value, trained_min: r.min, trained_max: r.max,
        issue: "outside training range",
      });
    }
  }
  return flags;
}

/**
 * Convert landmarks into absolute engineering quantities.
 *
 * This is the whole point of asking for geometry. Stress in MPa is N/mm^2, so a
 * diameter turns it into a force a person can act on; a gauge length turns an
 * elongation percentage into millimetres of travel. Nothing here is predicted --
 * it is arithmetic on the prediction.
 *
 * The caveat that rides along: elongation percentage is only comparable within a
 * gauge-length standard (A5 and A50mm differ for the same material), and
 * SteelBench does not report which standard its rows used. The extension in
 * millimetres therefore assumes the specimen is proportional to whatever
 * standard the training data used, and inherits that uncertainty.
 */
export function toEngineeringUnits(landmarks, geometry) {
  const { diameter_mm, gauge_length_mm } = geometry;
  const area_mm2 = Math.PI * (diameter_mm / 2) ** 2;

  const extension_mm = (landmarks.elongation / 100) * gauge_length_mm;
  return {
    area_mm2,
    yield_force_kN: (landmarks.yield_strength * area_mm2) / 1000,
    breaking_force_kN: (landmarks.tensile_strength * area_mm2) / 1000,
    // Elastic extension at yield, taking E = 205 GPa: essentially constant for
    // all steels, which is why the project does not model it (see docs).
    extension_at_yield_mm: (landmarks.yield_strength / 205000) * gauge_length_mm,
    extension_at_break_mm: extension_mm,
    final_length_mm: gauge_length_mm + extension_mm,
    mass_per_metre_kg: (area_mm2 * 1000 * 7.85) / 1e6,
  };
}

export const YOUNGS_MODULUS_MPA = 205000;

/**
 * A curve through the predicted landmarks, for the animation.
 *
 * IMPORTANT, and stated in the UI as well: this shape is NOT predicted. The
 * model outputs three landmarks; no open dataset found reports strain at UTS or
 * a hardening exponent, so the path between them is a Hollomon power law closed
 * with the Considere condition, constrained to pass through the points the model
 * did predict. It is a model-based reconstruction, not a learned curve.
 */
export function reconstructCurve(landmarks, nPoints = 220) {
  const { yield_strength: ys, tensile_strength: uts, elongation } = landmarks;
  const eps_total = elongation / 100;
  const eps_yield = ys / YOUNGS_MODULUS_MPA;

  // Considere: necking begins where the hardening exponent equals true strain.
  const n = Math.max(0.02, Math.min(0.30, Math.log(uts / ys) / 2 + 0.05));
  const eps_uts = Math.min(Math.max(n, eps_yield * 2), eps_total * 0.75);
  const K = uts / Math.pow(Math.max(eps_uts, 1e-6), n);

  const points = [];
  for (let i = 0; i < nPoints; i += 1) {
    const e = (i / (nPoints - 1)) * eps_total;
    let s;
    if (e <= eps_yield) {
      s = YOUNGS_MODULUS_MPA * e;                       // elastic
    } else if (e <= eps_uts) {
      s = Math.min(K * Math.pow(e, n), uts);            // work hardening
    } else {
      // Post-UTS softening: engineering stress falls as the neck develops.
      const t = (e - eps_uts) / Math.max(eps_total - eps_uts, 1e-9);
      s = uts * (1 - 0.35 * t * t);
    }
    points.push({ strain: e, stress: s });
  }
  return { points, eps_yield, eps_uts, eps_total, hardening_exponent: n };
}
