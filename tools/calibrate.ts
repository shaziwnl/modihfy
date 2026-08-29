// Phase 0d: pick the runtime matching rule from measured data.
//
// The stock dlib threshold of 0.6 is wrong for this problem. That number comes from
// LFW 1-vs-1 *verification* on clean frontal faces. Here the extension does open-set
// 1-vs-world matching against thousands of faces per browsing session, so the metric
// that matters is: how often does a random stranger score below the threshold?
//
// Recall is measured leave-one-out — each target face is scored against the OTHER
// target faces, never itself — which is an honest stand-in for "a new photo of him".
//
// Two decision rules are compared:
//   threshold only   — d_target < T
//   threshold+margin — d_target < T AND d_target + M < d_nearest_hard_negative
// The margin is a cheap likelihood-ratio test: a face sitting 0.45 from the target
// but 0.30 from some random person is probably that random person.
//
// The negative set is split in half. The rule is chosen on the FIT half and its
// false-positive rate is reported on the untouched HOLDOUT half, because the margin
// rule selects its own hard negatives and would otherwise be graded on exactly the
// data it was fitted to.

import {
  distance,
  minDistance,
  type Embedding,
  type NegativeFace,
  type NegativeSet,
  type TargetSet,
} from './faces.ts';
import fs from 'node:fs/promises';
import path from 'node:path';

const TARGET_FPR = 1 / 2000; // conservative posture: a meme on a stranger is the worst failure
const HARD_NEGATIVES = 500; // bundled into the extension for the margin test
const MARGINS = [0, 0.02, 0.05, 0.08, 0.12];

// LFW is built from mid-2000s news photography and contains world leaders — the
// target included. Left in, his own LFW portrait scores as the nearest "stranger"
// and the sweep is penalised for getting the answer right, which drags the
// threshold tighter than the evidence warrants.
const EXCLUDE_IDENTITY = /modi/i;

// Extra negative sources may be passed comma-separated. LFW alone is 2007 Western
// press photography, so a threshold calibrated on it is optimistic for a target of
// any other demographic — folding the audit set in measures against the faces that
// actually collide instead of discovering them afterwards.
const negFiles = (process.argv[2] ?? 'tools/out/negatives.json').split(',');
const outFile = process.argv[3] ?? 'extension/public/targets.json';
const targetFile = process.argv[4] ?? 'tools/out/target-set.json';

function histogram(
  values: number[],
  lo: number,
  hi: number,
  bins: number,
  width = 48,
): string {
  const counts: number[] = new Array(bins).fill(0);
  for (const v of values) {
    const i = Math.min(bins - 1, Math.max(0, Math.floor(((v - lo) / (hi - lo)) * bins)));
    counts[i]++;
  }
  const peak = Math.max(...counts);
  return counts
    .map((c, i) => {
      const edge = (lo + ((hi - lo) * i) / bins).toFixed(2);
      return `  ${edge}  ${String(c).padStart(5)} ${'#'.repeat(peak ? Math.round((c / peak) * width) : 0)}`;
    })
    .join('\n');
}

/** Deterministic shuffle so the fit/holdout split is reproducible across runs. */
function shuffled<T>(arr: T[], seed = 42): T[] {
  let s = seed;
  const rnd = () => ((s = (s * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff);
  return arr
    .map((v) => ({ v, k: rnd() }))
    .sort((a, b) => a.k - b.k)
    .map((x) => x.v);
}

/** The negatives nearest the target — the only ones that can ever win a margin test. */
interface HardNegative {
  identity: string;
  embedding: Embedding;
  d: number;
}

function selectHard(negatives: NegativeFace[], targets: Embedding[]): HardNegative[] {
  return negatives
    .map((n) => ({ identity: n.identity, embedding: n.embedding, d: minDistance(n.embedding, targets) }))
    .sort((a, b) => a.d - b.d)
    .slice(0, HARD_NEGATIVES);
}

/**
 * Distance to the nearest hard negative, skipping the subject's own identity.
 *
 * Without the exclusion, any negative that is itself in the hard set scores 0 and
 * the margin test rejects it for free — which is circular, and makes the rule look
 * far better than it is. At runtime a *different* photo of that person arrives, at
 * a real distance from the bundled vector, so the calibration must model that.
 */
function distanceToHard(
  embedding: ArrayLike<number>,
  identity: string | null,
  hard: HardNegative[],
): number {
  let best = Infinity;
  for (const h of hard) {
    if (h.identity === identity) continue;
    const d = distance(embedding, h.embedding);
    if (d < best) best = d;
  }
  return best;
}

async function main() {
  const targetSet = JSON.parse(await fs.readFile(targetFile, 'utf8')) as TargetSet;
  const targets = targetSet.faces.map((f) => f.embedding);

  // Sources differ in shape: the LFW set carries identity labels, the audit sets do
  // not. Synthesise labels for the latter so the same-identity exclusions downstream
  // still behave.
  const loaded: NegativeFace[] = [];
  const sourceNames: string[] = [];
  for (const file of negFiles) {
    const set = JSON.parse(await fs.readFile(file, 'utf8')) as Omit<NegativeSet, 'embeddings'> & {
      embeddings: Array<{ identity?: string; index?: number; embedding: Embedding }>;
    };
    const name = set.source ?? set.dataset ?? file;
    sourceNames.push(`${name} (${set.embeddings.length})`);
    for (const [i, e] of set.embeddings.entries()) {
      loaded.push({ identity: e.identity ?? `${name}#${e.index ?? i}`, embedding: e.embedding });
    }
  }

  const contaminated = loaded.filter((n) => EXCLUDE_IDENTITY.test(n.identity));
  const allNeg = loaded.filter((n) => !EXCLUDE_IDENTITY.test(n.identity));

  console.log(`Target embeddings: ${targets.length}`);
  console.log(`Negative embeddings: ${allNeg.length} from ${sourceNames.join(' + ')}`);
  if (contaminated.length) {
    console.log(
      `Excluded ${contaminated.length} as the target's own identity: ` +
        contaminated.map((c) => c.identity).join(', '),
    );
  }

  const mixed = shuffled(allNeg);
  const cut = mixed.length >> 1;
  const fit = mixed.slice(0, cut);
  const hold = mixed.slice(cut);
  console.log(`Split: ${fit.length} fit / ${hold.length} holdout\n`);

  // --- Distances ----------------------------------------------------------
  const targetLoo = targets.map((t, i) => minDistance(t, targets.filter((_, j) => j !== i)));
  const negD = new Map(allNeg.map((n) => [n, minDistance(n.embedding, targets)]));

  console.log('Target distance to nearest other target photo (recall side):');
  console.log(histogram(targetLoo, 0.2, 0.8, 12));
  console.log('\nStranger distance to nearest target photo (false-positive side):');
  console.log(histogram([...negD.values()], 0.2, 0.8, 12));

  const nearest = allNeg.reduce((a, b) => ((negD.get(b) as number) < (negD.get(a) as number) ? b : a));
  console.log(
    `\nHardest target photo: ${Math.max(...targetLoo).toFixed(3)}` +
      `   nearest stranger: ${(negD.get(nearest) as number).toFixed(3)} (${nearest.identity})`,
  );

  // Hard negatives come from the fit half only, so the holdout stays uncontaminated.
  const hardFit = selectHard(fit, targets);
  const dNegTargetFit = targets.map((t) => distanceToHard(t, null, hardFit));
  const dNegFit = new Map(fit.map((n) => [n, distanceToHard(n.embedding, n.identity, hardFit)]));
  const dNegHold = new Map(hold.map((n) => [n, distanceToHard(n.embedding, n.identity, hardFit)]));

  const passes = (d: number, dNeg: number, T: number, M: number): boolean =>
    d < T && (M === 0 || d + M < dNeg);

  // Every negative was inserted into these maps above, so the lookups cannot miss.
  // Naming that assumption once beats scattering non-null assertions through the sweep.
  const dTarget = (n: NegativeFace): number => negD.get(n) as number;
  const dHardFit = (n: NegativeFace): number => dNegFit.get(n) as number;
  const dHardHold = (n: NegativeFace): number => dNegHold.get(n) as number;

  // --- Sweep --------------------------------------------------------------
  console.log(
    `\nSweep on the fit half (FPR budget ${TARGET_FPR.toExponential(1)} = 1 in ${Math.round(1 / TARGET_FPR)}):\n`,
  );
  console.log('  margin  threshold   recall    fit FPR     holdout FPR   total FP');

  interface Candidate {
    T: number;
    M: number;
    recall: number;
    fp: number;
    fpr: number;
    holdFp: number;
    holdFpr: number;
    totalFp: number;
  }

  const candidates: Candidate[] = [];
  for (const M of MARGINS) {
    let best: Candidate | null = null;
    for (let T = 0.3; T <= 0.75; T += 0.005) {
      const recall =
        targets.filter((_, i) => passes(targetLoo[i], dNegTargetFit[i], T, M)).length / targets.length;
      const fp = fit.filter((n) => passes(dTarget(n), dHardFit(n), T, M)).length;
      const fpr = fp / fit.length;
      if (fpr > TARGET_FPR) continue;
      // Keep the SMALLEST threshold that reaches the best recall. Once recall has
      // saturated, a larger threshold buys nothing and gives away head-room against
      // faces unlike anything in the negative set.
      if (!best || recall > best.recall) {
        best = { T, M, recall, fp, fpr, holdFp: 0, holdFpr: 0, totalFp: 0 };
      }
    }
    if (!best) {
      console.log(`  ${M.toFixed(2)}    —         no threshold meets the FPR budget`);
      continue;
    }
    const chosenBest = best;
    best.holdFp = hold.filter((n) => passes(dTarget(n), dHardHold(n), chosenBest.T, chosenBest.M)).length;
    best.holdFpr = best.holdFp / hold.length;
    candidates.push(best);
    console.log(
      `  ${best.M.toFixed(2)}    ${best.T.toFixed(3)}     ${(best.recall * 100).toFixed(1)}%    ` +
        `${best.fpr.toExponential(2)}    ${best.holdFpr.toExponential(2)}      ${best.fp + best.holdFp}`,
    );
  }

  if (candidates.length === 0) {
    console.log(
      `\nGATE FAILED: no rule meets the FPR budget. Switch to ONNX Runtime Web +` +
        ` InsightFace ArcFace before writing extension code.`,
    );
    process.exit(2);
  }

  // Recall first, then TOTAL false positives across both halves — ranking on the
  // holdout alone stops discriminating once several rules reach zero there, and
  // would hide a rule that still admits a real false positive in the fit half.
  // Then prefer the simpler rule (margin 0 bundles no negatives) and the tighter
  // threshold.
  for (const c of candidates) c.totalFp = c.fp + c.holdFp;
  const chosen = candidates.sort(
    (a, b) => b.recall - a.recall || a.totalFp - b.totalFp || a.M - b.M || a.T - b.T,
  )[0];

  console.log(
    `\nChosen: threshold ${chosen.T.toFixed(3)}, margin ${chosen.M.toFixed(2)}` +
      ` -> recall ${(chosen.recall * 100).toFixed(1)}%, ` +
      (chosen.totalFp > 0
        ? `${chosen.totalFp} false positive(s) in ${allNeg.length} strangers (1 in ${Math.round(allNeg.length / chosen.totalFp)})`
        : `no false positives in ${allNeg.length} strangers`),
  );

  const missed = targetSet.faces.filter((_, i) => !passes(targetLoo[i], dNegTargetFit[i], chosen.T, chosen.M));
  if (missed.length) {
    console.log(`Missed ${missed.length} target photo(s): ${missed.map((m) => m.file).join(', ')}`);
  }

  // --- Emit the runtime artifact -----------------------------------------
  // Hard negatives are re-selected across ALL negatives for shipping: the split
  // existed to keep the evaluation honest, not to throw away half the evidence.
  const hardFinal = chosen.M > 0 ? selectHard(allNeg, targets).map((h) => h.embedding) : [];

  const artifact = {
    generatedAt: new Date().toISOString(),
    model: 'face_recognition_model (dlib ResNet-34, 128-d) via @vladmandic/face-api',
    rule: {
      threshold: Number(chosen.T.toFixed(4)),
      margin: Number(chosen.M.toFixed(4)),
      minFacePx: targetSet.params.MIN_FACE_PX,
    },
    calibration: {
      targetCount: targets.length,
      negativeCount: allNeg.length,
      negativeSources: sourceNames,
      measuredRecall: Number(chosen.recall.toFixed(4)),
      falsePositives: chosen.totalFp,
      holdoutFalsePositives: chosen.holdFp,
      holdoutSize: hold.length,
    },
    target: targets,
    hardNegatives: hardFinal,
  };

  await fs.mkdir(path.dirname(outFile), { recursive: true });
  await fs.writeFile(outFile, JSON.stringify(artifact));
  console.log(
    `\nWrote ${outFile} (${((await fs.stat(outFile)).size / 1024).toFixed(0)} KB,` +
      ` ${hardFinal.length} bundled hard negatives)`,
  );
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
