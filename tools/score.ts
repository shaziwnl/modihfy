// Score individual images against the shipped rule.
//
// Answers "why didn't this image work?" without loading the extension. Mirrors the
// decision in extension/entrypoints/offscreen/main.ts exactly — same detector, same
// size gate, same threshold and margin, read from the same targets.json — so a
// verdict here is the verdict the browser would reach.
//
// The one thing it cannot reproduce is discovery: whether the content script ever
// found the image on the page. An image that scores MATCH here but is not swapped in
// the browser is a discovery problem (hidden element, stylesheet-only background,
// unreachable frame), not a matching one.
//
// Usage:
//   node tools/score.js <file|dir> [...more]
//   node tools/score.js lund-test/
//   node tools/score.js a.jpg b.png --verbose    # every face, not just the best

import { loadModels, facesInImage, minDistance, type TargetArtifact } from './faces.ts';
import fs from 'node:fs/promises';
import path from 'node:path';

const ARTIFACT = 'extension/public/targets.json';
const DECODABLE = new Set(['.jpg', '.jpeg', '.png', '.webp', '.avif', '.cms', '.gif', '.bmp']);

const args = process.argv.slice(2);
const verbose = args.includes('--verbose');
const inputs = args.filter((a) => !a.startsWith('--'));

/** Expand directories one level; keep explicit files as given. */
async function expand(inputs: string[]): Promise<string[]> {
  const files: string[] = [];
  for (const input of inputs) {
    const stat = await fs.stat(input).catch(() => null);
    if (!stat) {
      console.log(`  ${input}: not found`);
      continue;
    }
    if (stat.isDirectory()) {
      const names = (await fs.readdir(input))
        .filter((n) => DECODABLE.has(path.extname(n).toLowerCase()))
        .sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));
      files.push(...names.map((n) => path.join(input, n)));
    } else {
      files.push(input);
    }
  }
  return files;
}

async function main() {
  if (inputs.length === 0) {
    console.log('Usage: node tools/score.js <file|dir> [...] [--verbose]');
    process.exit(1);
  }

  const artifact = JSON.parse(await fs.readFile(ARTIFACT, 'utf8')) as TargetArtifact;
  const { threshold, margin, minFacePx } = artifact.rule;
  const targets = artifact.target;
  const hard = artifact.hardNegatives ?? [];

  const files = await expand(inputs);
  if (files.length === 0) return;

  console.log(
    `Rule: distance < ${threshold}` +
      (margin > 0 ? `, margin ${margin} vs ${hard.length} hard negatives` : ', no margin') +
      `, min face ${minFacePx}px, ${targets.length} target vectors\n`,
  );

  await loadModels();

  let matched = 0;
  let missed = 0;
  let unusable = 0;

  for (const file of files) {
    const name = path.basename(file);
    let faces;
    try {
      faces = await facesInImage(file);
    } catch (err) {
      console.log(`  ${name.padEnd(16)} DECODE FAILED: ${(err as Error).message}`);
      unusable++;
      continue;
    }

    if (faces.length === 0) {
      // Worth distinguishing from a distance miss: no target set can fix this.
      console.log(`  ${name.padEnd(16)}    --        no face detected`);
      unusable++;
      continue;
    }

    // Score every face, then report the best. A group shot should not be judged on
    // whichever face happens to be largest.
    const scored = faces
      .map((f) => ({
        f,
        d: f.faceWidth < minFacePx ? null : minDistance(f.embedding, targets),
      }))
      .sort((a, b) => (a.d ?? Infinity) - (b.d ?? Infinity));

    const shown = verbose ? scored : scored.slice(0, 1);
    for (const { f, d } of shown) {
      const size = `${f.faceWidth}px`.padStart(6);
      const det = `det=${f.score.toFixed(2)}`;

      if (d === null) {
        console.log(`  ${name.padEnd(16)}${size} ${det}   face under ${minFacePx}px, not scored`);
        continue;
      }

      let verdict;
      if (d >= threshold) {
        verdict = `MISS   (needs ${(d - threshold).toFixed(3)} closer)`;
      } else if (margin > 0 && hard.length > 0 && d + margin >= minDistance(f.embedding, hard)) {
        verdict = 'MISS   (rejected by margin — nearer a known stranger)';
      } else {
        verdict = 'MATCH';
      }
      console.log(`  ${name.padEnd(16)}${size} ${det}   d=${d.toFixed(3)}   ${verdict}`);
    }

    const best = scored[0];
    if (best.d === null) unusable++;
    else if (
      best.d < threshold &&
      !(margin > 0 && hard.length > 0 && best.d + margin >= minDistance(best.f.embedding, hard))
    ) {
      matched++;
    } else {
      missed++;
    }
  }

  if (files.length > 1) {
    console.log(`\n${matched} match, ${missed} miss, ${unusable} unusable (no face / too small)`);
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
