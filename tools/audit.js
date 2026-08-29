// Phase 0d supplement: stress the calibrated threshold against a demographic that
// LFW barely covers.
//
// LFW is ~13k faces scraped from 2007 Western news photos. The target here is an
// Indian public figure, so the strangers most likely to trip a false positive —
// other South Asian faces, and older bearded men in particular — are severely
// under-represented. An FPR measured on LFW alone is therefore optimistic.
//
// This does NOT feed the headline FPR (the source has no identity labels, so trials
// would be correlated). It answers a narrower, more useful question: does any face
// in a South Asian face set land inside the chosen threshold?
//
// Source note: the obvious candidate (bollywood_celeb_faces) ships 64x64 pre-cropped
// thumbnails. Upscaling those to get a detection would embed a blurry crop and
// produce a meaningless answer, so it is unusable here. This dataset carries
// full-resolution Indian faces, including plenty of bearded men.
//
// Still necessary rather than sufficient: the sharpest hard negatives would be
// other Indian politicians specifically.
//
// Usage: node tools/audit.js [n] [dataset]

import { loadModels, facesInImage, distance } from './faces.js';
import fs from 'node:fs/promises';
import path from 'node:path';

const PAGE = 100;
const MIN_FACE_PX = 80;
const DUP_EPS = 0.15; // unlabelled source, so drop near-identical repeats

const n = Number(process.argv[2] ?? 1200);
const dataset = process.argv[3] ?? 'lokesh6309/indian_face-caption';
const outFile = `tools/out/audit-${dataset.split('/').pop()}.json`;

async function fetchPage(offset) {
  const url =
    `https://datasets-server.huggingface.co/rows?dataset=${encodeURIComponent(dataset)}` +
    `&config=default&split=train&offset=${offset}&length=${PAGE}`;
  for (let attempt = 1; attempt <= 6; attempt++) {
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = await res.json();
      return json.rows.map((r, i) => ({ index: offset + i, src: r.row.image.src }));
    } catch (err) {
      if (attempt === 6) return null;
      await new Promise((r) => setTimeout(r, Math.min(30000, 1000 * 2 ** attempt)));
    }
  }
}

const minDistance = (emb, set) => {
  let best = Infinity;
  for (const e of set) {
    const d = distance(emb, e);
    if (d < best) best = d;
  }
  return best;
};

async function main() {
  const artifact = JSON.parse(await fs.readFile('extension/public/targets.json', 'utf8'));
  const { threshold, margin } = artifact.rule;
  const targets = artifact.target;
  console.log(`Auditing ${n} faces from ${dataset}`);
  console.log(`Rule under test: threshold ${threshold}, margin ${margin}\n`);

  await loadModels();

  const kept = [];
  let scanned = 0;
  let noFace = 0;
  let tooSmall = 0;
  let dupes = 0;

  for (let offset = 0; kept.length < n; offset += PAGE) {
    const rows = await fetchPage(offset);
    if (rows === null) break;
    if (rows.length === 0) break;

    const buffers = await Promise.all(
      rows.map((r) => fetch(r.src).then((res) => res.arrayBuffer()).then(Buffer.from).catch(() => null)),
    );

    for (let i = 0; i < rows.length && kept.length < n; i++) {
      scanned++;
      if (!buffers[i]) continue;
      try {
        const faces = await facesInImage(buffers[i], `row-${rows[i].index}`);
        if (!faces.length) {
          noFace++;
          continue;
        }
        const best = faces.reduce((a, b) => (b.faceWidth > a.faceWidth ? b : a));
        if (best.faceWidth < MIN_FACE_PX) {
          tooSmall++;
          continue;
        }
        if (kept.some((k) => distance(k.embedding, best.embedding) < DUP_EPS)) {
          dupes++;
          continue;
        }
        kept.push({ index: rows[i].index, embedding: best.embedding });
      } catch {
        /* skip undecodable rows */
      }
    }
    process.stdout.write(`\r  scanned ${scanned}  kept ${kept.length}  (${noFace} no-face, ${tooSmall} small, ${dupes} dup)   `);
  }

  // Score every audited face against the target set.
  const scored = kept
    .map((k) => ({ index: k.index, d: minDistance(k.embedding, targets) }))
    .sort((a, b) => a.d - b.d);

  const hits = scored.filter((s) => s.d < threshold);

  console.log(`\n\nAudited ${scored.length} South Asian faces.`);
  console.log(`Closest to target: ${scored[0].d.toFixed(3)} (row ${scored[0].index})`);
  console.log(`Nearest five: ${scored.slice(0, 5).map((s) => `${s.d.toFixed(3)}`).join(', ')}`);
  console.log(`Median distance: ${scored[Math.floor(scored.length / 2)].d.toFixed(3)}`);

  if (hits.length === 0) {
    const headroom = scored[0].d - threshold;
    console.log(
      `\nPASS: no face fell inside the ${threshold} threshold.` +
        ` Headroom to the nearest: ${headroom.toFixed(3)}.`,
    );
  } else {
    console.log(
      `\nFAIL: ${hits.length} face(s) fell inside the threshold` +
        ` (rows ${hits.slice(0, 10).map((h) => h.index).join(', ')}).`,
    );
    console.log(
      `The LFW-derived threshold does not survive this demographic. Tighten it to` +
        ` below ${scored[0].d.toFixed(3)}, or fold these embeddings in as hard negatives` +
        ` and re-run calibration with a margin.`,
    );
  }

  await fs.mkdir(path.dirname(outFile), { recursive: true });
  await fs.writeFile(outFile, JSON.stringify({ dataset, count: kept.length, embeddings: kept }));
  console.log(`\nWrote ${outFile}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
