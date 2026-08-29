// Phase 0b: turn raw per-face detections into a clean target embedding set.
//
// Four filters, in this order:
//   1. size      — drop faces whose crop is too small to embed reliably
//   2. dedup     — collapse near-identical embeddings (catches re-encodes/resizes
//                  that byte-level md5 misses, without perceptual hashing)
//   3. cluster   — keep the largest identity cluster, discarding group-shot
//                  bystanders and any wrong-person photos
//   4. outliers  — drop cluster members far from the median, which would otherwise
//                  widen the match radius at runtime
//
// Usage: node tools/curate.js [rawFile] [outFile]

import { distance } from './faces.js';
import fs from 'node:fs/promises';
import path from 'node:path';

const MIN_FACE_PX = 80; // below this, descriptors get mushy
const DUP_EPS = 0.15; // same photo, re-encoded
const CLUSTER_EPS = 0.55; // same identity
const OUTLIER_EPS = 0.5; // from cluster median

const rawFile = process.argv[2] ?? 'tools/out/raw-faces.json';
const outFile = process.argv[3] ?? 'tools/out/target-set.json';

const label = (f) => `${f.file}[${f.bbox.x},${f.bbox.y} ${f.faceWidth}px]`;

/** Element-wise median — more robust to a stray bad embedding than the mean. */
function medianEmbedding(faces) {
  const dims = faces[0].embedding.length;
  const out = new Array(dims);
  const scratch = new Array(faces.length);
  for (let d = 0; d < dims; d++) {
    for (let i = 0; i < faces.length; i++) scratch[i] = faces[i].embedding[d];
    scratch.sort((a, b) => a - b);
    const mid = scratch.length >> 1;
    out[d] = scratch.length % 2 ? scratch[mid] : (scratch[mid - 1] + scratch[mid]) / 2;
  }
  return out;
}

async function main() {
  const { faces: raw } = JSON.parse(await fs.readFile(rawFile, 'utf8'));
  console.log(`Loaded ${raw.length} detected faces\n`);

  // 1. size filter
  const tooSmall = raw.filter((f) => f.faceWidth < MIN_FACE_PX);
  let faces = raw.filter((f) => f.faceWidth >= MIN_FACE_PX);
  console.log(`1. size >= ${MIN_FACE_PX}px: kept ${faces.length}, dropped ${tooSmall.length}`);
  for (const f of tooSmall) console.log(`     - ${label(f)}`);

  // 2. dedup in embedding space, keeping the largest crop of each duplicate group
  faces.sort((a, b) => b.faceWidth - a.faceWidth);
  const unique = [];
  const dupes = [];
  for (const f of faces) {
    const match = unique.find((u) => distance(u.embedding, f.embedding) < DUP_EPS);
    if (match) dupes.push([f, match]);
    else unique.push(f);
  }
  faces = unique;
  console.log(`\n2. dedup < ${DUP_EPS}: kept ${faces.length}, dropped ${dupes.length}`);
  for (const [f, keptAs] of dupes) console.log(`     - ${label(f)} == ${label(keptAs)}`);

  // 3. largest identity cluster, seeded from the most-connected face (the medoid).
  //    Seeding from the medoid rather than growing connected components avoids
  //    chaining two identities together through a borderline pair.
  const neighbours = faces.map(
    (f) => faces.filter((g) => g !== f && distance(f.embedding, g.embedding) < CLUSTER_EPS).length,
  );
  const seedIdx = neighbours.indexOf(Math.max(...neighbours));
  const seed = faces[seedIdx];
  const inCluster = faces.filter(
    (f) => f === seed || distance(seed.embedding, f.embedding) < CLUSTER_EPS,
  );
  const outCluster = faces.filter((f) => !inCluster.includes(f));
  console.log(
    `\n3. cluster < ${CLUSTER_EPS} around ${label(seed)} (${neighbours[seedIdx]} neighbours):` +
      ` kept ${inCluster.length}, dropped ${outCluster.length}`,
  );
  for (const f of outCluster) {
    console.log(`     - ${label(f)} d=${distance(seed.embedding, f.embedding).toFixed(3)}`);
  }

  // 4. outlier trim against the cluster median
  const median = medianEmbedding(inCluster);
  const scored = inCluster
    .map((f) => ({ ...f, dMedian: distance(median, f.embedding) }))
    .sort((a, b) => a.dMedian - b.dMedian);
  const kept = scored.filter((f) => f.dMedian <= OUTLIER_EPS);
  const outliers = scored.filter((f) => f.dMedian > OUTLIER_EPS);
  console.log(`\n4. outliers <= ${OUTLIER_EPS} from median: kept ${kept.length}, dropped ${outliers.length}`);
  for (const f of outliers) console.log(`     - ${label(f)} d=${f.dMedian.toFixed(3)}`);

  // Spread diagnostics: a set that is too tight will not generalise to hard photos.
  const dists = kept.map((f) => f.dMedian);
  console.log(
    `\nSpread from median: min ${Math.min(...dists).toFixed(3)}` +
      ` / mean ${(dists.reduce((a, b) => a + b, 0) / dists.length).toFixed(3)}` +
      ` / max ${Math.max(...dists).toFixed(3)}`,
  );

  await fs.mkdir(path.dirname(outFile), { recursive: true });
  await fs.writeFile(
    outFile,
    JSON.stringify(
      {
        params: { MIN_FACE_PX, DUP_EPS, CLUSTER_EPS, OUTLIER_EPS },
        count: kept.length,
        faces: kept.map((f) => ({
          file: f.file,
          faceWidth: f.faceWidth,
          dMedian: Number(f.dMedian.toFixed(4)),
          embedding: f.embedding,
        })),
      },
      null,
      2,
    ),
  );

  console.log(`\n${kept.length} embeddings -> ${outFile}`);
  if (kept.length < 15) {
    console.log(
      `\nGATE FAILED: fewer than 15 embeddings survived. The source set is too` +
        ` homogeneous or too low-resolution — add more varied photos (different eras,` +
        ` angles, glasses, lighting) before continuing.`,
    );
    process.exit(2);
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
