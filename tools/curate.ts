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
// Multiple raw files may be passed. Each is curated INDEPENDENTLY and the results
// are merged.
//
// That separation is what makes pose coverage possible. Steps 3 and 4 both assume a
// single tight blob of embeddings, which is exactly what ejects bystanders — but a
// profile of the target sits further from his frontal photos (measured: 0.530) than
// the outlier cutoff allows. Curated as one pile, the pipeline would silently delete
// the profile photos added to fix profiles and report a healthy count.
//
// Loosening the cutoffs is not an alternative: a stranger's frontal face is closer to
// the target's frontal photos (0.434) than the target's own profile is, so any radius
// wide enough to admit his profiles also admits other people.
//
// So the grouping is asserted per folder — the one thing the algorithm cannot infer —
// while clustering still cleans up bystanders within each group.
//
// Usage: node tools/curate.js [rawFile[,rawFile2,...]] [outFile]

import { distance, type CuratedFace, type DetectedFace, type Embedding, type RawFaces } from './faces.ts';
import fs from 'node:fs/promises';
import path from 'node:path';

const MIN_FACE_PX = 80; // below this, descriptors get mushy
const DUP_EPS = 0.15; // same photo, re-encoded
const CLUSTER_EPS = 0.55; // same identity
const OUTLIER_EPS = 0.5; // from cluster median

const rawFiles = (process.argv[2] ?? 'tools/out/raw-faces.json').split(',');
const outFile = process.argv[3] ?? 'tools/out/target-set.json';

const label = (f: DetectedFace): string => `${f.file}[${f.bbox.x},${f.bbox.y} ${f.faceWidth}px]`;

/** Element-wise median — more robust to a stray bad embedding than the mean. */
function medianEmbedding(faces: DetectedFace[]): Embedding {
  const dims = faces[0].embedding.length;
  const out: number[] = new Array(dims);
  const scratch: number[] = new Array(faces.length);
  for (let d = 0; d < dims; d++) {
    for (let i = 0; i < faces.length; i++) scratch[i] = faces[i].embedding[d];
    scratch.sort((a, b) => a - b);
    const mid = scratch.length >> 1;
    out[d] = scratch.length % 2 ? scratch[mid] : (scratch[mid - 1] + scratch[mid]) / 2;
  }
  return out;
}

interface CuratedGroup {
  kept: CuratedFace[];
  median: Embedding;
}

async function curateOne(rawFile: string): Promise<CuratedGroup> {
  const { faces: raw } = JSON.parse(await fs.readFile(rawFile, 'utf8')) as RawFaces;
  console.log(`\n=== ${rawFile}: ${raw.length} detected faces ===\n`);

  // 1. size filter
  const tooSmall = raw.filter((f) => f.faceWidth < MIN_FACE_PX);
  let faces = raw.filter((f) => f.faceWidth >= MIN_FACE_PX);
  console.log(`1. size >= ${MIN_FACE_PX}px: kept ${faces.length}, dropped ${tooSmall.length}`);
  for (const f of tooSmall) console.log(`     - ${label(f)}`);

  // 2. dedup in embedding space, keeping the largest crop of each duplicate group
  faces.sort((a, b) => b.faceWidth - a.faceWidth);
  const unique: DetectedFace[] = [];
  const dupes: Array<[DetectedFace, DetectedFace]> = [];
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

  return { kept, median };
}

async function main() {
  const groups: Array<CuratedGroup & { file: string }> = [];
  for (const f of rawFiles) groups.push({ file: f, ...(await curateOne(f)) });

  // Merge, skipping anything an earlier group already covers.
  const merged: CuratedFace[] = [];
  for (const g of groups) {
    for (const f of g.kept) {
      if (!merged.some((m) => distance(m.embedding, f.embedding) < DUP_EPS)) merged.push(f);
    }
  }

  if (groups.length > 1) {
    console.log(`\n=== merge ===`);
    const base = groups[0];
    for (const g of groups.slice(1)) {
      const spread = distance(base.median, g.median);
      console.log(`  ${g.file}: ${g.kept.length} kept, median ${spread.toFixed(3)} from ${base.file}`);
      // Medians for two poses of one person are legitimately far apart, so this
      // cannot be a hard failure. Past ~0.8 it is more likely two different people,
      // and a mislabelled folder would otherwise poison the set with no other signal.
      if (spread > 0.8) {
        console.log(
          `    WARNING: far enough apart to suspect a different person. Check the` +
            ` folder before trusting the calibration that follows.`,
        );
      }
    }
    console.log(`  merged total: ${merged.length}`);
  }

  await fs.mkdir(path.dirname(outFile), { recursive: true });
  await fs.writeFile(
    outFile,
    JSON.stringify(
      {
        params: { MIN_FACE_PX, DUP_EPS, CLUSTER_EPS, OUTLIER_EPS },
        sources: rawFiles,
        count: merged.length,
        faces: merged.map((f) => ({
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

  console.log(`\n${merged.length} embeddings -> ${outFile}`);
  if (merged.length < 15) {
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
