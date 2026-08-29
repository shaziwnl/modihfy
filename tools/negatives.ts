// Phase 0c: build the negative embedding set from LFW (13,233 images / 5,749 identities).
//
// Fetched through HuggingFace's datasets-server rows API rather than the canonical
// UMass tarball (unreachable) or the 188MB parquet mirror (needs a parquet reader).
// The API serves per-row image URLs, so we stream: fetch a page, embed it, discard
// the pixels, keep only the 128-d vectors.
//
// One image per identity — LFW is heavily skewed (530 images of one person), and
// duplicate identities would correlate the false-positive trials and flatter the FPR.
//
// Embeddings are checkpointed each page, so an interrupted run resumes cheaply.
//
// Usage: node tools/negatives.js [numIdentities] [outFile]

import { loadModels, facesInImage, type NegativeFace, type NegativeSet } from './faces.ts';
import fs from 'node:fs/promises';
import path from 'node:path';

const DATASET = 'bitmind/lfw';
const PAGE = 100; // API maximum
const MIN_FACE_PX = 80; // same gate as the target set
const TOTAL_ROWS = 13233;

const target = Number(process.argv[2] ?? 3000);
const outFile = process.argv[3] ?? 'tools/out/negatives.json';

const identityOf = (filename: string): string => filename.replace(/_\d+\.jpg$/i, '');

// The datasets-server 502s intermittently under load. A page that will not come
// back after patient retries is skipped rather than aborting the whole run —
// losing 100 rows out of 13k costs nothing, losing an hour of embedding costs a lot.
interface Row {
  filename: string;
  src: string;
}

async function fetchPage(offset: number): Promise<Row[] | null> {
  const url =
    `https://datasets-server.huggingface.co/rows?dataset=${encodeURIComponent(DATASET)}` +
    `&config=default&split=train&offset=${offset}&length=${PAGE}`;
  for (let attempt = 1; attempt <= 6; attempt++) {
    try {
      const res = await fetch(url);
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const json = (await res.json()) as { rows: Array<{ row: { filename: string; image: { src: string } } }> };
      return json.rows.map((r) => ({ filename: r.row.filename, src: r.row.image.src }));
    } catch (err) {
      if (attempt === 6) {
        process.stdout.write(`\n  page at offset ${offset} failed (${(err as Error).message}), skipping\n`);
        return null;
      }
      await new Promise((r) => setTimeout(r, Math.min(30000, 1000 * 2 ** attempt)));
    }
  }
  // Unreachable: the final attempt always returns. Present so the signature holds.
  return null;
}

async function download(src: string): Promise<Buffer> {
  const res = await fetch(src);
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  return Buffer.from(await res.arrayBuffer());
}

async function main() {
  console.log(`Building negative set: ${target} identities from LFW\n`);
  await loadModels();

  // Resume from a previous run's checkpoint if present.
  let embeddings: NegativeFace[] = [];
  let seen = new Set<string>();
  let startOffset = 0;
  try {
    const prior = JSON.parse(await fs.readFile(outFile, 'utf8')) as NegativeSet;
    embeddings = prior.embeddings ?? [];
    seen = new Set(prior.seenIdentities ?? embeddings.map((e) => e.identity));
    startOffset = prior.nextOffset ?? 0;
    if (embeddings.length) {
      console.log(`Resuming: ${embeddings.length} embeddings, ${seen.size} identities seen, offset ${startOffset}\n`);
    }
  } catch {
    // no checkpoint, start clean
  }

  let scanned = 0;
  let tooSmall = 0;
  let noFace = 0;
  let errors = 0;

  await fs.mkdir(path.dirname(outFile), { recursive: true });

  let offset = startOffset;
  for (; offset < TOTAL_ROWS && embeddings.length < target; offset += PAGE) {
    const rows = await fetchPage(offset);
    if (rows === null) continue;

    // One image per identity, chosen before we spend any inference on them.
    const wanted = rows.filter((r) => {
      const id = identityOf(r.filename);
      if (seen.has(id)) return false;
      seen.add(id);
      return true;
    });

    // Download the page concurrently; inference stays serial below.
    const buffers = await Promise.all(
      wanted.map((r) => download(r.src).catch(() => null)),
    );

    for (let i = 0; i < wanted.length && embeddings.length < target; i++) {
      scanned++;
      const buf = buffers[i];
      if (!buf) {
        errors++;
        continue;
      }
      try {
        const faces = await facesInImage(buf, wanted[i].filename);
        if (faces.length === 0) {
          noFace++;
          continue;
        }
        // Largest face only — LFW crops occasionally catch a bystander at the edge.
        const best = faces.reduce((a, b) => (b.faceWidth > a.faceWidth ? b : a));
        if (best.faceWidth < MIN_FACE_PX) {
          tooSmall++;
          continue;
        }
        embeddings.push({
          identity: identityOf(wanted[i].filename),
          faceWidth: best.faceWidth,
          embedding: best.embedding,
        });
      } catch {
        errors++;
      }
    }

    await fs.writeFile(
      outFile,
      JSON.stringify({
        source: DATASET,
        count: embeddings.length,
        nextOffset: offset + PAGE,
        seenIdentities: [...seen],
        embeddings,
      }),
    );
    process.stdout.write(
      `\r  offset ${offset + PAGE}/${TOTAL_ROWS}  embedded ${embeddings.length}` +
        `  (skipped: ${noFace} no-face, ${tooSmall} small, ${errors} errors)   `,
    );
  }

  console.log(
    `\n\n${embeddings.length} negative embeddings from ${seen.size} identities -> ${outFile}`,
  );
  console.log(`Scanned ${scanned}; skipped ${noFace} no-face, ${tooSmall} under ${MIN_FACE_PX}px, ${errors} errors.`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
