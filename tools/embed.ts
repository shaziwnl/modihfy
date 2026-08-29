// Phase 0a: detect and embed every face in a directory of images.
//
// Emits one record per DETECTED FACE, not per file — lund/ contains group shots,
// and we cannot assume the target is the only person in a photo.
//
// Usage: node tools/embed.js [inputDir] [outFile]

import { loadModels, facesInImage, type DetectedFace } from './faces.ts';
import fs from 'node:fs/promises';
import path from 'node:path';

const DECODABLE = new Set(['.jpg', '.jpeg', '.png', '.webp', '.avif', '.cms']);

const inputDir = process.argv[2] ?? 'lund';
const outFile = process.argv[3] ?? 'tools/out/raw-faces.json';

async function main() {
  const names = (await fs.readdir(inputDir))
    .filter((n) => DECODABLE.has(path.extname(n).toLowerCase()))
    .sort((a, b) => a.localeCompare(b, undefined, { numeric: true }));

  console.log(`Embedding ${names.length} images from ${inputDir}/`);
  await loadModels();

  const faces: DetectedFace[] = [];
  const noFace: string[] = [];
  const failed: Array<{ name: string; error: string }> = [];

  for (const name of names) {
    const file = path.join(inputDir, name);
    try {
      const found = await facesInImage(file);
      if (found.length === 0) {
        noFace.push(name);
        console.log(`  ${name.padEnd(16)} no face`);
      } else {
        faces.push(...found);
        const desc = found.map((f) => `${f.faceWidth}px@${f.score}`).join(' ');
        console.log(`  ${name.padEnd(16)} ${found.length} face(s)  ${desc}`);
      }
    } catch (err) {
      failed.push({ name, error: (err as Error).message });
      console.log(`  ${name.padEnd(16)} DECODE FAILED: ${(err as Error).message}`);
    }
  }

  await fs.mkdir(path.dirname(outFile), { recursive: true });
  await fs.writeFile(outFile, JSON.stringify({ inputDir, faces }, null, 2));

  console.log(`\n${faces.length} faces across ${names.length - noFace.length - failed.length} images -> ${outFile}`);
  if (noFace.length) console.log(`No face detected in ${noFace.length}: ${noFace.join(', ')}`);
  if (failed.length) console.log(`Failed to decode ${failed.length}: ${failed.map((f) => f.name).join(', ')}`);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
