// Shared face-detection helpers for the offline calibration harness.
//
// Decoding goes through sharp rather than tfjs-node's decodeImage or node-canvas:
// the source photos include AVIF and WebP, which tf.node.decodeImage cannot read.
// sharp handles every format in lund/ and applies EXIF orientation.

import './node-compat.ts'; // must precede tfjs-node
import * as tf from '@tensorflow/tfjs-node';
import sharp from 'sharp';
import faceapi from '@vladmandic/face-api';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
export const MODEL_DIR = path.join(ROOT, 'models');

/** A 128-d face descriptor, as plain numbers so it survives JSON round-trips. */
export type Embedding = number[];

export interface BoundingBox {
  x: number;
  y: number;
  width: number;
  height: number;
}

/** One detected face. Images can yield several — group shots are common. */
export interface DetectedFace {
  file: string;
  bbox: BoundingBox;
  faceWidth: number;
  /** Detector confidence, not match confidence. */
  score: number;
  embedding: Embedding;
}

/** What embed.ts writes and curate.ts reads. */
export interface RawFaces {
  inputDir: string;
  faces: DetectedFace[];
}

/** A curated target face, carrying its distance from the cluster median. */
export interface CuratedFace {
  file: string;
  faceWidth: number;
  dMedian: number;
  embedding: Embedding;
}

export interface TargetSet {
  params: Record<string, number>;
  sources?: string[];
  count: number;
  faces: CuratedFace[];
}

/** One stranger. `identity` groups multiple images of the same person. */
export interface NegativeFace {
  identity: string;
  faceWidth?: number;
  embedding: Embedding;
}

export interface NegativeSet {
  source?: string;
  dataset?: string;
  count: number;
  nextOffset?: number;
  seenIdentities?: string[];
  embeddings: NegativeFace[];
}

/** The rule the extension actually applies, and the evidence behind it. */
export interface MatchRule {
  threshold: number;
  margin: number;
  minFacePx: number;
}

export interface TargetArtifact {
  generatedAt: string;
  model: string;
  rule: MatchRule;
  calibration: Record<string, unknown>;
  target: Embedding[];
  hardNegatives: Embedding[];
}

// SSD rather than TinyFaceDetector: this runs offline where accuracy beats speed.
const DETECTOR_OPTS = new faceapi.SsdMobilenetv1Options({ minConfidence: 0.5 });

let loaded = false;

export async function loadModels(): Promise<void> {
  if (loaded) return;
  await faceapi.nets.ssdMobilenetv1.loadFromDisk(MODEL_DIR);
  await faceapi.nets.faceLandmark68Net.loadFromDisk(MODEL_DIR);
  await faceapi.nets.faceRecognitionNet.loadFromDisk(MODEL_DIR);
  loaded = true;
}

/**
 * Decode any image format to an RGB uint8 tensor of shape [h, w, 3].
 * Accepts a file path or a Buffer of encoded image bytes.
 */
export async function decodeToTensor(src: string | Buffer): Promise<tf.Tensor3D> {
  const { data, info } = await sharp(src)
    .rotate() // apply EXIF orientation before we measure anything
    .removeAlpha()
    .toColourspace('srgb')
    .raw()
    .toBuffer({ resolveWithObject: true });
  return tf.tensor3d(new Uint8Array(data), [info.height, info.width, 3]);
}

/**
 * Detect every face in an image and return one record per face.
 * Deliberately not one-per-file: lund/ contains group shots.
 *
 * `src` is a file path or a Buffer; `name` labels the output rows.
 */
export async function facesInImage(
  src: string | Buffer,
  name: string | null = null,
): Promise<DetectedFace[]> {
  const tensor = await decodeToTensor(src);
  try {
    const results = await faceapi
      .detectAllFaces(tensor as never, DETECTOR_OPTS)
      .withFaceLandmarks()
      .withFaceDescriptors();

    return results.map((r) => {
      const b = r.detection.box;
      return {
        file: name ?? (Buffer.isBuffer(src) ? '<buffer>' : path.basename(src)),
        bbox: {
          x: Math.round(b.x),
          y: Math.round(b.y),
          width: Math.round(b.width),
          height: Math.round(b.height),
        },
        faceWidth: Math.round(b.width),
        score: Number(r.detection.score.toFixed(4)),
        embedding: Array.from(r.descriptor),
      };
    });
  } finally {
    tensor.dispose();
  }
}

/** Euclidean distance between two embeddings. */
export function distance(a: ArrayLike<number>, b: ArrayLike<number>): number {
  let sum = 0;
  for (let i = 0; i < b.length; i++) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return Math.sqrt(sum);
}

/** Smallest distance from one embedding to any member of a set. */
export function minDistance(emb: ArrayLike<number>, set: readonly Embedding[]): number {
  let best = Infinity;
  for (const v of set) {
    const d = distance(emb, v);
    if (d < best) best = d;
  }
  return best;
}
