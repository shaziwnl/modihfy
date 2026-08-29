// Shared face-detection helpers for the offline calibration harness.
//
// Decoding goes through sharp rather than tfjs-node's decodeImage or node-canvas:
// the source photos include AVIF and WebP, which tf.node.decodeImage cannot read.
// sharp handles every format in lund/ and applies EXIF orientation.

import './node-compat.js'; // must precede tfjs-node
import * as tf from '@tensorflow/tfjs-node';
import sharp from 'sharp';
import faceapi from '@vladmandic/face-api';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const ROOT = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
export const MODEL_DIR = path.join(ROOT, 'models');

// SSD rather than TinyFaceDetector: this runs offline where accuracy beats speed.
const DETECTOR_OPTS = new faceapi.SsdMobilenetv1Options({ minConfidence: 0.5 });

let loaded = false;

export async function loadModels() {
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
export async function decodeToTensor(src) {
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
export async function facesInImage(src, name = null) {
  const tensor = await decodeToTensor(src);
  try {
    const results = await faceapi
      .detectAllFaces(tensor, DETECTOR_OPTS)
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
export function distance(a, b) {
  let sum = 0;
  for (let i = 0; i < a.length; i++) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return Math.sqrt(sum);
}
