// Offscreen document: the only place that touches pixels.
//
// Everything about this file's existence is forced by MV3:
//
//  - A service worker has no DOM, canvas or WebGL, so it cannot run the model.
//  - A content script has been subject to CORS since Chrome 85, so it cannot
//    fetch cross-origin image bytes, and drawing a cross-origin <img> into a
//    canvas taints it. Only an extension-origin context gets the privileged
//    fetch that host_permissions grants.
//  - chrome.runtime.sendMessage is JSON-only, so pixels cannot be shipped
//    between contexts anyway.
//
// Keeping fetch, decode and inference together here means one model instance
// for the whole browser rather than one per tab, and only strings crossing
// process boundaries.

import * as faceapi from '@vladmandic/face-api';

interface TargetArtifact {
  rule: { threshold: number; margin: number; minFacePx: number };
  calibration: Record<string, unknown>;
  target: number[][];
  hardNegatives: number[][];
}

interface Verdict {
  match: boolean;
  distance: number | null;
  reason: 'match' | 'no-face' | 'face-too-small' | 'above-threshold' | 'margin' | 'error';
  error?: string;
}

let artifact: TargetArtifact;
let ready: Promise<void> | null = null;

/**
 * SSD MobileNet v1, not TinyFaceDetector.
 *
 * The threshold in targets.json was calibrated against crops produced by this
 * detector plus the 68-point landmark aligner. A different detector yields
 * slightly different alignment, which shifts the embeddings and silently
 * invalidates the calibration. Swapping in the smaller detector means re-running
 * tools/calibrate.js, not just changing this line.
 */
const DETECTOR = new faceapi.SsdMobilenetv1Options({ minConfidence: 0.5 });

async function init(): Promise<void> {
  // A RELATIVE path, deliberately — not chrome.runtime.getURL('models').
  //
  // face-api's getModelUris() only strips a leading 'http://' or 'https://', then
  // rejoins the path with .split('/').filter(Boolean), which silently collapses the
  // '//' in any other scheme. An absolute 'chrome-extension://<id>/models' comes back
  // out as 'chrome-extension:/<id>/models' — one slash — and that URL is unfetchable,
  // surfacing as a bare "Failed to fetch" rather than a 404.
  //
  // This document is served from the extension root, so 'models' resolves against its
  // own base URL to the right place and face-api never sees a scheme to mangle. The
  // weight .bin files listed in the manifest resolve the same way.
  const modelUri = 'models';
  // Errors from here and from the image fetch are both bare "Failed to fetch",
  // so each is tagged to say which one actually died.
  try {
    await faceapi.nets.ssdMobilenetv1.loadFromUri(modelUri);
    await faceapi.nets.faceLandmark68Net.loadFromUri(modelUri);
    await faceapi.nets.faceRecognitionNet.loadFromUri(modelUri);
    artifact = await fetch(chrome.runtime.getURL('targets.json')).then((r) => r.json());
  } catch (err) {
    throw new Error(`model load failed from ${modelUri}: ${(err as Error)?.message ?? err}`);
  }

  console.info(
    `[modihfy] model ready on ${faceapi.tf.getBackend()};`,
    `${artifact.target.length} target embeddings,`,
    `threshold ${artifact.rule.threshold}, margin ${artifact.rule.margin}`,
  );
}

function ensureReady(): Promise<void> {
  // On failure the promise must be cleared. Caching a rejected promise would make
  // one transient model-load failure permanent for the life of the document, and
  // every subsequent image would fail with the same stale error.
  ready ??= init().catch((err) => {
    ready = null;
    console.error('[modihfy] model init failed', err);
    throw err;
  });
  return ready;
}

function distance(a: Float32Array | number[], b: number[]): number {
  let sum = 0;
  for (let i = 0; i < b.length; i++) {
    const d = a[i] - b[i];
    sum += d * d;
  }
  return Math.sqrt(sum);
}

function minDistance(emb: Float32Array, set: number[][]): number {
  let best = Infinity;
  for (const v of set) {
    const d = distance(emb, v);
    if (d < best) best = d;
  }
  return best;
}

/** Fetch and decode to a canvas. The fetch is privileged; the canvas is untainted. */
async function toCanvas(url: string): Promise<HTMLCanvasElement> {
  let res: Response;
  try {
    res = await fetch(url);
  } catch (err) {
    // Almost always a missing host permission for this origin, or another
    // extension's request blocking.
    throw new Error(`image fetch blocked: ${(err as Error)?.message ?? err}`);
  }
  if (!res.ok) throw new Error(`image fetch HTTP ${res.status}`);
  const bitmap = await createImageBitmap(await res.blob());

  const canvas = document.createElement('canvas');
  canvas.width = bitmap.width;
  canvas.height = bitmap.height;
  canvas.getContext('2d')!.drawImage(bitmap, 0, 0);
  bitmap.close();
  return canvas;
}

async function classify(url: string): Promise<Verdict> {
  await ensureReady();
  const canvas = await toCanvas(url);

  const results = await faceapi
    .detectAllFaces(canvas, DETECTOR)
    .withFaceLandmarks()
    .withFaceDescriptors();

  if (results.length === 0) return { match: false, distance: null, reason: 'no-face' };

  const { threshold, margin, minFacePx } = artifact.rule;
  let best: Verdict = { match: false, distance: null, reason: 'no-face' };

  for (const r of results) {
    // Small crops produce mushy descriptors that sit near everything, so they are
    // rejected before scoring rather than allowed to trip the threshold.
    if (r.detection.box.width < minFacePx) {
      if (best.reason === 'no-face') best = { match: false, distance: null, reason: 'face-too-small' };
      continue;
    }

    const d = minDistance(r.descriptor, artifact.target);
    if (best.distance === null || d < best.distance) {
      best = { match: false, distance: d, reason: 'above-threshold' };
    }
    if (d >= threshold) continue;

    // Margin test: a face closer to some stranger than to the target is that
    // stranger. Skipped when calibration chose margin 0 and bundled no negatives.
    if (margin > 0 && artifact.hardNegatives.length > 0) {
      if (d + margin >= minDistance(r.descriptor, artifact.hardNegatives)) {
        best = { match: false, distance: d, reason: 'margin' };
        continue;
      }
    }

    return { match: true, distance: d, reason: 'match' };
  }

  return best;
}

// Inference is serialised. Concurrent WebGL work on one context thrashes and ends
// up slower than running the same jobs back to back.
let queue: Promise<unknown> = Promise.resolve();
function enqueue<T>(job: () => Promise<T>): Promise<T> {
  const run = queue.then(job, job);
  queue = run.catch(() => {});
  return run;
}

chrome.runtime.onMessage.addListener(
  (msg: { target?: string; type?: string; url?: string }, _sender, sendResponse) => {
    if (msg?.target !== 'offscreen' || msg.type !== 'CHECK_IMAGE' || !msg.url) return false;

    enqueue(() => classify(msg.url!)).then(sendResponse, (err) => {
      console.error('[modihfy] classify failed', msg.url, err);
      sendResponse({
        match: false,
        distance: null,
        reason: 'error',
        error: String(err?.message ?? err),
      } satisfies Verdict);
    });
    return true; // response is async
  },
);
