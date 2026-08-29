# modihfy — Architecture

Two halves that meet at one file:

- An **offline pipeline** (Node) that turns photos into a calibrated matching rule.
- A **runtime extension** (Chrome MV3) that applies that rule to the live web.

`extension/public/targets.json` is the only thing that crosses between them.

---

## Offline pipeline

```
lund/*.{jpg,jpeg,webp,avif,cms}
        │
        ▼
  tools/embed.js ──────────► tools/out/raw-faces.json
   detect + landmarks           one record per FACE
   + 128-d descriptor           (group shots yield several)
        │
        ▼
  tools/curate.js ─────────► tools/out/target-set.json
   size → dedup → cluster       the target's embeddings only
   → outlier trim
        │
        │        LFW ──► tools/negatives.js ──► tools/out/negatives.json
        │                 one image per identity
        ▼                          │
  tools/calibrate.js ◄─────────────┘
   leave-one-out recall vs. measured stranger FPR
        │
        ▼
  extension/public/targets.json   ← embeddings + threshold + margin
        │
        ▼
  tools/audit.js
   stress the threshold against South Asian faces
```

### Why each curation step exists

| Step | Removes |
|---|---|
| Size filter (≥80px crop) | Faces too small to embed reliably; their descriptors sit near everything. |
| Dedup (<0.15 apart) | Re-encoded and resized copies of the same photo, which would over-weight one image. Catches what byte-level hashing misses. |
| Largest cluster (<0.55) | Bystanders in group shots, and any wrong-person photos — no manual review needed. |
| Outlier trim (>0.5 from median) | Stragglers that would widen the match radius at runtime. |

### Why calibration is the centre of gravity

dlib's well-known 0.6 threshold comes from **1-vs-1 verification** on clean
frontal faces. This extension does **open-set 1-vs-world** matching against
thousands of faces per session, so the only number that matters is the measured
false-positive rate against a large stranger set. That is what `calibrate.js`
produces, and it is the project's gate: if the separation is not there, no amount
of extension code fixes it.

---

## Runtime extension

```
       PAGE                    EXTENSION ORIGIN
  ┌──────────────┐
  │ content.ts   │   URL string
  │              │ ─────────────────►  ┌──────────────────┐
  │ • find <img> │                     │  background.ts   │
  │ • swap src   │ ◄───────────────── │  (service worker)│
  └──────────────┘   {match, distance} │                  │
                                       │ • verdict cache  │
                                       │ • offscreen life │
                                       └────────┬─────────┘
                                                │ URL string
                                                ▼
                                       ┌──────────────────┐
                                       │ offscreen/main.ts│
                                       │                  │
                                       │ • privileged     │
                                       │   fetch          │
                                       │ • decode→canvas  │
                                       │ • detect + embed │
                                       │ • apply rule     │
                                       │ • serial queue   │
                                       └──────────────────┘
                                          targets.json
                                          models/*.bin
```

### Why the offscreen document exists

This is the load-bearing decision, and it is forced three times over:

1. **A service worker cannot run the model.** No DOM, no canvas, no WebGL.

2. **A content script cannot get the pixels.** Content scripts have been subject
   to CORS since Chrome 85, so `host_permissions` does *not* give them a
   privileged fetch — that lives only in extension-origin contexts. And drawing a
   cross-origin `<img>` into a canvas taints it, so the pixels cannot be read
   back either.

3. **The bytes cannot be shipped between contexts.** `chrome.runtime.sendMessage`
   is JSON-serialised; ArrayBuffers and ImageBitmaps do not survive.

Keeping fetch, decode and inference together in the offscreen document means only
**strings out and booleans back**. It also means **one** model instance for the
whole browser instead of ~12 MB of weights and a WebGL context per tab.

The re-fetch is cheaper than it sounds: the page just loaded the same URL, so it
almost always hits the browser HTTP cache.

### Message flow

| From | To | Payload |
|---|---|---|
| content | background | `{ type: 'CHECK_IMAGE', url }` |
| background | offscreen | `{ target: 'offscreen', type: 'CHECK_IMAGE', url }` |
| offscreen | background → content | `{ match, distance, reason }` |

`reason` (`no-face`, `face-too-small`, `above-threshold`, `margin`, `match`,
`error`) exists so log-only mode is diagnosable.

### The matching rule

Applied in `offscreen/main.ts`, parameterised entirely by `targets.json`:

1. Reject face crops under `minFacePx`.
2. `d` = min distance to any target embedding.
3. Match if `d < threshold`.
4. If a margin is configured, additionally require `d + margin < ` the distance to
   the nearest bundled hard negative — a face closer to some stranger than to the
   target is that stranger.

### The swap

Four things fight a replaced image, and each is handled explicitly:

- **`srcset` outranks `src`** → cleared, along with `sizes`.
- **A parent `<picture>` outranks the `<img>`** → its `<source>` elements are blanked.
- **Framework re-renders reassign `src`** → a per-element `MutationObserver`
  re-applies the meme.
- **Aspect mismatch** → the meme is chosen from the bundled set by shape
  similarity (on `log(aspect)`, so too-wide and too-tall are penalised equally),
  then picked deterministically by hashing the image URL so it never flickers
  between memes.

### Performance strategy

| Mechanism | Effect |
|---|---|
| `IntersectionObserver` (200px margin) | Only visible images are ever inferred. |
| Minimum 50×50px | Icons and spacers are skipped. |
| Debounced `MutationObserver` (250ms) | A burst of feed insertions costs one scan. |
| URL-keyed cache in `chrome.storage.local` | Any image URL is decided **once, ever** — across pages and sessions. |
| Serial inference queue | Concurrent WebGL work on one context thrashes. |
| One offscreen document | One model load, not one per tab. |

### MV3 constraints baked into the manifest

- `content_security_policy.extension_pages` includes `'wasm-unsafe-eval'` —
  TensorFlow.js needs it, and without it model loading fails with an opaque error
  rather than a manifest warning.
- `web_accessible_resources` covers `memes/*` only. The content script writes
  those URLs into the page; models and `targets.json` are read by the offscreen
  document, which needs no such declaration.
- Model weights live in `public/` and are loaded via `chrome.runtime.getURL()`.
  They are never `import`ed — that would inline a multi-MB base64 blob or produce
  a hashed filename that cannot be referenced.

### Detector coupling

The runtime uses **SSD MobileNet v1**, the same detector as calibration. Face
descriptors depend on the aligned crop, so switching to the smaller
`TinyFaceDetector` shifts the embeddings and silently invalidates the threshold.
That swap requires re-running `tools/calibrate.js`, not just changing one line.

---

## Repository layout

```
lund/                     source photos of the target
memes/                    replacement images
models/                   face-api weights (offline pipeline)
tools/                    offline pipeline
  faces.js                shared decode + detect + embed
  node-compat.js          Node 26 shim for tfjs-node
  embed.js  curate.js  negatives.js  calibrate.js  audit.js
extension/
  wxt.config.ts           manifest, permissions, CSP
  src/                    protocol types, meme picker + manifest
  entrypoints/
    content.ts            image discovery + swap
    background.ts         routing + verdict cache
    offscreen/            the model, the fetch, the decision
    popup/                the on/off toggle
  public/
    models/               weights, shipped
    memes/                memes, shipped
    targets.json          the calibrated rule
spec/
```
