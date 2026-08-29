# modihfy

A Chrome extension that finds one specific person's face in images on any website
and replaces those images with memes of him.

All inference runs locally in the browser. No API calls, no server, nothing leaves
the machine — which is also what makes it free to run.

## How it works

No model is trained. A pretrained face-embedding network (dlib ResNet-34, via
[face-api](https://github.com/vladmandic/face-api)) turns any face into 128 numbers;
recognising one person is then a nearest-neighbour lookup against vectors computed
once, offline, from photos of him.

The interesting part is not recognition, it is **calibration** — choosing the
distance threshold. The published dlib threshold of 0.6 comes from 1-vs-1
verification on clean frontal faces. This extension does open-set 1-vs-world
matching against thousands of faces per browsing session, so the threshold was
measured against 5,187 stranger identities from LFW instead of assumed.

| | |
|---|---|
| Target embeddings | 41 |
| Negative set | 5,187 LFW identities |
| **Threshold** | **0.445** Euclidean, no margin |
| Recall | 100% (leave-one-out) |
| False positives | 1 in 5,187 |

Re-tested against 400 full-resolution Indian faces, since LFW skews Western and
would otherwise flatter the result: nearest non-target sat at 0.498, giving 0.053
of head-room.

See [`spec/architecture.md`](spec/architecture.md) for the full design and
[`spec/00-first-spec.md`](spec/00-first-spec.md) for requirements and how the
threshold was arrived at.

## Setup

```bash
npm install
cd extension && npm install && cd ..

# Model weights ship inside the face-api package rather than this repo.
mkdir -p models extension/public/models
cp node_modules/@vladmandic/face-api/model/{ssd_mobilenetv1,face_landmark_68,face_recognition}_model* models/
cp models/* extension/public/models/

cd extension && npm run build
```

Then load `extension/.output/chrome-mv3` as an unpacked extension at
`chrome://extensions` with Developer mode on.

Note that `.output` is a dot-folder, so macOS Finder hides it — use <kbd>Cmd</kbd>
+<kbd>Shift</kbd>+<kbd>G</kbd> in the file picker and paste the path.

## Using it

The extension ships in **log-only mode**: it decides but does not touch pages, and
writes every verdict to the console. That is deliberate. A false positive is
invisible once swapping is on — a meme is exactly what you expect to see — so the
only mode where the failure you care about is observable is the one where nothing
changes.

Browse for a while, watch for a match on anyone who isn't the target, then enable
**Replace matched images** from the popup.

The popup also toggles the verdict cache. With it on, each image URL is decided
once ever and repeat visits swap instantly; with it off every image is re-inferred,
so the original stays visible for a beat before the meme lands.

## Offline pipeline

Requires your own photos in `lund/` — the originals are scraped press images and
are not redistributed here. The calibrated output they produced is committed as
`extension/public/targets.json`, so the extension works without them.

```bash
npm run embed       # detect + embed every face, one record per face
npm run curate      # size filter → dedup → largest cluster → outlier trim
npm run negatives   # build the LFW negative set (~45 min, resumable)
npm run calibrate   # sweep thresholds, emit extension/public/targets.json
npm run audit       # re-test the threshold against Indian faces
```

`calibrate.js` guards three traps that each silently flatter the result: the target
appearing inside the negative set (LFW contains him), grading a rule on the same
data that selected its hard negatives, and a margin test where a negative scores
zero against itself.

## Known limitations

- **Side profiles are missed.** dlib's 2017 embedding model represents a profile of
  the target as *further* from his frontal photos than a random stranger is, so no
  threshold catches profiles without admitting false positives first. Fixing this
  needs profile photos in the target set, or a pose-invariant model (ArcFace).
- `<canvas>` elements and CSS `background-image` are not handled — neither can be
  re-fetched by URL.
- The runtime detector must match the one used for calibration. Swapping
  `SsdMobilenetv1` for `TinyFaceDetector` shifts the embeddings and invalidates the
  threshold; it requires re-running `npm run calibrate`.

## Credits

Face detection and embeddings: [@vladmandic/face-api](https://github.com/vladmandic/face-api) (MIT).
Negative set: [Labeled Faces in the Wild](https://vis-www.cs.umass.edu/lfw/).
Built with [WXT](https://wxt.dev).
