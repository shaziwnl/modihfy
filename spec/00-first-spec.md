# modihfy — First Spec

## What it is

A Chrome extension that watches the images on any page, recognises one specific
person's face, and replaces those images with memes of him.

## Product requirements

### Functional

| # | Requirement |
|---|---|
| F1 | Detect faces in images on any website the user visits. |
| F2 | Recognise one hardcoded target identity among those faces. |
| F3 | Replace matched images with one of the bundled memes. |
| F4 | The replacement survives scrolling, lazy-loading and framework re-renders. |
| F5 | A single toggle turns replacement on and off. |

### Fixed by the developer, not the user

The target identity and the meme set are baked into the build. There is no
configuration UI, no way to point it at a different person, and no way to add
memes. This is a deliberate product constraint, and it simplifies the whole
design: the target's embeddings are computed once, offline, and shipped as a
static file.

### Non-functional

| # | Requirement | Why |
|---|---|---|
| N1 | **Zero running cost.** No API calls, no server, no hosted inference. | The hard constraint. It forces local, in-browser inference. |
| N2 | **Nothing leaves the browser.** No telemetry, no image uploads. | Follows from N1, and it is the right default for something doing face recognition. |
| N3 | **Almost never matches a stranger.** Target ~1 false positive in 2000 faces. | A meme pasted over a random person's face is the failure that makes the extension feel broken. Missing the target in a hard photo is a shrug. |
| N4 | **No visible jank on image-heavy feeds.** | The extension is always on, across every site. |
| N5 | Layouts must not break when an image is replaced. | |

### Explicitly out of scope for v1

- `<canvas>` elements and CSS `background-image` — neither can be re-fetched by
  URL, and canvas is a materially harder problem.
- Video.
- Any user configuration beyond the on/off toggle.

## Why one-shot matching, not training

No model is trained. A pretrained face embedding network turns any face into a
128-d vector; recognising one person is then just a nearest-neighbour lookup
against vectors precomputed from photos of him. No labelled dataset, no training
loop, no GPU — and it is why N1 is achievable at all.

The interesting engineering problem is not recognition, it is **calibration**:
choosing the distance threshold that satisfies N3 on the open web, where the
extension sees thousands of unrelated faces per session.

## Steps to the first build

1. **Embed the source photos.** Run every image in `lund/` through the detector
   and embedding model. One record per detected face, not per file — some photos
   are group shots.

2. **Curate.** Drop faces too small to embed reliably, collapse duplicates in
   embedding space, keep the largest identity cluster (this ejects bystanders and
   any wrong-person photos automatically), and trim outliers.

3. **Build a negative set.** Embed thousands of strangers from LFW. This is what
   makes N3 a measured number rather than a guess.

4. **Calibrate the threshold.** Score strangers against the target set and pick
   the operating point at the required false-positive rate. Recall is measured
   leave-one-out so no photo is ever matched against itself. Output:
   `extension/public/targets.json`.

   *Gate: if no threshold satisfies N3 with usable recall, switch to a stronger
   embedding model before writing any extension code.*

5. **Audit against the right demographic.** LFW is mostly Western faces; the
   target is not. Re-check the threshold against a South Asian face set.

6. **Scaffold the extension.** WXT project with three entrypoints — content
   script, service worker, offscreen document — plus the models and memes as
   static assets. Verify the weights land as real files at predictable paths.

7. **Wire up inference** in the offscreen document, and run in **log-only mode**:
   verdicts to the console, pages untouched.

   *Gate: browse normally for a day. Any false positive on a stranger means going
   back to step 4.*

8. **Enable the swap.** Develop against a hostile site (Twitter, Reddit) rather
   than a static test page.

9. **Performance pass.** Visible-images-only scanning, debounced rescans, and a
   URL-keyed verdict cache so any image is decided once, ever.

## Where calibration landed

Steps 1–4 are done. The measured rule:

| | |
|---|---|
| Target embeddings | 41 (from 47 source photos) |
| Negative set | 5,187 LFW identities |
| **Threshold** | **0.445** Euclidean, no margin |
| Recall | 100% (leave-one-out over all 41) |
| False positives | 1 in 5,187 — inside the N3 budget of 1 in 2,000 |

Three things were worth the trouble:

- **LFW contains the target.** `Narendra_Modi` is an LFW identity, so his own
  portrait scored as the nearest "stranger" and the sweep was being penalised for
  getting the answer right. Excluded explicitly.
- **The separation is genuinely narrow.** The hardest target photo sits at 0.440
  and the nearest true stranger (`Raja_Qureshi`) at 0.434 — they overlap. Any rule
  with full recall admits that one face. That is the single false positive above,
  and it is why the threshold is 0.445 rather than something more relaxed.
- **The margin rule does not help here.** It appeared to eliminate the false
  positive, but only because negatives that were themselves in the hard-negative
  set scored a distance of zero against it and got rejected for free. With that
  circularity removed the margin earns nothing, so the shipped rule is a plain
  distance threshold with no bundled negatives.

### Demographic audit (step 5)

LFW is mostly Western faces from 2007 news photography, so a false-positive rate
measured on it alone is optimistic for this target. Re-tested against 400
full-resolution Indian faces:

| | |
|---|---|
| Nearest face to the target | 0.498 |
| Threshold | 0.445 |
| Head-room | 0.053 |
| Median distance | 0.615 |

**Pass** — nothing crossed the threshold. Note this set is ordinary people rather
than older bearded politicians specifically, so it is necessary evidence, not
sufficient. (The obvious alternative source, `bollywood_celeb_faces`, ships 64×64
pre-cropped thumbnails; upscaling those would embed a blurry crop and produce a
meaningless answer.)

See [architecture.md](architecture.md) for how the pieces fit together.
