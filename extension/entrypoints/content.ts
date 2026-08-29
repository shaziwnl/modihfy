// Content script: find candidate images, ask for a verdict, swap on a match.
//
// Deliberately thin. It owns no model, fetches no image bytes, and touches no
// canvas — content scripts have been subject to CORS since Chrome 85, so they
// cannot read pixels from cross-origin images anyway. Only a URL string goes out
// and a verdict comes back.
//
// Inference and swapping are driven by two SEPARATE observers:
//
//   prefetch (PREFETCH_MARGIN ahead)  →  decide the verdict
//   viewport (0 margin)               →  apply the swap
//
// Doing both at the viewport edge made the swap feel like a stall, because the
// user was watching inference happen. Deciding early and applying late means the
// answer is already in hand when the image scrolls in, so the flip is immediate
// and reads as a transition rather than lag. Whichever of the two happens second
// triggers the swap, so the ordering does not matter.

import type { CheckRequest, CheckResponse } from '../src/protocol';
import { pickMeme, type Meme } from '../src/pick-meme';

const MIN_DIMENSION = 50; // px; smaller than this is an icon, not a photo
const RESCAN_DEBOUNCE_MS = 250;

// How far ahead of the viewport to start inferring. Big enough to finish before
// the image arrives, small enough that a long page is not inferred all at once.
const PREFETCH_MARGIN = '1500px';

// Deliberate pause between an image becoming visible and the meme landing. The
// verdict is already known by this point, so this is not waiting on work — it is
// there so the reader registers the original first and the swap reads as a beat
// rather than a glitch.
const SWAP_DELAY_MS = 500;

// Log every verdict, not just matches. Without this, "no face found", "scored
// above threshold" and "the offscreen document threw" are all indistinguishable
// from silence, which makes log-only mode useless for diagnosis.
const VERBOSE = true;

interface Swapped {
  originalSrc: string;
  originalSrcset: string;
  memeUrl: string;
}

const decided = new WeakSet<HTMLImageElement>(); // verdict known, match or not
const inFlight = new WeakSet<HTMLImageElement>(); // request outstanding
const inView = new WeakSet<HTMLImageElement>(); // has reached the viewport
const pendingSwap = new WeakMap<HTMLImageElement, Meme>(); // matched, awaiting visibility
const scheduled = new WeakSet<HTMLImageElement>(); // swap timer already running
const swapped = new WeakMap<HTMLImageElement, Swapped>();

let swapEnabled = false; // Phase 1 runs in log-only mode; see architecture.md

/** The URL actually being displayed — `currentSrc` accounts for srcset selection. */
function sourceUrl(img: HTMLImageElement): string | null {
  const url = img.currentSrc || img.src;
  if (!url) return null;
  // blob: URLs belong to the page's own object store and cannot be refetched from
  // an extension context. data: URLs are self-contained and fetch fine.
  if (url.startsWith('blob:')) return null;
  return url;
}

function isCandidate(img: HTMLImageElement): boolean {
  if (decided.has(img) || inFlight.has(img) || swapped.has(img)) return false;
  const rect = img.getBoundingClientRect();
  if (rect.width < MIN_DIMENSION || rect.height < MIN_DIMENSION) return false;
  return sourceUrl(img) !== null;
}

/**
 * Swap when both conditions hold: we know it is a match, and it has reached the
 * viewport. Called from both observers, so whichever completes last does the work.
 */
function maybeSwap(img: HTMLImageElement) {
  if (!swapEnabled || swapped.has(img) || scheduled.has(img) || !inView.has(img)) return;
  const meme = pendingSwap.get(img);
  if (!meme) return;

  // Both observers call this, so the guard matters: without it an image that is
  // already visible when its verdict lands would arm two timers and swap twice.
  scheduled.add(img);
  setTimeout(() => swap(img, meme), SWAP_DELAY_MS);
}

/**
 * Replace the image, then hold the replacement in place.
 *
 * Three things fight this swap:
 *  - `srcset`, which the browser prefers over `src`, so it must be cleared
 *  - a parent `<picture>`, whose `<source>` children outrank the img entirely
 *  - framework re-renders, which reassign `src` from their own state
 */
function swap(img: HTMLImageElement, meme: Meme) {
  const memeUrl = chrome.runtime.getURL(`memes/${meme.file}`);

  swapped.set(img, {
    originalSrc: img.src,
    originalSrcset: img.srcset,
    memeUrl,
  });

  applyMeme(img, memeUrl);

  // Re-apply when a framework re-render reverts us. Scoped to this element and
  // to the two attributes that matter, so it stays cheap.
  const guard = new MutationObserver(() => {
    const state = swapped.get(img);
    if (!state) return;
    if (img.getAttribute('src') !== state.memeUrl || img.srcset) {
      applyMeme(img, state.memeUrl);
    }
  });
  guard.observe(img, { attributes: true, attributeFilter: ['src', 'srcset', 'sizes'] });
}

function applyMeme(img: HTMLImageElement, memeUrl: string) {
  // A <picture>'s <source> elements win over the <img>, so blank them first.
  const picture = img.parentElement;
  if (picture instanceof HTMLPictureElement) {
    for (const source of picture.querySelectorAll('source')) {
      source.removeAttribute('srcset');
      source.removeAttribute('sizes');
    }
  }

  // Order matters: clearing srcset before setting src avoids a flash of the
  // original at a different resolution.
  img.removeAttribute('srcset');
  img.removeAttribute('sizes');
  if (img.getAttribute('src') !== memeUrl) img.setAttribute('src', memeUrl);
}

async function check(img: HTMLImageElement) {
  const url = sourceUrl(img);
  if (!url) return;

  inFlight.add(img);
  try {
    const res: CheckResponse = await chrome.runtime.sendMessage<CheckRequest, CheckResponse>({
      type: 'CHECK_IMAGE',
      url,
    });
    decided.add(img);

    if (!res?.match) {
      if (VERBOSE) {
        const d = res?.distance != null ? ` d=${res.distance.toFixed(3)}` : '';
        const why = res?.error ? ` (${res.error})` : '';
        console.debug(`[modihfy] ${res?.reason ?? 'no-response'}${d}${why}`, url);
      }
      return;
    }

    const meme = pickMeme(url);
    pendingSwap.set(img, meme);
    console.info(
      `[modihfy] match d=${res.distance?.toFixed(3)} -> ${meme.file}`,
      swapEnabled ? '' : '(log-only mode, not swapping)',
      url,
    );
    maybeSwap(img); // no-op until it reaches the viewport
  } catch (err) {
    // Service worker asleep or extension reloading; leave undecided so a later
    // rescan retries.
    if (VERBOSE) console.debug('[modihfy] check failed', url, err);
  } finally {
    inFlight.delete(img);
  }
}

// Constructed inside main(), not at module scope: WXT imports this file in Node
// at build time to read the config below, where browser globals do not exist.
let prefetch: IntersectionObserver;
let viewport: IntersectionObserver;

function scan() {
  for (const img of document.images) {
    if (!decided.has(img) && !inFlight.has(img)) prefetch.observe(img);
    if (!inView.has(img)) viewport.observe(img);
  }
}

let rescanTimer: number | undefined;
function scheduleScan() {
  clearTimeout(rescanTimer);
  rescanTimer = setTimeout(scan, RESCAN_DEBOUNCE_MS) as unknown as number;
}

export default defineContentScript({
  matches: ['<all_urls>'],
  runAt: 'document_idle',

  async main() {
    const stored = await chrome.storage.local.get('swapEnabled');
    swapEnabled = stored.swapEnabled === true;
    console.info(`[modihfy] active (${swapEnabled ? 'swapping' : 'log-only'})`);

    // Decide early, ahead of the viewport, so the answer is ready on arrival.
    // Registering an image costs nothing; only intersecting ones are inferred,
    // so a long page still never infers more than the reader approaches.
    prefetch = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (!entry.isIntersecting) continue;
          const img = entry.target as HTMLImageElement;
          prefetch.unobserve(img);
          if (isCandidate(img)) void check(img);
        }
      },
      { rootMargin: PREFETCH_MARGIN },
    );

    // Apply late, exactly at the viewport edge. Zero margin on purpose: this is
    // what makes the change visible to the reader rather than happening offscreen.
    viewport = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        const img = entry.target as HTMLImageElement;
        viewport.unobserve(img);
        inView.add(img);
        maybeSwap(img);
      }
    });

    scan();

    // Feeds insert images constantly; debounce so a burst costs one scan.
    new MutationObserver(scheduleScan).observe(document.documentElement, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ['src', 'srcset'],
    });
  },
});
