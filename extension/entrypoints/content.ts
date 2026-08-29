// Content script: find candidate images, ask for a verdict, swap on a match.
//
// Deliberately thin. It owns no model, fetches no image bytes, and touches no
// canvas — content scripts have been subject to CORS since Chrome 85, so they
// cannot read pixels from cross-origin images anyway. Only a URL string goes out
// and a verdict comes back.
//
// Two kinds of target are handled:
//   <img>                          — the obvious case
//   CSS background-image elements  — plugins like imgLiquid read an <img>, move its
//                                    src onto the parent's background-image, and set
//                                    the <img> to display:none. The visible pixels
//                                    then belong to no image element at all, and the
//                                    hidden <img> measures 0x0.
//
// Inference and swapping are driven by two SEPARATE observers:
//   prefetch (PREFETCH_MARGIN ahead)  →  decide the verdict
//   viewport (0 margin)               →  apply the swap
// Deciding early and applying late means the answer is in hand when the element
// scrolls in, so the flip is a transition rather than a stall. Whichever finishes
// second triggers the swap, so ordering does not matter.

import type { CheckRequest, CheckResponse } from '../src/protocol';
import { pickMeme, type Meme } from '../src/pick-meme';

const MIN_DIMENSION = 50; // px; smaller than this is an icon, not a photo
const RESCAN_DEBOUNCE_MS = 250;
const PREFETCH_MARGIN = '1500px';

// Deliberate pause between an element becoming visible and the meme landing. The
// verdict is already known by then, so this is not waiting on work — it is there so
// the reader registers the original first.
const SWAP_DELAY_MS = 250;

const VERBOSE = true;

type Target = HTMLElement;

interface Swapped {
  memeUrl: string;
  isImg: boolean;
}

// Keyed by the URL that was judged, not just the element. Lazy-loaded images are
// often checked while still showing a placeholder; marking the ELEMENT decided meant
// the real image, once loaded, was never looked at again.
const decidedUrl = new WeakMap<Target, string>();
const inFlight = new WeakSet<Target>();
const inView = new WeakSet<Target>();
const pendingSwap = new WeakMap<Target, Meme>();
const scheduled = new WeakSet<Target>();
const swapped = new WeakMap<Target, Swapped>();

let swapEnabled = false;

const isImg = (el: Target): el is HTMLImageElement => el instanceof HTMLImageElement;

/** The first url(...) in an element's background-image, if any. */
function backgroundUrl(el: Target): string | null {
  const raw = el.style.backgroundImage || getComputedStyle(el).backgroundImage;
  if (!raw || raw === 'none') return null;
  const m = /url\(\s*(['"]?)(.*?)\1\s*\)/.exec(raw);
  if (!m?.[2]) return null;
  try {
    return new URL(m[2], document.baseURI).href;
  } catch {
    return null;
  }
}

function sourceUrl(el: Target): string | null {
  const url = isImg(el) ? el.currentSrc || el.src : backgroundUrl(el);
  if (!url) return null;
  // blob: belongs to the page's own object store and cannot be refetched from an
  // extension context. SVGs have no intrinsic raster size and fail to decode.
  if (url.startsWith('blob:')) return null;
  if (/\.svg($|[?#])/i.test(url)) return null;
  return url;
}

function needsCheck(el: Target): boolean {
  if (inFlight.has(el) || swapped.has(el)) return false;
  const url = sourceUrl(el);
  if (url === null) return false;
  if (decidedUrl.get(el) === url) return false;
  const rect = el.getBoundingClientRect();
  return rect.width >= MIN_DIMENSION && rect.height >= MIN_DIMENSION;
}

function maybeSwap(el: Target) {
  if (!swapEnabled || swapped.has(el) || scheduled.has(el) || !inView.has(el)) return;
  const meme = pendingSwap.get(el);
  if (!meme) return;
  // Both observers call this, so without the guard an element already visible when
  // its verdict lands would arm two timers.
  scheduled.add(el);
  setTimeout(() => swap(el, meme), SWAP_DELAY_MS);
}

function swap(el: Target, meme: Meme) {
  const memeUrl = chrome.runtime.getURL(`memes/${meme.file}`);
  swapped.set(el, { memeUrl, isImg: isImg(el) });
  applyMeme(el, memeUrl);

  // Re-apply when a framework re-render or plugin reverts us.
  const guard = new MutationObserver(() => {
    const state = swapped.get(el);
    if (state) applyMeme(el, state.memeUrl);
  });
  guard.observe(el, {
    attributes: true,
    attributeFilter: isImg(el) ? ['src', 'srcset', 'sizes'] : ['style'],
  });
}

function applyMeme(el: Target, memeUrl: string) {
  if (!isImg(el)) {
    // Compare before writing: setting backgroundImage retriggers our own observer.
    if (backgroundUrl(el) !== memeUrl) {
      el.style.setProperty('background-image', `url("${memeUrl}")`, 'important');
    }
    return;
  }

  // A <picture>'s <source> elements win over the <img>, so blank them first.
  const picture = el.parentElement;
  if (picture instanceof HTMLPictureElement) {
    for (const source of picture.querySelectorAll('source')) {
      source.removeAttribute('srcset');
      source.removeAttribute('sizes');
    }
  }
  // Order matters: clearing srcset before setting src avoids a flash of the original.
  el.removeAttribute('srcset');
  el.removeAttribute('sizes');
  if (el.getAttribute('src') !== memeUrl) el.setAttribute('src', memeUrl);
}

async function check(el: Target) {
  const url = sourceUrl(el);
  if (!url) return;

  inFlight.add(el);
  try {
    const res: CheckResponse = await chrome.runtime.sendMessage<CheckRequest, CheckResponse>({
      type: 'CHECK_IMAGE',
      url,
    });
    decidedUrl.set(el, url);

    if (!res?.match) {
      if (VERBOSE) {
        const d = res?.distance != null ? ` d=${res.distance.toFixed(3)}` : '';
        const why = res?.error ? ` (${res.error})` : '';
        console.debug(`[modihfy] ${res?.reason ?? 'no-response'}${d}${why}`, url);
      }
      return;
    }

    const meme = pickMeme(url);
    pendingSwap.set(el, meme);
    console.info(
      `[modihfy] match d=${res.distance?.toFixed(3)} -> ${meme.file}`,
      swapEnabled ? '' : '(log-only mode, not swapping)',
      url,
    );
    maybeSwap(el);
  } catch (err) {
    if (VERBOSE) console.debug('[modihfy] check failed', url, err);
  } finally {
    inFlight.delete(el);
  }
}

// Constructed inside main(): WXT imports this file in Node at build time to read the
// config below, where browser globals do not exist.
let prefetch: IntersectionObserver;
let viewport: IntersectionObserver;

/**
 * Every element worth judging.
 *
 * Background images are found via the inline-style selector rather than by walking
 * the DOM and calling getComputedStyle on everything — that would be far too slow on
 * a large page. It catches the JS-driven cases that matter (imgLiquid, lazysizes and
 * friends all write inline styles); backgrounds set purely in a stylesheet are missed.
 */
function collectTargets(): Target[] {
  return [
    ...document.images,
    ...document.querySelectorAll<HTMLElement>('[style*="background-image"]'),
  ];
}

function scan() {
  for (const el of collectTargets()) {
    if (needsCheck(el)) prefetch.observe(el);
    if (!inView.has(el)) viewport.observe(el);
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

  // Embedded content — tweets, video cards, most social widgets — renders inside a
  // cross-origin iframe with its own document. Without this the script only ever
  // sees the top frame, and every image inside an embed is invisible to it.
  // Each frame runs its own copy, but they all talk to the same offscreen document,
  // so this costs observers rather than model loads.
  allFrames: true,

  async main() {
    const stored = await chrome.storage.local.get('swapEnabled');
    swapEnabled = stored.swapEnabled === true;
    console.info(`[modihfy] active (${swapEnabled ? 'swapping' : 'log-only'})`);

    // Decide early, ahead of the viewport, so the answer is ready on arrival.
    prefetch = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (!entry.isIntersecting) continue;
          const el = entry.target as Target;
          if (needsCheck(el)) {
            prefetch.unobserve(el);
            void check(el);
          }
        }
      },
      { rootMargin: PREFETCH_MARGIN },
    );

    // Apply late, at the viewport edge, so the change is visible to the reader.
    viewport = new IntersectionObserver((entries) => {
      for (const entry of entries) {
        if (!entry.isIntersecting) continue;
        const el = entry.target as Target;
        viewport.unobserve(el);
        inView.add(el);
        maybeSwap(el);
      }
    });

    scan();

    // Feeds insert images constantly; debounce so a burst costs one scan.
    new MutationObserver(scheduleScan).observe(document.documentElement, {
      childList: true,
      subtree: true,
      attributes: true,
      attributeFilter: ['src', 'srcset', 'style'],
    });
  },
});
