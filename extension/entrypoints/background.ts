// Service worker: owns the offscreen document's lifecycle and the verdict cache.
//
// It does no inference itself — a service worker has no DOM, no canvas and no
// WebGL. It is a router with a cache in front of it.

import type { CheckRequest, CheckResponse } from '../src/protocol';

const OFFSCREEN_PATH = 'offscreen.html';
const CACHE_PREFIX = 'v:';
const CACHE_LIMIT = 5000;

/**
 * Guards against the race where several tabs ask for a verdict at once: only one
 * offscreen document may exist, and a second createDocument() call throws.
 */
let creating: Promise<void> | null = null;

async function ensureOffscreen(): Promise<void> {
  const contexts = await chrome.runtime.getContexts({
    contextTypes: [chrome.runtime.ContextType.OFFSCREEN_DOCUMENT],
  });
  if (contexts.length > 0) return;

  if (!creating) {
    creating = chrome.offscreen
      .createDocument({
        url: OFFSCREEN_PATH,
        reasons: [chrome.offscreen.Reason.BLOBS],
        justification:
          'Fetches and decodes page images to run local face recognition; needs a DOM canvas, which a service worker lacks.',
      })
      .finally(() => {
        creating = null;
      });
  }
  await creating;
}

// --- Cache toggle --------------------------------------------------------
// Held in memory and refreshed from storage events rather than read per request:
// the service worker restarts freely, so the null state means "not yet loaded"
// and triggers one read.

let cacheEnabled: boolean | null = null;

async function isCacheEnabled(): Promise<boolean> {
  if (cacheEnabled === null) {
    const stored = await chrome.storage.local.get('cacheEnabled');
    cacheEnabled = stored.cacheEnabled !== false; // default on
  }
  return cacheEnabled;
}

chrome.storage.onChanged.addListener((changes, area) => {
  if (area === 'local' && 'cacheEnabled' in changes) {
    cacheEnabled = changes.cacheEnabled.newValue !== false;
  }
});

// --- Verdict cache -------------------------------------------------------
// Keyed by image URL, so any given image is decided once and never re-inferred.
// Most of the web reuses the same asset URLs across pages and sessions, so this
// is the single largest performance win available.

async function cacheGet(url: string): Promise<CheckResponse | undefined> {
  const key = CACHE_PREFIX + url;
  const stored = await chrome.storage.local.get(key);
  return stored[key];
}

async function cacheSet(url: string, verdict: CheckResponse): Promise<void> {
  await chrome.storage.local.set({ [CACHE_PREFIX + url]: verdict });
  void trimCache();
}

let trimming = false;
async function trimCache(): Promise<void> {
  if (trimming) return;
  trimming = true;
  try {
    const all = await chrome.storage.local.get(null);
    const keys = Object.keys(all).filter((k) => k.startsWith(CACHE_PREFIX));
    if (keys.length <= CACHE_LIMIT) return;
    // No access ordering is tracked, so drop an arbitrary slice. The cache is a
    // pure optimisation — evicting the wrong entry costs one re-inference.
    await chrome.storage.local.remove(keys.slice(0, keys.length - CACHE_LIMIT));
  } finally {
    trimming = false;
  }
}

// --- Routing -------------------------------------------------------------

async function handleCheck(url: string): Promise<CheckResponse> {
  const useCache = await isCacheEnabled();
  if (useCache) {
    const cached = await cacheGet(url);
    if (cached) return cached;
  }

  await ensureOffscreen();

  // createDocument() resolving only means the document exists — its module script
  // may not have run yet, so its onMessage listener may not be registered. A
  // message sent into that window rejects with "Could not establish connection",
  // which would otherwise surface as a permanent failure for the first images on
  // the first page load. Retry briefly to let the listener come up.
  let verdict: CheckResponse | undefined;
  for (let attempt = 0; attempt < 5; attempt++) {
    try {
      verdict = await chrome.runtime.sendMessage({
        target: 'offscreen',
        type: 'CHECK_IMAGE',
        url,
      });
      if (verdict) break;
    } catch (err) {
      if (attempt === 4) throw err;
      await new Promise((r) => setTimeout(r, 150 * (attempt + 1)));
      await ensureOffscreen(); // it may have been torn down mid-flight
    }
  }
  if (!verdict) return { match: false, distance: null, reason: 'error', error: 'no response from offscreen' };

  // Errors are transient (network, a torn-down offscreen doc), so they are not
  // cached — a later visit should get another go.
  if (useCache && verdict.reason !== 'error') await cacheSet(url, verdict);
  return verdict;
}

chrome.runtime.onMessage.addListener((msg: CheckRequest & { target?: string }, _sender, sendResponse) => {
  if (msg?.type !== 'CHECK_IMAGE' || msg.target === 'offscreen') return false;

  handleCheck(msg.url).then(sendResponse, (err) =>
    sendResponse({
      match: false,
      distance: null,
      reason: 'error',
      error: String(err?.message ?? err),
    } satisfies CheckResponse),
  );
  return true; // response is async
});

export default defineBackground(() => {
  console.info('[modihfy] service worker ready');
});
