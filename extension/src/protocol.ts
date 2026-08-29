// Messages crossing the content-script / background / offscreen boundaries.
//
// chrome.runtime.sendMessage is JSON-serialized, so nothing binary can travel
// this way — no ArrayBuffers, no ImageBitmaps. That constraint is the reason
// image fetching and decoding live in the offscreen document rather than the
// content script: only a URL string goes out and a verdict comes back.

export interface CheckRequest {
  type: 'CHECK_IMAGE';
  url: string;
}

export interface CheckResponse {
  match: boolean;
  /** Distance to the nearest target embedding; null when no face was usable. */
  distance: number | null;
  reason: 'match' | 'no-face' | 'face-too-small' | 'above-threshold' | 'margin' | 'error';
  /** Populated only when reason is 'error', so failures are diagnosable from the page console. */
  error?: string;
}

export type Request = CheckRequest;
