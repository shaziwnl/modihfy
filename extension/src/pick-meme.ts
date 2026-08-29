// Choosing which meme replaces a given image.
//
// Every meme is equally likely — the pick is a uniform choice across the whole set,
// with no weighting by shape.
//
// It is keyed off the image URL rather than Math.random(), which gives the same
// uniform spread but makes the choice STABLE. That matters: on a single-page app the
// element gets torn down and re-created as you scroll, and a fresh roll each time
// would visibly cycle the meme under the reader. Hashing the URL means one image
// always lands on one meme.
//
// Trade-off accepted here: a 1.81-wide meme can land in a square avatar slot, where
// `object-fit: cover` will crop it to a sliver. Shape-aware selection is in git
// history if that turns out to matter more than an even spread.

import memes from './memes.json' with { type: 'json' };

export interface Meme {
  file: string;
  width: number;
  height: number;
  aspect: number;
}

export const MEMES: Meme[] = memes;

/** FNV-1a, 32-bit. Small, dependency-free, and well spread for short strings. */
function hash(str: string): number {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193);
  }
  return h >>> 0;
}

/** Pick a meme, uniformly across the set, stable for a given image URL. */
export function pickMeme(url: string): Meme {
  return MEMES[hash(url) % MEMES.length];
}
