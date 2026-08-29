import { defineConfig } from 'wxt';

export default defineConfig({
  manifest: {
    name: 'modihfy',

    // PNG only — Chrome rejects webp for extension icons. Generated from logo.webp
    // by cover-cropping to square; the source is 720x685, so almost nothing is lost.
    icons: {
      16: 'icon/16.png',
      32: 'icon/32.png',
      48: 'icon/48.png',
      128: 'icon/128.png',
    },
    description: 'Replaces images of one specific face with memes. All processing is local.',
    version: '0.1.0',

    // storage: the URL-keyed verdict cache.
    // offscreen: the document that owns the model — see architecture.md.
    permissions: ['storage', 'offscreen'],

    // Needed for the privileged fetch of page images. Content scripts have been
    // subject to CORS since Chrome 85, so the fetch happens in the offscreen
    // document (an extension-origin context), which this permission covers.
    host_permissions: ['<all_urls>'],

    // The content script rewrites img.src to these, so the page must be allowed
    // to load them.
    web_accessible_resources: [
      { resources: ['memes/*'], matches: ['<all_urls>'] },
    ],

    // TensorFlow.js needs wasm-unsafe-eval. Without this the offscreen document
    // fails at model load with an opaque CSP error rather than a manifest warning.
    content_security_policy: {
      extension_pages: "script-src 'self' 'wasm-unsafe-eval'; object-src 'self'",
    },
  },
});
