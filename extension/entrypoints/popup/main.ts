// The only UI. Flips the content script between log-only mode (Phase 1, where
// verdicts go to the console and nothing on the page changes) and swapping.

const swapBox = document.getElementById('swap') as HTMLInputElement;
const hint = document.getElementById('hint') as HTMLParagraphElement;
const cacheBox = document.getElementById('cache') as HTMLInputElement;
const cacheHint = document.getElementById('cacheHint') as HTMLParagraphElement;

function renderSwap(enabled: boolean) {
  swapBox.checked = enabled;
  hint.textContent = enabled
    ? 'Matched images are replaced. Reload a tab to apply.'
    : 'Log-only: verdicts go to the console, pages are untouched.';
}

function renderCache(enabled: boolean) {
  cacheBox.checked = enabled;
  cacheHint.textContent = enabled
    ? 'Each image URL is decided once, ever. Repeat visits swap instantly.'
    : 'Every image is re-inferred, so the original stays visible for a beat before swapping. Slower.';
}

chrome.storage.local.get(['swapEnabled', 'cacheEnabled']).then((s) => {
  renderSwap(s.swapEnabled === true);
  renderCache(s.cacheEnabled !== false); // default on
});

swapBox.addEventListener('change', async () => {
  await chrome.storage.local.set({ swapEnabled: swapBox.checked });
  renderSwap(swapBox.checked);
});

cacheBox.addEventListener('change', async () => {
  await chrome.storage.local.set({ cacheEnabled: cacheBox.checked });
  renderCache(cacheBox.checked);
});
