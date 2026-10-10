'use strict';
// Only expose comparisons backed by a completed inference manifest.
(async () => {
  const root = document.getElementById('depth-comparison');
  const base = 'assets/comparisons/depth-dav2/';
  let manifest;
  try {
    const response = await fetch(base + 'manifest.json');
    if (!response.ok) return;
    manifest = await response.json();
    if (!Array.isArray(manifest.samples) || !manifest.samples.length) return;
  } catch (_) { return; }
  const images = [...root.querySelectorAll('[data-compare-image]')];
  const status = document.getElementById('compare-status');
  let scene = 'meeting_room';
  let revision = 0;
  function render() {
    const sample = manifest.samples.find(item => item.scene === scene && item.time_seconds.toFixed(1) === '4.5');
    if (!sample) return;
    const current = ++revision;
    root.setAttribute('aria-busy', 'true');
    status.textContent = 'Loading matched predictions…';
    const loaded = images.map(image => new Promise((resolve, reject) => {
      const pending = new Image();
      pending.onload = () => resolve({image, src: pending.src});
      pending.onerror = reject;
      pending.src = base + sample.id + '-' + image.dataset.compareImage + '.png';
    }));
    Promise.all(loaded).then(results => {
      if (current !== revision) return;
      results.forEach(({image, src}) => { image.src = src; });
      status.textContent = `${scene === 'meeting_room' ? 'Our meeting room' : 'Living room'} · ${sample.time_seconds.toFixed(2)} s · Matched video frame · Existing GeoBridgeNet result · Qualitative comparison`;
      const firstReveal = root.hidden;
      root.hidden = false;
      root.setAttribute('aria-busy', 'false');
      if (firstReveal && location.hash === '#depth-comparison') root.scrollIntoView();
    }).catch(() => {
      if (current === revision) {
        root.setAttribute('aria-busy', 'false');
        status.textContent = 'This comparison could not load. Reload the page to try again.';
      }
    });
  }
  root.querySelectorAll('[data-compare-scene]').forEach(button => {
    button.addEventListener('click', () => {
      scene = button.dataset.compareScene;
      root.querySelectorAll('[data-compare-scene]').forEach(item => item.setAttribute('aria-pressed', String(item === button)));
      render();
    });
  });
  document.getElementById('compare-zoom').addEventListener('change', event => {
    root.classList.toggle('is-zoomed', event.target.checked);
  });
  images.forEach(image => image.parentElement.addEventListener('pointermove', event => {
    const bounds = image.parentElement.getBoundingClientRect();
    const x = Math.max(0, Math.min(100, (event.clientX - bounds.left) / bounds.width * 100));
    const y = Math.max(0, Math.min(100, (event.clientY - bounds.top) / bounds.height * 100));
    root.style.setProperty('--detail-x', `${x}%`);
    root.style.setProperty('--detail-y', `${y}%`);
  }));
  render();
})();
