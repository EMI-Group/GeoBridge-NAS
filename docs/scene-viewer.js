'use strict';
// Local, static Gaussian assets: no model server or external CDN is required.
(() => {
  const loadedScripts = new Map();
  function comparisonPose(asset, queryIndex = 0) {
    if (asset.queries) return window.DDN_SEMANTIC_POSES?.[asset.label]?.[asset.queries[queryIndex].label] || asset.queries[queryIndex].camera;
    const [a, b] = asset.cameras;
    const midpoint = key => a[key].map((value, i) => (value + b[key][i]) / 2);
    return {position: midpoint('position'), target: midpoint('target'), up: midpoint('up'), fov: (a.fov + b.fov) / 2, focus_distance: (a.focus_distance + b.focus_distance) / 2};
  }
  let syncingCameras = false;
  function syncCameras(source) {
    const group = source.element.dataset.syncGroup;
    if (!group || !source.viewer || source.busy || syncingCameras) return;
    syncingCameras = true;
    try {
      for (const target of Object.values(window.DDNViewers || {})) {
        if (target === source || target.element.dataset.syncGroup !== group || !target.viewer || target.busy) continue;
        target.activePose = source.activePose;
        target.viewer.camera.position.copy(source.viewer.camera.position);
        target.viewer.camera.up.copy(source.viewer.camera.up);
        target.viewer.camera.quaternion.copy(source.viewer.camera.quaternion);
        target.viewer.controls.target.copy(source.viewer.controls.target);
        target.viewer.controls.update();
        target.updateFov();
        target.viewer.forceRenderNextFrame();
        if (target.visible && !target.viewer.selfDrivenModeRunning) target.viewer.start();
        target.status.textContent = 'Camera linked · Drag either view to compare.';
        source.status.textContent = 'Camera linked · Drag either view to compare.';
      }
    } finally { syncingCameras = false; }
  }
  function loadScript(src) {
    if (!loadedScripts.has(src)) loadedScripts.set(src, new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = src;
      script.onload = resolve;
      script.onerror = () => { loadedScripts.delete(src); script.remove(); reject(new Error('Scene asset could not be loaded.')); };
      document.head.appendChild(script);
    }));
    return loadedScripts.get(src);
  }
  async function decodeScene(asset) {
    const bytes = Uint8Array.from(atob(asset.base64), char => char.charCodeAt(0));
    const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('gzip'));
    return new Uint8Array(await new Response(stream).arrayBuffer());
  }
  class SceneViewer {
    constructor(element) {
      this.element = element;
      this.host = element.querySelector('.scene-canvas');
      this.status = element.querySelector('.scene-status');
      this.viewer = null;
      this.busy = false;
      this.cameraIndex = 0;
      this.mode = element.dataset.semanticCompare ? 'highlight' : 'rgb';
      this.visible = true;
      this.host.addEventListener('contextmenu', e => e.preventDefault());
      this.host.addEventListener('pointerdown', () => this.host.focus({preventScroll: true}));
      this.host.addEventListener('keydown', e => this.move(e));
      element.querySelector('[data-reset]').addEventListener('click', () => {
        if (!this.asset || this.busy) return;
        this.element.dataset.syncGroup ? this.setPose(comparisonPose(this.asset, this.queryIndex || 0)) : this.setCamera(0);
      });
      element.querySelectorAll('[data-display]').forEach(button => button.addEventListener('click', () => {
        this.mode = button.dataset.display;
        this.paintQuery();
      }));
      element.querySelector('.semantic-queries')?.addEventListener('click', event => {
        const button = event.target.closest('[data-semantic-query]');
        if (button && !this.busy) {
          this.queryIndex = Number(button.dataset.semanticQuery);
          this.setPose(this.element.dataset.semanticCompare ? comparisonPose(this.asset, this.queryIndex) : this.asset.queries[this.queryIndex].camera);
          this.paintQuery();
        }
      });
      element.querySelector('[data-fullscreen]').addEventListener('click', async () => {
        try { if (document.fullscreenElement === element) await document.exitFullscreen(); else await element.requestFullscreen(); } catch (_) { this.status.textContent = 'Full screen is unavailable. You can still explore the scene here.'; }
      });
      new ResizeObserver(() => this.updateFov()).observe(this.host);
      this.observer = new IntersectionObserver(entries => {
        this.visible = entries[0].isIntersecting;
        if (this.viewer && !this.busy) this.visible ? this.viewer.start() : this.viewer.stop();
      });
      this.observer.observe(element);
      document.addEventListener('visibilitychange', () => {
        if (this.viewer && !this.busy) document.hidden ? this.viewer.stop() : (this.visible && this.viewer.start());
      });
    }
    setBusy(busy, message = '') {
      this.busy = busy;
      this.element.setAttribute('aria-busy', String(busy));
      this.element.querySelectorAll('button').forEach(b => b.disabled = busy);
      this.status.textContent = message;
      this.element.classList.toggle('is-loading', busy);
    }
    async load(name) {
      if (this.busy) return;
      this.setBusy(true, 'Loading interactive 3D scene…');
      try {
        await Promise.all([loadScript('assets/vendor/gaussian-viewer.js'), loadScript(`assets/scenes3d/${name}.js`)]);
        this.asset = window.DDN_SCENES[name];
        this.raw = await decodeScene(this.asset);
        this.name = name;
        await this.build(this.raw);
        if (this.asset.queries) this.prepareQueries();
        this.element.querySelector('.scene-label').textContent = this.asset.label;
        this.element.dataset.loadedScene = name;
        this.setBusy(false, 'Drag to explore the 3D scene.');
        if (this.element.dataset.syncGroup) {
          const peer = Object.values(window.DDNViewers).find(item => item !== this && item.element.dataset.syncGroup === this.element.dataset.syncGroup && item.viewer && !item.busy);
          if (peer) {
            syncCameras(peer);
            if (this.asset.queries && peer.asset?.queries) {
              this.queryIndex = peer.queryIndex;
              this.mode = peer.mode;
            }
          }
        }
        if (this.asset.queries) this.paintQuery();
        delete this.element.dataset.error;
      } catch (error) {
        this.setBusy(false, 'This browser could not open the 3D scene. See the GeoBridgeNet-A3 results below.');
        this.element.dataset.error = error.message;
        console.error(error);
      }
    }
    async build(bytes) {
      const { GS } = window.DDN3D;
      this.host.replaceChildren();
      const initial = this.element.dataset.syncGroup ? comparisonPose(this.asset) : this.asset.cameras[0];
      this.activePose = initial;
      this.viewer = new GS.Viewer({
        rootElement: this.host,
        cameraUp: initial.up,
        initialCameraPosition: initial.position,
        initialCameraLookAt: initial.target,
        sharedMemoryForWorkers: false,
        gpuAcceleratedSort: false,
        integerBasedSort: false,
        ignoreDevicePixelRatio: true,
        renderMode: GS.RenderMode.OnChange,
        sceneRevealMode: GS.SceneRevealMode.Instant,
        sphericalHarmonicsDegree: 0,
        inMemoryCompressionLevel: 0,
        optimizeSplatData: false,
        antialiased: false,
        kernel2DSize: this.kernelSize(),
        showLoadingUI: false
      });
      // Restrict keyboard capture to the focused viewer and omit upstream debug shortcuts.
      this.viewer.removeEventHandlers();
      this.viewer.controls.stopListenToKeyEvents();
      this.viewer.controls.listenToKeyEvents(this.host);
      this.viewer.controls.enableDamping = false;
      this.viewer.controls.addEventListener('change', () => syncCameras(this));
      // Defer until OrbitControls has applied the pointer/wheel update.
      for (const type of ['pointermove', 'pointerup', 'wheel']) {
        this.host.addEventListener(type, () => requestAnimationFrame(() => syncCameras(this)), {passive: true});
      }
      this.viewer.controls.minPolarAngle = 0.02;
      this.viewer.controls.maxPolarAngle = Math.PI - 0.02;
      this.viewer.controls.rotateSpeed = 0.45;
      this.viewer.controls.minDistance = this.asset.cameras[0].focus_distance * 0.08;
      this.viewer.controls.maxDistance = this.asset.cameras[0].focus_distance * 8;
      this.cameraIndex = 0;
      this.updateFov();
      this.viewer.renderer.domElement.setAttribute('aria-label', `${this.asset.label}: interactive Gaussian reconstruction`);
      const url = URL.createObjectURL(new Blob([bytes], {type: 'application/octet-stream'}));
      try {
        await this.viewer.addSplatScene(url, {format: GS.SceneFormat.Splat, splatAlphaRemovalThreshold: this.asset.queries ? 0 : 1, showLoadingUI: false, progressiveLoad: false});
        // Expose the screen-space footprint as a uniform so full screen and
        // responsive resizing preserve the source-resolution rendering convention.
        const material = this.viewer.splatMesh.material;
        material.vertexShader = 'uniform float ddnKernelSize;\n' + material.vertexShader.replace(/(cov2Dm\[[01]\]\[[01]\] \+= )[^;]+;/g, '$1ddnKernelSize;');
        material.uniforms.ddnKernelSize = {value: 0.3};
        material.needsUpdate = true;
        this.updateFov();
        this.viewer.renderer.setClearColor(0xf1f3f5, 1);
        if (this.visible && !document.hidden) this.viewer.start();
      } finally { URL.revokeObjectURL(url); }
    }
    updateFov() {
      if (!this.viewer || !this.asset) return;
      const aspect = this.host.clientWidth / Math.max(1, this.host.clientHeight);
      // Keep the semantic reference fully in view; frame NVS to its captured width.
      const nativeFov = this.activePose.fov;
      const nativeAspect = this.asset.nativeAspect || 1;
      const scale = this.asset.queries ? Math.max(1, nativeAspect / aspect) : 1 / Math.max(1, aspect / nativeAspect);
      this.viewer.camera.fov = 2 * Math.atan(Math.tan(nativeFov * Math.PI / 360) * scale) * 180 / Math.PI;
      this.viewer.camera.aspect = aspect;
      this.viewer.camera.updateProjectionMatrix();
      const kernel = this.viewer.splatMesh?.material?.uniforms?.ddnKernelSize;
      if (kernel) kernel.value = this.kernelSize();
      this.viewer.forceRenderNextFrame();
    }
    kernelSize() {
      const pixels = this.asset.queries ? this.host.clientWidth : Math.max(this.host.clientWidth, this.host.clientHeight);
      return 0.3 * Math.max(1, pixels / (this.asset.nativeWidth || 256)) ** 2;
    }
    prepareQueries() {
      const mesh = this.viewer.splatMesh;
      const base = mesh.splatDataTextures.baseData;
      const n = this.asset.count;
      if (mesh.getSplatCount() !== n) throw new Error('Gaussian count does not match the semantic masks.');
      // The loader spatially buckets Gaussians. Match exact float32 centers back
      // to the source order before applying any mask. The exporter verifies that
      // coincident centers always share the same membership in every query mask.
      const source = new Uint32Array(this.raw.buffer, this.raw.byteOffset, this.raw.byteLength / 4);
      const index = new Map();
      for (let i = 0; i < n; i++) index.set(`${source[i*8]},${source[i*8+1]},${source[i*8+2]}`, i);
      const centers = new Uint32Array(base.centers.buffer, base.centers.byteOffset, n * 3);
      this.sourceIndices = new Uint32Array(n);
      for (let i = 0; i < n; i++) {
        const sourceIndex = index.get(`${centers[i*3]},${centers[i*3+1]},${centers[i*3+2]}`);
        if (sourceIndex === undefined) throw new Error('A Gaussian could not be matched to its semantic mask.');
        this.sourceIndices[i] = sourceIndex;
      }
      this.originalColors = base.colors.slice();
      this.queryMasks = this.asset.queries.map(q => Uint8Array.from(atob(q.mask), c => c.charCodeAt(0)));
      this.queryIndex = 0;
      const choices = this.element.querySelector('.semantic-queries');
      choices.replaceChildren();
      this.asset.queries.forEach((q, i) => {
        const button = document.createElement('button');
        button.type = 'button';button.dataset.semanticQuery = String(i);button.textContent = `“${q.label}”`;
        choices.appendChild(button);
      });
    }
    paintQuery(sync = true) {
      if (!this.asset?.queries || this.busy) return;
      const query = this.asset.queries[this.queryIndex];
      const mask = this.queryMasks[this.queryIndex];
      const texture = this.viewer.splatMesh.splatDataTextures.centerColors;
      const original = this.originalColors;
      let selectedCount = 0;
      for (let i = 0; i < this.asset.count; i++) {
        const sourceIndex = this.sourceIndices[i];
        const selected = (mask[sourceIndex >> 3] >> (sourceIndex & 7)) & 1;
        selectedCount += selected;
        const j = i * 4;
        let r = original[j], g = original[j+1], b = original[j+2], a = original[j+3];
        if (this.mode === 'highlight') {
          if (selected) { r = Math.round(r*.3+28*.7);g = Math.round(g*.3+182*.7);b = Math.round(b*.3+146*.7); }
          else { const gray = Math.round((r*.2126+g*.7152+b*.0722)*.35+230*.65);r = g = b = gray; }
        } else if (this.mode === 'isolate' && !selected) a = 0;
        // Update only packed RGBA; position, covariance and depth sorting are untouched.
        texture.data[j] = (r | (g << 8) | (b << 16) | (a << 24)) >>> 0;
      }
      if (selectedCount !== query.count) throw new Error('Semantic mask correspondence is incomplete.');
      texture.texture.needsUpdate = true;
      this.viewer.forceRenderNextFrame();
      this.element.querySelectorAll('[data-semantic-query]').forEach(b => b.setAttribute('aria-pressed', String(Number(b.dataset.semanticQuery) === this.queryIndex)));
      this.element.querySelectorAll('[data-display]').forEach(b => b.setAttribute('aria-pressed', String(b.dataset.display === this.mode)));
      const methodLabel = (this.asset.method || 'GeoBridgeNet-A3').replace(/\bDDN\b/g, 'GeoBridgeNet');
      this.status.textContent = `${methodLabel} · “${query.label}” · ${this.mode === 'isolate' ? 'Selected object only' : this.mode === 'highlight' ? 'Selected object highlighted' : 'Original scene'}`;
      this.element.dataset.query = query.label;
      this.element.dataset.mode = this.mode;
      this.element.dataset.selectedCount = String(selectedCount);
      if (sync && this.element.dataset.semanticCompare) {
        for (const peer of Object.values(window.DDNViewers)) {
          if (peer === this || peer.busy || !peer.asset?.queries || peer.element.dataset.syncGroup !== this.element.dataset.syncGroup) continue;
          const index = peer.asset.queries.findIndex(item => item.label === query.label);
          if (index < 0) continue;
          peer.queryIndex = index;
          peer.mode = this.mode;
          peer.paintQuery(false);
        }
      }
    }
    setCamera(index) {
      if (!this.viewer || this.busy) return;
      const pose = this.asset.cameras[index];
      if (!pose) return;
      this.cameraIndex = index;
      this.setPose(pose);
    }
    setPose(pose) {
      if (!pose || !this.viewer || this.busy) return;
      this.activePose = pose;
      this.viewer.camera.position.fromArray(pose.position);
      this.viewer.camera.up.fromArray(pose.up);
      this.viewer.controls.target.fromArray(pose.target);
      this.viewer.camera.lookAt(this.viewer.controls.target);
      this.viewer.controls.update();
      this.updateFov();
    }
    move(event) {
      if (!this.viewer || this.busy || !['w','a','s','d','q','e'].includes(event.key.toLowerCase())) return;
      event.preventDefault();
      const { THREE } = window.DDN3D;
      const camera = this.viewer.camera;
      const step = this.asset.cameras[0].focus_distance * (event.shiftKey ? 0.07 : 0.025);
      const forward = new THREE.Vector3();camera.getWorldDirection(forward);
      const right = new THREE.Vector3().crossVectors(forward, camera.up).normalize();
      const key = event.key.toLowerCase();
      const delta = ({w: forward, s: forward.clone().negate(), d: right, a: right.clone().negate(), q: camera.up.clone(), e: camera.up.clone().negate()})[key].multiplyScalar(step);
      camera.position.add(delta);this.viewer.controls.target.add(delta);this.viewer.controls.update();this.viewer.forceRenderNextFrame();
    }
  }
  window.DDNViewers = {};
  document.querySelectorAll('[data-semantic-scene]').forEach(button => {
    button.addEventListener('click', () => {
      document.querySelectorAll('[data-semantic-scene]').forEach(item => item.setAttribute('aria-pressed', String(item === button)));
      document.querySelectorAll('[data-semantic-panel]').forEach(panel => {
        panel.hidden = panel.dataset.semanticPanel !== button.dataset.semanticScene;
      });
    });
  });
  document.querySelectorAll('[data-nvs-compare]').forEach(button => {
    button.addEventListener('click', () => {
      document.querySelectorAll('[data-nvs-compare]').forEach(item => item.setAttribute('aria-pressed', String(item === button)));
      document.querySelectorAll('[data-nvs-compare-panel]').forEach(panel => {
        panel.hidden = panel.dataset.nvsComparePanel !== button.dataset.nvsCompare;
      });
    });
  });
  document.querySelectorAll('[data-gaussian-viewer]').forEach(element => {
    const controller = new SceneViewer(element);
    window.DDNViewers[element.id] = controller;
    if (element.hasAttribute('data-lazy')) {
      const observer = new IntersectionObserver(entries => {
        if (entries[0].isIntersecting) { observer.disconnect();controller.load(element.dataset.gaussianViewer); }
      }, {rootMargin: '150px'});
      observer.observe(element);
    } else controller.load(element.dataset.gaussianViewer);
  });
})();
