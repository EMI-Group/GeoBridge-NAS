'use strict';
// GeoBridgeNet-A3 manuscript examples: each query has its own reference view and mask.
const queryScenes = {
  sheep: {scene: 'Teatime', query: 'sheep'},
  bear: {scene: 'Teatime', query: 'stuffed bear'},
  pouf: {scene: 'Teatime', query: 'yellow pouf'},
  pot: {scene: 'Waldo Kitchen', query: 'pot'}
};
const comparison = document.querySelector('#comparison');
const reveal = document.querySelector('#reveal');
reveal.addEventListener('input', () => comparison.style.setProperty('--split', `${reveal.value}%`));
document.querySelectorAll('[data-query]').forEach(button => {
  button.addEventListener('click', () => {
    const key = button.dataset.query;
    const {scene, query} = queryScenes[key];
    document.querySelectorAll('[data-query]').forEach(item => item.setAttribute('aria-pressed', String(item === button)));
    document.querySelector('#scene-name').textContent = scene;
    const reference = document.querySelector('#scene-image');
    reference.src = `assets/grounding/${key}-reference.jpg`;
    reference.alt = `Reference view for the query “${query}” in ${scene}`;
    document.querySelector('#predicted-mask').src = `assets/grounding/${key}-mask.png`;
  });
});

const videoStates = new WeakMap();
function switchVideo(id, path, label) {
  const v=document.getElementById(id);
  const state=videoStates.get(v);
  if (state) { state.resumeTime=0; state.wantsPlay=true; state.src=path+'.mp4'; }
  v.pause();v.poster=path+'.jpg';v.querySelector('source').src=path+'.mp4';
  v.setAttribute('aria-label',label);v.load();v.play().catch(()=>{});
}
document.querySelectorAll('[data-depth]').forEach(button=>button.addEventListener('click',()=>{const key=button.dataset.depth;document.querySelectorAll('[data-depth]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));switchVideo('depth-video',`assets/demos/depth-${key}`,`${key==='living_room'?'Living room':'Meeting room'} continuous input footage and predicted depth`);document.getElementById('depth-credit').textContent=key==='living_room'?'Footage: Curtis Adams / Pexels':'Footage: Rui Wang · Hangzhou Normal University';}));
document.querySelectorAll('[data-nvs]').forEach(button=>button.addEventListener('click',()=>{const key=button.dataset.nvs;document.querySelectorAll('[data-nvs]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));switchVideo('nvs-video',`assets/demos/nvs-${key}`,'GeoBridgeNet-A3 continuously rendered camera trajectory');[0,1].forEach(i=>document.getElementById(`context-${i}`).src=`assets/demos/nvs-${key}-context-${i}.jpg`);document.getElementById('nvs-target').src=`assets/demos/nvs-${key}-target.jpg`;}));
// Do not keep six independent video decoders alive while exploring 3D scenes.
// Reload on re-entry so Chromium receives a fresh video surface, with a poster
// while loading. Keep the playback position and respect a manual pause.
function resumeVideo(video, state) {
  const source=video.querySelector('source');
  if (!source.hasAttribute('src')) { source.src=state.src; video.load(); }
  if (state.wantsPlay) video.play().catch(()=>{});
}
function suspendVideo(video, state) {
  if (video.readyState>0) state.resumeTime=video.currentTime;
  video.pause();
  video.querySelector('source').removeAttribute('src');
  video.load();
}
const videoObserver = new IntersectionObserver(entries => {
  entries.forEach(entry => {
    const video=entry.target;
    const state=videoStates.get(video);
    state.visible=entry.isIntersecting;
    if (state.visible && !document.hidden) {
      resumeVideo(video,state);
    } else {
      suspendVideo(video,state);
    }
  });
}, {threshold:0});
document.querySelectorAll('video').forEach(video => {
  const state={visible:false,wantsPlay:true,resumeTime:0,src:video.querySelector('source').getAttribute('src')};
  videoStates.set(video,state);
  video.muted=true;
  video.addEventListener('loadedmetadata', () => {
    if (state.resumeTime>0 && Number.isFinite(video.duration)) {
      video.currentTime=Math.min(state.resumeTime,Math.max(0,video.duration-.05));
    }
  });
  video.addEventListener('play',()=>{state.wantsPlay=true;});
  video.addEventListener('pause',()=>{
    if (state.visible && !document.hidden && video.readyState>0) state.wantsPlay=false;
  });
  videoObserver.observe(video);
});
document.addEventListener('visibilitychange',()=>{
  document.querySelectorAll('video').forEach(video=>{
    const state=videoStates.get(video);
    if (document.hidden) {
      suspendVideo(video,state);
    } else if (state.visible) resumeVideo(video,state);
  });
});
document.querySelectorAll('[data-hero-video]').forEach(button => {
  const video = document.getElementById(button.dataset.heroVideo);
  const label = button.getAttribute('aria-label').replace(/^Pause /, '');
  const update = () => {
    const action = video.paused ? 'Play' : 'Pause';
    button.textContent = action;
    button.setAttribute('aria-label', `${action} ${label}`);
  };
  button.addEventListener('click', () => video.paused ? video.play().catch(()=>{}) : video.pause());
  video.addEventListener('play', update);
  video.addEventListener('pause', update);
  update();
});

// Keep the depth table and qualitative figure on the same selected benchmark.
document.querySelectorAll('[data-benchmark]').forEach(button => {
  button.addEventListener('click', () => {
    document.querySelectorAll('[data-benchmark]').forEach(item => item.setAttribute('aria-pressed', String(item === button)));
    document.querySelectorAll('[data-benchmark-panel]').forEach(panel => { panel.hidden = panel.dataset.benchmarkPanel !== button.dataset.benchmark; });
  });
});
