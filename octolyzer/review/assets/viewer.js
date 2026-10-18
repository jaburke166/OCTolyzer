(() => {
'use strict';

const manifest = JSON.parse(document.getElementById('octolyzer-manifest').textContent);
const payload = JSON.parse(document.getElementById('octolyzer-payload').textContent);
const embedded = window.parent !== window;
const app = document.getElementById('app');

const STRUCTURES = [
  ['retinal_layers', 'Retinal layers'], ['choroid_boundary', 'Choroid boundary'],
  ['choroid_vessels', 'Choroid vessels'], ['oct_fovea', 'OCT fovea'],
  ['slo_vessels', 'SLO vessels'], ['artery_vein', 'Artery/vein'],
  ['optic_disc', 'Optic disc'], ['slo_fovea', 'SLO fovea'], ['image_quality', 'Image quality'],
];
const GRADES = [['pass', 'Pass', 'P', 'check'], ['borderline', 'Borderline', 'B', 'approx'], ['fail', 'Fail', 'F', 'cross']];
const COLORS = {
  ILM: '#ffd166', RNFL: '#ff9f1c', GCL: '#ff6b6b', IPL: '#f472b6', INL: '#c084fc', OPL: '#818cf8',
  ELM: '#38bdf8', PR1: '#22d3ee', PR2: '#2dd4bf', RPE: '#a3e635', BM: '#f8fafc',
  CHORupper: '#34d399', CHORlower: '#fb7185', choroid_fill: '#34d399', chor_vessels: '#ff3b30',
  oct_fovea: '#ffffff', roi: '#3ddc84', slo_vessels: '#c084fc', artery: '#ff4d4f', vein: '#4d8dff',
  disc: '#3ddc84', slo_fovea: '#ffd166', acq: '#3ddc84', zones: '#f59e0b', map: '#ffffff', grid: '#ffffff',
};
const SPARE_COLORS = ['#e879f9', '#facc15', '#60a5fa', '#f97316', '#4ade80', '#f43f5e'];
// Colour maps for en-face maps as [position, [r, g, b]] stops; 'hot' follows matplotlib.
const COLORMAPS = {
  hot: { label: 'Hot', stops: [[0, [11, 0, 0]], [0.365, [255, 0, 0]], [0.746, [255, 255, 0]], [1, [255, 255, 255]]] },
  viridis: { label: 'Viridis', stops: [[0, [68, 1, 84]], [0.25, [59, 82, 139]], [0.5, [33, 145, 140]], [0.75, [94, 201, 98]], [1, [253, 231, 37]]] },
};
const DEFAULT_PREFS = {
  theme: 'dark', brightness: 0, contrast: 1, gamma: 1, invert: false, opacity: 0.9, lineWidth: 1.25,
  aspect: 'pixel', split: false, swapped: false, expanded: null, railHidden: false,
  visible: {}, colors: {}, map: null, colormap: 'hot', closed: {},
};
const PREFS_KEY = 'octolyzer-review-prefs';
const VERDICT_KEY = `octolyzer-review-verdict:${manifest.filename}`;

// ---------------------------------------------------------------- utilities
function h(tag, attrs, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs || {})) {
    if (value === null || value === undefined || value === false) continue;
    if (key === 'class') node.className = value;
    else if (key === 'text') node.textContent = value;
    else if (key.startsWith('on')) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? '' : value);
  }
  for (const child of children.flat()) {
    if (child !== null && child !== undefined && child !== false) node.append(child);
  }
  return node;
}
const clamp = (value, low, high) => Math.min(high, Math.max(low, value));
function iconButton(name, tip, onclick, extra = {}) {
  return h('button', { class: 'icon quiet', 'data-tip': tip, 'aria-label': tip, onclick, ...extra }, OctUI.icon(name));
}
const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
function store(key, value) {
  try {
    if (value === undefined) return JSON.parse(localStorage.getItem(key));
    localStorage.setItem(key, JSON.stringify(value));
  } catch (error) { /* storage is unavailable for some local files */ }
  return null;
}
function format(value, digits = 3, sentinel = false) {
  if (value === null || value === undefined || value === '') return '–';
  if (typeof value !== 'number') return String(value);
  if (sentinel && value === -1) return '–';
  if (Number.isInteger(value)) return String(value);
  return Math.abs(value) >= 100 ? value.toFixed(0) : Number(value.toPrecision(digits)).toString();
}
function post(message) {
  if (embedded) window.parent.postMessage({ octolyzer: 'viewer', ...message }, '*');
}
function loadImage(uri) {
  return new Promise((resolve, reject) => {
    const image = new Image();
    image.onload = () => resolve(image);
    image.onerror = () => reject(new Error('Unable to decode an embedded image.'));
    image.src = uri;
  });
}
async function inflate(text) {
  const bytes = Uint8Array.from(atob(text), (char) => char.charCodeAt(0));
  const stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream('deflate'));
  return new Uint8Array(await new Response(stream).arrayBuffer());
}
// Mirror of encode.pack_uint16: rows are delta-encoded little-endian uint16.
async function unpack16(text, rows, cols) {
  const raw = await inflate(text);
  const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  const out = new Uint16Array(rows * cols);
  for (let r = 0; r < rows; r++) {
    let acc = 0;
    for (let c = 0; c < cols; c++) {
      acc = (acc + view.getUint16(2 * (r * cols + c), true)) & 0xffff;
      out[r * cols + c] = acc;
    }
  }
  return out;
}
function grayscale(image) {
  const canvas = document.createElement('canvas');
  canvas.width = image.naturalWidth;
  canvas.height = image.naturalHeight;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(image, 0, 0);
  const rgba = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
  const gray = new Uint8Array(canvas.width * canvas.height);
  for (let i = 0; i < gray.length; i++) gray[i] = rgba[4 * i];
  return gray;
}
function tint(image, color) {
  const canvas = document.createElement('canvas');
  canvas.width = image.naturalWidth || image.width;
  canvas.height = image.naturalHeight || image.height;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(image, 0, 0);
  ctx.globalCompositeOperation = 'source-in';
  ctx.fillStyle = color;
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  return canvas;
}

// ------------------------------------------------------------------- state
const prefs = { ...DEFAULT_PREFS, ...(embedded ? {} : store(PREFS_KEY) || {}) };
const state = { slice: 0, hover: null, blink: false, gray: null, sloGray: null, verdict: emptyVerdict() };
let layers = [];
let traces = {};
let octImages = [];
let vesselImages = [];
let maskImages = {};
let mapData = {};
let tintCache = new WeakMap();
const mapCache = new Map();
let octPane = null, sloPane = null, cleanPane = null;
const ui = { expand: {} };

function emptyVerdict() { return { grade: '', flags: {}, note: '' }; }
function savePrefs() {
  if (embedded) post({ type: 'prefs', prefs });
  else store(PREFS_KEY, prefs);
}
function applyTheme() { document.documentElement.dataset.theme = prefs.theme; }

// ------------------------------------------------------------------- panes
class Pane {
  constructor(host, width, height, view) {
    this.host = host;
    this.w = width; this.h = height; this.ax = 1; this.ay = 1;
    this.view = view || { s: 1, x: 0, y: 0, fitted: true };
    this.base = h('canvas');
    this.over = h('canvas');
    host.append(this.base, this.over);
    this.source = null;
    this.onOverlay = null; this.onHover = null; this.onClick = null; this.onScroll = null;
    this.peers = [];
    this.cw = 1; this.ch = 1;
    this.bind();
    new ResizeObserver(() => this.resize()).observe(host);
  }
  bind() {
    let drag = null;
    this.host.addEventListener('pointerdown', (event) => {
      if (event.button !== 0) return;
      drag = { x: event.clientX, y: event.clientY, moved: false };
      this.host.setPointerCapture(event.pointerId);
    });
    this.host.addEventListener('pointermove', (event) => {
      const rect = this.host.getBoundingClientRect();
      const sx = event.clientX - rect.left, sy = event.clientY - rect.top;
      if (drag) {
        const dx = event.clientX - drag.x, dy = event.clientY - drag.y;
        if (drag.moved || Math.hypot(dx, dy) > 3) {
          drag.moved = true;
          this.host.classList.add('dragging');
          this.view.x += dx; this.view.y += dy; this.view.fitted = false;
          drag.x = event.clientX; drag.y = event.clientY;
          this.drawAll();
        }
        return;
      }
      if (this.onHover) this.onHover(...this.toImage(sx, sy), sx, sy);
    });
    this.host.addEventListener('pointerup', (event) => {
      const moved = drag && drag.moved;
      drag = null;
      this.host.classList.remove('dragging');
      if (!moved && this.onClick) {
        const rect = this.host.getBoundingClientRect();
        this.onClick(event.clientX - rect.left, event.clientY - rect.top);
      }
    });
    this.host.addEventListener('pointerleave', () => { if (this.onHover) this.onHover(null); });
    this.host.addEventListener('dblclick', () => { this.fit(); this.drawAll(); });
    this.host.addEventListener('wheel', (event) => {
      event.preventDefault();
      if (this.onScroll && !event.ctrlKey && !event.metaKey && this.onScroll(event.deltaY)) return;
      const rect = this.host.getBoundingClientRect();
      this.zoomAt(event.clientX - rect.left, event.clientY - rect.top, Math.exp(-event.deltaY * 0.0025));
    }, { passive: false });
  }
  resize() {
    const rect = this.host.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    const dpr = window.devicePixelRatio || 1;
    this.cw = rect.width; this.ch = rect.height;
    for (const canvas of [this.base, this.over]) {
      canvas.width = Math.round(rect.width * dpr);
      canvas.height = Math.round(rect.height * dpr);
    }
    if (this.view.fitted) this.fit();
    this.draw();
  }
  fit() {
    const s = Math.min(this.cw / (this.w * this.ax), this.ch / (this.h * this.ay)) * 0.97;
    this.view.s = s;
    this.view.x = (this.cw - this.w * this.ax * s) / 2;
    this.view.y = (this.ch - this.h * this.ay * s) / 2;
    this.view.fitted = true;
  }
  zoomAt(sx, sy, factor) {
    const [ix, iy] = this.toImage(sx, sy);
    this.view.s = clamp(this.view.s * factor, 0.05, 60);
    this.view.x = sx - ix * this.ax * this.view.s;
    this.view.y = sy - iy * this.ay * this.view.s;
    this.view.fitted = false;
    this.drawAll();
  }
  toScreen(x, y) { return [this.view.x + x * this.ax * this.view.s, this.view.y + y * this.ay * this.view.s]; }
  toImage(sx, sy) { return [(sx - this.view.x) / (this.ax * this.view.s), (sy - this.view.y) / (this.ay * this.view.s)]; }
  rect() { return [...this.toScreen(0, 0), this.w * this.ax * this.view.s, this.h * this.ay * this.view.s]; }
  context(canvas) {
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, this.cw, this.ch);
    ctx.imageSmoothingEnabled = this.view.s * Math.max(this.ax, this.ay) < 3;
    return ctx;
  }
  drawBase() {
    const ctx = this.context(this.base);
    if (this.source) ctx.drawImage(this.source, ...this.rect());
  }
  drawOverlay() {
    const ctx = this.context(this.over);
    if (this.onOverlay) this.onOverlay(ctx, this);
  }
  draw() { this.drawBase(); this.drawOverlay(); }
  drawAll() { this.draw(); for (const peer of this.peers) peer.draw(); }
}

function renderGray(gray, width, height, canvas) {
  const lut = new Uint8ClampedArray(256);
  for (let i = 0; i < 256; i++) {
    let v = (i / 255 - 0.5) * prefs.contrast + 0.5 + prefs.brightness;
    v = Math.pow(clamp(v, 0, 1), 1 / prefs.gamma);
    lut[i] = 255 * (prefs.invert ? 1 - v : v);
  }
  canvas.width = width; canvas.height = height;
  const ctx = canvas.getContext('2d');
  const image = ctx.createImageData(width, height);
  const data = image.data;
  for (let i = 0; i < gray.length; i++) {
    const v = lut[gray[i]];
    data[4 * i] = v; data[4 * i + 1] = v; data[4 * i + 2] = v; data[4 * i + 3] = 255;
  }
  ctx.putImageData(image, 0, 0);
  return canvas;
}

// ------------------------------------------------------------------ layers
const oct = payload.oct;
const slo = payload.slo;
const N = oct ? oct.width : 0;
const M = oct ? oct.height : 0;
const S = oct ? oct.images.length : 0;
const lines = oct && oct.acquisition && oct.acquisition.lines;
const circle = oct && oct.acquisition && oct.acquisition.circle;

function layer(id) { return layers.find((item) => item.id === id); }
function visible(id) {
  const item = layer(id);
  if (!item) return false;
  return id in prefs.visible ? prefs.visible[id] : item.on;
}
function color(id) { return prefs.colors[id] || (layer(id) || {}).color || '#ffffff'; }
function defineLayers() {
  const list = [];
  let spare = 0;
  const add = (id, pane, label, on = true, title = '') => list.push({
    id, pane, label, on, title, color: COLORS[id] || SPARE_COLORS[spare++ % SPARE_COLORS.length],
  });
  for (const item of oct.layers) add(item.key, 'oct', item.key, true, item.name);
  if (traces.CHORupper && traces.CHORlower) add('choroid_fill', 'oct', 'Choroid region', false);
  if (oct.vessels) add('chor_vessels', 'oct', 'Choroid vessels');
  if (oct.foveas.some(Boolean)) add('oct_fovea', 'oct', 'Fovea');
  if (oct.roi_um) add('roi', 'oct', `Measured region (±${oct.roi_um} µm)`);
  if (slo) {
    const masks = slo.masks || {};
    const split = Boolean(masks.artery || masks.vein);
    if (masks.vessels) add('slo_vessels', 'slo', 'Vessels (binary)', !split);
    if (masks.artery) add('artery', 'slo', 'Arteries');
    if (masks.vein) add('vein', 'slo', 'Veins');
    if (masks.disc) add('disc', 'slo', 'Optic disc');
    if (slo.fovea) add('slo_fovea', 'slo', 'Fovea');
    if (lines || circle) add('acq', 'slo', circle ? 'Scan circle and centre' : 'Acquisition lines');
    if (slo.optic_disc && manifest.location === 'peripapillary') add('zones', 'slo', 'Zones B and C');
    if (payload.maps.length) add('map', 'slo', 'En-face map', false);
    if (payload.grid) add('grid', 'slo', 'ETDRS grid', false);
  }
  layers = list;
}

// ------------------------------------------------------------ data helpers
function traceAt(key, index, scan = state.slice) {
  const data = traces[key];
  if (!data || index < 0 || index >= N) return NaN;
  return data[scan * N + index];
}
function thickness(upper, lower, index, scan = state.slice) {
  return (traceAt(lower, index, scan) - traceAt(upper, index, scan)) * oct.scale_y;
}
function profilePairs() {
  const pairs = [];
  if (traces.ILM && traces.BM) pairs.push(['Retina', 'ILM', 'BM', cssVar('--text')]);
  if (traces.ILM && traces.RNFL) pairs.push(['RNFL', 'ILM', 'RNFL', color('RNFL')]);
  if (traces.CHORupper && traces.CHORlower) pairs.push(['Choroid', 'CHORupper', 'CHORlower', color('CHORlower')]);
  return pairs;
}
function flaggedSlices() {
  const set = new Set();
  for (const flag of manifest.flags) for (const index of flag.slices || []) set.add(index);
  return [...set].sort((a, b) => a - b);
}
function sloPoint(index, scan = state.slice) {
  const t = N > 1 ? index / (N - 1) : 0;
  if (circle) {
    const angle = 2 * Math.PI * t;
    return [circle.cx + circle.r * Math.cos(angle), circle.cy + circle.r * Math.sin(angle)];
  }
  if (lines && lines[scan]) {
    const [x0, y0, x1, y1] = lines[scan];
    return [x0 + t * (x1 - x0), y0 + t * (y1 - y0)];
  }
  return null;
}
function nearestLine(x, y) {
  let best = { index: -1, distance: Infinity, t: 0 };
  (lines || []).forEach(([x0, y0, x1, y1], index) => {
    const dx = x1 - x0, dy = y1 - y0;
    const t = clamp(((x - x0) * dx + (y - y0) * dy) / (dx * dx + dy * dy || 1), 0, 1);
    const distance = Math.hypot(x - (x0 + t * dx), y - (y0 + t * dy));
    if (distance < best.distance) best = { index, distance, t };
  });
  return best;
}
function niceLength(target, options) {
  let best = options[0];
  for (const option of options) if (option <= target) best = option;
  return best;
}

// --------------------------------------------------------------- overlays
function strokeMarker(ctx, sx, sy, fill, size = 5) {
  ctx.lineCap = 'round';
  for (const [stroke, width] of [['#000', 4], [fill, 1.75]]) {
    ctx.strokeStyle = stroke; ctx.lineWidth = width;
    ctx.beginPath();
    ctx.moveTo(sx - size, sy - size); ctx.lineTo(sx + size, sy + size);
    ctx.moveTo(sx - size, sy + size); ctx.lineTo(sx + size, sy - size);
    ctx.stroke();
  }
}
function scaleBar(ctx, pane, micronsX, micronsY) {
  const span = 110;
  const lengthX = niceLength(span / (pane.ax * pane.view.s) * micronsX, [50, 100, 200, 500, 1000, 2000, 5000]);
  const px = lengthX / micronsX * pane.ax * pane.view.s;
  const x = 14, y = pane.ch - 14;
  ctx.strokeStyle = 'rgba(255,255,255,0.75)'; ctx.fillStyle = 'rgba(255,255,255,0.75)';
  ctx.lineWidth = 1; ctx.font = '10px ui-monospace, Menlo, monospace';
  ctx.beginPath(); ctx.moveTo(x, y); ctx.lineTo(x + px, y);
  let label = `${lengthX} µm`;
  if (micronsY) {
    const lengthY = niceLength(60 / (pane.ay * pane.view.s) * micronsY, [20, 50, 100, 200, 500, 1000]);
    ctx.moveTo(x, y); ctx.lineTo(x, y - lengthY / micronsY * pane.ay * pane.view.s);
    label += ` × ${lengthY} µm`;
  }
  ctx.stroke();
  ctx.fillText(label, x + 6, y - 5);
}
function drawMask(ctx, pane, image, id) {
  if (!image) return;
  let cached = tintCache.get(image);
  if (!cached || cached.color !== color(id)) {
    cached = { color: color(id), canvas: tint(image, color(id)) };
    tintCache.set(image, cached);
  }
  ctx.drawImage(cached.canvas, ...pane.rect());
}
function drawTrace(ctx, pane, key, first, last) {
  const data = traces[key];
  const base = state.slice * N;
  ctx.beginPath();
  let pen = false;
  for (let i = first; i <= last; i++) {
    const y = data[base + i];
    if (y !== y) { pen = false; continue; }
    const [sx, sy] = pane.toScreen(i + 0.5, y + 0.5);
    if (pen) ctx.lineTo(sx, sy); else { ctx.moveTo(sx, sy); pen = true; }
  }
  ctx.strokeStyle = color(key);
  ctx.stroke();
}
function drawFill(ctx, pane, upper, lower, first, last) {
  const base = state.slice * N;
  let run = [];
  const flush = () => {
    if (run.length > 1) {
      ctx.beginPath();
      run.forEach((i, n) => { const p = pane.toScreen(i + 0.5, traces[upper][base + i] + 0.5); n ? ctx.lineTo(...p) : ctx.moveTo(...p); });
      for (let n = run.length - 1; n >= 0; n--) ctx.lineTo(...pane.toScreen(run[n] + 0.5, traces[lower][base + run[n]] + 0.5));
      ctx.closePath(); ctx.fill();
    }
    run = [];
  };
  for (let i = first; i <= last; i++) {
    const a = traces[upper][base + i], b = traces[lower][base + i];
    if (a === a && b === b) run.push(i); else flush();
  }
  flush();
}
function octCursorX() {
  if (!state.hover) return null;
  return state.hover.octX === undefined ? null : state.hover.octX;
}
function drawOct(ctx, pane) {
  const first = clamp(Math.floor(pane.toImage(0, 0)[0]) - 1, 0, N - 1);
  const last = clamp(Math.ceil(pane.toImage(pane.cw, 0)[0]) + 1, 0, N - 1);
  const fovea = oct.foveas[state.slice];
  if (!state.blink) {
    ctx.save();
    ctx.beginPath(); ctx.rect(...pane.rect()); ctx.clip();
    ctx.globalAlpha = prefs.opacity;
    ctx.lineJoin = 'round';
    if (visible('roi') && fovea) {
      const half = oct.roi_um / oct.scale_x;
      const [x0, y0] = pane.toScreen(fovea[0] + 0.5 - half, 0);
      const [x1, y1] = pane.toScreen(fovea[0] + 0.5 + half, M);
      ctx.fillStyle = color('roi');
      ctx.globalAlpha = prefs.opacity * 0.07;
      ctx.fillRect(x0, y0, x1 - x0, y1 - y0);
      ctx.globalAlpha = prefs.opacity * 0.8;
      ctx.strokeStyle = color('roi'); ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
      ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x0, y1); ctx.moveTo(x1, y0); ctx.lineTo(x1, y1); ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = prefs.opacity;
    }
    if (visible('choroid_fill')) {
      ctx.fillStyle = color('choroid_fill');
      ctx.globalAlpha = prefs.opacity * 0.3;
      drawFill(ctx, pane, 'CHORupper', 'CHORlower', first, last);
      ctx.globalAlpha = prefs.opacity;
    }
    if (visible('chor_vessels')) {
      ctx.globalAlpha = prefs.opacity * 0.6;
      drawMask(ctx, pane, vesselImages[state.slice], 'chor_vessels');
      ctx.globalAlpha = prefs.opacity;
    }
    ctx.lineWidth = prefs.lineWidth;
    for (const item of oct.layers) if (visible(item.key)) drawTrace(ctx, pane, item.key, first, last);
    if (visible('oct_fovea') && fovea) strokeMarker(ctx, ...pane.toScreen(fovea[0] + 0.5, fovea[1] + 0.5), color('oct_fovea'));
    ctx.restore();
  }
  const cursor = octCursorX();
  if (cursor !== null) {
    const [sx] = pane.toScreen(cursor + 0.5, 0);
    ctx.strokeStyle = 'rgba(255,255,255,0.35)'; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(sx, 0); ctx.lineTo(sx, pane.ch); ctx.stroke();
  }
  scaleBar(ctx, pane, oct.scale_x, oct.scale_y);
}

function colormap() { return COLORMAPS[prefs.colormap] || COLORMAPS.hot; }
function colormapLut(stops) {
  const lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0, k = 0; i < 256; i++) {
    const t = i / 255;
    while (k < stops.length - 2 && t > stops[k + 1][0]) k++;
    const [t0, c0] = stops[k], [t1, c1] = stops[k + 1];
    const f = clamp((t - t0) / (t1 - t0), 0, 1);
    for (let c = 0; c < 3; c++) lut[3 * i + c] = c0[c] + f * (c1[c] - c0[c]);
  }
  return lut;
}
function colormapGradient(stops) {
  return `linear-gradient(90deg, ${stops.map(([t, c]) => `rgb(${c.join(',')}) ${100 * t}%`).join(',')})`;
}
function setColormap(name) {
  prefs.colormap = name;
  mapCache.clear();
  savePrefs(); updateMapUi(); redraw(1);
}
function mapCanvas(entry) {
  if (mapCache.has(entry.key)) return mapCache.get(entry.key);
  const lut = colormapLut(colormap().stops);
  const data = mapData[entry.key];
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = entry.size;
  const ctx = canvas.getContext('2d');
  const image = ctx.createImageData(entry.size, entry.size);
  const span = entry.max - entry.min || 1;
  for (let i = 0; i < data.length; i++) {
    if (!data[i]) continue;
    const level = Math.round(clamp((data[i] / entry.scale - entry.min) / span, 0, 1) * 255);
    for (let c = 0; c < 3; c++) image.data[4 * i + c] = lut[3 * level + c];
    image.data[4 * i + 3] = 255;
  }
  ctx.putImageData(image, 0, 0);
  mapCache.set(entry.key, canvas);
  return canvas;
}
function currentMap() {
  return payload.maps.find((entry) => entry.key === prefs.map) || payload.maps[0] || null;
}
function gridAxis() {
  // Orientation of the fovea-centred B-scan on the SLO, folded into (-90, 90].
  const line = lines && lines[oct.fovea_slice === null ? Math.floor(S / 2) : oct.fovea_slice];
  if (!line) return 0;
  let angle = Math.atan2(line[3] - line[1], line[2] - line[0]);
  if (angle > Math.PI / 2) angle -= Math.PI;
  if (angle <= -Math.PI / 2) angle += Math.PI;
  return angle;
}
function gridValues() {
  const table = payload.measurements.etdrs_measurements;
  const entry = currentMap();
  if (!table || !entry || !visible('map')) return null;
  const name = table.columns.indexOf('map_name');
  const row = table.rows.find((item) => item[name] === entry.key);
  if (!row) return null;
  return Object.fromEntries(table.columns.map((column, index) => [column, row[index]]));
}
function drawGrid(ctx, pane) {
  const grid = payload.grid;
  const [cx, cy] = grid.centre;
  const radii = grid.diameters_um.map((diameter) => diameter / 2 / slo.scale);
  const axis = gridAxis();
  ctx.strokeStyle = color('grid'); ctx.lineWidth = 1;
  for (const radius of radii) {
    const [sx, sy] = pane.toScreen(cx + 0.5, cy + 0.5);
    ctx.beginPath(); ctx.arc(sx, sy, radius * pane.view.s, 0, 2 * Math.PI); ctx.stroke();
  }
  for (let k = 0; k < 4; k++) {
    const angle = axis + Math.PI / 4 + k * Math.PI / 2;
    ctx.beginPath();
    ctx.moveTo(...pane.toScreen(cx + 0.5 + radii[0] * Math.cos(angle), cy + 0.5 + radii[0] * Math.sin(angle)));
    ctx.lineTo(...pane.toScreen(cx + 0.5 + radii[2] * Math.cos(angle), cy + 0.5 + radii[2] * Math.sin(angle)));
    ctx.stroke();
  }
  const values = gridValues();
  if (!values) return;
  const nasal = manifest.eye === 'Left' ? Math.PI : 0;
  const directions = { nasal, temporal: nasal + Math.PI, superior: -Math.PI / 2, inferior: Math.PI / 2 };
  const digits = (value) => (typeof value === 'number' && Math.abs(value) < 5 ? value.toFixed(2) : format(value));
  ctx.font = '11px ui-monospace, Menlo, monospace'; ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  const label = (text, x, y) => {
    const [sx, sy] = pane.toScreen(x + 0.5, y + 0.5);
    ctx.lineWidth = 3; ctx.strokeStyle = 'rgba(0,0,0,0.8)'; ctx.strokeText(text, sx, sy);
    ctx.fillStyle = '#fff'; ctx.fillText(text, sx, sy);
  };
  label(digits(values.central), cx, cy);
  [['inner', (radii[0] + radii[1]) / 2], ['outer', (radii[1] + radii[2]) / 2]].forEach(([ring, radius]) => {
    for (const [side, direction] of Object.entries(directions)) {
      const value = values[`${ring}_${side}`];
      if (value === undefined) continue;
      const angle = direction + axis;
      label(value ? digits(value) : '–', cx + radius * Math.cos(angle), cy + radius * Math.sin(angle));
    }
  });
  ctx.textAlign = 'start'; ctx.textBaseline = 'alphabetic';
}
function drawSlo(ctx, pane) {
  if (!state.blink) {
    ctx.save();
    ctx.beginPath(); ctx.rect(...pane.rect()); ctx.clip();
    ctx.globalAlpha = prefs.opacity;
    const entry = currentMap();
    if (visible('map') && entry && mapData[entry.key]) {
      ctx.globalAlpha = prefs.opacity * 0.75;
      ctx.drawImage(mapCanvas(entry), ...pane.rect());
      ctx.globalAlpha = prefs.opacity;
    }
    ctx.globalAlpha = prefs.opacity * 0.7;
    for (const [id, name] of [['slo_vessels', 'vessels'], ['artery', 'artery'], ['vein', 'vein'], ['disc', 'disc']]) {
      if (visible(id)) {
        ctx.globalAlpha = prefs.opacity * (id === 'disc' ? 0.35 : 0.7);
        drawMask(ctx, pane, maskImages[name], id);
      }
    }
    ctx.globalAlpha = prefs.opacity;
    if (visible('zones')) {
      const disc = slo.optic_disc;
      const [sx, sy] = pane.toScreen(disc.cx + 0.5, disc.cy + 0.5);
      ctx.strokeStyle = color('zones'); ctx.fillStyle = color('zones'); ctx.lineWidth = 1;
      ctx.font = '10px ui-monospace, Menlo, monospace';
      [[1, true, ''], [2, false, 'B'], [3, false, ''], [5, true, 'C']].forEach(([factor, dashed, text]) => {
        ctx.setLineDash(dashed ? [5, 4] : []);
        ctx.beginPath(); ctx.arc(sx, sy, factor * disc.r * pane.view.s, 0, 2 * Math.PI); ctx.stroke();
        if (text) ctx.fillText(text, sx + 4, sy - factor * disc.r * pane.view.s - 3);
      });
      ctx.setLineDash([]);
    }
    if (visible('grid')) drawGrid(ctx, pane);
    if (visible('acq')) {
      ctx.strokeStyle = color('acq'); ctx.fillStyle = color('acq');
      if (circle) {
        const [sx, sy] = pane.toScreen(circle.cx + 0.5, circle.cy + 0.5);
        ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.arc(sx, sy, circle.r * pane.view.s, 0, 2 * Math.PI); ctx.stroke();
        ctx.beginPath(); ctx.arc(...pane.toScreen(circle.cx + circle.r + 0.5, circle.cy + 0.5), 3, 0, 2 * Math.PI); ctx.fill();
        // Centre of the scan circle, as placed by the operator at acquisition
        for (const [stroke, width] of [['#000', 3.5], [color('acq'), 1.5]]) {
          ctx.strokeStyle = stroke; ctx.lineWidth = width;
          ctx.beginPath();
          ctx.moveTo(sx - 7, sy); ctx.lineTo(sx + 7, sy); ctx.moveTo(sx, sy - 7); ctx.lineTo(sx, sy + 7);
          ctx.stroke();
        }
      } else if (lines) {
        lines.forEach(([x0, y0, x1, y1], index) => {
          const current = index === state.slice;
          ctx.globalAlpha = prefs.opacity * (current ? 1 : 0.45);
          ctx.lineWidth = current ? 1.75 : 1;
          ctx.beginPath(); ctx.moveTo(...pane.toScreen(x0 + 0.5, y0 + 0.5)); ctx.lineTo(...pane.toScreen(x1 + 0.5, y1 + 0.5)); ctx.stroke();
          if (current) { ctx.beginPath(); ctx.arc(...pane.toScreen(x0 + 0.5, y0 + 0.5), 3, 0, 2 * Math.PI); ctx.fill(); }
        });
        ctx.globalAlpha = prefs.opacity;
      }
    }
    if (visible('disc') && slo.optic_disc) {
      strokeMarker(ctx, ...pane.toScreen(slo.optic_disc.cx + 0.5, slo.optic_disc.cy + 0.5), color('disc'), 4);
    }
    if (visible('slo_fovea')) strokeMarker(ctx, ...pane.toScreen(slo.fovea[0] + 0.5, slo.fovea[1] + 0.5), color('slo_fovea'));
    ctx.restore();
  }
  const cursor = octCursorX();
  const point = cursor === null ? null : sloPoint(cursor);
  if (point) {
    const [sx, sy] = pane.toScreen(point[0] + 0.5, point[1] + 0.5);
    ctx.strokeStyle = '#000'; ctx.lineWidth = 3;
    ctx.beginPath(); ctx.arc(sx, sy, 5, 0, 2 * Math.PI); ctx.stroke();
    ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(sx, sy, 5, 0, 2 * Math.PI); ctx.stroke();
  }
  scaleBar(ctx, pane, slo.scale, 0);
}

// -------------------------------------------------------- plot and scrubber
function sizeCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  if (canvas.width !== Math.round(rect.width * dpr) || canvas.height !== Math.round(rect.height * dpr)) {
    canvas.width = Math.round(rect.width * dpr); canvas.height = Math.round(rect.height * dpr);
  }
  const ctx = canvas.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, rect.width, rect.height);
  return [ctx, rect.width, rect.height];
}
const PLOT = { left: 34, right: 10, top: 20, bottom: 8 };
function drawPlot() {
  if (!ui.plot || !ui.plot.offsetParent) return;
  const [ctx, width, height] = sizeCanvas(ui.plot);
  const pairs = profilePairs();
  const muted = cssVar('--muted'), faint = cssVar('--line');
  ctx.font = '10px ui-monospace, Menlo, monospace';
  if (!pairs.length) { ctx.fillStyle = muted; ctx.fillText('No thickness profile available', 12, 16); return; }
  const series = pairs.map(([, upper, lower]) => Float32Array.from({ length: N }, (_, i) => thickness(upper, lower, i)));
  let top = 0;
  for (const values of series) for (const value of values) if (value === value && value > top) top = value;
  top = Math.max(50, Math.ceil(top / 50) * 50);
  const px = (i) => PLOT.left + (i / Math.max(1, N - 1)) * (width - PLOT.left - PLOT.right);
  const py = (v) => height - PLOT.bottom - (v / top) * (height - PLOT.top - PLOT.bottom);
  ctx.strokeStyle = faint; ctx.fillStyle = muted; ctx.lineWidth = 1;
  for (const tick of [0, top / 2, top]) {
    ctx.beginPath(); ctx.moveTo(PLOT.left, py(tick)); ctx.lineTo(width - PLOT.right, py(tick)); ctx.stroke();
    ctx.fillText(String(Math.round(tick)), 6, py(tick) + 3);
  }
  let legendX = PLOT.left;
  ctx.fillText('µm', 6, 12);
  pairs.forEach(([name, , , stroke], n) => {
    ctx.fillStyle = stroke; ctx.fillText(name, legendX, 12);
    legendX += ctx.measureText(name).width + 12;
    ctx.strokeStyle = stroke; ctx.lineWidth = 1.25; ctx.beginPath();
    let pen = false;
    for (let i = 0; i < N; i++) {
      const value = series[n][i];
      if (value !== value) { pen = false; continue; }
      if (pen) ctx.lineTo(px(i), py(value)); else { ctx.moveTo(px(i), py(value)); pen = true; }
    }
    ctx.stroke();
  });
  const fovea = oct.foveas[state.slice];
  if (fovea) {
    ctx.strokeStyle = muted; ctx.setLineDash([2, 3]); ctx.lineWidth = 1;
    const marks = [fovea[0]];
    if (oct.roi_um) marks.push(fovea[0] - oct.roi_um / oct.scale_x, fovea[0] + oct.roi_um / oct.scale_x);
    for (const mark of marks) { ctx.beginPath(); ctx.moveTo(px(mark), PLOT.top); ctx.lineTo(px(mark), height - PLOT.bottom); ctx.stroke(); }
    ctx.setLineDash([]);
  }
  const cursor = octCursorX();
  if (cursor !== null) {
    ctx.strokeStyle = cssVar('--text'); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(px(cursor), PLOT.top); ctx.lineTo(px(cursor), height - PLOT.bottom); ctx.stroke();
  }
}
function drawStrip() {
  if (!ui.strip || S < 2) return;
  const [ctx, width, height] = sizeCanvas(ui.strip);
  const px = (index) => 4 + (index / (S - 1)) * (width - 8);
  ctx.strokeStyle = cssVar('--line'); ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(0, height - 6.5); ctx.lineTo(width, height - 6.5); ctx.stroke();
  const series = [['retina', cssVar('--muted')], ['choroid', color('CHORlower')]];
  for (const [name, stroke] of series) {
    const values = (oct.slice_thickness || {})[name];
    if (!values) continue;
    const finite = values.filter((value) => value !== null);
    if (finite.length < 2) continue;
    const low = Math.min(...finite), high = Math.max(...finite), span = high - low || 1;
    ctx.strokeStyle = stroke; ctx.beginPath();
    let pen = false;
    values.forEach((value, index) => {
      if (value === null) { pen = false; return; }
      const y = height - 10 - ((value - low) / span) * (height - 14);
      if (pen) ctx.lineTo(px(index), y); else { ctx.moveTo(px(index), y); pen = true; }
    });
    ctx.stroke();
  }
  ctx.fillStyle = cssVar('--fail');
  for (const index of flaggedSlices()) ctx.fillRect(px(index) - 1, height - 5, 2, 5);
  if (oct.fovea_slice !== null && oct.fovea_slice !== undefined) {
    ctx.fillStyle = cssVar('--text');
    ctx.beginPath(); ctx.arc(px(oct.fovea_slice), height - 6.5, 2.5, 0, 2 * Math.PI); ctx.fill();
  }
  ctx.strokeStyle = cssVar('--text'); ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(px(state.slice), 0); ctx.lineTo(px(state.slice), height); ctx.stroke();
}

// ----------------------------------------------------------------- redraws
let pending = 0;
function redraw(level = 1) {
  pending = Math.max(pending, level);
  if (redraw.queued) return;
  redraw.queued = true;
  requestAnimationFrame(() => {
    redraw.queued = false;
    const full = pending > 1;
    pending = 0;
    for (const pane of [octPane, sloPane, cleanPane]) {
      if (!pane) continue;
      if (full) pane.draw(); else pane.drawOverlay();
    }
    drawPlot(); drawStrip(); updateReadout();
  });
}
function mainPane() {
  if (prefs.expanded === 'slo' && sloPane) return sloPane;
  if (prefs.expanded === 'oct') return octPane;
  return prefs.swapped && sloPane ? sloPane : octPane;
}
function toggleExpand(id) {
  prefs.expanded = prefs.expanded === id ? null : id;
  savePrefs(); applyLayout();
}
function toggleRail() {
  prefs.railHidden = !prefs.railHidden;
  savePrefs(); applyLayout();
}
function expandButton(id) {
  const name = id === 'oct' ? 'B-scan' : 'en-face';
  const button = h('button', { class: 'icon tool', 'aria-label': `Expand the ${name} pane` });
  // Keep clicks on the button from starting a pan or selecting a B-scan underneath.
  for (const type of ['pointerdown', 'pointerup', 'dblclick']) button.addEventListener(type, (event) => event.stopPropagation());
  button.addEventListener('click', () => toggleExpand(id));
  button.refresh = () => {
    const expanded = prefs.expanded === id;
    button.replaceChildren(OctUI.icon(expanded ? 'collapse' : 'expand', 14));
    button.dataset.tip = expanded ? 'Show both panes again (E)' : `Expand the ${name} pane (E)`;
    button.setAttribute('aria-pressed', String(expanded));
  };
  ui.expand[id] = button;
  return button;
}
function renderDisplay() {
  if (state.gray) octPane.source = renderGray(state.gray, N, M, octPane.source || document.createElement('canvas'));
  if (sloPane && state.sloGray) sloPane.source = renderGray(state.sloGray, slo.width, slo.height, sloPane.source || document.createElement('canvas'));
  if (cleanPane) cleanPane.source = mainPane().source;
  redraw(2);
}
let sliceToken = 0;
async function setSlice(index) {
  index = clamp(Math.round(index), 0, S - 1);
  state.slice = index;
  const token = ++sliceToken;
  updateSliceUi();
  redraw(1);
  if (!octImages[index]) octImages[index] = loadImage(oct.images[index]);
  if (oct.vessels && oct.vessels[index] && !vesselImages[index]) {
    loadImage(oct.vessels[index]).then((image) => { vesselImages[index] = image; if (state.slice === index) redraw(1); });
  }
  const image = await octImages[index];
  if (token !== sliceToken) return;
  state.gray = grayscale(image);
  renderDisplay();
  for (const next of [index + 1, index - 1]) {
    if (next >= 0 && next < S && !octImages[next]) octImages[next] = loadImage(oct.images[next]);
  }
}
function stepFlagged(direction) {
  const flagged = flaggedSlices();
  if (!flagged.length) return;
  const next = direction > 0 ? flagged.find((index) => index > state.slice) : [...flagged].reverse().find((index) => index < state.slice);
  setSlice(next === undefined ? (direction > 0 ? flagged[0] : flagged[flagged.length - 1]) : next);
}
function applyAspect() {
  octPane.ax = prefs.aspect === 'physical' ? oct.scale_x / oct.scale_y : 1;
  octPane.view.fitted = true;
  if (cleanPane && !prefs.swapped) cleanPane.ax = octPane.ax;
  octPane.resize();
  if (cleanPane) cleanPane.resize();
}
function applyLayout() {
  const main = ui.main, side = ui.side;
  const big = mainPane();
  const small = big === octPane ? sloPane : octPane;
  ui.stage.classList.toggle('solo', Boolean(prefs.expanded) && Boolean(sloPane));
  ui.stage.classList.toggle('no-rail', Boolean(prefs.railHidden));
  if (ui.railButton) ui.railButton.setAttribute('aria-pressed', String(!prefs.railHidden));
  for (const button of Object.values(ui.expand)) button.refresh();
  main.prepend(big.host);
  if (small) side.prepend(small.host);
  if (cleanPane) {
    cleanPane.host.hidden = !prefs.split;
    cleanPane.host.style.display = prefs.split ? '' : 'none';
    cleanPane.w = big.w; cleanPane.h = big.h; cleanPane.ax = big.ax; cleanPane.ay = big.ay;
    cleanPane.view = big.view;
    cleanPane.source = big.source;
    big.peers = [cleanPane]; if (small) small.peers = [];
    cleanPane.peers = [big];
  }
  for (const pane of [octPane, sloPane]) if (pane) pane.view.fitted = true;
}

// --------------------------------------------------------------- rail UI
function section(id, title, ...body) {
  const node = h('details', { open: !prefs.closed[id] }, h('summary', { text: title }), h('div', { class: 'body' }, ...body));
  node.addEventListener('toggle', () => { prefs.closed[id] = !node.open; savePrefs(); });
  return node;
}
function slider(glyph, label, key, min, max, step, onChange) {
  const output = h('output', { text: prefs[key].toFixed(2) });
  const input = h('input', { type: 'range', min, max, step, value: prefs[key], 'aria-label': label });
  input.addEventListener('input', () => {
    prefs[key] = Number(input.value);
    output.textContent = prefs[key].toFixed(2);
    onChange(); savePrefs();
  });
  input.addEventListener('change', () => input.blur());
  ui.sliders = ui.sliders || {};
  ui.sliders[key] = { input, output };
  return h('div', { class: 'field' }, h('span', { class: 'glyph', 'data-tip': label }, OctUI.icon(glyph)), input, output);
}
function toggleButton(glyph, tip, isOn, onToggle) {
  const button = h('button', { class: 'icon', 'aria-pressed': String(isOn()), 'data-tip': tip, 'aria-label': tip }, OctUI.icon(glyph));
  button.addEventListener('click', () => { onToggle(); button.setAttribute('aria-pressed', String(isOn())); });
  button.refresh = () => button.setAttribute('aria-pressed', String(isOn()));
  return button;
}
function layerRow(item, hotkey) {
  const swatch = h('input', { type: 'color', class: 'swatch', value: color(item.id), 'aria-label': `${item.label} colour` });
  swatch.addEventListener('click', (event) => event.stopPropagation());
  swatch.addEventListener('input', () => { prefs.colors[item.id] = swatch.value; tintCache = new WeakMap(); savePrefs(); redraw(1); });
  const row = h('div', { class: 'layer', 'data-tip': item.title || null, role: 'checkbox', tabindex: '0' },
    swatch, h('span', { class: 'name', text: item.label }), hotkey ? h('kbd', { text: hotkey }) : null);
  row.refresh = () => { row.classList.toggle('off', !visible(item.id)); row.setAttribute('aria-checked', String(visible(item.id))); };
  row.addEventListener('click', () => toggleLayer(item.id));
  row.addEventListener('keydown', (event) => { if (event.key === 'Enter') toggleLayer(item.id); });
  row.refresh();
  item.row = row;
  return row;
}
function toggleLayer(id, value) {
  prefs.visible[id] = value === undefined ? !visible(id) : value;
  const item = layer(id);
  if (item && item.row) item.row.refresh();
  savePrefs(); redraw(1); updateMapUi();
}
function toggleAll() {
  const paneLayers = layers.filter((item) => item.pane === 'oct');
  const anyOn = paneLayers.some((item) => visible(item.id));
  for (const item of paneLayers) { prefs.visible[item.id] = !anyOn; item.row.refresh(); }
  savePrefs(); redraw(1);
}
function resetDisplay() {
  for (const key of ['brightness', 'contrast', 'gamma', 'opacity', 'lineWidth']) {
    prefs[key] = DEFAULT_PREFS[key];
    ui.sliders[key].input.value = prefs[key];
    ui.sliders[key].output.textContent = prefs[key].toFixed(2);
  }
  prefs.invert = false; ui.invert.refresh();
  savePrefs(); renderDisplay();
}
function updateMapUi() {
  if (!ui.mapSelect) return;
  const entry = currentMap();
  ui.mapSelect.value = visible('map') && entry ? entry.key : '';
  ui.legend.style.display = visible('map') && entry ? '' : 'none';
  for (const button of ui.colormaps) button.setAttribute('aria-pressed', String(button.dataset.colormap === (COLORMAPS[prefs.colormap] ? prefs.colormap : 'hot')));
  if (entry) {
    ui.legend.querySelector('.legend').style.background = colormapGradient(colormap().stops);
    const unit = entry.unit ? ` ${entry.unit.replace('um', 'µm')}` : '';
    ui.legend.querySelector('.low').textContent = format(entry.min) + unit;
    ui.legend.querySelector('.high').textContent = format(entry.max) + unit;
  }
}
function cycleMap() {
  if (!payload.maps.length) return;
  const keys = ['', ...payload.maps.map((entry) => entry.key)];
  const current = visible('map') ? (currentMap() || {}).key || '' : '';
  selectMap(keys[(keys.indexOf(current) + 1) % keys.length]);
}
function selectMap(key) {
  if (key) prefs.map = key;
  toggleLayer('map', Boolean(key));
  if (key && layer('grid')) toggleLayer('grid', true);
}
function table(data, sheet) {
  const scanColumn = data.columns.indexOf('scan_number');
  const rows = data.rows.map((row) => {
    const node = h('tr', {}, row.map((value) => h('td', { text: format(value, 3, true) })));
    if (scanColumn >= 0) node.dataset.scan = row[scanColumn];
    return node;
  });
  const title = sheet.replace(/_/g, ' ').replace(/^slo measurements/, 'SLO ·').replace(/^oct measurements/, 'OCT')
    .replace(/^etdrs/, 'ETDRS');
  return h('div', {}, h('div', { class: 'group-label', text: title }),
    h('div', { class: 'table-wrap' }, h('table', {},
      h('thead', {}, h('tr', {}, data.columns.map((column) => h('th', { text: column.replace(/_\[(.*)\]/, ' $1').replace(/_/g, ' ') })))),
      h('tbody', {}, rows))));
}
function buildRail() {
  ui.invert = toggleButton('invert', 'Invert greyscale (I)', () => prefs.invert, () => { prefs.invert = !prefs.invert; savePrefs(); renderDisplay(); });
  ui.aspect = toggleButton('aspect', 'Equal microns per pixel in both directions (A)', () => prefs.aspect === 'physical', () => {
    prefs.aspect = prefs.aspect === 'physical' ? 'pixel' : 'physical'; savePrefs(); applyAspect();
  });
  ui.splitButton = toggleButton('split', 'Show the scan without overlays alongside (S)', () => prefs.split,
    () => { prefs.split = !prefs.split; savePrefs(); applyLayout(); });
  ui.swapButton = toggleButton('swap', 'Swap the OCT and SLO panes (X)', () => prefs.swapped,
    () => { prefs.swapped = !prefs.swapped; savePrefs(); applyLayout(); });
  const display = section('display', 'Display',
    slider('sun', 'Brightness', 'brightness', -0.5, 0.5, 0.01, renderDisplay),
    slider('contrast', 'Contrast', 'contrast', 0.3, 3, 0.01, renderDisplay),
    slider('gamma', 'Gamma', 'gamma', 0.3, 3, 0.01, renderDisplay),
    slider('opacity', 'Overlay opacity', 'opacity', 0, 1, 0.01, () => redraw(1)),
    slider('line', 'Boundary line width', 'lineWidth', 0.5, 4, 0.05, () => redraw(1)),
    h('div', { class: 'row' }, ui.invert, ui.aspect, ui.splitButton, slo ? ui.swapButton : null,
      iconButton('reset', 'Reset display and view (Shift + R)', () => { resetDisplay(); for (const p of [octPane, sloPane]) if (p) { p.fit(); } redraw(2); })));

  const octLayers = layers.filter((item) => item.pane === 'oct');
  const sloLayers = layers.filter((item) => item.pane === 'slo');
  const layerSection = section('layers', 'Layers',
    octLayers.length ? h('div', { class: 'group-label', text: 'OCT' }) : null,
    octLayers.map((item, index) => layerRow(item, index < 9 ? String(index + 1) : '')),
    sloLayers.length ? h('div', { class: 'group-label', text: 'SLO' }) : null,
    sloLayers.map((item) => layerRow(item, '')),
    payload.unavailable.length ? h('div', { class: 'muted', style: 'margin-top:8px', text: `Not saved for this file: ${payload.unavailable.join(', ').toLowerCase()}.` }) : null);

  const sections = [display, layerSection];
  if (payload.maps.length) {
    ui.mapSelect = h('select', { 'aria-label': 'En-face map', style: 'width:100%' },
      h('option', { value: '', text: 'None' }),
      payload.maps.map((entry) => h('option', { value: entry.key, text: entry.label })));
    ui.mapSelect.addEventListener('change', () => selectMap(ui.mapSelect.value));
    ui.legend = h('div', {}, h('div', { class: 'legend' }),
      h('div', { class: 'legend-scale' }, h('span', { class: 'low' }), h('span', { class: 'high' })));
    ui.colormaps = Object.entries(COLORMAPS).map(([name, { label, stops }]) => h('button', {
      class: 'cmap', 'data-colormap': name, 'data-tip': `${label} colour map (C)`, 'aria-label': `${label} colour map`,
      'aria-pressed': 'false', style: `background:${colormapGradient(stops)}`, onclick: () => setColormap(name),
    }));
    sections.push(section('map', 'En-face map', ui.mapSelect, h('div', { class: 'row' }, ui.colormaps), ui.legend));
  }
  ui.readout = h('div', { class: 'kv' });
  sections.push(section('readout', 'Thickness at cursor', ui.readout));
  if (manifest.flags.length) {
    sections.push(section('flags', `Automatic flags (${manifest.flags.length})`, manifest.flags.map((flag) => {
      const jump = flag.slices && flag.slices.length && S > 1;
      return h('div', { class: `flag${jump ? ' jump' : ''}`, onclick: jump ? () => jumpFlag(flag) : null },
        h('div', { class: 'head' }, h('span', { class: `dot ${['', '', 'borderline', 'fail'][flag.severity] || ''}` }), h('span', { text: flag.label })),
        h('div', { class: 'detail', text: flag.detail }));
    })));
  }
  const sheets = Object.entries(payload.measurements);
  if (sheets.length) {
    ui.tables = h('div', {}, sheets.map(([sheet, data]) => table(data, sheet)));
    sections.push(section('measurements', 'Measurements', ui.tables));
  }
  const meta = Object.entries(payload.metadata).filter(([, value]) => value !== null && value !== '');
  sections.push(section('metadata', 'Metadata', h('div', { class: 'kv' },
    meta.flatMap(([key, value]) => [h('span', { class: 'k', text: key.replace(/_/g, ' '), title: key }), h('span', { class: 'v', text: format(value, 5) })]))));
  if (payload.log.length) {
    if (!('log' in prefs.closed)) prefs.closed.log = true;
    sections.push(section('log', 'Processing log', h('div', { class: 'log', text: payload.log.join('\n').trim() })));
  }
  return h('aside', { class: 'rail' }, sections);
}
function jumpFlag(flag) {
  const next = flag.slices.find((index) => index > state.slice);
  setSlice(next === undefined ? flag.slices[0] : next);
}
function updateSliceUi() {
  if (ui.count) {
    const quality = oct.quality[state.slice];
    ui.count.textContent = `B-scan ${state.slice + 1} / ${S}` + (typeof quality === 'number' ? ` · Q ${quality.toFixed(1)}` : '');
  }
  if (ui.tables) {
    for (const row of ui.tables.querySelectorAll('tr[data-scan]')) {
      const current = Number(row.dataset.scan) === state.slice;
      row.classList.toggle('current', current);
      row.classList.toggle('other', !current);
    }
  }
}
function updateReadout() {
  if (!ui.readout) return;
  const cursor = octCursorX();
  const parts = [];
  const rows = [];
  if (cursor !== null) {
    const index = clamp(Math.round(cursor), 0, N - 1);
    parts.push(`A-scan ${index + 1}`);
    const fovea = oct.foveas[state.slice];
    if (fovea) parts.push(`${Math.round((index - fovea[0]) * oct.scale_x)} µm from fovea`);
    for (const [name, upper, lower] of profilePairs()) {
      const value = thickness(upper, lower, index);
      if (value === value) parts.push(`${name.toLowerCase()} ${Math.round(value)} µm`);
    }
    const shown = oct.layers.filter((item) => visible(item.key)).map((item) => item.key);
    for (let i = 0; i + 1 < shown.length; i++) {
      const value = thickness(shown[i], shown[i + 1], index);
      rows.push([`${shown[i]} – ${shown[i + 1]}`, value === value ? `${Math.round(value)} µm` : '–']);
    }
  }
  if (state.hover && state.hover.map !== undefined) parts.push(state.hover.map);
  ui.status.textContent = parts.length ? parts.join(' · ') : ui.statusDefault;
  const signature = rows.map((row) => row.join('=')).join('|');
  if (signature !== ui.readout.dataset.signature) {
    ui.readout.dataset.signature = signature;
    ui.readout.replaceChildren(...(rows.length
      ? rows.flatMap(([key, value]) => [h('span', { class: 'k', text: key }), h('span', { class: 'v', text: value })])
      : [h('span', { class: 'k', text: 'Hover over the B-scan' })]));
  }
}

// ----------------------------------------------------------------- verdict
function renderVerdict() {
  for (const button of ui.grades) button.setAttribute('aria-pressed', String(button.dataset.grade === state.verdict.grade));
  for (const button of ui.structures) button.setAttribute('aria-pressed', String(Boolean(state.verdict.flags[button.dataset.structure])));
  if (document.activeElement !== ui.note) ui.note.value = state.verdict.note || '';
}
function commitVerdict(advance = false) {
  renderVerdict();
  if (embedded) {
    post({ type: 'verdict', filename: manifest.filename, verdict: state.verdict, advance });
  } else {
    store(VERDICT_KEY, state.verdict);
    ui.saved.textContent = 'Saved in browser';
  }
}
function setGrade(grade, fromKeyboard = false) {
  state.verdict.grade = state.verdict.grade === grade ? '' : grade;
  commitVerdict(fromKeyboard && state.verdict.grade === 'pass');
}
function csvCell(value) {
  const text = String(value === null || value === undefined ? '' : value);
  return /[",\n\r]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}
function exportVerdict() {
  const header = ['Filename', 'scan_type', 'eye', 'verdict', ...STRUCTURES.map(([id]) => `flag_${id}`), 'note', 'auto_flags', 'severity', 'reviewer', 'timestamp'];
  const row = [manifest.source || `${manifest.filename}.vol`, manifest.scan_type, manifest.eye, state.verdict.grade,
    ...STRUCTURES.map(([id]) => (state.verdict.flags[id] ? 1 : 0)), state.verdict.note,
    manifest.flags.map((flag) => flag.id).join(';'), manifest.severity, '', new Date().toISOString()];
  const blob = new Blob([header.map(csvCell).join(',') + '\n' + row.map(csvCell).join(',') + '\n'], { type: 'text/csv' });
  const link = h('a', { href: URL.createObjectURL(blob), download: `${manifest.filename}_qc.csv` });
  link.click();
  URL.revokeObjectURL(link.href);
}
function buildVerdict() {
  ui.grades = GRADES.map(([grade, label, key, glyph]) => h('button', {
    class: 'icon', 'data-grade': grade, 'data-tip': `${label} (${key})`, 'aria-label': label, 'aria-pressed': 'false',
    onclick: () => setGrade(grade),
  }, OctUI.icon(glyph, 16)));
  ui.structures = STRUCTURES.filter(([id]) => manifest.structures.includes(id)).map(([id, label]) => h('button', {
    class: 'icon', 'data-structure': id, 'aria-pressed': 'false', 'aria-label': `Problem with ${label.toLowerCase()}`,
    'data-tip': `Problem with ${label.toLowerCase()}`,
    onclick: () => { state.verdict.flags[id] = !state.verdict.flags[id]; commitVerdict(); },
  }, OctUI.icon(id, 16)));
  ui.note = h('input', { type: 'text', placeholder: 'Note (N)', 'aria-label': 'Review note' });
  ui.note.addEventListener('input', () => { state.verdict.note = ui.note.value; commitVerdict(); });
  ui.saved = h('span', { class: 'saved' });
  return h('footer', { class: 'verdict' },
    h('div', { class: 'grades' }, ui.grades), h('div', { class: 'structures' }, ui.structures), ui.note,
    embedded ? null : iconButton('download', 'Download this verdict as CSV', exportVerdict),
    ui.saved);
}

// ---------------------------------------------------------------- keyboard
const HELP = [
  ['↑ ↓ / wheel', 'Previous / next B-scan'], ['[ ]', 'Previous / next flagged B-scan'],
  ['Ctrl + wheel, + −', 'Zoom'], ['Drag', 'Pan'], ['Double-click, R', 'Fit to window'],
  ['1 – 9', 'Toggle OCT layer'], ['0', 'Toggle all OCT layers'], ['Space (hold)', 'Hide overlays'],
  ['S', 'Split view with clean scan'], ['X', 'Swap OCT and SLO panes'], ['E', 'Expand the main pane / show both'],
  [',  .', 'Hide or show the scan list (hub) / settings panel'], ['A', 'Pixel / 1:1 micron aspect'],
  ['I', 'Invert'], ['M', 'Cycle en-face maps'], ['C', 'Switch en-face colour map'], ['Shift + R', 'Reset display'],
  ['P  B  F', 'Pass / Borderline / Fail'], ['N', 'Write a note'], ['J  K', 'Next / previous file (hub)'],
  ['T', 'Light / dark theme'], ['?', 'This help'],
];
function handleKey(type, event) {
  const key = event.key;
  if (type === 'keyup') {
    if (key === ' ' && state.blink) { state.blink = false; redraw(1); }
    return false;
  }
  if (event.ctrlKey || event.metaKey || event.altKey) return false;
  if (ui.help.classList.contains('open')) { ui.help.classList.remove('open'); return true; }
  const lower = key.length === 1 ? key.toLowerCase() : key;
  switch (lower) {
    case 'ArrowUp': setSlice(state.slice - 1); return true;
    case 'ArrowDown': setSlice(state.slice + 1); return true;
    case 'PageUp': setSlice(state.slice - 5); return true;
    case 'PageDown': setSlice(state.slice + 5); return true;
    case 'Home': setSlice(0); return true;
    case 'End': setSlice(S - 1); return true;
    case '[': stepFlagged(-1); return true;
    case ']': stepFlagged(1); return true;
    case ' ': if (!state.blink) { state.blink = true; redraw(1); } return true;
    case '0': toggleAll(); return true;
    case '+': case '=': mainPane().zoomAt(mainPane().cw / 2, mainPane().ch / 2, 1.25); return true;
    case '-': case '_': mainPane().zoomAt(mainPane().cw / 2, mainPane().ch / 2, 0.8); return true;
    case 'r':
      if (event.shiftKey) resetDisplay();
      for (const pane of [octPane, sloPane]) if (pane) pane.fit();
      redraw(2); return true;
    case 's': ui.splitButton.click(); return true;
    case 'x': if (sloPane) ui.swapButton.click(); return true;
    case 'a': ui.aspect.click(); return true;
    case 'i': ui.invert.click(); return true;
    case 'm': cycleMap(); return true;
    case 'c': {
      if (!payload.maps.length) return true;
      const names = Object.keys(COLORMAPS);
      setColormap(names[(names.indexOf(prefs.colormap) + 1) % names.length]);
      return true;
    }
    case 't': prefs.theme = prefs.theme === 'dark' ? 'light' : 'dark'; applyTheme(); savePrefs(); redraw(2); return true;
    case 'p': setGrade('pass', true); return true;
    case 'b': setGrade('borderline', true); return true;
    case 'f': setGrade('fail', true); return true;
    case 'n': ui.note.focus(); return true;
    case 'e': toggleExpand(mainPane() === sloPane ? 'slo' : 'oct'); return true;
    case '.': toggleRail(); return true;
    case ',': post({ type: 'panel' }); return true;
    case 'j': post({ type: 'nav', delta: 1 }); return true;
    case 'k': post({ type: 'nav', delta: -1 }); return true;
    case '?': ui.help.classList.add('open'); return true;
    default:
      if (/^[1-9]$/.test(key)) {
        const item = layers.filter((entry) => entry.pane === 'oct')[Number(key) - 1];
        if (item) toggleLayer(item.id);
        return true;
      }
  }
  return false;
}
function bindKeys() {
  const typing = (target) => target && (target.tagName === 'INPUT' && target.type === 'text' || target.tagName === 'TEXTAREA' || target.tagName === 'SELECT');
  for (const type of ['keydown', 'keyup']) {
    window.addEventListener(type, (event) => {
      if (typing(event.target)) {
        if (type === 'keydown' && (event.key === 'Escape' || event.key === 'Enter')) event.target.blur();
        return;
      }
      if (payload.failed && !/^[pbfnjkt?,]$/i.test(event.key)) return;
      if (handleKey(type, event)) event.preventDefault();
    });
  }
  window.addEventListener('blur', () => { if (state.blink) { state.blink = false; redraw(1); } });
  document.addEventListener('click', (event) => {
    const button = event.target.closest && event.target.closest('button');
    if (button) button.blur();
  });
}
function bindHub() {
  window.addEventListener('message', (event) => {
    const message = event.data;
    if (!message || message.octolyzer !== 'hub') return;
    if (message.type === 'init') {
      Object.assign(prefs, message.prefs || {});
      mapCache.clear();
      state.verdict = { ...emptyVerdict(), ...(message.verdict || {}) };
      state.verdict.flags = { ...(state.verdict.flags || {}) };
      applyTheme();
      if (ui.ready) ui.ready();
      if (ui.grades) renderVerdict();
    } else if (message.type === 'key' && (!payload.failed || /^[pbfnt?]$/i.test(message.key))) {
      handleKey(message.event, message);
    } else if (message.type === 'saved') {
      ui.saved.textContent = message.text || '';
    }
  });
}

// -------------------------------------------------------------------- boot
function header() {
  const info = [manifest.scan_type, manifest.eye && `${manifest.eye} eye`, S > 1 && `${S} B-scans`].filter(Boolean).join(' · ');
  return h('header', { class: 'bar' },
    h('span', { class: 'title', text: manifest.filename }), h('span', { class: 'muted', text: info }),
    h('div', { class: 'chips' }, manifest.flags.map((flag) => {
      const jump = flag.slices && flag.slices.length && S > 1;
      return h('span', {
        class: `chip s${flag.severity}${jump ? ' jump' : ''}`, 'data-tip': `${flag.label}\n${flag.detail}`, 'aria-label': flag.label,
        onclick: jump ? () => jumpFlag(flag) : null,
      }, OctUI.icon('flag', 12));
    })),
    h('span', { class: 'spacer' }),
    payload.failed ? null : (ui.railButton = h('button', {
      class: 'icon quiet', 'data-tip': 'Hide or show the settings panel (.)', 'aria-label': 'Settings panel',
      'aria-pressed': String(!prefs.railHidden), onclick: toggleRail,
    }, OctUI.icon('panel_right'))),
    iconButton('moon', 'Light / dark theme (T)', () => handleKey('keydown', { key: 't' })),
    iconButton('help', 'Keyboard and mouse shortcuts (?)', () => ui.help.classList.add('open')));
}
function helpCard() {
  ui.help = h('div', { class: 'help', onclick: () => ui.help.classList.remove('open') },
    h('div', { class: 'card' }, h('h2', { text: 'Keyboard and mouse' }),
      HELP.flatMap(([keys, text]) => [h('span', { class: 'mono muted', text: keys }), h('span', { text })])));
  return ui.help;
}
function bootFailed() {
  app.append(header(),
    h('div', { class: 'message' }, h('h1', { text: 'Analysis failed' }),
      h('p', { class: 'muted', text: 'OCTolyzer could not analyse this file, so there are no segmentations to review.' }),
      h('div', { class: 'log', text: (payload.log || []).join('\n').trim() })),
    h('div'), buildVerdict(), helpCard());
}
async function boot() {
  applyTheme();
  bindHub();
  bindKeys();
  if (payload.failed) {
    bootFailed();
  } else {
    if (typeof DecompressionStream === 'undefined') {
      app.append(h('div', { class: 'message' }, h('h1', { text: 'Browser not supported' }),
        h('p', { class: 'muted', text: 'Please open this file in a current version of Chrome, Edge, Firefox or Safari.' })));
      return;
    }
    for (const item of oct.layers) {
      const raw = await unpack16(item.data, S, N);
      const values = new Float32Array(raw.length);
      for (let i = 0; i < raw.length; i++) values[i] = raw[i] ? raw[i] / oct.trace_scale : NaN;
      traces[item.key] = values;
    }
    for (const entry of payload.maps) mapData[entry.key] = await unpack16(entry.data, entry.size, entry.size);
    defineLayers();

    const octHost = h('section', { class: 'pane' }, h('span', { class: 'tag', text: 'OCT' }), slo ? expandButton('oct') : null);
    const cleanHost = h('section', { class: 'pane' }, h('span', { class: 'tag', text: 'Clean' }));
    ui.main = h('div', { class: 'main' }, octHost, cleanHost);
    ui.plot = h('canvas');
    ui.side = h('aside', { class: 'side' }, h('div', { class: 'plot' }, ui.plot));
    ui.strip = h('canvas');
    ui.count = h('span', { class: 'count mono' });
    ui.statusDefault = S > 1 ? 'Scroll or use ↑ ↓ to move through B-scans · ? for shortcuts' : 'Hover for thickness · ? for shortcuts';
    ui.status = h('span', { class: 'status', text: ui.statusDefault });
    const stripHost = h('div', { class: 'strip' }, ui.strip);
    const scrub = h('div', { class: `scrub${S > 1 ? '' : ' single'}` }, ui.count, stripHost, ui.status);
    const stage = ui.stage = h('main', { class: `stage${slo ? '' : ' no-side'}` }, ui.main, ui.side);
    app.append(header(), stage, scrub, buildVerdict(), helpCard());

    octPane = new Pane(octHost, N, M);
    cleanPane = new Pane(cleanHost, N, M, octPane.view);
    octPane.onOverlay = drawOct;
    let wheel = 0;
    octPane.onScroll = (delta) => {
      if (S < 2) return false;
      wheel += delta;
      if (Math.abs(wheel) >= 30) { setSlice(state.slice + Math.sign(wheel)); wheel = 0; }
      return true;
    };
    octPane.onHover = (x) => {
      state.hover = x === null || x < 0 || x >= N ? null : { octX: clamp(x - 0.5, 0, N - 1) };
      redraw(1);
    };
    if (slo) {
      const sloHost = h('section', { class: 'pane' }, h('span', { class: 'tag', text: 'SLO' }), expandButton('slo'));
      ui.side.prepend(sloHost);
      sloPane = new Pane(sloHost, slo.width, slo.height);
      sloPane.onOverlay = drawSlo;
      sloPane.onHover = (x, y) => {
        if (x === null) { state.hover = null; redraw(1); return; }
        const hover = {};
        const hit = nearestLine(x, y);
        if (hit.index === state.slice && hit.distance * sloPane.view.s < 10) hover.octX = hit.t * (N - 1);
        if (circle && Math.abs(Math.hypot(x - circle.cx, y - circle.cy) - circle.r) * sloPane.view.s < 10) {
          const angle = (Math.atan2(y - circle.cy, x - circle.cx) + 2 * Math.PI) % (2 * Math.PI);
          hover.octX = angle / (2 * Math.PI) * (N - 1);
        }
        const entry = currentMap();
        if (visible('map') && entry && mapData[entry.key] && x >= 0 && y >= 0 && x < slo.width && y < slo.height) {
          const column = Math.floor(x / slo.width * entry.size), row = Math.floor(y / slo.height * entry.size);
          const raw = mapData[entry.key][row * entry.size + column];
          if (raw) hover.map = `${entry.label} ${format(raw / entry.scale)}${entry.unit ? ' ' + entry.unit.replace('um', 'µm') : ''}`;
        }
        state.hover = hover;
        redraw(1);
      };
      sloPane.onClick = (sx, sy) => {
        if (!lines || S < 2) return;
        const hit = nearestLine(...sloPane.toImage(sx, sy));
        if (hit.index >= 0 && hit.distance * sloPane.view.s < 12) setSlice(hit.index);
      };
      loadImage(slo.image).then((image) => { state.sloGray = grayscale(image); renderDisplay(); });
      for (const [name, uri] of Object.entries(slo.masks || {})) {
        if (uri) loadImage(uri).then((image) => { maskImages[name] = image; redraw(1); });
      }
    }
    stage.append(buildRail());
    updateMapUi();

    const pick = (event) => {
      const rect = stripHost.getBoundingClientRect();
      setSlice(((event.clientX - rect.left - 4) / (rect.width - 8)) * (S - 1));
    };
    stripHost.addEventListener('pointerdown', (event) => { stripHost.setPointerCapture(event.pointerId); pick(event); });
    stripHost.addEventListener('pointermove', (event) => { if (event.buttons & 1) pick(event); });
    ui.plot.addEventListener('pointermove', (event) => {
      const rect = ui.plot.getBoundingClientRect();
      const t = (event.clientX - rect.left - PLOT.left) / (rect.width - PLOT.left - PLOT.right);
      state.hover = t < 0 || t > 1 ? null : { octX: t * (N - 1) };
      redraw(1);
    });
    ui.plot.addEventListener('pointerleave', () => { state.hover = null; redraw(1); });
    new ResizeObserver(() => redraw(1)).observe(ui.plot);
    new ResizeObserver(() => redraw(1)).observe(ui.strip);

    ui.ready = () => {
      applyAspect(); applyLayout();
      for (const item of layers) if (item.row) item.row.refresh();
      for (const key of Object.keys(ui.sliders)) { ui.sliders[key].input.value = prefs[key]; ui.sliders[key].output.textContent = prefs[key].toFixed(2); }
      for (const button of [ui.invert, ui.aspect, ui.splitButton, ui.swapButton]) button.refresh();
      updateMapUi(); renderDisplay();
    };
    ui.ready();
    await setSlice(oct.fovea_slice === null || oct.fovea_slice === undefined ? 0 : oct.fovea_slice);
  }
  if (!embedded) {
    state.verdict = { ...emptyVerdict(), ...(store(VERDICT_KEY) || {}) };
    state.verdict.flags = { ...(state.verdict.flags || {}) };
  }
  renderVerdict();
  post({ type: 'ready', manifest });
}

boot().catch((error) => {
  console.error(error);
  app.append(h('div', { class: 'message' }, h('h1', { text: 'Unable to open this review file' }), h('p', { class: 'muted', text: String(error.message || error) })));
});
})();
