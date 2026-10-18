(() => {
'use strict';

const STRUCTURES = ['retinal_layers', 'choroid_boundary', 'choroid_vessels', 'oct_fovea', 'slo_vessels',
  'artery_vein', 'optic_disc', 'slo_fovea', 'image_quality'];
const COLUMNS = ['Filename', 'scan_type', 'eye', 'verdict', ...STRUCTURES.map((id) => `flag_${id}`),
  'note', 'auto_flags', 'severity', 'reviewer', 'timestamp'];
const CSV_NAME = 'qc_review.csv';
const MANIFEST = /<script type="application\/json" id="octolyzer-manifest">([\s\S]*?)<\/script>/;
const MANIFEST_WINDOW = 65536;
const PREFS_KEY = 'octolyzer-review-prefs';
const REVIEWER_KEY = 'octolyzer-review-reviewer';
const SIDEBAR_KEY = 'octolyzer-review-list-hidden';
const root = document.getElementById('hub');

const state = {
  entries: [], current: null, rows: new Map(), reviewer: '', directory: null, folder: '',
  prefs: {}, dirty: false, url: null, search: '', type: '', status: '', sort: 'name',
};
const ui = {};

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
function store(key, value) {
  try {
    if (value === undefined) return JSON.parse(localStorage.getItem(key));
    localStorage.setItem(key, JSON.stringify(value));
  } catch (error) { /* storage is unavailable for some local files */ }
  return null;
}
function csvCell(value) {
  const text = String(value === null || value === undefined ? '' : value);
  return /[",\n\r]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
}
function parseCsv(text) {
  const rows = [];
  let row = [], cell = '', quoted = false;
  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    if (quoted) {
      if (char === '"' && text[i + 1] === '"') { cell += '"'; i++; }
      else if (char === '"') quoted = false;
      else cell += char;
    } else if (char === '"') quoted = true;
    else if (char === ',') { row.push(cell); cell = ''; }
    else if (char === '\n' || char === '\r') {
      if (char === '\r' && text[i + 1] === '\n') i++;
      row.push(cell); cell = '';
      if (row.some((value) => value !== '')) rows.push(row);
      row = [];
    } else cell += char;
  }
  row.push(cell);
  if (row.some((value) => value !== '')) rows.push(row);
  if (!rows.length) return [];
  const header = rows.shift().map((name) => name.replace(/^﻿/, '').trim());
  return rows.map((values) => Object.fromEntries(header.map((name, index) => [name, values[index] || ''])));
}
const rowKey = (filename, reviewer) => `${filename}\u0000${reviewer}`;
const source = (entry) => entry.manifest.source || `${entry.manifest.filename}.vol`;

// ------------------------------------------------------------------ verdicts
function verdictOf(entry) {
  const row = state.rows.get(rowKey(source(entry), state.reviewer));
  if (!row) return { grade: '', flags: {}, note: '' };
  return {
    grade: row.verdict || '',
    flags: Object.fromEntries(STRUCTURES.filter((id) => String(row[`flag_${id}`]) === '1').map((id) => [id, true])),
    note: row.note || '',
  };
}
function setVerdict(entry, verdict) {
  const key = rowKey(source(entry), state.reviewer);
  const flagged = STRUCTURES.some((id) => verdict.flags && verdict.flags[id]);
  if (!verdict.grade && !flagged && !verdict.note) {
    state.rows.delete(key);
  } else {
    const row = {
      Filename: source(entry), scan_type: entry.manifest.scan_type, eye: entry.manifest.eye,
      verdict: verdict.grade || '', note: verdict.note || '',
      auto_flags: (entry.manifest.flags || []).map((flag) => flag.id).join(';'),
      severity: entry.manifest.severity || 0, reviewer: state.reviewer, timestamp: new Date().toISOString(),
    };
    for (const id of STRUCTURES) row[`flag_${id}`] = verdict.flags && verdict.flags[id] ? 1 : 0;
    state.rows.set(key, row);
  }
  scheduleSave();
}
function serialise() {
  const rows = [...state.rows.values()].sort((a, b) =>
    a.Filename.localeCompare(b.Filename, undefined, { numeric: true }) || String(a.reviewer).localeCompare(String(b.reviewer)));
  return [COLUMNS, ...rows.map((row) => COLUMNS.map((name) => row[name]))]
    .map((cells) => cells.map(csvCell).join(',')).join('\r\n') + '\r\n';
}
function mergeRows(rows) {
  for (const row of rows) {
    if (!row.Filename) continue;
    const key = rowKey(row.Filename, row.reviewer || '');
    const existing = state.rows.get(key);
    if (!existing || String(row.timestamp) > String(existing.timestamp)) state.rows.set(key, row);
  }
}
let saveTimer = null;
function scheduleSave() {
  state.dirty = true;
  store(`octolyzer-review-rows:${state.folder}`, [...state.rows.values()]);
  clearTimeout(saveTimer);
  saveTimer = setTimeout(save, 350);
}
async function save() {
  if (!state.directory) {
    setStatus('Not saved to disk · export when done', true);
    return;
  }
  try {
    const handle = await state.directory.getFileHandle(CSV_NAME, { create: true });
    const writable = await handle.createWritable();
    await writable.write(serialise());
    await writable.close();
    state.dirty = false;
    const text = `Saved ${new Date().toLocaleTimeString()}`;
    setStatus(`${CSV_NAME} · ${text.toLowerCase()}`);
    sendToViewer({ type: 'saved', text });
  } catch (error) {
    setStatus(`Could not write ${CSV_NAME} (is it open in another program?)`, true);
  }
}
function exportCsv() {
  const link = h('a', { href: URL.createObjectURL(new Blob([serialise()], { type: 'text/csv' })), download: CSV_NAME });
  link.click();
  URL.revokeObjectURL(link.href);
  if (!state.directory) { state.dirty = false; setStatus('Exported'); }
}
function setStatus(text, warn = false) {
  ui.status.textContent = text;
  ui.status.title = text;
  ui.status.classList.toggle('warn', warn);
}

// -------------------------------------------------------------- file access
async function readManifest(file) {
  const match = MANIFEST.exec(await file.slice(0, MANIFEST_WINDOW).text());
  if (!match) return null;
  try {
    const manifest = JSON.parse(match[1]);
    return manifest && manifest.filename ? manifest : null;
  } catch (error) { return null; }
}
async function loadFiles(items, directory, folder) {
  setStatus('Reading files…');
  const entries = [];
  let csvText = '';
  for (const item of items) {
    if (item.name === CSV_NAME) { csvText = await (await item.getFile()).text(); continue; }
    if (!/\.html?$/i.test(item.name)) continue;
    const manifest = await readManifest(await item.getFile());
    if (manifest) entries.push({ manifest, getFile: item.getFile });
  }
  entries.sort((a, b) => a.manifest.filename.localeCompare(b.manifest.filename, undefined, { numeric: true }));
  state.entries = entries;
  state.directory = directory || null;
  state.folder = folder || '';
  state.rows = new Map();
  mergeRows(parseCsv(csvText));
  mergeRows(store(`octolyzer-review-rows:${state.folder}`) || []);
  state.dirty = false;
  ui.folder.textContent = folder || 'No folder';
  ui.folder.title = folder || '';
  const types = [...new Set(entries.map((entry) => entry.manifest.scan_type).filter(Boolean))].sort();
  ui.type.replaceChildren(h('option', { value: '', text: 'All scan types' }), ...types.map((type) => h('option', { value: type, text: type })));
  setStatus(entries.length
    ? (directory ? `Verdicts save to ${CSV_NAME}` : 'Not saved to disk · export when done')
    : 'No review files found in this folder', !entries.length || !directory);
  renderList();
  const first = visibleEntries().find((entry) => !verdictOf(entry).grade) || visibleEntries()[0];
  if (first) select(first);
  else showWelcome('No review files were found. Choose the OCTolyzer output folder (the one containing analysis_output.xlsx).');
}
async function collectDirectory(handle) {
  const items = [];
  const add = (entry) => items.push({ name: entry.name, getFile: () => entry.getFile() });
  for await (const entry of handle.values()) {
    if (entry.kind === 'file') add(entry);
    else if (entry.name === 'review') for await (const child of entry.values()) if (child.kind === 'file' && child.name !== CSV_NAME) add(child);
  }
  return items;
}
async function openFolder() {
  if (state.dirty && !state.directory && !confirm('Verdicts have not been exported. Open another folder anyway?')) return;
  if (!window.showDirectoryPicker) { ui.picker.click(); return; }
  let handle;
  try {
    handle = await window.showDirectoryPicker({ id: 'octolyzer-review', mode: 'readwrite' });
  } catch (error) {
    if (error.name === 'AbortError') return;
    try { handle = await window.showDirectoryPicker({ id: 'octolyzer-review' }); } catch (inner) { return; }
  }
  let writable = false;
  try { writable = (await handle.queryPermission({ mode: 'readwrite' })) === 'granted'; } catch (error) { writable = false; }
  askReviewer();
  await loadFiles(await collectDirectory(handle), writable ? handle : null, handle.name);
}
function askReviewer() {
  if (state.reviewer) return;
  const answer = prompt('Reviewer initials (saved with every verdict):', '');
  state.reviewer = (answer || '').trim();
  ui.reviewer.value = state.reviewer;
  store(REVIEWER_KEY, state.reviewer);
}
function openFallback(fileList) {
  const files = [...fileList];
  if (!files.length) return;
  askReviewer();
  const folder = (files[0].webkitRelativePath || '').split('/')[0];
  const items = files.filter((file) => {
    const parts = (file.webkitRelativePath || file.name).split('/').slice(1);
    return parts.length === 1 || (parts.length === 2 && parts[0] === 'review' && file.name !== CSV_NAME);
  }).map((file) => ({ name: file.name, getFile: async () => file }));
  loadFiles(items, null, folder);
}

// ---------------------------------------------------------------------- list
const SHORT = { Ppole: 'Volume', Peripapillary: 'Peripap.' };
function visibleEntries() {
  const search = state.search.trim().toLowerCase();
  const entries = state.entries.filter((entry) => {
    const grade = verdictOf(entry).grade;
    if (search && !entry.manifest.filename.toLowerCase().includes(search)) return false;
    if (state.type && entry.manifest.scan_type !== state.type) return false;
    if (state.status === 'unreviewed') return !grade;
    if (state.status === 'flagged') return (entry.manifest.flags || []).length > 0;
    if (state.status) return grade === state.status;
    return true;
  });
  if (state.sort === 'severity') entries.sort((a, b) => (b.manifest.severity || 0) - (a.manifest.severity || 0));
  if (state.sort === 'unreviewed') entries.sort((a, b) => Boolean(verdictOf(a).grade) - Boolean(verdictOf(b).grade));
  return entries;
}
function renderList() {
  const entries = visibleEntries();
  const reviewed = state.entries.filter((entry) => verdictOf(entry).grade).length;
  ui.count.textContent = state.entries.length ? `${reviewed} / ${state.entries.length} reviewed` : '';
  ui.bar.style.width = state.entries.length ? `${(100 * reviewed) / state.entries.length}%` : '0';
  if (!entries.length) {
    ui.list.replaceChildren(h('div', { class: 'empty-list', text: state.entries.length ? 'No scans match the filters.' : 'No folder open.' }));
    return;
  }
  ui.list.replaceChildren(...entries.map((entry) => {
    const manifest = entry.manifest;
    const flags = manifest.flags || [];
    const worst = flags.reduce((max, flag) => Math.max(max, flag.severity || 0), 0);
    const info = [manifest.failed ? 'Failed' : SHORT[manifest.scan_type] || manifest.scan_type, (manifest.eye || '').slice(0, 1)].filter(Boolean).join(' ');
    const node = h('div', { class: `item${entry === state.current ? ' current' : ''}`, role: 'option', onclick: () => select(entry) },
      h('span', { class: `dot ${verdictOf(entry).grade}` }),
      h('span', { class: 'name', text: manifest.filename }),
      h('span', { class: 'info' }, info, flags.length
        ? h('span', { class: `auto s${worst}`, 'data-tip': flags.map((flag) => flag.label).join('\n') }, OctUI.icon('flag', 11), String(flags.length))
        : null));
    entry.node = node;
    return node;
  }));
  if (state.current && state.current.node && state.current.node.isConnected) state.current.node.scrollIntoView({ block: 'nearest' });
}
function showWelcome(text) {
  ui.frame.style.display = 'none';
  ui.welcome.style.display = '';
  ui.welcomeText.textContent = text;
}
async function select(entry) {
  state.current = entry;
  renderList();
  const file = await entry.getFile();
  if (state.current !== entry) return;
  if (state.url) URL.revokeObjectURL(state.url);
  state.url = URL.createObjectURL(new Blob([file], { type: 'text/html' }));
  ui.welcome.style.display = 'none';
  ui.frame.style.display = '';
  ui.frame.src = state.url;
}
function sendToViewer(message) {
  if (ui.frame.contentWindow && state.current) ui.frame.contentWindow.postMessage({ octolyzer: 'hub', ...message }, '*');
}
function sendInit() {
  if (!state.current) return;
  sendToViewer({ type: 'init', verdict: verdictOf(state.current), prefs: state.prefs, reviewer: state.reviewer });
  ui.frame.contentWindow.focus();
}
function move(delta) {
  const entries = visibleEntries();
  if (!entries.length) return;
  const index = entries.indexOf(state.current);
  const next = entries[Math.min(entries.length - 1, Math.max(0, (index < 0 ? 0 : index) + delta))];
  if (next && next !== state.current) select(next);
}

// -------------------------------------------------------------------- events
function applyTheme() { document.documentElement.dataset.theme = state.prefs.theme || 'dark'; }
function toggleSidebar(hidden = !root.classList.contains('collapsed')) {
  root.classList.toggle('collapsed', hidden);
  ui.toggle.setAttribute('aria-pressed', String(!hidden));
  store(SIDEBAR_KEY, hidden);
}
window.addEventListener('message', (event) => {
  const message = event.data;
  if (!message || message.octolyzer !== 'viewer' || event.source !== ui.frame.contentWindow) return;
  if (message.type === 'ready') sendInit();
  else if (message.type === 'prefs') {
    state.prefs = message.prefs || {};
    store(PREFS_KEY, state.prefs);
    applyTheme();
  } else if (message.type === 'nav') move(message.delta);
  else if (message.type === 'panel') toggleSidebar();
  else if (message.type === 'verdict' && state.current) {
    const order = message.advance ? visibleEntries() : null;
    setVerdict(state.current, message.verdict);
    if (message.advance) {
      const index = order.indexOf(state.current);
      const next = [...order.slice(index + 1), ...order.slice(0, Math.max(index, 0))].find((entry) => !verdictOf(entry).grade);
      renderList();
      if (next) select(next); else setStatus('All scans in this list are reviewed');
    } else renderList();
  }
});
for (const type of ['keydown', 'keyup']) {
  window.addEventListener(type, (event) => {
    const target = event.target;
    if (target && (target.tagName === 'INPUT' || target.tagName === 'SELECT' || target.tagName === 'TEXTAREA')) {
      if (type === 'keydown' && (event.key === 'Escape' || event.key === 'Enter')) target.blur();
      return;
    }
    if (event.ctrlKey || event.metaKey || event.altKey) return;
    if (type === 'keydown' && (event.key === 'j' || event.key === 'J')) { move(1); event.preventDefault(); return; }
    if (type === 'keydown' && (event.key === 'k' || event.key === 'K')) { move(-1); event.preventDefault(); return; }
    if (event.key === ',') { if (type === 'keydown') toggleSidebar(); event.preventDefault(); return; }
    if (!state.current) {
      if (type === 'keydown' && (event.key === 't' || event.key === 'T')) {
        state.prefs.theme = state.prefs.theme === 'light' ? 'dark' : 'light';
        store(PREFS_KEY, state.prefs); applyTheme();
      }
      return;
    }
    sendToViewer({ type: 'key', event: type, key: event.key, shiftKey: event.shiftKey });
    if ([' ', 'ArrowUp', 'ArrowDown', 'PageUp', 'PageDown', 'Home', 'End'].includes(event.key)) event.preventDefault();
  });
}
window.addEventListener('beforeunload', (event) => {
  if (state.dirty && !state.directory) { event.preventDefault(); event.returnValue = ''; }
});

// ---------------------------------------------------------------------- boot
function build() {
  state.prefs = store(PREFS_KEY) || {};
  state.reviewer = store(REVIEWER_KEY) || '';
  applyTheme();

  ui.picker = h('input', { type: 'file', webkitdirectory: true, multiple: true, style: 'display:none' });
  ui.picker.addEventListener('change', () => openFallback(ui.picker.files));
  ui.folder = h('div', { class: 'folder', text: 'No folder' });
  ui.reviewer = h('input', { type: 'text', value: state.reviewer, placeholder: 'Initials', maxlength: '12', 'aria-label': 'Reviewer initials' });
  ui.reviewer.addEventListener('change', () => {
    state.reviewer = ui.reviewer.value.trim();
    store(REVIEWER_KEY, state.reviewer);
    renderList();
    if (state.current) sendInit();
  });
  ui.count = h('span', { class: 'muted' });
  ui.bar = h('div');
  ui.search = h('input', { type: 'text', placeholder: 'Filter by name', 'aria-label': 'Filter by name' });
  ui.search.addEventListener('input', () => { state.search = ui.search.value; renderList(); });
  ui.type = h('select', { 'aria-label': 'Scan type' }, h('option', { value: '', text: 'All scan types' }));
  ui.type.addEventListener('change', () => { state.type = ui.type.value; renderList(); });
  ui.statusFilter = h('select', { 'aria-label': 'Review status' },
    [['', 'All'], ['unreviewed', 'Unreviewed'], ['flagged', 'Auto-flagged'], ['pass', 'Pass'], ['borderline', 'Borderline'], ['fail', 'Fail']]
      .map(([value, text]) => h('option', { value, text })));
  ui.statusFilter.addEventListener('change', () => { state.status = ui.statusFilter.value; renderList(); });
  ui.sort = h('select', { 'aria-label': 'Sort order' },
    [['name', 'Sort by name'], ['severity', 'Most flagged first'], ['unreviewed', 'Unreviewed first']]
      .map(([value, text]) => h('option', { value, text })));
  ui.sort.addEventListener('change', () => { state.sort = ui.sort.value; renderList(); });
  ui.list = h('div', { class: 'list', role: 'listbox', 'aria-label': 'Scans' });
  ui.status = h('span', { class: 'state', text: 'No folder open' });
  ui.frame = h('iframe', { title: 'Scan review', style: 'display:none' });
  ui.frame.addEventListener('load', sendInit);
  ui.welcomeText = h('p', { text: 'Choose the OCTolyzer output folder (the one containing analysis_output.xlsx). Each scan opens with its segmentations as layers; verdicts are saved to qc_review.csv in that folder.' });
  ui.welcome = h('div', { class: 'welcome' }, h('h1', { text: 'Review segmentations' }), ui.welcomeText,
    h('button', { text: 'Open folder', onclick: openFolder }),
    window.showDirectoryPicker ? null : h('p', { text: 'This browser cannot write to the folder directly. Verdicts are kept in the browser; use the download button to save them. Chrome and Edge save automatically.' }));

  ui.toggle = h('button', {
    class: 'icon quiet toggle', 'data-tip': 'Hide or show the scan list (,)', 'aria-label': 'Scan list', 'aria-pressed': 'true',
    onclick: () => toggleSidebar(),
  }, OctUI.icon('panel_left'));
  root.append(
    h('aside', { class: 'sidebar' },
      h('div', { class: 'top' },
        h('div', { class: 'brand' }, h('span', { text: 'OCTolyzer review' }),
          h('button', { class: 'icon quiet', 'data-tip': 'Open an OCTolyzer output folder', 'aria-label': 'Open folder', onclick: openFolder }, OctUI.icon('folder')),
          ui.toggle),
        ui.folder,
        h('div', { class: 'line' }, h('label', { 'data-tip': 'Reviewer initials, saved with every verdict' }, OctUI.icon('user')), ui.reviewer, ui.count)),
      h('div', {}, h('div', { class: 'progress' }, ui.bar),
        h('div', { class: 'filters' }, ui.search, h('div', { class: 'line' }, ui.type, ui.statusFilter), ui.sort)),
      ui.list,
      h('div', { class: 'bottom' }, ui.status,
        h('button', { class: 'icon quiet', 'data-tip': 'Download qc_review.csv', 'aria-label': 'Export CSV', onclick: exportCsv }, OctUI.icon('download')))),
    h('main', { class: 'content' }, ui.frame, ui.welcome), ui.picker);
  renderList();
  toggleSidebar(Boolean(store(SIDEBAR_KEY)));
}
build();
// Entry point for loading files programmatically (used by tests).
window.octolyzerHub = { loadFiles, state };
})();
