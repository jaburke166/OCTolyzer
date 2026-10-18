// Shared icon set and hover tooltips for the review viewer and hub.
const OctUI = (() => {
  'use strict';
  const PATHS = {
    check: '<path d="M5 12.5l4.5 4.5L19 7.5"/>',
    approx: '<path d="M5 10c2.5-3 4.500-3 7 0s4.500 3 7 0M5 15.500c2.500-3 4.500-3 7 0s4.500 3 7 0"/>',
    cross: '<path d="M6 6l12 12M18 6L6 18"/>',
    moon: '<path d="M20 14.500A8 8 0 0 1 9.500 4a8 8 0 1 0 10.500 10.500z"/>',
    help: '<circle cx="12" cy="12" r="9"/><path d="M9.500 9.500a2.500 2.500 0 1 1 3.500 2.300c-.700.400-1 .900-1 1.700"/><path d="M12 17h.010"/>',
    sun: '<circle cx="12" cy="12" r="3.500"/><path d="M12 3v2.500M12 18.500V21M3 12h2.500M18.500 12H21M5.600 5.600l1.800 1.800M16.600 16.600l1.800 1.800M5.600 18.400l1.800-1.800M16.600 7.400l1.800-1.800"/>',
    contrast: '<circle cx="12" cy="12" r="8.500"/><path d="M12 3.500a8.500 8.500 0 0 1 0 17z" fill="currentColor"/>',
    gamma: '<path d="M4 4v16h16"/><path d="M4 20c9 0 14-5 16-15"/>',
    opacity: '<rect x="4" y="4" width="11" height="11" rx="1"/><rect x="9" y="9" width="11" height="11" rx="1" fill="currentColor" fill-opacity=".35"/>',
    line: '<path d="M4 6h16" stroke-width="1"/><path d="M4 12h16" stroke-width="2.250"/><path d="M4 18.500h16" stroke-width="3.750"/>',
    invert: '<rect x="4" y="4" width="16" height="16" rx="1"/><path d="M4 20L20 4v16z" fill="currentColor"/>',
    aspect: '<rect x="3.500" y="7" width="17" height="10" rx="1"/><path d="M8 12h8M8 12l2-2M8 12l2 2M16 12l-2-2M16 12l-2 2"/>',
    split: '<rect x="3.500" y="5" width="17" height="14" rx="1"/><path d="M12 5v14"/>',
    swap: '<path d="M4 8h14M18 8l-3-3M18 8l-3 3M20 16H6M6 16l3-3M6 16l3 3"/>',
    reset: '<path d="M4.500 12a7.500 7.500 0 1 0 2.400-5.500"/><path d="M4 4.500V9h4.500"/>',
    download: '<path d="M12 4v11M12 15l-4-4M12 15l4-4M5 19.500h14"/>',
    folder: '<path d="M3.500 7.500A1.500 1.500 0 0 1 5 6h4l2 2.500h8a1.500 1.500 0 0 1 1.500 1.500v7.500A1.500 1.500 0 0 1 19 19H5a1.500 1.500 0 0 1-1.500-1.500z"/>',
    flag: '<path d="M6 21V4M6 4.500h11l-2.500 4 2.500 4H6"/>',
    expand: '<path d="M9 4H4v5M15 4h5v5M20 15v5h-5M4 15v5h5"/>',
    collapse: '<path d="M4 9h5V4M20 9h-5V4M15 20v-5h5M9 20v-5H4"/>',
    panel_left: '<rect x="3.500" y="5" width="17" height="14" rx="1"/><path d="M9 5v14"/><path d="M5.500 9h1.500M5.500 12h1.500" stroke-width="1.200"/>',
    panel_right: '<rect x="3.500" y="5" width="17" height="14" rx="1"/><path d="M15 5v14"/><path d="M17 9h1.500M17 12h1.500" stroke-width="1.200"/>',
    user: '<circle cx="12" cy="8" r="3.500"/><path d="M5 20c.500-4 3.500-6 7-6s6.500 2 7 6"/>',
    retinal_layers: '<path d="M3 7c4-2 6 2 9 0s5-2 9 0M3 12c4-2 6 2 9 0s5-2 9 0M3 17c4-2 6 2 9 0s5-2 9 0"/>',
    choroid_boundary: '<path d="M3 8c4-2 6 2 9 0s5-2 9 0"/><path d="M3 16c3 2.500 6 3.500 9 3.500s6-1 9-3.500" stroke-dasharray="2.500 2.500"/>',
    choroid_vessels: '<path d="M3 6c4-2 6 2 9 0s5-2 9 0"/><circle cx="7.500" cy="14" r="2.200"/><circle cx="13.500" cy="16.500" r="2.800"/><circle cx="18" cy="12.500" r="1.600"/>',
    oct_fovea: '<path d="M3 8h4c2.500 0 3 5 5 5s2.500-5 5-5h4"/><path d="M12 16.500V20M10.250 18.250h3.500"/>',
    slo_vessels: '<path d="M12 21v-7M12 14c0-4-3-4-5-8M12 14c0-4 3-4 6-9M9.500 9.500C8 9 6.500 9.500 4.500 11M14.800 9.200c1.700 0 3 .800 4.700 2.300"/>',
    artery_vein: '<path d="M8 21V10c0-3-2-4-3.500-6.500M8 12c0-3 1.500-4 3-5.500"/><path d="M16 21V10c0-3 2-4 3.500-6.500M16 12c0-3-1.500-4-3-5.500" stroke-dasharray="2.500 2"/>',
    optic_disc: '<circle cx="12" cy="12" r="7.500"/><circle cx="12" cy="12" r="3"/>',
    slo_fovea: '<circle cx="12" cy="12" r="4.500"/><path d="M12 3v4M12 17v4M3 12h4M17 12h4"/>',
    image_quality: '<rect x="3.500" y="5" width="17" height="14" rx="1"/><path d="M3.500 16l4.500-4.500 3.500 3.500 3-3 6 6"/><circle cx="8.500" cy="9" r="1.200"/>',
  };

  function icon(name, size = 15) {
    const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
    svg.setAttribute('viewBox', '0 0 24 24');
    svg.setAttribute('width', size);
    svg.setAttribute('height', size);
    svg.setAttribute('aria-hidden', 'true');
    svg.setAttribute('class', 'icon-svg');
    svg.innerHTML = PATHS[name] || '';
    return svg;
  }

  // One floating tooltip for every element carrying a data-tip attribute.
  let node = null, timer = null, owner = null;
  function hide() {
    clearTimeout(timer);
    owner = null;
    if (node) node.style.display = 'none';
  }
  function show(target) {
    if (!target.isConnected) return;
    if (!node) { node = document.createElement('div'); node.className = 'tip'; node.setAttribute('role', 'tooltip'); document.body.append(node); }
    node.textContent = target.dataset.tip;
    node.style.display = 'block';
    const rect = target.getBoundingClientRect();
    const width = node.offsetWidth, height = node.offsetHeight;
    let top = rect.bottom + 6;
    if (top + height > window.innerHeight - 4) top = rect.top - height - 6;
    const left = Math.min(window.innerWidth - width - 4, Math.max(4, rect.left + rect.width / 2 - width / 2));
    node.style.top = `${Math.max(4, top)}px`;
    node.style.left = `${left}px`;
  }
  document.addEventListener('pointerover', (event) => {
    const target = event.target.closest && event.target.closest('[data-tip]');
    if (target === owner) return;
    hide();
    if (!target) return;
    owner = target;
    timer = setTimeout(() => show(target), 200);
  });
  for (const type of ['pointerdown', 'keydown', 'wheel']) document.addEventListener(type, hide, true);
  window.addEventListener('blur', hide);
  document.addEventListener('pointerleave', hide);

  return { icon, hideTip: hide };
})();
