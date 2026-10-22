"""Assemble self-contained review HTML files and the review hub.

A review file embeds the scan, every segmentation as a separate layer and the
viewer itself, so it opens offline in any modern browser.
"""

from __future__ import annotations

import datetime
import json
import os
from pathlib import Path

import numpy as np

from . import encode, flags as flag_module

SCHEMA_VERSION = 1
ASSETS = Path(__file__).resolve().parent / "assets"
REVIEW_DIRECTORY = "review"
HUB_FILENAME = "review.html"
# The hub lists files by reading only the start of each one, so the manifest
# must stay within this many bytes of the top of the document.
MANIFEST_WINDOW = 65536

LAYER_NAMES = {
    "ILM": "Inner limiting membrane",
    "RNFL": "Retinal nerve fibre layer",
    "GCL": "Ganglion cell layer",
    "IPL": "Inner plexiform layer",
    "INL": "Inner nuclear layer",
    "OPL": "Outer plexiform layer",
    "ELM": "External limiting membrane",
    "PR1": "Photoreceptor layer 1",
    "PR2": "Photoreceptor layer 2",
    "RPE": "Retinal pigment epithelium",
    "BM": "Bruch's membrane",
    "CHORupper": "Choroid upper boundary",
    "CHORlower": "Choroid-sclera boundary",
}
MAP_LABELS = {
    "retina": "Retina thickness",
    "inner_retina": "Inner retina thickness",
    "outer_retina": "Outer retina thickness",
    "choroid": "Choroid thickness",
    "choroid_vessel": "Choroid vessel density",
    "choroid_CVI": "Choroid vascular index",
}

_TEMPLATE = """<!doctype html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="generator" content="OCTolyzer review">
<title>{title}</title>
<script type="application/json" id="octolyzer-manifest">{manifest}</script>
<style>
{css}
</style>
</head>
<body>
<div id="app"></div>
<script type="application/json" id="octolyzer-payload">{payload}</script>
<script>
{script}
</script>
</body>
</html>
"""


def _json(value) -> str:
    """Serialise for embedding inside a <script> element."""
    text = json.dumps(value, separators=(",", ":"), allow_nan=False, ensure_ascii=False)
    return text.replace("</", "<\\/").replace("<!--", "<\\u0021--")


def _asset(name: str) -> str:
    return (ASSETS / name).read_text(encoding="utf-8")


def _round(values, digits=2):
    return [None if value is None else round(float(value), digits) for value in values]


def _structures(data) -> list[str]:
    traces = data.get("traces") or {}
    slo_masks = data.get("slo_masks") or {}
    present = []
    if any(not key.startswith("CHOR") for key in traces):
        present.append("retinal_layers")
    if any(key.startswith("CHOR") for key in traces):
        present.append("choroid_boundary")
    if data.get("vessels") is not None:
        present.append("choroid_vessels")
    if any(point is not None for point in data.get("foveas") or []):
        present.append("oct_fovea")
    if "vessels" in slo_masks:
        present.append("slo_vessels")
    if "artery" in slo_masks or "vein" in slo_masks:
        present.append("artery_vein")
    if "disc" in slo_masks:
        present.append("optic_disc")
    if data.get("slo_fovea") is not None:
        present.append("slo_fovea")
    present.append("image_quality")
    return present


def build_manifest(data, review_flags) -> dict:
    return {
        "schema": SCHEMA_VERSION,
        "generator": "OCTolyzer review",
        "filename": data["filename"],
        "source": data.get("source") or f"{data['filename']}.vol",
        "scan_type": data.get("scan_type") or "",
        "location": data.get("location") or "",
        "eye": data.get("eye") or "",
        "n_scans": int(data.get("n_scans") or 0),
        "failed": bool(data.get("failed")),
        "structures": ["image_quality"] if data.get("failed") else _structures(data),
        "flags": review_flags,
        "severity": flag_module.severity_score(review_flags),
        "built": datetime.datetime.now().isoformat(timespec="seconds"),
    }


def _oct_payload(data) -> dict:
    height, width = data["bscan_shape"]
    scale_x, scale_y, scale_z = data["bscan_scale"]
    traces = data.get("traces") or {}
    names = {**LAYER_NAMES, **(data.get("layer_names") or {})}
    vessels = data.get("vessels")
    thickness = {}
    for name, (upper, lower) in {"retina": ("ILM", "BM"), "choroid": ("CHORupper", "CHORlower")}.items():
        values = flag_module.mean_thickness(traces, upper, lower, scale_y)
        if values is not None:
            thickness[name] = [None if not np.isfinite(v) else round(float(v), 1) for v in values]
    return {
        "width": int(width),
        "height": int(height),
        "scale_x": float(scale_x),
        "scale_y": float(scale_y),
        "scale_z": float(scale_z),
        "images": [encode.encode_image(bscan) for bscan in data["bscans"]],
        "quality": data.get("quality") or [],
        "trace_scale": encode.TRACE_SCALE,
        "layers": [{"key": key, "name": names.get(key, key), "data": encode.pack_traces(values)}
                   for key, values in traces.items()],
        "vessels": None if vessels is None else [encode.encode_mask(mask) for mask in vessels],
        "foveas": [None if point is None else _round(point, 1) for point in data.get("foveas") or []],
        "fovea_slice": data.get("fovea_slice"),
        "roi_um": data.get("roi_um"),
        "acquisition": data.get("acquisition") or {},
        "slice_thickness": thickness,
    }


def _slo_payload(data) -> dict | None:
    slo = data.get("slo")
    if slo is None:
        return None
    return {
        "width": int(slo.shape[1]),
        "height": int(slo.shape[0]),
        "scale": float(data.get("slo_scale") or 1.0),
        "image": encode.encode_image(slo),
        "masks": {name: encode.encode_mask(mask) for name, mask in (data.get("slo_masks") or {}).items()},
        "fovea": data.get("slo_fovea"),
        "optic_disc": data.get("optic_disc"),
    }


def _maps_payload(data) -> list[dict]:
    packed = []
    for item in data.get("maps") or []:
        entry = encode.pack_map(item["array"])
        if entry is not None:
            entry.update(key=item["key"], unit=item.get("unit") or "",
                         label=MAP_LABELS.get(item["key"], item["key"].replace("_", " ") + " thickness"))
            packed.append(entry)
    return packed


def build_payload(data) -> dict:
    if data.get("failed"):
        return {"failed": True, "log": data.get("log") or [], "metadata": data.get("metadata") or {}}
    return {
        "oct": _oct_payload(data),
        "slo": _slo_payload(data),
        "maps": _maps_payload(data),
        "grid": data.get("grid"),
        "metadata": data.get("metadata") or {},
        "measurements": data.get("measurements") or {},
        "log": data.get("log") or [],
        "unavailable": data.get("unavailable") or [],
    }


def render_html(data) -> str:
    """Return the complete review document for a review data dictionary."""
    manifest = build_manifest(data, flag_module.compute_flags(data))
    return _TEMPLATE.format(
        title=f"{manifest['filename']} - OCTolyzer review",
        manifest=_json(manifest),
        css=_asset("theme.css") + "\n" + _asset("viewer.css"),
        payload=_json(build_payload(data)),
        script=_asset("icons.js") + "\n" + _asset("viewer.js"),
    )


def _write_atomic(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)
    return path


def review_path(output_directory, fname: str) -> Path:
    return Path(output_directory) / REVIEW_DIRECTORY / f"{fname}.html"


def write_review(data, output_directory) -> Path:
    return _write_atomic(review_path(output_directory, data["filename"]), render_html(data))


def build_review(vol_path, output_directory, preprocess=True) -> Path:
    """Build the review file for an analysed .vol file.

    ``output_directory`` is the batch output folder; the saved results are read
    from ``<output_directory>/<fname>/``.
    """
    from .loader import load_review_data

    fname = Path(vol_path).stem
    data = load_review_data(vol_path, Path(output_directory) / fname, preprocess=preprocess)
    return write_review(data, output_directory)


def build_failed_stub(vol_path, output_directory, log=None) -> Path:
    """Write a placeholder review file so failed scans still show up in the hub."""
    data = {"filename": Path(vol_path).stem, "source": Path(vol_path).name, "failed": True,
            "log": [str(line) for line in (log or [])], "metadata": {}}
    return write_review(data, output_directory)


def write_hub(output_directory) -> Path:
    """Copy the review hub next to the collated results."""
    document = (_asset("hub.html")
                .replace("/*__THEME_CSS__*/", _asset("theme.css"))
                .replace("/*__HUB_CSS__*/", _asset("hub.css"))
                .replace("/*__HUB_JS__*/", _asset("icons.js") + "\n" + _asset("hub.js")))
    return _write_atomic(Path(output_directory) / HUB_FILENAME, document)
