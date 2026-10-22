"""Heuristic quality-control flags used to prioritise scans for manual review.

The flags only point a reviewer at scans (and B-scans) that are more likely to
contain a segmentation problem; they never change any measurement.
"""

from __future__ import annotations

import numpy as np

# Anatomical order, top of the retina to the choroid-sclera boundary.
LAYER_SEQUENCE = (
    "ILM", "RNFL", "GCL", "IPL", "INL", "OPL", "ELM",
    "PR1", "PR2", "RPE", "BM", "CHORupper", "CHORlower",
)
CHOROID_LAYERS = ("CHORupper", "CHORlower")

THRESHOLDS = {
    # Fraction of A-scans without a boundary before a B-scan is flagged.
    "retina_gap_fraction": 0.25,
    "choroid_gap_fraction": 0.25,
    # Missing run (in A-scans) between two valid stretches of a boundary.
    "interior_gap_px": 5,
    # A deeper layer drawn above a shallower one, tolerance in pixels and
    # minimum fraction of A-scans affected.
    "crossing_tolerance_px": 1.5,
    "crossing_fraction": 0.005,
    # Vertical step between neighbouring A-scans, in microns.
    "jump_um": 60.0,
    # Slice-to-slice thickness outliers in volumes.
    "outlier_min_scans": 7,
    "outlier_window": 5,
    "outlier_mad": 4.0,
    "outlier_relative": 0.15,
    # Choroid-sclera boundary touching the bottom of the image.
    "edge_margin_px": 3,
    "edge_fraction": 0.02,
    # Heidelberg quality index (dB).
    "low_quality": 15.0,
}

SEVERITY_CAP = 3


def _flag(identifier, label, severity, detail="", slices=None):
    return {
        "id": identifier,
        "label": label,
        "severity": int(min(severity, SEVERITY_CAP)),
        "detail": detail,
        "slices": sorted({int(index) for index in (slices or [])}),
    }


def _valid(trace):
    return np.isfinite(trace) & (trace > 0)


def _as_bool(value):
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes"}
    try:
        return bool(value) and not (isinstance(value, float) and np.isnan(value))
    except (TypeError, ValueError):
        return False


def order_layers(traces):
    """Order layer keys anatomically; unknown keys are placed by median depth."""
    known = [key for key in LAYER_SEQUENCE if key in traces]
    unknown = [key for key in traces if key not in LAYER_SEQUENCE]
    if not unknown:
        return known

    def depth(key):
        values = np.asarray(traces[key], dtype=np.float64)
        mask = _valid(values)
        return float(np.median(values[mask])) if mask.any() else float("inf")

    return sorted(known + unknown, key=depth) if unknown else known


def _slice_label(slices, n_scans):
    if n_scans <= 1:
        return ""
    shown = ", ".join(str(index + 1) for index in slices[:8])
    more = f" (+{len(slices) - 8} more)" if len(slices) > 8 else ""
    return f" B-scan {shown}{more}."


def trace_gap_flags(traces, n_scans):
    gap_slices, interior_slices, layers_hit = set(), set(), set()
    for key, values in traces.items():
        limit = THRESHOLDS["choroid_gap_fraction" if key in CHOROID_LAYERS else "retina_gap_fraction"]
        for index, row in enumerate(np.atleast_2d(values)):
            mask = _valid(row)
            if not mask.any():
                gap_slices.add(index)
                layers_hit.add(key)
                continue
            if 1.0 - mask.mean() > limit:
                gap_slices.add(index)
                layers_hit.add(key)
            first, last = np.flatnonzero(mask)[[0, -1]]
            inner = ~mask[first:last + 1]
            if inner.any():
                padded = np.concatenate([[0], inner.astype(np.int8), [0]])
                edges = np.flatnonzero(np.diff(padded))
                if (edges[1::2] - edges[::2]).max() >= THRESHOLDS["interior_gap_px"]:
                    interior_slices.add(index)
                    layers_hit.add(key)
    flags = []
    layer_text = ", ".join(sorted(layers_hit))
    if gap_slices:
        flags.append(_flag("trace_gap", "Incomplete boundary", 2,
                           f"Boundary missing over a large part of the scan ({layer_text})."
                           + _slice_label(sorted(gap_slices), n_scans), gap_slices))
    if interior_slices - gap_slices:
        flags.append(_flag("trace_hole", "Hole in boundary", 1,
                           f"Boundary interrupted mid-scan ({layer_text})."
                           + _slice_label(sorted(interior_slices), n_scans), interior_slices))
    return flags


def crossing_flags(traces, n_scans):
    order = order_layers(traces)
    slices, pairs = set(), set()
    for upper, lower in zip(order[:-1], order[1:]):
        top = np.atleast_2d(np.asarray(traces[upper], dtype=np.float64))
        bottom = np.atleast_2d(np.asarray(traces[lower], dtype=np.float64))
        both = _valid(top) & _valid(bottom)
        crossed = both & (top - bottom > THRESHOLDS["crossing_tolerance_px"])
        counts = both.sum(axis=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            fraction = np.where(counts > 0, crossed.sum(axis=1) / counts, 0.0)
        hit = np.flatnonzero(fraction > THRESHOLDS["crossing_fraction"])
        if hit.size:
            slices.update(hit.tolist())
            pairs.add(f"{upper}/{lower}")
    if not slices:
        return []
    return [_flag("layer_crossing", "Layers cross", 3,
                  f"Boundaries out of anatomical order ({', '.join(sorted(pairs))})."
                  + _slice_label(sorted(slices), n_scans), slices)]


def jump_flags(traces, scale_y, n_scans):
    limit_px = THRESHOLDS["jump_um"] / max(float(scale_y), 1e-6)
    slices, layers_hit = set(), set()
    for key, values in traces.items():
        rows = np.atleast_2d(np.asarray(values, dtype=np.float64))
        valid = _valid(rows)
        step = np.abs(np.diff(rows, axis=1))
        step[~(valid[:, 1:] & valid[:, :-1])] = 0
        hit = np.flatnonzero((step > limit_px).any(axis=1))
        if hit.size:
            slices.update(hit.tolist())
            layers_hit.add(key)
    if not slices:
        return []
    return [_flag("trace_jump", "Abrupt boundary step", 2,
                  f"Step larger than {THRESHOLDS['jump_um']:.0f} microns between neighbouring A-scans "
                  f"({', '.join(sorted(layers_hit))})." + _slice_label(sorted(slices), n_scans), slices)]


def mean_thickness(traces, upper, lower, scale_y):
    """Mean vertical thickness (microns) per B-scan between two boundaries."""
    if upper not in traces or lower not in traces:
        return None
    top = np.atleast_2d(np.asarray(traces[upper], dtype=np.float64))
    bottom = np.atleast_2d(np.asarray(traces[lower], dtype=np.float64))
    both = _valid(top) & _valid(bottom)
    difference = np.where(both, bottom - top, np.nan)
    with np.errstate(invalid="ignore"):
        counts = both.sum(axis=1)
        means = np.where(counts > 0, np.nansum(difference, axis=1) / np.maximum(counts, 1), np.nan)
    return means * float(scale_y)


def outlier_flags(traces, scale_y, n_scans):
    if n_scans < THRESHOLDS["outlier_min_scans"]:
        return []
    half = THRESHOLDS["outlier_window"] // 2
    slices, names = set(), set()
    for name, (upper, lower) in {"retina": ("ILM", "BM"), "choroid": CHOROID_LAYERS}.items():
        thickness = mean_thickness(traces, upper, lower, scale_y)
        if thickness is None or np.isfinite(thickness).sum() < THRESHOLDS["outlier_min_scans"]:
            continue
        residual = np.full(n_scans, np.nan)
        local = np.full(n_scans, np.nan)
        for index in range(n_scans):
            window = np.delete(thickness[max(0, index - half):index + half + 1],
                               min(index, half))
            window = window[np.isfinite(window)]
            if window.size and np.isfinite(thickness[index]):
                local[index] = np.median(window)
                residual[index] = thickness[index] - local[index]
        finite = np.isfinite(residual)
        if not finite.any():
            continue
        mad = 1.4826 * np.median(np.abs(residual[finite] - np.median(residual[finite])))
        limit = np.maximum(THRESHOLDS["outlier_mad"] * mad, THRESHOLDS["outlier_relative"] * local)
        hit = np.flatnonzero(finite & (np.abs(residual) > limit))
        if hit.size:
            slices.update(hit.tolist())
            names.add(name)
    if not slices:
        return []
    return [_flag("slice_outlier", "Thickness outlier between B-scans", 2,
                  f"Mean {' and '.join(sorted(names))} thickness differs from neighbouring B-scans."
                  + _slice_label(sorted(slices), n_scans), slices)]


def edge_flags(traces, height, n_scans):
    if "CHORlower" not in traces or not height:
        return []
    rows = np.atleast_2d(np.asarray(traces["CHORlower"], dtype=np.float64))
    valid = _valid(rows)
    touching = valid & (rows >= height - 1 - THRESHOLDS["edge_margin_px"])
    counts = np.maximum(valid.sum(axis=1), 1)
    hit = np.flatnonzero(touching.sum(axis=1) / counts > THRESHOLDS["edge_fraction"])
    if not hit.size:
        return []
    return [_flag("choroid_edge", "Choroid at image edge", 2,
                  "Choroid-sclera boundary reaches the bottom of the B-scan."
                  + _slice_label(hit.tolist(), n_scans), hit.tolist())]


def quality_flags(quality, n_scans):
    values = np.asarray(quality if quality is not None else [], dtype=np.float64)
    values = values[np.isfinite(values)] if values.size else values
    if not values.size:
        return []
    low = np.flatnonzero(np.asarray(quality, dtype=np.float64) < THRESHOLDS["low_quality"])
    if not low.size:
        return []
    severity = 2 if low.size > max(1, n_scans // 4) else 1
    return [_flag("low_quality", "Low image quality", severity,
                  f"Device quality below {THRESHOLDS['low_quality']:.0f} dB "
                  f"(lowest {values.min():.1f})." + _slice_label(low.tolist(), n_scans), low.tolist())]


def metadata_flags(metadata, foveas, scan_type):
    flags = []
    missing_fovea = _as_bool(metadata.get("bscan_missing_fovea"))
    slices = []
    if foveas is not None and scan_type in {"H-line", "V-line", "Radial"}:
        points = np.atleast_2d(np.asarray(
            [(np.nan, np.nan) if point is None else point for point in foveas], dtype=np.float64))
        slices = np.flatnonzero(~np.isfinite(points).all(axis=1) | (points.sum(axis=1) == 0)).tolist()
        missing_fovea = missing_fovea or bool(slices)
    if missing_fovea:
        flags.append(_flag("oct_fovea_missing", "OCT fovea not detected", 3,
                           "The fovea could not be located on the B-scan, so measurements were skipped.",
                           slices))
    if _as_bool(metadata.get("slo_missing_fovea")):
        flags.append(_flag("slo_fovea_missing", "SLO fovea not detected", 2,
                           "The fovea could not be located on the SLO image."))
    missing = [name for name in ("retinal", "choroid")
               if _as_bool(metadata.get(f"missing_{name}_oct_measurements"))]
    if missing:
        flags.append(_flag("missing_measurements", "Missing measurements", 2,
                           f"Some {' and '.join(missing)} measurements could not be computed."))
    if _as_bool(metadata.get("optic_disc_overlap_warning")):
        overlap = metadata.get("optic_disc_overlap_index_%", "")
        flags.append(_flag("od_offcentre", "Acquisition off-centre", 2,
                           f"Peripapillary circle is off-centre from the optic disc ({overlap}% of disc diameter)."))
    return flags


def log_flags(log_lines):
    flags = []
    text = [str(line).strip() for line in (log_lines or [])]
    warnings = [line for line in text if "warning" in line.lower()]
    if any("avr value exceeds 1" in line.lower() for line in warnings):
        flags.append(_flag("avr_high", "AVR above 1", 2,
                           "Arteriolar-venular ratio exceeds 1; check the artery-vein classification."))
    if any("optic disc not detected" in line.lower() for line in text):
        flags.append(_flag("od_missing", "Optic disc not detected", 2,
                           "The optic disc was not found on the SLO image."))
    if any("failed to analyse slo" in line.lower() for line in text):
        flags.append(_flag("slo_failed", "SLO analysis failed", 3,
                           "The SLO analysis raised an error; see the log."))
    other = [line for line in warnings if "avr value exceeds 1" not in line.lower()]
    if other:
        flags.append(_flag("pipeline_warning", "Pipeline warnings", 1,
                           f"{len(other)} warning(s) in the processing log."))
    return flags


def compute_flags(data):
    """Return the list of QC flags for a review data dictionary."""
    if data.get("failed"):
        return [_flag("failed", "Analysis failed", 3, "OCTolyzer could not analyse this file.")]

    traces = data.get("traces") or {}
    n_scans = int(data.get("n_scans") or 1)
    scale_y = (data.get("bscan_scale") or (1.0, 1.0, 1.0))[1]
    height = data.get("bscan_shape", (0, 0))[0]
    metadata = data.get("metadata") or {}

    flags = metadata_flags(metadata, data.get("foveas"), data.get("scan_type"))
    if not traces:
        flags.append(_flag("no_layers", "No layer segmentations", 3,
                           "No retinal or choroidal boundaries were available for this scan."))
    else:
        flags += crossing_flags(traces, n_scans)
        flags += trace_gap_flags(traces, n_scans)
        flags += jump_flags(traces, scale_y, n_scans)
        flags += outlier_flags(traces, scale_y, n_scans)
        flags += edge_flags(traces, height, n_scans)
    flags += quality_flags(data.get("quality"), n_scans)
    flags += log_flags(data.get("log"))
    return sorted(flags, key=lambda flag: -flag["severity"])


def severity_score(flags):
    return int(sum(flag["severity"] for flag in flags))
