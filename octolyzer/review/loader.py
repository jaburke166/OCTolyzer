"""Collect everything needed for a review file from a .vol file and its saved results.

The review file is always built from what OCTolyzer saved to disk (plus the
pixel data of the .vol), so the same code serves the pipeline and files that
were analysed in the past.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .flags import order_layers

VESSEL_THRESHOLD = 0.5
# Grey levels written to <fname>_slo_avod_map.png by analyse_slo.
AVOD_VALUES = {"artery": 191, "vein": 127, "disc": 255}
ETDRS_DIAMETERS_UM = (1000, 3000, 6000)
LINE_SCAN_TYPES = ("H-line", "V-line", "Radial")


def _clean(value):
    """Convert numpy/pandas scalars to JSON-friendly Python values."""
    if value is None:
        return None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if value is pd.NaT:
        return None
    return value if isinstance(value, (int, str)) else str(value)


def _table(frame: pd.DataFrame) -> dict:
    return {
        "columns": [str(column) for column in frame.columns],
        "rows": [[_clean(value) for value in row] for row in frame.itertuples(index=False, name=None)],
    }


def _read_grey(path: Path):
    if not path.is_file():
        return None
    return cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)


def _numbers(value) -> list[float]:
    """Parse a metadata entry holding one number or a comma-separated list."""
    if value is None:
        return []
    parts = str(value).split(",")
    numbers = []
    for part in parts:
        try:
            number = float(part)
        except ValueError:
            return []
        if not np.isfinite(number):
            return []
        numbers.append(number)
    return numbers


def _point(metadata, x_key, y_key):
    x, y = _numbers(metadata.get(x_key)), _numbers(metadata.get(y_key))
    if len(x) != 1 or len(y) != 1 or (x[0] == 0 and y[0] == 0):
        return None
    return [x[0], y[0]]


def read_results(result_dir: Path, fname: str) -> dict:
    """Read the per-file workbook written by ``analyse.analyse``."""
    workbook = result_dir / f"{fname}_output.xlsx"
    if not workbook.is_file():
        raise FileNotFoundError(f"No saved results found for {fname}: {workbook}")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sheets = pd.read_excel(workbook, sheet_name=None)

    metadata = {}
    if "metadata" in sheets and len(sheets["metadata"]):
        metadata = {str(key): _clean(value) for key, value in sheets["metadata"].iloc[0].items()}

    traces = {}
    for name, frame in sheets.items():
        if name.startswith("segmentations_"):
            values = frame.drop(columns=["scan_number"], errors="ignore").to_numpy(dtype=np.float64)
            values[~np.isfinite(values) | (values <= 0)] = np.nan
            traces[name[len("segmentations_"):]] = values
    traces = {key: traces[key] for key in order_layers(traces)}

    layer_names = {}
    if "layer_keys" in sheets:
        layer_names = {str(row.key): str(row.layer) for row in sheets["layer_keys"].itertuples()}

    measurements = {name: _table(frame) for name, frame in sheets.items() if "measurements" in name}
    return {"metadata": metadata, "traces": traces, "layer_names": layer_names,
            "measurements": measurements}


def read_vessels(result_dir: Path, fname: str, shape: tuple[int, int, int]):
    """Binary choroid vessel masks ``(n_scans, M, N)`` or ``None`` if not saved."""
    stack_path = result_dir / f"{fname}_chorvessel_maps.npy"
    single_path = result_dir / f"{fname}_chorvessel_mask.png"
    if stack_path.is_file():
        vessels = np.load(stack_path)
    elif single_path.is_file():
        vessels = _read_grey(single_path).astype(np.float64)[np.newaxis] / 255.0
    else:
        return None
    if vessels.shape != tuple(shape):
        return None
    return vessels > VESSEL_THRESHOLD


def read_slo_masks(result_dir: Path, fname: str, shape: tuple[int, int]) -> dict:
    masks = {}
    binary = _read_grey(result_dir / f"{fname}_slo_binary_map.png")
    if binary is not None and binary.shape == tuple(shape):
        masks["vessels"] = binary > 127
    avod = _read_grey(result_dir / f"{fname}_slo_avod_map.png")
    if avod is not None and avod.shape == tuple(shape):
        for name, level in AVOD_VALUES.items():
            masks[name] = avod == level
    fovea = _read_grey(result_dir / f"{fname}_slo_fovea_map.png")
    if fovea is not None and fovea.shape == tuple(shape):
        masks["fovea"] = fovea > 127
    return {name: mask for name, mask in masks.items() if mask.any()}


def read_maps(result_dir: Path, fname: str, measurements: dict) -> list[dict]:
    """Posterior pole en-face maps saved as .npy, registered to the SLO."""
    units = {}
    table = measurements.get("etdrs_measurements") or measurements.get("square_measurements")
    if table and "map_name" in table["columns"] and "units" in table["columns"]:
        name_index = table["columns"].index("map_name")
        unit_index = table["columns"].index("units")
        units = {row[name_index]: (row[unit_index] or "") for row in table["rows"]}

    maps = []
    for path in sorted((result_dir / "thickness_maps").glob(f"{fname}_*_map.npy")):
        key = path.name[len(fname) + 1:-len("_map.npy")]
        for suffix in ("_thickness", "_area", "_"):
            if key.endswith(suffix):
                key = key[:-len(suffix)]
                break
        try:
            values = np.load(path, allow_pickle=True).astype(np.float64)
        except (ValueError, OSError):
            continue
        if values.ndim == 2:
            maps.append({"key": key, "unit": str(units.get(key, "")).strip("[]"), "array": values})
    priority = {"retina": 0, "choroid": 1, "choroid_CVI": 2, "choroid_vessel": 3}
    return sorted(maps, key=lambda item: (priority.get(item["key"], 4), item["key"]))


def _fovea_slice(lines: np.ndarray, slo_fovea) -> int:
    """B-scan whose acquisition line passes closest to the fovea on the SLO."""
    if slo_fovea is None:
        return len(lines) // 2
    point = np.asarray(slo_fovea, dtype=np.float64)
    start, end = lines[:, 0], lines[:, 1]
    direction = end - start
    length = np.maximum((direction ** 2).sum(axis=1), 1e-9)
    position = np.clip(((point - start) * direction).sum(axis=1) / length, 0, 1)
    nearest = start + position[:, np.newaxis] * direction
    return int(((nearest - point) ** 2).sum(axis=1).argmin())


def load_review_data(vol_path, result_dir, preprocess=True) -> dict:
    """Assemble the review data dictionary for one analysed .vol file."""
    from octolyzer import utils  # imported lazily: pulls in the processing stack

    vol_path, result_dir = Path(vol_path), Path(result_dir)
    fname = vol_path.stem
    results = read_results(result_dir, fname)
    metadata, traces = results["metadata"], results["traces"]

    bscans, _, slo_output, _, _, acquisition = utils.load_volfile(
        str(vol_path), preprocess=bool(preprocess), custom_maps=[], logging=[],
        verbose=False, return_acquisition=True)
    slo = slo_output[0]
    n_scans, height, width = bscans.shape
    points = np.asarray(acquisition["points_px"], dtype=np.float64)

    scan_type = str(metadata.get("bscan_type") or "")
    location = str(metadata.get("location") or "")
    unavailable = []

    traces = {key: values for key, values in traces.items() if values.shape == (n_scans, width)}
    if not traces:
        unavailable.append("Layer boundaries")

    vessels = None
    if location != "peripapillary":
        vessels = read_vessels(result_dir, fname, bscans.shape)
        if vessels is None:
            unavailable.append("Choroid vessels")

    slo_fovea = _point(metadata, "slo_fovea_x", "slo_fovea_y")
    slo_masks = read_slo_masks(result_dir, fname, slo.shape)
    # The fovea is drawn as a point; fall back to the saved mask when the
    # coordinates were not written to the metadata (peripapillary scans).
    fovea_mask = slo_masks.pop("fovea", None)
    if slo_fovea is None and fovea_mask is not None:
        rows, columns = np.nonzero(fovea_mask)
        slo_fovea = [float(columns.mean()), float(rows.mean())]
    if "vessels" not in slo_masks:
        unavailable.append("SLO vessel segmentations")

    # Acquisition geometry and fovea position on each B-scan
    foveas = [None] * n_scans
    fovea_slice = None
    if scan_type == "Peripapillary":
        edge, centre = points[0]
        geometry = {"circle": {"cx": float(centre[0]), "cy": float(centre[1]),
                               "r": float(abs(centre[0] - edge[0]))}}
    else:
        geometry = {"lines": [[float(v) for v in line.reshape(-1)] for line in points]}
        xs, ys = _numbers(metadata.get("bscan_fovea_x")), _numbers(metadata.get("bscan_fovea_y"))
        if scan_type == "Ppole":
            fovea_slice = _fovea_slice(points, slo_fovea)
            if len(xs) == 1 and len(ys) == 1:
                foveas[fovea_slice] = [xs[0], ys[0]]
        elif len(xs) == n_scans and len(ys) == n_scans:
            foveas = [[x, y] if (x or y) else None for x, y in zip(xs, ys)]
            fovea_slice = n_scans // 2

    optic_disc = None
    disc_centre = _point(metadata, "optic_disc_x", "optic_disc_y")
    disc_radius = _numbers(metadata.get("optic_disc_radius_px"))
    if disc_centre is not None and len(disc_radius) == 1 and disc_radius[0] > 0:
        optic_disc = {"cx": disc_centre[0], "cy": disc_centre[1], "r": disc_radius[0]}

    maps = read_maps(result_dir, fname, results["measurements"]) if scan_type == "Ppole" else []
    maps = [item for item in maps if item["array"].shape == slo.shape]
    grid = None
    if maps and slo_fovea is not None:
        angle = _numbers(metadata.get("acquisition_angle_degrees"))
        grid = {"type": "etdrs", "centre": slo_fovea, "angle": angle[0] if len(angle) == 1 else 0.0,
                "diameters_um": list(ETDRS_DIAMETERS_UM)}

    roi = _numbers(metadata.get("linescan_area_ROI_microns"))
    log_path = result_dir / f"{fname}_log.txt"
    log = log_path.read_text(encoding="utf-8", errors="replace").splitlines() if log_path.is_file() else []

    return {
        "filename": fname,
        "source": str(metadata.get("Filename") or vol_path.name),
        "scan_type": scan_type,
        "location": location,
        "eye": str(metadata.get("eye") or ""),
        "failed": False,
        "n_scans": n_scans,
        "bscans": bscans,
        "bscan_shape": (height, width),
        "bscan_scale": (float(metadata.get("bscan_scale_x") or 1.0),
                        float(metadata.get("bscan_scale_y") or 1.0),
                        float(metadata.get("bscan_scale_z") or 0.0)),
        "quality": [_clean(np.float64(value)) for value in acquisition["quality"]],
        "traces": traces,
        "layer_names": results["layer_names"],
        "vessels": vessels,
        "foveas": foveas,
        "fovea_slice": fovea_slice,
        "roi_um": roi[0] if len(roi) == 1 and scan_type in LINE_SCAN_TYPES else None,
        "acquisition": geometry,
        "slo": slo,
        "slo_scale": float(metadata.get("slo_scale_xy") or 1.0),
        "slo_masks": slo_masks,
        "slo_fovea": slo_fovea,
        "optic_disc": optic_disc,
        "maps": maps,
        "grid": grid,
        "metadata": metadata,
        "measurements": results["measurements"],
        "log": log,
        "unavailable": unavailable,
    }
