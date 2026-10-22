"""Compact encodings used to embed images, masks, traces and maps in review HTML.

Only depends on numpy and OpenCV so it can be imported (and tested) without the
scientific processing stack.
"""

from __future__ import annotations

import base64
import zlib

import cv2
import numpy as np

# Layer boundaries are stored as fixed-point values (pixels * TRACE_SCALE).
TRACE_SCALE = 10
MAP_SIZE = 256
UINT16_MAX = 65535


def _data_uri(buffer: bytes, mime: str) -> str:
    return f"data:{mime};base64,{base64.b64encode(buffer).decode('ascii')}"


def to_uint8(img: np.ndarray) -> np.ndarray:
    """Convert a greyscale image in [0, 1] or [0, 255] to uint8."""
    arr = np.asarray(img)
    if arr.dtype == np.uint8:
        return arr
    arr = np.nan_to_num(arr.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if arr.size and arr.max() <= 1.0:
        arr = arr * 255.0
    return np.clip(np.round(arr), 0, 255).astype(np.uint8)


def encode_image(img: np.ndarray, quality: int = 90) -> str:
    """Encode a greyscale image as a lossy WebP data URI, falling back to PNG."""
    arr = to_uint8(img)
    try:
        ok, buffer = cv2.imencode(".webp", arr, [cv2.IMWRITE_WEBP_QUALITY, int(quality)])
    except cv2.error:
        ok = False
    if ok:
        return _data_uri(buffer.tobytes(), "image/webp")
    ok, buffer = cv2.imencode(".png", arr)
    if not ok:
        raise ValueError("Unable to encode image for the review file.")
    return _data_uri(buffer.tobytes(), "image/png")


def encode_mask(mask: np.ndarray | None) -> str | None:
    """Encode a binary mask as a lossless PNG whose alpha channel is the mask.

    Returns ``None`` for a missing or empty mask so the viewer can skip it.
    """
    if mask is None:
        return None
    binary = np.asarray(mask) > 0
    if not binary.any():
        return None
    rgba = np.full((*binary.shape, 4), 255, dtype=np.uint8)
    rgba[..., 3] = binary.astype(np.uint8) * 255
    ok, buffer = cv2.imencode(".png", rgba, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if not ok:
        raise ValueError("Unable to encode mask for the review file.")
    return _data_uri(buffer.tobytes(), "image/png")


def pack_uint16(values: np.ndarray) -> str:
    """Delta-encode rows of a 2D uint16 array, deflate and base64 it."""
    arr = np.ascontiguousarray(values, dtype=np.uint16)
    if arr.ndim != 2:
        raise ValueError("pack_uint16 expects a 2D array.")
    delta = arr.copy()
    delta[:, 1:] = arr[:, 1:] - arr[:, :-1]  # wraps modulo 2**16
    raw = delta.astype("<u2").tobytes()
    return base64.b64encode(zlib.compress(raw, 9)).decode("ascii")


def unpack_uint16(text: str, shape: tuple[int, int]) -> np.ndarray:
    """Inverse of :func:`pack_uint16`; mirrors the decoder in ``viewer.js``."""
    raw = zlib.decompress(base64.b64decode(text))
    delta = np.frombuffer(raw, dtype="<u2").reshape(shape)
    return np.cumsum(delta, axis=1, dtype=np.uint16)


def pack_traces(traces: np.ndarray, scale: int = TRACE_SCALE) -> str:
    """Pack layer boundary heights ``(n_scans, n_ascans)``; missing values become 0."""
    arr = np.asarray(traces, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[np.newaxis]
    valid = np.isfinite(arr) & (arr > 0)
    fixed = np.zeros(arr.shape, dtype=np.float64)
    fixed[valid] = np.clip(np.round(arr[valid] * scale), 1, UINT16_MAX)
    return pack_uint16(fixed.astype(np.uint16))


def unpack_traces(text: str, shape: tuple[int, int], scale: int = TRACE_SCALE) -> np.ndarray:
    values = unpack_uint16(text, shape).astype(np.float64) / scale
    values[values == 0] = np.nan
    return values


def pack_map(values: np.ndarray, size: int = MAP_SIZE) -> dict | None:
    """Downsample an en-face map and pack it as fixed-point uint16.

    Pixels that are not finite or not positive are treated as "no data" and
    stored as 0; valid pixels are stored as ``value * scale`` (at least 1).
    """
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim != 2:
        return None
    valid = np.isfinite(arr) & (arr > 0)
    if not valid.any():
        return None
    data = np.where(valid, arr, 0.0)
    small_values = cv2.resize(data, (size, size), interpolation=cv2.INTER_AREA)
    small_weight = cv2.resize(valid.astype(np.float64), (size, size), interpolation=cv2.INTER_AREA)
    keep = small_weight > 0.5
    small = np.zeros((size, size), dtype=np.float64)
    small[keep] = small_values[keep] / small_weight[keep]
    if not keep.any():
        return None

    vmax = float(small[keep].max())
    scale = (UINT16_MAX - 1) / vmax if vmax > 0 else 1.0
    fixed = np.zeros((size, size), dtype=np.uint16)
    fixed[keep] = np.clip(np.round(small[keep] * scale), 1, UINT16_MAX).astype(np.uint16)
    low, high = np.percentile(small[keep], [1, 99])
    return {
        "size": size,
        "source_width": int(arr.shape[1]),
        "source_height": int(arr.shape[0]),
        "scale": scale,
        "min": float(low),
        "max": float(high),
        "data": pack_uint16(fixed),
    }
