import numpy as np
import skimage as sk
import matplotlib.pyplot as plt
import logging
import pandas as pd
import pickle
import os
import torch
from PIL import Image
from scipy import interpolate as interp
from skimage import exposure
from skimage import measure as meas
from skimage import morphology as morph
from sklearn.linear_model import LinearRegression


def curve_length(curve, scale=(11.49,3.87)):
    """
    Calculate the length of a curve in microns based on its coordinates.

    This function computes the curve's length using the Euclidean distance between 
    consecutive points and scales the pixel distances into microns.

    Parameters
    ----------
    curve : np.ndarray
        A 2D array of coordinates (x, y) defining the curve.
        
    scale : tuple[float, float], default=(11.49, 3.87)
        Scaling factors in microns-per-pixel for the x and y directions.

    Returns
    -------
    length : float
        The total length of the curve in microns.
    """
    # Scale constants
    xum_per_pix, yum_per_pix = scale

    # Calculate difference between pairwise consecutive coordinates of curve
    diff = np.abs((curve[1:] - curve[:-1]).astype(np.float64))

    # Convert pixel difference to micron difference
    diff[:, 0] *= xum_per_pix
    diff[:, 1] *= yum_per_pix

    # Length is total euclidean distance between all pairwise-micron-movements
    length = np.sum(np.sqrt(np.sum((diff) ** 2, axis=1)))

    return length


def curve_location(curve, distance=2000, ref_idx=400, scale=(11.49,3.87)):
    """
    Find coordinates on a curve that are a specified distance away from a reference point.

    This function identifies two points on a curve that are approximately `distance` microns 
    away from a reference point, indexed by `ref_idx`. It uses cumulative Euclidean distance 
    scaled by the specified microns-per-pixel values.

    Parameters
    ----------
    curve : np.ndarray
        A 2D array of coordinates (x, y) defining the curve.
        
    distance : float, default=2000
        The distance in microns from the reference point.
        
    ref_idx : int, default=400
        Index of the reference coordinate on the curve.
        
    scale : tuple[float, float], default=(11.49, 3.87)
        Scaling factors in microns per pixel for the x and y directions.
        
    verbose : int or bool, default=0
        Verbosity level. If greater than 0, prints warnings when segmentation is insufficient.

    Returns
    -------
    idx_l : int or None
        Index of the coordinate to the left of `ref_idx` at the specified distance.
        
    idx_r : int or None
        Index of the coordinate to the right of `ref_idx` at the specified distance.

    Notes
    -----
    If the segmentation length is insufficient for the specified `distance`, 
    the function returns `None` or `np.nan` and may print warnings based on the `verbose` flag.
    """
    # Work out number of microns per unit pixel movement
    try:
        N = curve.shape[0]
        
        # Scale constants
        xum_per_pix, yum_per_pix = scale

        # Calculate difference between pairwise consecutive coordinates of curve
        diff_r = np.abs((curve[1 + ref_idx:] - curve[ref_idx:-1]).astype(np.float64))
        diff_l = np.abs((curve[::-1][1 + (N - ref_idx):] - curve[::-1][(N - ref_idx):-1]).astype(np.float64))

        # Convert pixel difference to micron difference
        diff_r[:, 0] *= xum_per_pix
        diff_r[:, 1] *= yum_per_pix
        diff_l[:, 0] *= xum_per_pix
        diff_l[:, 1] *= yum_per_pix

        # length per movement is euclidean distance between pairwise-micron-movements
        length_l = np.sqrt(np.sum((diff_l) ** 2, axis=1))
        cumsum_l = np.cumsum(length_l)
        length_r = np.sqrt(np.sum((diff_r) ** 2, axis=1))
        cumsum_r = np.cumsum(length_r)

        # Work out largest index in cumulative length sum where it is smaller than *distance*
        idx_l = ref_idx - np.argmin(cumsum_l < distance)
        idx_r = ref_idx + np.argmin(cumsum_r < distance)
        if (idx_l == ref_idx) and distance > 200:
            return None, np.nan
        if (idx_r == ref_idx) and distance > 200:
            return np.nan, None
        return idx_l, idx_r
    except:
        return None, None


def _check_offset(offset, offsets_lr, N_pts):
    """
    Validate and adjust offsets to ensure they are within valid bounds.

    Parameters
    ----------
    offset : int
        The desired offset value.
        
    offsets_lr : tuple[int, int]
        A tuple containing the left and right offset indices.
        
    N_pts : int
        Total number of points in the trace.

    Returns
    -------
    offset_l : int
        Adjusted left offset index.
        
    offset_r : int
        Adjusted right offset index.
    """
    (offset_l, offset_r) = offsets_lr
    if offset_l < 0:
        offset_l = 0
        logging.warning(f"Offset {offset} too far to the left, choosing index {offset_l}")
        
    if offset_r >= N_pts:
        offset_r = N_pts-1
        logging.warning(f"Offset {offset} too far to the right, choosing index {offset_r}")

    return offset_l, offset_r


def nearest_coord(trace, coord, offset=15, columnwise=False):
    """
    Find the nearest coordinate on a trace to a given point and compute offsets.

    This function identifies the point on a trace closest to a reference coordinate 
    and calculates offset points for defining a tangent line.

    Parameters
    ----------
    trace : np.ndarray
        A 2D array of coordinates (x, y) defining the trace.
        
    coord : np.ndarray
        A pixel coordinate (x, y) to find the nearest point on the trace.
        
    offset : int, default=15
        Offset distance in indices on either side of the reference point for downstream tangent computation.
        
    columnwise : bool, default=False
        If True, finds the nearest point in the same column (x-coordinate).

    Returns
    -------
    trace_refpt : np.ndarray
        The point on the trace closest to `coord`.
        
    offset_pts : np.ndarray
        Points on the trace at the specified offset distances.
    """
    N_pts = trace.shape[0]

    # Work out closest coordinate on trace to coord
    if not columnwise:
        fovea_argmin = np.argmin(np.sum((trace - coord) ** 2, axis=1))
        trace_refpt = trace[fovea_argmin]
    else:
        fovea_argmin = coord[0] - trace[0,0]
        trace_refpt = trace[fovea_argmin]

    # Prevent error by choosing maximum offset, if offset is too large for given trace
    offset_l, offset_r = fovea_argmin - offset, fovea_argmin + offset
    offset_l, offset_r = _check_offset(offset, (offset_l, offset_r), N_pts)
    offset_pts = trace[[offset_l, offset_r]]
    
    return trace_refpt, offset_pts



def construct_line(p1, p2):
    """
    Compute the gradient and intercept of a straight line between two points.

    Parameters
    ----------
    p1 : np.ndarray or list
        A 2D coordinate (x, y) representing the first point.
        
    p2 : np.ndarray or list
        A 2D coordinate (x, y) representing the second point.

    Returns
    -------
    m : float
        The gradient (slope) of the line.
        
    c : float
        The y-intercept of the line.

    Notes
    -----
    If the line is vertical, the function returns `m` and `c` as `np.inf`.
    """
    # Measure difference between x- and y-coordinates of p1 and p2
    delta_x = (p2[0] - p1[0])
    delta_y = (p2[1] - p1[1])

    # Compute gradient and intercept
    try:
        assert delta_x != 0
        m = delta_y / delta_x
        c = p2[1] - m * p2[0]
    except AssertionError:
        m = np.inf
        c = np.inf

    return m, c



def generate_perp_line(pt1, pt2=None, N=None, ref_pt=None):
    """
    Generates a perpendicular line to a given line (tangent) defined by two points.
    The line is evaluated far enough to ensure its intersection with a defined boundary 
    (e.g., Choroid-Sclera boundary) after being rotated by 90 degrees around a reference point.

    Parameters:
    -----------
    pt1 : numpy.ndarray
        A 2D array representing the first point of the tangent line. It must have shape (2,).

    pt2 : numpy.ndarray, optional, default=None
        A 2D array representing the second point of the tangent line. If not provided, 
        a linear model is generated from a single point (`pt1`).

    N : int, optional, default=None
        A scalar value determining how far along the tangent line to evaluate. It defines the 
        range of the tangent line in the x-direction.

    ref_pt : tuple of int, optional, default=None
        A tuple containing the x and y coordinates of the reference point around which the tangent 
        line will be rotated to generate the perpendicular line.

    Returns:
    --------
    tuple of numpy.ndarray
        A tuple containing two 1D numpy arrays: the x and y coordinates of the generated perpendicular line.

    Notes:
    ------
    - The function first fits a linear regression model to the tangent line defined by `pt1` and `pt2` (or just `pt1`).
    - If `N` and `ref_pt` are provided, the function generates the perpendicular line by rotating the tangent line 
      by 90 degrees around `ref_pt` and evaluates the line over the defined range.

    Example:
    --------
    perp_line = generate_perp_line(np.array([1, 2]), pt2=np.array([3, 4]), N=10, ref_pt=(2, 3))
    # This will return the coordinates of the perpendicular line at a given range, rotated around (2, 3).
    """
    # Fit linear model at reference points along tangent
    if pt2 is None:
        X, y = pt1[:,0].reshape(-1,1), pt1[:,1]
    else:
        X, y = np.array([pt1[0], pt2[0]]).reshape(-1,1), np.array([pt1[1], pt2[1]])    
    output = LinearRegression().fit(X, y)

    # Generate perpendicular line if reference point and sample size provided
    if N is not None and ref_pt is not None:

        # Evaluate across tangent
        ref_x, ref_y = ref_pt
        xtan_grid = np.array([ref_x, X[-1,0]+N])
        ytan_grid = output.predict(xtan_grid.reshape(-1,1)).astype(int)

        # Rotate at reference point 90 degrees
        perp_x = (-(ytan_grid - ref_y) + ref_x).reshape(-1,)
        perp_y = (xtan_grid - ref_x + ref_y).reshape(-1,)
        output = (perp_x, perp_y)

        # build output of perpendicular line
        y_grid = np.arange(perp_y[0], perp_y[1])
        x_grid = np.interp(y_grid, perp_y, perp_x)
        output = (x_grid, y_grid)
        
    return output



def detect_orthogonal_coords(reference_pts, traces, offset=15, tol=2, preserve_shape=False):
    """
    Detects coordinates along the lower boundary that intersect with perpendicular lines
    drawn from tangent lines at reference points along the upper boundary. The function calculates
    the points on the lower boundary where the perpendicular lines from the upper boundary's tangent
    lines intersect, within a given tolerance.

    If preserve_shape is True, the function keeps one output row per input
    reference point. Failed matches are marked with np.nan instead of being
    removed.

    Parameters:
    -----------
    reference_pts : numpy.ndarray
        A 2D array of reference points along the upper boundary. Each point defines the location
        from which a tangent and corresponding perpendicular line will be drawn.

    traces : tuple of numpy.ndarray
        A tuple containing two 2D arrays: the upper and lower boundaries of the segmented layer in xy-space.
        The upper boundary (`top_lyr`) and the lower boundary (`bot_lyr`) are both 2D arrays of shape (N, 2).

    offset : int, optional, default=15
        The distance (in pixels) on either side of each reference point used to define the tangent line.
        This controls how local the tangent lines are defined.

    tol : int, optional, default=2
        The threshold (in pixels) to detect pixels along the lower boundary which intersect as close to the
        perpendicular lines.

    preserve_shape : bool, optional, default=False
        If True, the function will return an output array with the same number of rows as `reference_pts`,
        where rows corresponding to failed matches (no intersection within tolerance) are filled with np.nan.

    Returns:
    --------
    tuple of numpy.ndarray
        - chorscl_pts : The coordinates along the lower choroid boundary where perpendicular lines from the 
          upper boundary intersect, within the given tolerance.
        - reference_pts : The original reference points along the upper boundary that correspond to the 
          detected intersection points on the lower boundary.
        - perps : The perpendicular lines corresponding to each reference point along the upper boundary, 
          truncated to the detected intersection points.

    Notes:
    ------
    - The function works by generating tangent lines at each reference point on the upper boundary and 
      calculating the corresponding perpendicular lines.
    - These perpendicular lines are then compared to the lower boundary to find the intersection points.
    - The intersection points are accepted if their Euclidean distance to the lower boundary is within the given tolerance.

    Example:
    --------
    chorscl_pts, reference_pts, perps = detect_orthogonal_coords(reference_pts, (top_lyr, bot_lyr), offset=20, tol=3)
    # This will return the coordinates where the perpendicular lines intersect the lower boundary,
    # along with the corresponding reference points on the upper boundary.
    """
    # Extract traces    
    top_lyr, bot_lyr = traces
    toplyr_stx, botlyr_stx = top_lyr[0, 0], bot_lyr[0, 0]

    # total number of candidate points at each reference point to compare with 
    # Choroid-Sclera boundary
    N = max([bot_lyr[ref_x-botlyr_stx, 1] - ref_y for (ref_x, ref_y) in reference_pts])
    perps = []
    for ref_pt in reference_pts:
    
        # Work out local tangent line for each reference point
        # and rotate orthogonally
        ref_x, ref_y = ref_pt
        ref_xidx = ref_x - toplyr_stx
        tan_pt1, tan_pt2 = top_lyr[[ref_xidx - offset, ref_xidx + offset]] 
        (perp_x, perp_y) = generate_perp_line(tan_pt1, tan_pt2, N, ref_pt)
        perps.append(np.array([perp_x, perp_y]))
    
    # Vectorised search for points along Choroid-Sclera boundary where orthogonal 
    # lines from RPE-Choroid intersect

    perps = np.array(perps, dtype=np.float64)
    bot_cropped = bot_lyr[(perps[:, 0].astype(int) - botlyr_stx).clip(0, bot_lyr.shape[0] - 1)]
    bot_perps_residuals = np.transpose(perps, (0, 2, 1)) - bot_cropped
    bot_perps_distances = np.sqrt((bot_perps_residuals ** 2).sum(axis=-1))
    endpoint_valid = np.min(bot_perps_distances, axis=-1) <= tol
    botlyr_indexes = np.argmin(bot_perps_distances, axis=1)
    botlyr_pts = perps[np.arange(botlyr_indexes.shape[0]), :, botlyr_indexes]

    if preserve_shape:
        botlyr_pts = botlyr_pts.astype(np.float64, copy=True)
        perps = perps.astype(np.float64, copy=True)

        botlyr_pts[~endpoint_valid] = np.nan
        perps[~endpoint_valid] = np.nan

        outputs = (botlyr_pts, reference_pts.copy(), perps)
    else:
        outputs = (
            botlyr_pts[endpoint_valid].astype(int),
            reference_pts[endpoint_valid],
            perps[endpoint_valid].astype(int),
        )

    return outputs
