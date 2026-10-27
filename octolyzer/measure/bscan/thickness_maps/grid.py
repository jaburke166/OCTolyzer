import logging
import os
import numpy as np
import sys
import math
import matplotlib.pyplot as plt
import seaborn as sns
import skimage.feature as feature
import skimage.measure as meas 
import skimage.transform as trans
import skimage.morphology as morph
from skimage import segmentation
from scipy import interpolate
from skimage import draw
from sklearn.linear_model import LinearRegression
import matplotlib.ticker as ticker
import pandas as pd
import matplotlib

from octolyzer import utils
from octolyzer.measure.bscan import utils as bscan_utils


def rotate_point(point, origin, angle):
    """
    Rotate a point counterclockwise by a specified angle around a given origin.

    Parameters
    ----------
    point : tuple or list of float
        Coordinates of the point to be rotated, in the format (x, y).
        
    origin : tuple or list of float
        Coordinates of the origin point around which the rotation is performed, in the format (x, y).
        
    angle : float
        Rotation angle in radians. Positive values rotate anticlockwise.

    Returns
    -------
    qx, qy : float
        Coordinates of the rotated point.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy



def create_circular_mask(img_shape=(768,768), center=None, radius=None):
    """
    Create a binary circular mask with a specified center, radius, and image shape.

    Parameters
    ----------
    img_shape : tuple of int, optional
        Shape of the output image, specified as (height, width). Default is (768, 768).
        
    center : tuple of int, optional
        (x,y)-coordinates of the circle's center. 
        If None, the center is set to the middle of the image.
        
    radius : int, optional
        Radius of the circle in pixels. If None, the radius is set to the largest value
        that fits within the image boundaries based on the center.

    Returns
    -------
    mask : numpy.ndarray
        Binary mask with the same dimensions as `img_shape`, where pixels inside the circle
        are `True` and others are `False`.

    Examples
    --------
    - 
    """
    h, w = img_shape
    if center is None: # use the middle of the image
        center = (int(h/2), int(w/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask



def create_circular_grids(circle_mask, angle=0):
    """
    Split a binary circular mask into four equally sized quadrants based on a given 
    angle of rotation.

    This function divides the mask diagonally and can optionally rotate the 
    quadrants based on the specified angle. The resulting quadrants are ordered 
    according to the specified angle, permitting angles of rotation within 
    [-120, 120] degrees.

    Parameters
    ----------
    circle_mask : numpy.ndarray
        A binary mask of a filled circle. The mask is assumed to be circular.

    angle : float, optional, default=0
        The angle (in degrees) used to rotate the quadrants. Positive angles 
        rotate counterclockwise.

    Returns
    -------
    etdrs_masks : list of numpy.ndarray
        A list containing four binary masks, each representing one quadrant 
        of the circular mask, ordered based on the specified angle.
    """
    # Detect all angles from center of circular mask indexes
    output_shape = circle_mask.shape
    c_y, c_x = meas.centroid(circle_mask).astype(int)
    radius = int(circle_mask[:,c_x].sum()/2)
    M, N = output_shape
    circ_idx = np.array(np.where(circle_mask)).T
    x, y = circ_idx[:,1], circ_idx[:,0]
    all_angles = 180/np.pi*np.arctan((c_x-x)/(c_y-y+1e-8))

    # Deal with angles of rotation between [-120, 120]
    relabel = 0
    angle_sign = np.sign(angle)
    if angle != 0:
        angle = angle % (angle_sign*360)
    if abs(angle) > 44:
        rem = (abs(angle)-1) // 44
        angle += -1*angle_sign * 89 
        relabel += rem

    # Select pixels which represent superior and inferior regions, based on angle of elevation of points
    # along circular mask relative to horizontal axis (above 45* and below -45*)
    topbot_idx = np.ma.masked_where((all_angles < 45-angle) & (all_angles > -45-angle), 
                                      np.arange(circ_idx.shape[0])).mask

    # Generate superior-inferior and temporal-nasal subregions of circular mask
    top_bot = np.zeros_like(circle_mask)
    topbot_circidx = circ_idx[topbot_idx].copy()
    top_bot[topbot_circidx[:,0], topbot_circidx[:,1]] = 1
    right_left = np.zeros_like(circle_mask)
    rightleft_circidx = circ_idx[~topbot_idx].copy()
    right_left[rightleft_circidx[:,0], rightleft_circidx[:,1]] = 1

    # Split superior-inferior and temporal-nasal into quadrants
    topbot_split = np.concatenate(2*[np.zeros_like(circle_mask)[np.newaxis]]).astype(int)
    rightleft_split = np.concatenate(2*[np.zeros_like(circle_mask)[np.newaxis]]).astype(int)

    # Split two quadrants up - they're connected by a single pixel so 
    # temporarily remove and then replace
    top_bot[c_y, c_x] = 0
    topbot_props = meas.regionprops(meas.label(top_bot))  
    leftright_props = meas.regionprops(meas.label(right_left))  
    for i,(reg_tb, reg_rl) in enumerate(zip(topbot_props, leftright_props)):
        topbot_split[i, reg_tb.coords[:,0], reg_tb.coords[:,1]] = 1
        rightleft_split[i, reg_rl.coords[:,0], reg_rl.coords[:,1]] = 1
    topbot_split[0][c_y, c_x] = 1

    # Order quadrants consistently dependent on angle
    etdrs_masks = [*topbot_split, *rightleft_split]
    if angle >= 0:
        etdrs_masks = [etdrs_masks[i] for i in [0,2,1,3]]
    else:
        etdrs_masks = [etdrs_masks[i] for i in [0,3,1,2]]

    # Relabelling if angle is outwith [-44, 44]
    if relabel == 1:
        if angle_sign > 0:
            etdrs_masks = [etdrs_masks[i] for i in [3,2,1,0]]
        elif angle_sign < 0:
            etdrs_masks = [etdrs_masks[i] for i in [1,2,3,0]]
    elif relabel == 2:
        if angle_sign > 0:
            etdrs_masks = [etdrs_masks[i] for i in [3,2,1,0]]
        elif angle_sign < 0:
            etdrs_masks = [etdrs_masks[i] for i in [1,2,3,0]]
                
    return etdrs_masks



def create_peripapillary_grid(radius, centre, img_shape=(768,768), angle=0, eye='Right'):
    """
    Create a peripapillary grid to represent the average thickness profile 
    around the optic disc. The grid is divided into quadrants based on the 
    specified radius, centre, and angle, and the resulting regions are 
    adjusted for laterality.

    The function splits the circle into quadrants and further divides 
    superior and inferior quadrants into two sections. The resulting grid 
    provides masks for each region.

    Parameters
    ----------
    radius : int
        The radius of the peripapillary circular region of interest in pixels.
    
    centre : tuple of int
        The (x,y)-coordinates of the centre of the acquisition location, should theoretically
        be the optic disc center.
    
    img_shape : tuple of int, optional, default=(768, 768)
        The shape of the image (height, width), typically representing the SLO image resolution.
    
    angle : float, optional, default=0
        The angle (in degrees) used to rotate the quadrants, modifying the segmentation. 

    eye : str, optional, default='Right'
        Laterality ('Right' or 'Left'). This determines the ordering of the grid 
        quadrants so that the temporal quadrant is always listed first.

    Returns
    -------
    grid_masks : list of numpy.ndarray
        A list of binary masks representing different subfields of the peripapillary grid. 
        The regions include the temporal, supero-temporal, supero-nasal, nasal, infero-nasal, 
        infero-temporal subfields, as well as a smaller central circle concentric with the
        acqusition location.
    """
    angle += 1e-8
    circle_mask = create_circular_mask(img_shape, centre, radius).astype(int)
    centre_mask = create_circular_mask(img_shape, centre, int(radius//3)).astype(int)
    N, M = img_shape
    circ_4grids = create_circular_grids(circle_mask, angle)
    
    grid_masks = []
    for idx, quad in enumerate(circ_4grids):
        # Don't split temporal and nasal quadrant
        if idx in [1, 3]:
            grid_masks.append(quad * (1-centre_mask))

        # For superior and inferior quadrants, split in half lengthways
        else:
            # Generate line between centroid and centre of circle
            centroid = meas.centroid(quad)[[1,0]]
            m, c = bscan_utils.construct_line(centroid, centre)
            x_grid = np.arange(0, N)
            y_grid = m*x_grid+c
            x_grid = x_grid[(y_grid > 0) & (y_grid < N)].astype(int)
            y_grid = y_grid[(y_grid > 0) & (y_grid < N)].astype(int)

            # Loop over quadrant pixel coordinate and store which is below/above
            reg_coords = meas.regionprops(meas.label(quad))[0].coords
            left = []
            right = []
            left_mask = np.zeros_like(circle_mask)
            right_mask = np.zeros_like(circle_mask)
            for (y,x) in reg_coords:
                if y <= m*x + c:
                    right.append([x,y])
                else:
                    left.append([x,y])
            right = np.array(right)
            left = np.array(left)
            left_mask[(left[:,1], left[:,0])] = 1
            right_mask[(right[:,1], right[:,0])] = 1
            if idx == 0:
                grid_masks.append(left_mask.astype(int) * (1-centre_mask))
                grid_masks.append(right_mask.astype(int) * (1-centre_mask))
            else: 
                grid_masks.append(left_mask.astype(int) * (1-centre_mask))
                grid_masks.append(right_mask.astype(int) * (1-centre_mask))

    # Order according to eye type, so it's always temporal -> supero-temporal -> ... -> infero-temporal
    grid_masks.append(centre_mask)
    if eye == 'Right':
        grid_masks = np.array(grid_masks)[[2,1,0,5,3,4,6]]
    elif eye == 'Left':
        grid_masks = np.array(grid_masks)[[5,0,1,2,4,3,6]]

    return grid_masks




def create_etdrs_grid(scale=11.49, center=(384,384), img_shape=(768,768), 
                      angle=0, etdrs_microns=[1000,3000,6000]):
    """
    Create an ETDRS (Early Treatment Diabetic Retinopathy Study) grid for analysing
    thickness/density maps. The grid is created using binary masks 
    based on predefined circle radii, segmented into regions corresponding to 
    the central, inner, and outer areas of the grid.ß

    Parameters
    ----------
    scale : float, optional, default=11.49
        The scale factor in microns-per-pixel, used to convert from pixel units to 
        physical units (e.g., microns). This scale affects the size of the ETDRS grid
        on the corresponding SLO image.
    
    center : tuple of int, optional, default=(384,384)
        (x,y)-coordinates representing the center of the grid which will be the fovea on
        the SLO image. This point serves as the origin for drawing the circles.
    
    img_shape : tuple of int, optional, default=(768,768)
        The shape of the image (height, width), typically representing the image pixel resolution.
    
    angle : float, optional, default=0
        The angle (in degrees) used to rotate the quadrants of the grid for alignment.
    
    etdrs_microns : list of int, optional, default=[1000, 3000, 6000]
        The diameters of the circles in the ETDRS grid, measured in microns. The grid consists
        of concentric rings corresponding to these radii.

    Returns
    -------
    (circles, quadrants) : tuple of list of numpy.ndarray
        A tuple containing two lists:
        - circles: A list of binary masks representing concentric circles corresponding 
          to the different radii in the ETDRS grid.
        - quadrants: A list of binary masks representing quadrants for each circle.
    
    (central, inner_circle, outer_circle) : tuple of numpy.ndarray
        A tuple containing:
        - central: A binary mask for the innermost circle.
        - inner_circle: A binary mask representing the region between the central and 
          the inner rings of the ETDRS grid.
        - outer_circle: A binary mask representing the outer ring of the grid.

    (inner_regions, outer_regions) : tuple of list of numpy.ndarray
        A tuple containing:
        - inner_regions: A list of binary masks for the inner quadrants.
        - outer_regions: A list of binary masks for the outer quadrants.
    """
    # Standard diameter measureents of ETDRS study grid.
    etdrs_radii = [int(np.ceil((N/scale)/2)) for N in etdrs_microns]

    # Draw circles and quadrants
    circles = [create_circular_mask(img_shape, center, radius=r) for r in etdrs_radii]
    quadrants = [create_circular_grids(circle, angle) for circle in circles[1:]]

    # Subtract different sized masks to get individual binary masks of ETDRS study grid
    central = circles[0]
    inner_regions = [(q-central).clip(0,1) for q in quadrants[0]]
    inner_circle = np.sum(np.array(inner_regions), axis=0).clip(0,1)
    outer_regions = [(q-inner-central).clip(0,1) for (q,inner) in zip(quadrants[1],inner_regions)]
    outer_circle = np.sum(np.array(outer_regions), axis=0).clip(0,1)

    return (circles, quadrants), (central, inner_circle, outer_circle), (inner_regions, outer_regions)



def create_square_grid(scalex=11.48, center=None, img_shape=(768,768), angle=0, N_grid=8, grid_size=7000, logging=[]):
    '''
    Create an N x N square grid-based binary mask for measuring thickness/density maps.
    The number of rows and columns in the grid is specified by N_grid, and the grid_size
    determines the size of the entire region of interest in mm.
    
    Parameters
    ----------
    scalex : float, optional, default=11.48
        The transverse/lateral scaling factor in microns per pixel for converting pixel distances into mm.
    
    center : tuple of int, optional, default=None
        The (x,y)-coordinate of the grid center. If None, the grid is centered in the middle of the image.
    
    img_shape : tuple of int, optional, default=(768,768)
        The pixel resolution of the SLO image (height, width).
    
    angle : float, optional, default=0
        The rotation angle of the grid in degrees. If 0, the grid is not rotated.
    
    N_grid : int, optional, default=8
        The number of rows and columns in the grid.
    
    grid_size : float, optional, default=7000
        The total width of the grid in mm. This defines the overall size of the grid area.

    logging : list, optional, default=[]
        A list to collect log messages for tracking the process and any issues that arise during execution.

    Returns
    -------
    grid_masks : list of numpy.ndarray
        A list of binary masks for each grid cell in the square grid. Each mask represents one cell.
    
    labels : list of tuple of int
        A list of labels, where each label corresponds to the row and column indices of the grid cell.
    
    logging : list
        The logging list, updated with any messages generated during execution.
    
    Notes
    -----
    If the grid is too large for the given image size, a warning is logged and the pipeline will not measure
    the thickness/density maps using the posterior pole grid.
    '''
    # Force center
    h, w = img_shape
    if center is None: # use the middle of the image
        center = (int(h/2), int(w/2))

    # Size of each cell in mm, and width of grid in pixels
    cell_size = (grid_size/1e3) / N_grid
    width = grid_size / scalex
    
    # Pixel rows and columns defining grid lines of cells and ensure grid fits within image shape
    box_idx_lr = np.linspace(center[1]-width/2, center[1]+width/2, N_grid+1).astype(int)
    box_idx_ud = np.linspace(center[0]-width/2, center[0]+width/2, N_grid+1).astype(int) 
    condition = np.all((box_idx_lr > 0) & (box_idx_lr < w)) and np.all((box_idx_ud > 0) & (box_idx_ud < h))   

    # If specified grid is too large, the whole image is taken as the ROI instead, warning user that it
    # won't be fovea/optic disc centred which may lead to unstandardised measurements.
    try:
        assert condition, f"{N_grid}x{N_grid} {cell_size}mm grid unavailable given field of view."
    except AssertionError as msg:
        print(msg)
        logging.append(msg.args[0])
        msg = f"Failed to measure square grid with a width of {grid_size/1e3}mm."
        print(msg)
        logging.append(msg)
        msg = "Measuring entire SLO image, using centre of image, not fovea/optic-disc. See metadata sheet for field of view of scan."
        print(msg)
        logging.append(msg)
        return np.ones(img_shape), logging
            
    # Initialise list of masks and corresponding labels
    grid_masks = []
    labels = []

    # Simple creation of square masks if not rotation
    if angle == 0:
        # Create square mask
        rr,cc = draw.rectangle(start=(box_idx_lr[0], box_idx_ud[0]), extent=width)
        all_mask = np.zeros(img_shape)
        all_mask[rr,cc] = 1

        # Split square into cells 
        for i,(x1,x2) in enumerate(zip(box_idx_lr[:-1],box_idx_lr[1:])):
            for j,(y1,y2) in enumerate(zip(box_idx_ud[:-1], box_idx_ud[1:])):
                grid = all_mask.copy()
                grid[x1:x2, y1:y2] = 2
                grid_masks.append(grid-all_mask)
                labels.append((i,j))
                
    # If rotation, create meshgrid of cell vertices, rotate and then join the rotated
    # vertices to form a 2d polygons (which are just rhombi, i.e. equal-sided parallelograms)
    else:
        grid_xy = np.swapaxes(np.transpose(np.array(np.meshgrid(box_idx_lr, box_idx_ud))), 0, 1).reshape(-1,2)
        gridxy_rotate = np.array([rotate_point(xy, center, (angle*np.pi/180)) for xy in grid_xy]).astype(int)
        gridxy_rotate = gridxy_rotate.reshape(N_grid+1, N_grid+1, 2)

        for i in range(N_grid):
            for j in range(N_grid):
                arr = np.array([gridxy_rotate[[i,i+1],j], gridxy_rotate[[i,i+1],j+1]]).reshape(-1,2)[:,[1,0]]
                grid = draw.polygon2mask(polygon=arr[[2,3,1,0]], image_shape=img_shape)
                grid_masks.append(grid)
                labels.append((i,j))

    return grid_masks, labels, logging



def interp_missing(ctmask):
    """
    Interpolate missing values in a thickness/density map using nearest neighbour interpolation.
    This function is designed to handle missing values in one of the ETDRS study grids, where missing
    values are represented by NaN values.

    Parameters
    ----------
    ctmask : numpy.ndarray
        A 2D array representing the thickness/density map, where missing values are NaNs

    Returns
    -------
    new_ctmask : numpy.ndarray
        The CT map with interpolated values for the missing regions.
    
    Notes
    -----
    - Missing values are those where the CT map contains NaN or zero.
    - The interpolation method uses `scipy.interpolate.NearestNDInterpolator` for nearest neighbour interpolation.
    """
    # Detect where values to be interpolated, values 
    # with known CT measurements, and values outside subregion
    ctmap_nanmask = np.isnan(ctmask)
    ctmap_ctmask = ctmask > 0
    ctmap_ctnone = ctmask != 0

    # Extract relevant coordinates to interpolate and evaluate at
    all_coords = np.array(np.where(ctmap_ctnone)).T
    ct_coords = np.array(np.where(ctmap_ctmask)).T
    ct_data = ctmask[ct_coords[:,0],ct_coords[:,1]]

    # Build new subregion mask with interpolated valuee
    new_ctmask = np.zeros_like(ctmask)
    interp_func = interpolate.NearestNDInterpolator
    ctmask_interp = interp_func(ct_coords, ct_data)
    new_ctmask[all_coords[:,0], all_coords[:,1]] = ctmask_interp(all_coords[:,0], all_coords[:,1])

    return new_ctmask



def measure_grid(thick_map, fovea, scale, eye, interp=True, rotate=0,
                measure_type="etdrs", grid_kwds={"etdrs_microns":(1000,3000,6000)},
                plot=False, slo=None, dtype=np.uint64, fname=None, save_path=""):
    """
    Measure average thickness/density per subfield in the prescribed grid (ETDRS/Square).

    Parameters
    ----------
    thick_map : numpy.ndarray
        A 2D array representing the thickness/density map.
    
    fovea : tuple of int
        The (x, y)-coordinates of the fovea on the SLO.
    
    scale : float
        The scale of the image in microns-per-pixel.
    
    eye : str
        Laterality ("Right" or "Left") for determining the quadrant arrangement.
    
    interp : bool, optional, default=True
        If True, missing values in the grid will be interpolated.
    
    rotate : int, optional, default=0
        The angle by which to rotate the grid (in degrees).
    
    measure_type : str, optional, default="etdrs"
        The type of grid to use for measuring ("etdrs" or "square").
    
    grid_kwds : dict, optional, default={"etdrs_microns":(1000,3000,6000)}
        Additional keyword arguments for the grid creation, such as the diameters for ETDRS or the grid size for square grids.
    
    plot : bool, optional, default=False
        If True, plots the grid over the map and optionally an SLO image.
    
    slo : numpy.ndarray, optional, default=None
        The SLO image to be plotted with the grid, if provided.
    
    dtype : numpy.dtype, optional, default=np.uint64
        The data type to use for the grid values (e.g., uint64 for most metrics, float64 for CVI).
    
    fname : str, optional, default=None
        The filename for logging or saving the plot, if specified.
    
    save_path : str, optional, default=""
        The path where the plot should be saved, if specified.

    Returns
    -------
    grid_dict : dict
        A dictionary containing the average measurements for each subfield.
    
    gridvol_dict : dict
        A dictionary containing the total volume (based on interpolated subfield area and thickness) 
        for each subfield.
    
    logging_list : list
        A list of logging messages that indicate the progress or any issues during processing.

    Notes
    -----
    - The function handles both square and ETDRS grid types, measuring average thickness/density per subfield.
    - Missing values within the grid are interpolated using nearest neighbour interpolation.
    """
    # Initialise logging, specify xy-scaling (if vessel density, this is in square microns, otherwise microns)
    logging_list = []
    delta_xy = scale / 1e9
    if fname is not None:
        if "vessel" not in fname:
            delta_xy *= scale

    # Get image shape and ensure fovea is defined properly
    img_shape = thick_map.shape
    if isinstance(fovea, int):
        fovea = (fovea, fovea)

    # Generate subfield binary masks and labelling (according to laterality) for posterior pole square chess grid
    if measure_type == "square":
        grid_masks, labels, log = create_square_grid(scale, fovea, img_shape, rotate, **grid_kwds)
        logging_list.extend(log)
        grid_size = grid_kwds["N_grid"]
        grid_subgrids = []
        for l in labels:
            ud, lr = l
            ud_str = grid_size - ud
            lr_str = grid_size - lr if eye == 'Left' else lr+1
            quadrant = f"{ud_str}.{lr_str}"
            grid_subgrids.append(quadrant)

    # Generate subfield binary masks and labelling (according to laterality) for the ETDRS grid
    elif measure_type == "etdrs":
        output = create_etdrs_grid(scale, fovea, img_shape, rotate, **grid_kwds)
        (circles,_), (central, _, _), (inner_regions, outer_regions) = output
        if eye == 'Right':
            etdrs_locs = ["superior", "temporal", "inferior", "nasal"]
        elif eye == 'Left':
            etdrs_locs = ["superior", "nasal", "inferior", "temporal"]
        etdrs_regions = ["inner", "outer"]
        grid_masks = [central] + inner_regions + outer_regions
        grid_subgrids = ["central"] + ["_".join([grid, loc]) for grid in etdrs_regions for loc in etdrs_locs]

    # All mask is the entire ROI (whole 6mm circle for ETDRS, whole 7mm square grid for Ppole grid)
    all_mask = (np.sum(np.array(grid_masks), axis=0) > 0).astype(int)

    # Most features are measured as integer, except for CVI
    if dtype == np.uint64:
        round_idx = 0
    elif dtype == np.float64:
        round_idx = 3

    # If interpolating (strongly recommended, otherwise -1s will be used in calculation for any missing regions
    # under-estimating true average value)
    all_subr_vals = []
    if interp:
        grid_dict = {}
        gridvol_dict = {}

        # Loop over subfield binary masks and labelling
        for sub,mask in zip(grid_subgrids, grid_masks):
            bool_mask = mask.astype(bool)
            mapmask = thick_map.copy()

            # Check for missing values, replace -1s to NaNs and interpolate using nearest neighbour
            # output to end-user proportion of missing data relative to the size of the subfield.
            if np.any(mapmask[bool_mask] == -1):
                prop_missing = np.round(100*np.sum(mapmask[bool_mask] == -1) / bool_mask.sum(),2)
                msg = f"{prop_missing}% missing values in {sub} region in {measure_type} grid. Interpolating using nearest neighbour."
                logging_list.append(msg)
                logging.warning(msg)
                mapmask[~bool_mask] = 0
                mapmask[mapmask == -1] = np.nan
                mapmask = interp_missing(mapmask)

            # Extract subfield values and interpolate to volume (if not measuring CVI) and take average
            mapmask[~bool_mask] = -1
            all_subr_vals.append(mapmask)
            subr_vals = mapmask[bool_mask]
            if dtype == np.uint64:
                gridvol_dict[sub] = np.round((delta_xy*subr_vals).sum(),3)
            grid_dict[sub] = np.round(dtype(subr_vals.mean()),round_idx)

        # Work out average thickness in the entire grid
        for mapmask in all_subr_vals:
            mapmask[mapmask == -1] = 0
        all_subr_mask = thick_map[all_mask.astype(bool)]
        max_val_etdrs = all_subr_mask.max()
        if dtype == np.uint64:
            gridvol_dict["all"] = np.round((delta_xy*all_subr_mask).sum(),3)
        grid_dict["all"] = np.round(dtype(all_subr_mask.mean()),round_idx)

    # Same procedure as above but without interpolating - DO NOT RECOMMEND
    else:
        grid_dict = {sub : np.round(dtype(thick_map[mask.astype(bool)].mean()),round_idx) for (sub,mask) in zip(grid_subgrids, grid_masks)}
        all_subr_mask = thick_map[all_mask.astype(bool)]
        max_val_etdrs = all_subr_mask.max()
        grid_dict["all"] = np.round(dtype(all_subr_mask.mean()),round_idx)
        if dtype == np.uint64:
            gridvol_dict = {sub : np.round((delta_xy*thick_map[mask.astype(bool)]).sum(),3) for (sub,mask) in zip(grid_subgrids, grid_masks)}
            gridvol_dict["all"] = np.round((delta_xy*all_subr_mask).sum(),3)

    # Clipping value for visualisation is measured as 99.5th percentile to
    # ensure no extreme outliers at the edge of the map are included in calculation. 
    clip_val = np.quantile(thick_map[thick_map != -1], q=0.995)

    # Plot grid onto map and SLO
    if plot:
        if slo is None:
            print("SLO image not specified. Skipping plot.")
            return grid_dict
        _ = plot_grid(slo, thick_map, grid_dict, grid_masks, rotate=rotate,
                      measure_type=measure_type, grid_kwds=grid_kwds,
                      fname=fname, save_path=save_path, clip=clip_val)

    return grid_dict, gridvol_dict, logging_list



def plot_grid(slo, ctmap, grid_data, masks=None, scale=11.49, clip=None, eye="Right", fovea=np.array([384,384]),
              rotate=0, measure_type="etdrs", grid_kwds={"etdrs_microns":(1000,3000,6000)},
              cbar=True, img_shape=(768,768), with_grid=True, fname=None, save_path=None, transparent=False):
    """
    Plot the grid boundaries and associated average values on top of the SLO and thickness/density map.

    Parameters
    ----------
    slo : numpy.ndarray, optional
        The SLO image to be plotted beneath the grid and thickness/density map.
    
    ctmap : numpy.ndarray
        The thickness/density map.
    
    grid_data : dict
        A dictionary containing the grid's measurement values (average thickness or other metrics) for each subfield.
    
    masks : list of numpy.ndarray, optional
        List of binary masks defining subfield, used if the grid is precomputed or passed externally.
    
    scale : float, optional, default=11.49
        The scale of the SLO image in microns-per-pixel.
    
    clip : float, optional, default=None
        The maximum value for clipping the thickness/density map. If None, the 99.5 percentile value is used.
    
    eye : str, optional, default="Right"
        Laterality ("Right" or "Left") for determining the quadrant arrangement in the grid.
    
    fovea : numpy.ndarray, optional, default=np.array([384,384])
        (x,y)-coordinates of the fovea on the SLO.
    
    rotate : int, optional, default=0
        The angle by which to rotate the grid (in degrees).
    
    measure_type : str, optional, default="etdrs"
        The grid type for measuring ("etdrs" or "square").
    
    grid_kwds : dict, optional, default={"etdrs_microns":(1000,3000,6000)}
        Keyword arguments for grid generation, such as radii for ETDRS or grid size for square grids.
    
    cbar : bool, optional, default=True
        Whether to include a color bar in the plot.
    
    img_shape : tuple of int, optional, default=(768,768)
        The shape of the image (height, width).
    
    with_grid : bool, optional, default=True
        Whether to overlay the grid on the image.
    
    fname : str, optional, default=None
        The filename for saving the plot.
    
    save_path : str, optional, default=None
        The path where the plot should be saved.
    
    transparent : bool, optional, default=False
        Whether to save the plot with a transparent background.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated plot figure.

    Notes
    -----
    - The function supports both "etdrs" and "square" grid types.
    - It overlays grid measurements and boundaries onto the thickness map and optionally onto the SLO image.
    - The plot can be saved to a file if `fname` and `save_path` are provided.
    """
    # Build grid masks 
    if masks is None:
        if slo is not None:
            img_shape = slo.shape
        if measure_type == "etdrs":
            output = create_etdrs_grid(scale, fovea, img_shape, rotate, **grid_kwds)
            (_, _), (central, _, _), (inner_regions, outer_regions) = output
            if eye =='Right':
                etdrs_locs = ["superior", "temporal", "inferior", "nasal"]
            elif eye == 'Left':
                etdrs_locs = ["superior", "nasal", "inferior", "temporal"]
            masks = [central] + inner_regions + outer_regions
        elif measure_type == "square":
            masks, _, _ = create_square_grid(scale, fovea, img_shape, rotate, **grid_kwds)
    M, N = img_shape
    
    # Detect centroids of masks
    centroids = [meas.centroid(region)[[1,0]] for region in masks]
    all_centroid = np.array([centroids[-1][0], centroids[-4][1]])

    # Generate grid boundaries
    bounds = np.sum(np.array([segmentation.find_boundaries(mask.astype(bool)) for mask in masks]), axis=0).clip(0,1)
    bounds = morph.dilation(bounds, footprint=morph.disk(radius=2))
    bounds = utils.generate_imgmask(bounds)

    # if clipping heatmap
    mask = ctmap < 0
    if clip is None:
        vmax = np.quantile(ctmap[ctmap != -1], q=0.995)
    else:
        vmax = clip

    # Plot grid on top of thickness map, ontop of SLO
    if slo is not None and with_grid and cbar:
        figsize=(9,7)
    else:
        figsize=(9,9)
    fig, ax = plt.subplots(1,1,figsize=figsize)
    hmax = sns.heatmap(ctmap,
                    cmap = "rainbow",
                    alpha = 0.5,
                    zorder = 2,
                    vmax = vmax,
                    mask=mask,
                    cbar=cbar,
                    ax = ax)
    if slo is not None:
        hmax.imshow(slo, cmap="gray",
                aspect = hmax.get_aspect(),
                extent = hmax.get_xlim() + hmax.get_ylim(),
                zorder = 1)
    ax.set_axis_off()
    if with_grid:
        ax.imshow(bounds, zorder=3)
        for (ct, coord) in zip(grid_data.values(), centroids):
            if isinstance(ct, str):
                fontsize=20
            else:
                if ct // 1 == 0:
                    fontsize=13.5 + (2-2*cbar)
                elif ct // 1000 == 0:
                    fontsize=16 + (2-2*cbar)
                else:
                    fontsize=14 + (2-2*cbar)
            ax.text(s=f"{ct}", x=coord[0], y=coord[1], zorder=4,
                    fontdict={"fontsize":fontsize, 
                              "fontweight":"bold", "ha":"center", "va":"center"})
            
        # Plot average CT across whole grid
        ax.text(s=grid_data["all"], 
                x=all_centroid[0] - 50*np.sign(N//2-all_centroid[0]), 
                y=all_centroid[1] - 50*np.sign(M//2-all_centroid[1]),
                zorder=4, fontdict={"fontsize":fontsize, "fontweight":"bold", "ha":"center", "va":"center"})

    # Save out
    if (save_path is not None) and (fname is not None): 
        fig.savefig(os.path.join(save_path, fname), bbox_inches="tight", transparent=transparent, pad_inches=0)
        plt.close()

    return fig



def plot_multiple_grids(all_dict):
    """
    Plot multiple ETDRS grid thickness/density values on top of SLO.

    Parameters
    ----------
    all_dict : dict
        A dictionary containing multiple grid data. The first key ('core') should hold the core image data (SLO, filename, and save path), 
        and subsequent keys should contain the maps and corresponding grid data for each grid to be plotted.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure with multiple grids plotted.

    Notes
    -----
    - The function supports plotting multiple grids (e.g., for comparison) in a single figure.
    - Each grid's measurements (average thickness and grid volume) are overlaid onto both the thickness map and SLO image.
    """
    # Core plotting args
    with_grid = True
    transparent = False
    cbar = False
    measure_type = 'etdrs'
    grid_kwds = {'etdrs_microns':[1000,3000,6000]}
    interp = True

    # Core map and SLO args
    slo, fname, save_path = all_dict['core']
    img_shape = slo.shape
    map_keys = list(all_dict.keys())[1:]
    fovea, scale, eye, rotate = all_dict[map_keys[0]][1:5]

    # Work out plotting figure subplots
    N = len(list(all_dict.keys()))-1
    if N == 3:
        figsize=(2,2)
        fig, axes = plt.subplots(2,2, figsize=(14,14))
    elif N == 2:
        figsize=(1,3)
        fig, axes = plt.subplots(1,3, figsize=(21,7))
    elif N == 1:
        figsize=(1,2)
        fig, axes = plt.subplots(1,2, figsize=(14,7))

    # Build grid masks 
    output = create_etdrs_grid(scale, fovea, img_shape, rotate, **grid_kwds)
    (_, _), (central, _, _), (inner_regions, outer_regions) = output
    if eye == 'Right':
        etdrs_locs = ["superior", "temporal", "inferior", "nasal"]
    elif eye == 'Left':
        etdrs_locs = ["superior", "nasal", "inferior", "temporal"]
    masks = [central] + inner_regions + outer_regions

    # Detect centroids of masks
    centroids = [meas.centroid(region)[[1,0]] for region in masks]
    all_centroid = np.array([centroids[-1][0], centroids[-4][1]])

    # Generate grid boundaries
    bounds = np.sum(np.array([segmentation.find_boundaries(mask.astype(bool)) for mask in masks]), axis=0).clip(0,1)
    bounds = morph.dilation(bounds, footprint=morph.disk(radius=2))
    bounds = utils.generate_imgmask(bounds)

    plt_indexes = list(np.ndindex(figsize))
    if figsize[0]==1:
        ax = axes[0]
    else:
        ax = axes[plt_indexes[0]]
    ax.imshow(slo, cmap='gray')
    ax.set_axis_off()
    for idx, plt_key in enumerate(map_keys):
        (ctmap, _, _, _, _, dtype, grid_data, gridvol_data) = all_dict[plt_key]

        if figsize[0]==1:
            ax = axes[plt_indexes[idx+1][1]]
        else:
            ax = axes[plt_indexes[idx+1]]
        ax.set_title(plt_key, fontsize=18)

        # clipping heatmap
        mask = ctmap < 0
        vmax = np.quantile(ctmap[ctmap != -1], q=0.995)

        # Plot grid on top of thickness map, ontop of SLO
        hmax = sns.heatmap(ctmap,
                        cmap = "rainbow",
                        alpha = 0.5,
                        zorder = 2,
                        vmax = vmax,
                        mask=mask,
                        cbar=cbar,
                        ax = ax)
        if slo is not None:
            hmax.imshow(slo, cmap="gray",
                    aspect = hmax.get_aspect(),
                    extent = hmax.get_xlim() + hmax.get_ylim(),
                    zorder = 1)
        ax.set_axis_off()
        if with_grid:
            ax.imshow(bounds, zorder=3)
            for (ct, coord) in zip(grid_data.values(), centroids):
                if isinstance(ct, str):
                    fontsize=20
                else:
                    if ct // 1 == 0:
                        fontsize=11
                    elif ct // 1000 == 0:
                        fontsize=12
                    else:
                        fontsize=10
                ax.text(s=f"{ct}", x=coord[0], y=coord[1], zorder=4,
                        fontdict={"fontsize":fontsize, 
                                  "fontweight":"bold", "ha":"center", "va":"center"})
                
            # Plot average CT across whole grid
            ax.text(s=grid_data["all"], 
                    x=all_centroid[0] - 50*np.sign(384-all_centroid[0]), 
                    y=all_centroid[1] - 50*np.sign(384-all_centroid[1]),
                    zorder=4, fontdict={"fontsize":fontsize, "fontweight":"bold", "ha":"center", "va":"center"})


    return fig



def plot_peripapillary_grid(slo, slo_acq, metadata, grid_values, fovea_at_slo, 
                            raw_thicknesses, ma_thicknesses, key=None, fname=None, save_path=None):
    """
    Plot the peripapillary grid and thickness profile together.

    Parameters
    ----------
    slo : numpy.ndarray
        SLO image.
        
    slo_acq : numpy.ndarray
        The SLO image which has the peripapillary circular acquisition location overlaid.
        
    metadata : dict
        Metadata containing acquisition parameters such as acquisition radius and optic disc center.
        
    grid_values : dict
        The grid values containing thickness measurements for different subfields.
        
    fovea_at_slo : numpy.ndarray
        (x,y)-coordinates of the fovea in the SLO image.
        
    raw_thicknesses : numpy.ndarray
        The raw thickness measurements across the grid.
        
    ma_thicknesses : numpy.ndarray
        The smoothed thickness measurements across the grid.
        
    key : str, optional
        The layer key for the plot title, if applicable.
        
    fname : str, optional
        The filename to save the plot.
        
    save_path : str, optional
        The path where the plot should be saved.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure with the peripapillary grid and thickness profile.

    Notes
    -----
    - This function combines the visualisation of a peripapillary grid with the layers thickness profile.
    - The SLO image is annotated with the peripapillary grid and key anatomical landmarks.
    """
    M, N = slo.shape
    circ_mask = slo_acq[...,1] == 1
    circ_mask_dilate = morph.dilation(circ_mask, footprint=morph.disk(radius=2))
    acq_radius = metadata["acquisition_radius_px"]
    acq_center = np.array([metadata["acquisition_optic_disc_center_x"], 
                           metadata["acquisition_optic_disc_center_y"]]).astype(int)

    Xy =  np.concatenate([acq_center[np.newaxis], fovea_at_slo[np.newaxis]], axis=0)
    linmod = LinearRegression().fit(Xy[:,0].reshape(-1,1), Xy[:,1])
    theta = (np.arctan(linmod.coef_[0])*180)/np.pi
    grid_masks = create_peripapillary_grid(centre=acq_center, img_shape=slo.shape,
                                                radius=acq_radius, angle=theta, eye=metadata['eye'])

    # Detect centroids of masks
    centroids = [meas.centroid(region)[[1,0]] for region in grid_masks]
    
    # Generate grid boundaries
    bounds = np.sum(np.array([segmentation.find_boundaries(mask.astype(bool)) for mask in grid_masks]), axis=0).clip(0,1)
    bounds = morph.dilation(bounds, footprint=morph.disk(radius=2))
    bounds = utils.generate_imgmask(bounds)

    # Organise the grid values
    grid_values = pd.DataFrame(grid_values, index=[0])
    grid_values = dict(grid_values[['temporal_[um]','supero_temporal_[um]','supero_nasal_[um]',
                                'nasal_[um]','infero_nasal_[um]','infero_temporal_[um]', 'All_[um]',
                                'PMB_[um]', 'N/T']].iloc[0])

    # Subplot with the peripapillary grid and thickness profile
    fig, (ax0,ax) = plt.subplots(2,1,figsize=(12,12))

    # Plot SLO with peripapillary grid and annotations
    if key is not None:
        ax0.set_title(f"Layer: {key}", fontsize=20)
    ax0.imshow(slo, cmap='gray')
    ax0.imshow(bounds)
    ax0.imshow(utils.generate_imgmask(circ_mask_dilate,None,1))
    ax0.scatter(fovea_at_slo[0], fovea_at_slo[1], marker='X', edgecolors=(0,0,0), s=200, color='r')
    
    fontsize=20
    for (ct, coord) in zip(grid_values.values(), centroids):
        ax0.text(s=f"{int(ct)}", x=coord[0], y=coord[1], zorder=4,
                fontdict={"fontsize":fontsize, 'color':'darkred',
                          "fontweight":"bold", "ha":"center", "va":"center"})
    # Show N/T
    ax0.text(s=f'N/T: {np.round(grid_values["N/T"], 2)}',
            x=0.15*N, 
            y=0.2*N,
            zorder=4, fontdict={"fontsize":fontsize, "fontweight":"bold",'color':'darkred', 
                                "ha":"center", "va":"center"})
    # Show PMB value
    ax0.text(s=f'PMB: {int(grid_values["PMB_[um]"],)}', 
            x=0.15*N, 
            y=0.1*N,
            zorder=4, fontdict={"fontsize":fontsize, "fontweight":"bold",'color':'darkred',
                                "ha":"center", "va":"center"})
    ax0.set_axis_off()
    
    
    # Plot the thickness profile as a subplot underneath the peripapillary grid
    ax.plot(raw_thicknesses[:,0], raw_thicknesses[:,1], linewidth=1, linestyle="--", color="b")
    ax.plot(ma_thicknesses[:,0], ma_thicknesses[:,1], linewidth=3, linestyle="-", color="g")
    ax.set_ylabel(r"thickness ($\mu$m)", fontsize=18)
    ax.tick_params(labelsize=16)
    ax.set_xlim([-180,180])
    
    
    ax2 = ax.twiny()
    ax2.spines["bottom"].set_position(("axes", -0.10))
    ax2.tick_params('both', length=0, width=0, which='minor')
    ax2.tick_params('both', direction='in', which='major')
    ax2.xaxis.set_ticks_position("bottom")
    ax2.xaxis.set_label_position("bottom")
    
    grid_cutoffs = np.array([0, 45, 90, 135, 225, 270, 315, 360]) - 180
    xaxis_locs = np.array([22.5, 67.5, 112.5, 180, 247.5, 292.5, 337.5]) - 180
    ax2.set_xticks(grid_cutoffs)
    for g in grid_cutoffs[1:-1]:
        ax.axvline(g, color='k', linestyle='--')
    ax.set_xticks(list(grid_cutoffs[:4]) + [0] + list(grid_cutoffs[4:]))
    ax.set_xticklabels(list(np.abs(grid_cutoffs[:4])) + [0] + list(grid_cutoffs[4:]))
    rnfl_anatomical_locs = ["Nasal", "Infero Nasal", "Infero Temporal", "Temporal", 
                            "Supero Temporal", "Supero Nasal", "Nasal"]
    ax2.xaxis.set_major_formatter(ticker.NullFormatter())
    ax2.xaxis.set_minor_locator(ticker.FixedLocator(xaxis_locs))
    ax2.xaxis.set_minor_formatter(ticker.FixedFormatter(rnfl_anatomical_locs))
    ax2.tick_params(labelsize=20)
    fig.tight_layout()

    # Save out with transparent BG
    if save_path is not None and fname is not None:
        fig.savefig(os.path.join(save_path, f"peripapillary_grid_{fname}.png"), 
                    bbox_inches="tight", pad_inches=0)
        plt.close()


def plot_thickness_profile(raw_thicknesses, ma_thicknesses, 
                            key=None, fname=None, save_path=None):
    """
    Plot the thickness profile.

    Parameters
    ----------
    raw_thicknesses : numpy.ndarray
        The raw thickness measurements across the grid.
        
    ma_thicknesses : numpy.ndarray
        The smoothed thickness measurements across the grid.
        
    key : str, optional
        The layer key for the plot title, if applicable.
        
    fname : str, optional
        The filename to save the plot.
        
    save_path : str, optional
        The path where the plot should be saved.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure with the thickness profile.

    Notes
    -----
    - This function plots raw and smoothed thickness profiles with grid annotations.
    """
    # Subplot with the peripapillary grid and thickness profile
    fig,ax = plt.subplots(1,1,figsize=(12,5))

    # Plot SLO with peripapillary grid and annotations
    if key is not None:
        ax.set_title(f"Layer: {key}", fontsize=20)
    
    # Plot the thickness profile as a subplot underneath the peripapillary grid
    ax.plot(raw_thicknesses[:,0], raw_thicknesses[:,1], linewidth=1, linestyle="--", color="b")
    ax.plot(ma_thicknesses[:,0], ma_thicknesses[:,1], linewidth=3, linestyle="-", color="g")
    ax.set_ylabel(r"thickness ($\mu$m)", fontsize=18)
    ax.tick_params(labelsize=16)
    ax.set_xlim([-180,180])
    
    ax2 = ax.twiny()
    ax2.spines["bottom"].set_position(("axes", -0.10))
    ax2.tick_params('both', length=0, width=0, which='minor')
    ax2.tick_params('both', direction='in', which='major')
    ax2.xaxis.set_ticks_position("bottom")
    ax2.xaxis.set_label_position("bottom")
    
    grid_cutoffs = np.array([0, 45, 90, 135, 225, 270, 315, 360]) - 180
    xaxis_locs = np.array([22.5, 67.5, 112.5, 180, 247.5, 292.5, 337.5]) - 180
    ax2.set_xticks(grid_cutoffs)
    for g in grid_cutoffs[1:-1]:
        ax.axvline(g, color='k', linestyle='--')
    ax.set_xticks(list(grid_cutoffs[:4]) + [0] + list(grid_cutoffs[4:]))
    ax.set_xticklabels(list(np.abs(grid_cutoffs[:4])) + [0] + list(grid_cutoffs[4:]))
    rnfl_anatomical_locs = ["Nasal", "Infero Nasal", "Infero Temporal", "Temporal", 
                            "Supero Temporal", "Supero Nasal", "Nasal"]
    ax2.xaxis.set_major_formatter(ticker.NullFormatter())
    ax2.xaxis.set_minor_locator(ticker.FixedLocator(xaxis_locs))
    ax2.xaxis.set_minor_formatter(ticker.FixedFormatter(rnfl_anatomical_locs))
    ax2.tick_params(labelsize=20)
    fig.tight_layout()

    # Save out with transparent BG
    if save_path is not None and fname is not None:
        fig.savefig(os.path.join(save_path, f"thickness_profile_{fname}.png"), 
                    bbox_inches="tight", pad_inches=0)
    plt.close()