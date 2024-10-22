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
from octolyzer.measure.bscan.thickness_maps.utils import (extract_bounds, interp_trace, 
                                                                smart_crop, generate_imgmask)
from octolyzer.measure.bscan import utils as bscan_utils
import matplotlib.ticker as ticker
import pandas as pd
from octolyzer import utils
import matplotlib


def rotate_point(point, origin, angle):
    """
    Rotate a point counterclockwise by a given angle around a given origin.

    The angle should be given in radians.
    """
    ox, oy = origin
    px, py = point

    qx = ox + math.cos(angle) * (px - ox) - math.sin(angle) * (py - oy)
    qy = oy + math.sin(angle) * (px - ox) + math.cos(angle) * (py - oy)
    return qx, qy



def create_circular_mask(img_shape=(768,768), center=None, radius=None):
    """
    Given a center, radius and image shape, draw a filled circle
    as a binary mask.
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
    Given a binary mask of a filled circle,
    split it into four quadrants diagonally.
    """
    
    # Detect all angles from center of circular mask indexes
    output_shape = circle_mask.shape
    c_y, c_x = meas.centroid(circle_mask).astype(int)
    radius = int(circle_mask[:,c_x].sum()/2)
    M, N = output_shape
    circ_idx = np.array(np.where(circle_mask)).T
    x, y = circ_idx[:,1], circ_idx[:,0]
    all_angles = 180/np.pi*np.arctan((c_x-x)/(c_y-y+1e-8))

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
    '''
    Create peripapillary average thickness profile grid
    '''
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
    Create ETDRS study grid using image binary masks to quickly compute discretised grid
    of choroid thickness and vessel amsp
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
    Create an N x N grid based binary mask for measuring thickness maps. N_grid is the number
    of rows and columns, and grid_size is the width of the grid in mm.
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
            

    grid_masks = []
    labels = []
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



def interp_missing(ctmask, mode="nearest"):
    """
    Nearest neighbour interpolation of CT map if missing values
    in one of the ETDRS study grids
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
    if mode == "nearest":
        interp_func = interpolate.NearestNDInterpolator
    ctmask_interp = interp_func(ct_coords, ct_data)
    new_ctmask[all_coords[:,0], all_coords[:,1]] = ctmask_interp(all_coords[:,0], all_coords[:,1])

    return new_ctmask



def measure_grid(map, fovea, scale, eye, interp=True, rotate=0,
                measure_type="etdrs", grid_kwds={"etdrs_microns":(1000,3000,6000)}, # measure_type="square", grid_kwds={"N_grid":8, "grid_size":7000},
                plot=False, slo=None, dtype=np.uint64, fname=None, save_path=""):
    """
    Measure average choroid thickness per grid in the ETDRS study grid.
    """
    # Extract masks from map
    logging_list = []
    delta_xy = scale / 1e9
    if fname is not None:
        if "vessel" not in fname:
            delta_xy *= scale
    img_shape = map.shape
    if isinstance(fovea, int):
        fovea = (fovea, fovea)
        
    if measure_type == "square":

        grid_masks, labels, log = create_square_grid(scale, fovea, img_shape, rotate, **grid_kwds)
        logging_list.extend(log)
        # ud_locs = ["inferior", "superior"]
        # if eye == 'Left':
        #     lr_locs = ["nasal", "temporal"]
        # elif eye == 'Right':
        #     lr_locs = ["temporal", "nasal"]
            
        # grid_size_half = int(grid_size // 2)
        grid_size = grid_kwds["N_grid"]
        grid_subgrids = []
        # grid_subgrids_old = []
        for l in labels:
            ud, lr = l
            ud_str = grid_size - ud
            lr_str = grid_size - lr if eye == 'Left' else lr+1
            # lr_quad = lr_locs[lr < grid_size_half]
            # ud_quad = ud_locs[ud < grid_size_half]
            # lr_key = grid_size_half - (lr % grid_size_half) if lr < grid_size_half else (lr % grid_size_half)+1
            # ud_key = grid_size_half - (ud % grid_size_half) if ud < grid_size_half else (ud % grid_size_half)+1
            # quadrant_old = f"{lr_quad}({lr_key})-{ud_quad}({ud_key})"
            # grid_subgrids_old.append(quadrant_old)
            quadrant = f"{ud_str}.{lr_str}"
            grid_subgrids.append(quadrant)
            
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

    all_mask = (np.sum(np.array(grid_masks), axis=0) > 0).astype(int)
    if dtype == np.uint64:
        round_idx = 0
    elif dtype == np.float64:
        round_idx = 3

    all_subr_vals = []
    if interp:
        grid_dict = {}
        gridvol_dict = {}
        for sub,mask in zip(grid_subgrids, grid_masks):
            bool_mask = mask.astype(bool)
            mapmask = map.copy()
            if np.any(mapmask[bool_mask] == -1):
                prop_missing = np.round(100*np.sum(mapmask[bool_mask] == -1) / bool_mask.sum(),2)
                msg = f"{prop_missing}% missing values in {sub} region in {measure_type} grid. Interpolating using nearest neighbour."
                logging_list.append(msg)
                logging.warning(msg)
                mapmask[~bool_mask] = 0
                mapmask[mapmask == -1] = np.nan
                mapmask = interp_missing(mapmask)
            mapmask[~bool_mask] = -1
            all_subr_vals.append(mapmask)
            subr_vals = mapmask[bool_mask]
            if dtype == np.uint64:
                gridvol_dict[sub] = np.round((delta_xy*subr_vals).sum(),3)
            grid_dict[sub] = np.round(dtype(subr_vals.mean()),round_idx)

        # Work out average thickness in the entire grid
        for mapmask in all_subr_vals:
            mapmask[mapmask == -1] = 0
        all_subr_mask = map[all_mask.astype(bool)]
        max_val_etdrs = all_subr_mask.max()
        if dtype == np.uint64:
            gridvol_dict["all"] = np.round((delta_xy*all_subr_mask).sum(),3)
        grid_dict["all"] = np.round(dtype(all_subr_mask.mean()),round_idx)

    else:
        grid_dict = {sub : np.round(dtype(map[mask.astype(bool)].mean()),round_idx) for (sub,mask) in zip(grid_subgrids, grid_masks)}
        all_subr_mask = map[all_mask.astype(bool)]
        max_val_etdrs = all_subr_mask.max()
        grid_dict["all"] = np.round(dtype(all_subr_mask.mean()),round_idx)
        if dtype == np.uint64:
            gridvol_dict = {sub : np.round((delta_xy*map[mask.astype(bool)]).sum(),3) for (sub,mask) in zip(grid_subgrids, grid_masks)}
            gridvol_dict["all"] = np.round((delta_xy*all_subr_mask).sum(),3)
        
    clip_val = np.quantile(map[map != -1], q=0.995)

    # Plot grid onto map and SLO
    if plot:
        if slo is None:
            print("SLO image not specified. Skipping plot.")
            return grid_dict
        _ = plot_grid(slo, map, grid_dict, grid_masks, rotate=rotate,
                      measure_type=measure_type, grid_kwds=grid_kwds,
                      fname=fname, save_path=save_path, clip=clip_val)

    return grid_dict, gridvol_dict, logging_list



def plot_grid(slo, ctmap, grid_data, masks=None, scale=11.49, clip=None, eye="Right", fovea=np.array([384,384]),
              rotate=0, measure_type="etdrs", grid_kwds={"etdrs_microns":(1000,3000,6000)}, # measure_type="square", grid_kwds={"N_grid":8, "grid_size":7000},
              cbar=True, img_shape=(768,768), with_grid=True, fname=None, save_path=None, transparent=False):
    """
    Plot the etdrs grid thickness values ontop of SLO and Thickness map
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
    bounds = generate_imgmask(bounds)

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
    Plot the etdrs grid thickness values ontop of SLO and Thickness map
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
    bounds = generate_imgmask(bounds)

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
                            raw_thicknesses, ma_thicknesses, 
                            key=None, fname=None, save_path=None):
    '''
    Plot peripapillary grid and thickness profile together
    '''
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
    bounds = generate_imgmask(bounds)

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
    ax0.imshow(generate_imgmask(circ_mask_dilate,None,1))
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
    ax.set_ylabel("thickness ($\mu$m)", fontsize=18)
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
        # matplotlib.rcParams.update({'axes.facecolor':[1,1,1,1],
        #                         'figure.facecolor':[1,1,1,0],
        #                         'savefig.facecolor':[1,1,1,0]})
        fig.savefig(os.path.join(save_path, f"peripapillary_grid_{fname}.png"), 
                    bbox_inches="tight", pad_inches=0)
        plt.close()
        # matplotlib.rcParams.update({'axes.facecolor':'white',
        #                         'figure.facecolor':'white',
        #                         'savefig.facecolor': 'auto'})


def plot_thickness_profile(raw_thicknesses, ma_thicknesses, 
                            key=None, fname=None, save_path=None):

    # Subplot with the peripapillary grid and thickness profile
    fig,ax = plt.subplots(1,1,figsize=(12,5))

    # Plot SLO with peripapillary grid and annotations
    if key is not None:
        ax.set_title(f"Layer: {key}", fontsize=20)
    
    # Plot the thickness profile as a subplot underneath the peripapillary grid
    ax.plot(raw_thicknesses[:,0], raw_thicknesses[:,1], linewidth=1, linestyle="--", color="b")
    ax.plot(ma_thicknesses[:,0], ma_thicknesses[:,1], linewidth=3, linestyle="-", color="g")
    ax.set_ylabel("thickness ($\mu$m)", fontsize=18)
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
        # matplotlib.rcParams.update({'axes.facecolor':[1,1,1,1],
        #                         'figure.facecolor':[1,1,1,0],
        #                         'savefig.facecolor':[1,1,1,0]})
        fig.savefig(os.path.join(save_path, f"thickness_profile_{fname}.png"), 
                    bbox_inches="tight", pad_inches=0)
    plt.close()
    # matplotlib.rcParams.update({'axes.facecolor':'white',
    #                         'figure.facecolor':'white',
    #                         'savefig.facecolor': 'auto'})