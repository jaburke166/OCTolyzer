import os
import pandas as pd
import PIL.Image as Image
import numpy as np
import torch
import shutil
from skimage import measure, segmentation, morphology, exposure
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import SimpleITK as sitk

import eyepy
from eyepy.core import utils as eyepy_utils
from eyepy.io.he import vol_reader

from octolyzer.measure.bscan.thickness_maps import grid
from octolyzer.measure.slo import feature_measurement
from octolyzer.segment.octseg import choroidalyzer_inference
from octolyzer.segment.sloseg import avo_inference

# Taken directly from EyePy volreader.py 
# (https://github.com/MedVisBonn/eyepy/blob/master/src/eyepy/io/he/vol_reader.py)
# PR1 and EZ map to 14 and PR2 and IZ map to 15. Hence both names can be used
# to access the same data
SEG_MAPPING = {
    'ILM': 0,
    'BM': 1,
    'RNFL': 2,
    'GCL': 3,
    'IPL': 4,
    'INL': 5,
    'OPL': 6,
    'ONL': 7,
    'ELM': 8,
    'IOS': 9,
    'OPT': 10,
    'CHO': 11,
    'VIT': 12,
    'ANT': 13,
    'PR1': 14,
    'PR2': 15,
    'RPE': 16,
}


def load_volfile(vol_path, preprocess=False, custom_maps=[], logging=[], verbose=True):
    """
    Loads and formats OCT+SLO data from a .vol file, extracting pixel data, metadata, and retinal layers.

    Parameters:
    -----------
    vol_path : str
        Path to the .vol file.
        
    preprocess : bool, default=False
        If True, preprocesses B-scans to compensate for superficial retinal vessel shadows and 
        improves choroid visualisation.
        
    custom_maps : list, default=[]
        List of custom retinal layer maps to be generated if inner retinal layers are detected.
        
    logging : list, default=[]
        List to append logging messages detailing actions and any issues during processing.
        
    verbose : bool, default=True
        If True, prints logging messages to the console.

    Returns:
    --------
    bscan_data : np.ndarray
        Array containing processed B-scan data.
        
    metadata : dict
        Comprehensive and detailed metadata about the scan, including resolution, type, and eye information.
        
    slo_images : tuple
        Tuple of SLO images including:
        - Raw SLO image
        - SLO with all B-scan acquisition locations superimposed
        - SLO with fovea-centered B-scan acquisition location superimposed.
        
    layer_pairwise : dict
        Dictionary containing retinal layer segmentations for valid layer pairs.
        
    logging : list
        Updated logging list with detailed processing and diagnostic messages.
    """
    fname = os.path.split(vol_path)[1]
    msg = f"Reading file {fname}..."
    logging.append(msg)
    if verbose:
        print(msg)
    
    # Catch whether .vol file is a peripapillary or macular scan. Other locations, i.e. radial "star-shaped" scans
    # currently not supported.
    scan_type = "Macular"
    radial = False
    eyepy_import = False
    try: 
        voldata = eyepy.import_heyex_vol(vol_path)
        eyepy_import = True

        # pixel data
        bscan_data = voldata.data / 255
        N_scans, M, N = bscan_data.shape
        fovea_slice_num = N_scans // 2
        
    except ValueError as msg:
        if len(msg.args) > 0 and msg.args[0] == "The EyeVolume object does not support scan pattern 2 (one Circular B-scan).":
            voldata = vol_reader.HeVolReader(vol_path)
            scan_type = "Peripapillary"

            # pixel data
            pixel_data = voldata.parsed_file.bscans[0].data
            bscan_data = (eyepy_utils.from_vol_intensity(pixel_data.copy()) / 255).reshape(1,*pixel_data.shape)
            N_scans, M, N = bscan_data.shape
            fovea_slice_num = None

        elif len(msg.args) > 0 and msg.args[0] == "The EyeVolume object does not support scan pattern 5 (Radial scan - star pattern).":
            voldata = vol_reader.HeVolReader(vol_path)
            radial = True

            # pixel data
            pixel_data = [arr.data for arr in voldata.parsed_file.bscans]
            bscan_data = np.asarray([eyepy_utils.from_vol_intensity(arr.copy()) / 255 for arr in pixel_data])
            N_scans, M, N = bscan_data.shape
            fovea_slice_num = N_scans // 2
            
        else:
            logging.append(msg)
            raise msg

    # slo data and metadata
    slo = voldata.localizer.data.astype(float) / 255
    slo_N = slo.shape[0]
    slo_metadict = voldata.localizer.meta.as_dict()
    slo_metadict["slo_resolution_px"] = slo_N
    slo_metadict["field_of_view_mm"] = slo_metadict["scale_x"] * slo_N
    
    # bscan metadata
    vol_metadata = voldata.meta.as_dict()
    eye = vol_metadata["laterality"]
    scale_z, scale_x, scale_y = vol_metadata["scale_z"], vol_metadata["scale_x"], vol_metadata["scale_y"]
    bscan_meta = vol_metadata["bscan_meta"]
    
    # Detect scan pattern
    if scan_type == "Peripapillary":
        bscan_type = scan_type
        msg = f"Loaded a peripapillary (circular) B-scan."
        logging.append(msg)
        if verbose:
            print(msg)
    elif scan_type == "Macular" and scale_z != 0:
        if radial == 0:
            bscan_type = "Ppole"
            msg = f"Loaded a posterior pole scan with {N_scans} B-scans."
        else:
            bscan_type = "Radial"
            msg = f"Loaded a radial scan with {N_scans} B-scans."
        logging.append(msg)
        if verbose:
            print(msg)
    else:
        stp = bscan_meta[0]["start_pos"][0]
        enp = bscan_meta[0]["end_pos"][1]
        if np.allclose(stp,0,atol=1e-3):
            bscan_type = "H-line"
        elif np.allclose(enp,0,atol=1e-3):
            bscan_type = "V-line"
        else:
            bscan_type = "AV-line"
        msg = f"Loaded a single {bscan_type} B-scan."
        logging.append(msg)
        if verbose:
            print(msg)

    # Optional to try compensate for vessel shadows and brighten B-scans for improved
    # choroid visualisation. 
    # When bscan_type is "AV-line", we do not compensate for shadows
    if preprocess:
        if bscan_type != "AV-line":
            msg = "Preprocessing by compensating for vessel shadows and brightening choroid..."
            bscan_data = np.array([normalise_brightness(shadow_compensate(img)) for img in bscan_data])
        elif bscan_type == "AV-line":
            msg = "AV-line scans are not shadow-compensated and left raw for further processing."
        logging.append(msg)
        if verbose:
            print(msg)

    # retinal layers
    retinal_layer_raw = voldata.layers
    N_rlayers = len(retinal_layer_raw)
    if N_rlayers == 2:
        msg = ".vol file only has ILM and BM layer segmentations."
    elif N_rlayers == 17:
        msg = ".vol file contains all retinal layer segmentations."
    else:
        msg = ".vol file contains ILM and BM, but fewer than all inner retinal layer segmentations."
    logging.append(msg)
    if verbose:
        print(msg)

    # Collect all available retinal layer keys
    try:
        msg = "Processing retinal layer segmentations..."
        logging.append(msg)
        if verbose:
            print(msg)
        x_grid_all = np.repeat(np.arange(N).reshape(-1,1), N_scans, axis=1).T
        
        # Dealing with retinal layer segmentations using primitive loader from EyePy
        if not eyepy_import:
            
            # Collect retinal layer segmentations based on .vol mapping from EyePy
            retinal_layers = {}
            for name, i in SEG_MAPPING.items():
                msg = None
                if i >= retinal_layer_raw.shape[0]:
                    msg = 'The volume contains less layers than expected. The naming might not be correct.'
                    break
                retinal_layers[name] = retinal_layer_raw[i]
        
        # Dealing with retinal layer segmentations loaded and organised from EyePy
        else:
            retinal_layers = {name:val.data for name,val in retinal_layer_raw.items()}
                
        # Create pairwise retinal layers
        layer_keys = []
        for key in retinal_layers.keys():
            if not np.all(np.isnan(retinal_layers[key])):
                layer_keys.append(key) 
        layer_keys = layer_keys[:1] + layer_keys[2:] + ["BM"]
        N_rlayers = len(layer_keys)
        layer_key_pairwise = [f"{key1}_{key2}" for key1,key2 in zip(layer_keys[:-1], layer_keys[1:])]
        
        # By default we always provide the whole retina
        layer_key_pairwise.append("ILM_BM")
    
        # If macular scan, allow custom retinal layers to be created
        if bscan_type != 'Peripapillary':
    
            # If ELM traces exist, then we will provide measurements for inner and outer retina. 
            if 'ELM' in layer_keys:
                custom_maps += ["ILM_ELM", "ELM_BM"]
            custom_maps = list(set(custom_maps))
        
            # Add custom retinal layers if they exist
            if N_rlayers > 2 and len(custom_maps) > 0:
                for key_pair in custom_maps:
                    layer_key_pairwise.append(key_pair)
    
        # Collect retinal layer segmentations
        layer_pairwise = {}
        for key in layer_key_pairwise:
            key1, key2 = key.split("_")
            lyr1 = np.concatenate([x_grid_all[...,np.newaxis],
                                retinal_layers[key1][...,np.newaxis]], axis=-1)
            lyr2 = np.concatenate([x_grid_all[...,np.newaxis],
                                retinal_layers[key2][...,np.newaxis]], axis=-1)
            lyr12_xy_all = np.concatenate([lyr1[:,np.newaxis], lyr2[:,np.newaxis]], axis=1)
            layer_pairwise[key] = [remove_nans(tr) for tr in lyr12_xy_all]   
    
        # Check to make sure all B-scans have inner retinal layer segmentations. If not,
        # only return ILM and BM layers - this does not apply for peripapillary scans, where
        # only three layers are segmented.
        if N_rlayers > 2 and bscan_type == "Ppole":    
            check_layer = "ILM_RNFL"
            check_segs = [trace.shape[1]==0 for trace in layer_pairwise[check_layer]]
            N_missing = int(np.sum(check_segs))
            if N_missing > 0:
                msg = f"WARNING: Found {N_rlayers} retinal layer segmentations, but {N_missing}/{N_scans} B-scans have not been fully segmented for all retinal layers."
                logging.append(msg)
                if verbose:
                    print(msg)    
            if N_missing > N_scans//2:
                msg = f"""Over half of the B-scans are missing inner retinal layer segmentations.
        Falling back to default state of only analysing whole retina, and removing inner retinal layers in output.
        Please segment inner retinal layers for remaining scans to analyse all retinal layers."""
                logging.append(msg)
                if verbose:
                    print(msg)
                newlayer_pairwise = {}
                newlayer_pairwise["ILM_BM"] = layer_pairwise["ILM_BM"]
                layer_pairwise = newlayer_pairwise
                N_rlayers = 2        
        msg = f"Found {N_rlayers} valid retinal layer segmentations for all B-scans."
        logging.append(msg)
        if verbose:
            print(msg)
    
    except Exception as e:
        message = f"An exception of type {type(e).__name__} occurred. Error description:\n{e}"
        tell_user = "Unexpected error locating retinal layer segmentations. Please check B-scan for quality or metadata. Ignoring retinal layers."
        logging.extend([message, tell_user])
        if verbose:
            print(message)
            print(tell_user)
        N_rlayers = 0
        layer_key_pairwise = []
        layer_pairwise = {}

    # Construct slo-acquisition image and extract quality of B-scan    
    msg = "Accessing IR-SLO and organising metadata..."
    logging.append(msg)
    if verbose:
        print(msg)
    all_mm_points = []
    all_quality = []
    for m in bscan_meta:
        all_quality.append(m["quality"])
        st = m["start_pos"]
        en = m["end_pos"]
        point = np.array([st, en])
        all_mm_points.append(point)
    
    # Only relevant for Ppole data
    quality_mu = np.mean(all_quality)
    quality_sig = np.std(all_quality)
    
    # Convert start and end B-scan locations from mm to pixel
    all_px_points = []
    for point in all_mm_points:
        all_px_points.append(slo_N * point / slo_metadict["field_of_view_mm"])
    all_px_points = np.array(all_px_points)

    # Python indexing versus .vol all_px_points indexing
    all_px_points[:,1,0] -= 1

    # Create a (potentially) larger copy of the SLO 
    # to contain all acquisition locations
    slo_acq = np.concatenate(3*[slo[...,np.newaxis]], axis=-1)
    slo_acq_fixed = slo_acq.copy()
    slo_minmax_x = all_px_points[:,:,0].min(), all_px_points[:,:,0].max()
    slo_minmax_y = all_px_points[:,:,1].min(), all_px_points[:,:,1].max()
    
    # Work out padding dimensions to ensure the entire fovea-centred acquisition line fits onto slo_fov_max
    pad_y = int(np.ceil(abs(min(0,slo_minmax_y[0])))), int(np.ceil(abs(max(0,slo_minmax_y[1]-slo_N))))
    pad_x = int(np.ceil(abs(min(0,slo_minmax_x[0])))), int(np.ceil(abs(max(0,slo_minmax_x[1]-slo_N))))
    slo_acq = np.pad(slo_acq, (pad_y, pad_x, (0,0)), mode='constant')
    
    # For peripapillary scans, we draw a circular ROI
    if bscan_type == "Peripapillary":
        peripapillary_coords = all_px_points[0].astype(int)

        OD_edge, OD_center = peripapillary_coords
        circular_radius = np.abs(OD_center[0] - OD_edge[0])
        circular_mask = grid.create_circular_mask(img_shape=(slo_N,slo_N), 
                                     center=OD_center, 
                                     radius=circular_radius)
        circular_bnd_mask = segmentation.find_boundaries(circular_mask)
        slo_acq[circular_bnd_mask,:] = 0
        slo_acq[circular_bnd_mask,1] = 1
        slo_metadict["stxy_coord"] = f"{OD_edge[0]},{OD_edge[1]}"
        slo_metadict["acquisition_radius_px"] = circular_radius
        slo_metadict["acquisition_radius_mm"] = np.round(circular_radius*slo_metadict["scale_x"],2)
        slo_metadict["acquisition_optic_disc_center_x"] = OD_center[0]
        slo_metadict["acquisition_optic_disc_center_y"] = OD_center[1]

    # For macular scans, we generate a line for each B-scan location and 
    # superimpose the acquisition line onto a copy of the en face SLO.
    else:

        # Colour palette for acquisition lines, helpful for Ppole map registration onto SLO
        # Use green for single-line scans
        if N_scans == 1:
            acq_palette = [np.array([0,1,0])]

        # Use a spectrum of colour for Ppole/radial scans 
        else:
            acq_palette = np.array(plt.get_cmap("nipy_spectral")(np.linspace(0.1, 0.9, N_scans)))[...,:-1]

        # Loop across acquisition line endpoints, draw lines on SLO
        for idx, point in enumerate(all_px_points):
            loc_colour = acq_palette[idx] #np.array([0,1,0])
            if bscan_type == 'Radial':
                x_idx, y_idx = [[1,0], [0,1]][idx == N_scans//2]  
            else:
                x_idx, y_idx = [[1,0], [0,1]][bscan_type != "V-line"]
            X, y = point[:,x_idx].reshape(-1,1).astype(int), point[:,y_idx].astype(int)
            linmod = LinearRegression().fit(X, y)
            x_grid = np.linspace(X[0,0], X[1,0], 800).astype(int)
            y_grid = linmod.predict(x_grid.reshape(-1,1)).astype(int)
            for (x,y) in zip(x_grid, y_grid):
                if bscan_type == 'Radial':
                    x_idx, y_idx = [[x,y], [y,x]][idx == N_scans//2]  
                else:
                    x_idx, y_idx = [[y,x], [x,y]][bscan_type != "V-line"]

                # Overlay acquisition locations
                if (0 <= x_idx < slo_N) & (0 <= y_idx < slo_N):
                    slo_acq_fixed[y_idx, x_idx] = loc_colour
                slo_acq[pad_y[0]+y_idx, pad_x[0]+x_idx] = loc_colour
                    
    # Work out region of interest (ROI) captured by B-scan, helps determine maximum ROI to measure
    if scan_type != "Peripapillary":

        # For macular scans, use fovea-centred scan endpoints to work out acquistion ROI
        ROI_pts = all_px_points[fovea_slice_num]
        ROI_xy = np.abs(np.diff(ROI_pts, axis=0)) * np.array([scale_x, scale_x])
        ROI_mm = np.sqrt(np.square(ROI_xy).sum())

    else:
        # For peripapillary scans, use circumference of circular acquisition location using
        # OD centre and acquisition circular edge (forming a radius)
        ROI_mm = 2*np.pi*np.abs(np.diff(all_mm_points[0][:,0]))[0]
                    
    # Create DataFrame of metadata
    bscan_metadict = {}
    bscan_metadict["Filename"] = fname
    if eye  == 'OD':
        bscan_metadict["eye"] = 'Right'
    elif eye == 'OS':
        bscan_metadict["eye"] = 'Left'
    bscan_metadict["bscan_type"] = bscan_type
    bscan_metadict["bscan_resolution_x"] = N
    bscan_metadict["bscan_resolution_y"] = M
    bscan_metadict["bscan_scale_z"] = 1e3*scale_z
    bscan_metadict["bscan_scale_x"] = 1e3*scale_x
    bscan_metadict["bscan_scale_y"] = 1e3*scale_y
    bscan_metadict["bscan_ROI_mm"] = ROI_mm
    bscan_metadict["scale_units"] = "microns_per_pixel"
    bscan_metadict["avg_quality"] = quality_mu
    bscan_metadict["retinal_layers_N"] = N_rlayers

    # Remove duplicates: store scales as microns-per-pixel, laterality=eye
    slo_metadict["slo_scale_xy"] = 1e3*slo_metadict["scale_x"]
    for key in ["laterality", "scale_x", "scale_y", "scale_unit"]:
        del slo_metadict[key]
    slo_metadict["location"] = scan_type.lower()
    slo_metadict["slo_modality"] = slo_metadict.pop("modality")
    slo_metadict["field_size_degrees"] = slo_metadict.pop("field_size")
        
    # Combine metadata and return with data
    metadata = {**bscan_metadict, **slo_metadict}
    msg = "Done!"
    logging.append(msg)
    if verbose:
        print(msg)

    # collect SLO output
    slo_output = (slo, slo_acq_fixed, slo_acq, (pad_x,pad_y))
        
    return bscan_data, metadata, slo_output, layer_pairwise, logging


def load_img(path, ycutoff=0, xcutoff=0, pad=False, pad_factor=32):
    '''
    Helper function to load and normalise an image, and optionally
    pad to have dimensions divisible by pad_factor
    '''    """
    Load and normalize an image, with optional cropping and padding.

    Parameters:
    -----------
    path : str
        Path to the image file to load.

    ycutoff : int, default=0
        Number of pixels to crop from the top of the image along the vertical axis.

    xcutoff : int, default=0
        Number of pixels to crop from the left of the image along the horizontal axis.

    pad : bool, default=False
        Whether to pad the image dimensions to be divisible by `pad_factor`.

    pad_factor : int, default=32
        Factor by which the padded image dimensions should be divisible, if `pad=True`.

    Returns:
    --------
    numpy.ndarray
        Loaded and normalized image. If `pad=True`, the dimensions are padded accordingly.

    Example:
    --------
    ```python
    img = load_img("example.png", ycutoff=10, xcutoff=20, pad=True, pad_factor=16)
    ```
    """
    img = np.array(Image.open(path))[ycutoff:, xcutoff:]/255.0
    if pad:
        ndim = img.ndim
        M, N = img.shape[:2]
        pad_M = (pad_factor - M%pad_factor) % pad_factor
        pad_N = (pad_factor - N%pad_factor) % pad_factor
    
        # Assuming third color channel is last axis
        if ndim == 2:
            return np.pad(img, ((0, pad_M), (0, pad_N)))
        else: 
            return np.pad(img, ((0, pad_M), (0, pad_N), (0,0)))
    else:
        return img



def plot_img(img_data, traces=None, cmap=None, fovea=None, save_path=None, 
             fname=None, sidebyside=False, rnfl=False, close=False, 
             trace_kwds={'c':"r", 'linestyle':"--", 'linewidth':2}):
    """
    Helper function to visualise an OCT B-scan with optional overlays like layer segmentations,
    color maps, and markers.

    Parameters:
    -----------
    img_data : numpy.ndarray
        Input image data to display.

    traces : numpy.ndarray or list of numpy.ndarray, optional
        Trace data to overlay on the image. Can be a single trace or a pair of traces.

    cmap : numpy.ndarray, optional
        Color map data to overlay on the image. Should be the same image dimensions as `img_data`.

    fovea : tuple of (int, int), optional
        Pixel coordinates of the fovea to mark on the image.

    save_path : str, optional
        Directory path to save the image.

    fname : str, optional
        File name for saving the image. If `None`, the image is not saved.

    sidebyside : bool, default=False
        Whether to display the image and its overlay side by side.

    rnfl : bool, default=False
        If `True`, adjust figure size for peripapillary B-scan visualisation.

    close : bool, default=False
        If `True`, close the figure after plotting. If `False`, return the figure object.

    trace_kwds : dict, optional
        Keyword arguments for trace styling (e.g., color, linestyle, linewidth).

    Returns:
    --------
    matplotlib.figure.Figure or None
        The figure object if `close` is `False`, otherwise `None`.
    """
    img = img_data.copy().astype(np.float64)
    img -= img.min()
    img /= img.max()
    M, N = img.shape
    
    if rnfl:
        figsize=(15,6)
    else:
        figsize=(6,6)

    if sidebyside:
        if rnfl:
            figsize = (2*figsize[0], figsize[1])
        else:
            figsize = (figsize[0], 2*figsize[1])
    
    if sidebyside:
        if rnfl:
            fig, (ax0, ax) = plt.subplots(1,2,figsize=figsize)
        else:
            fig, (ax0, ax) = plt.subplots(2,1,figsize=figsize)
        ax0.imshow(img, cmap="gray", zorder=1, vmin=0, vmax=1)
        ax0.set_xticklabels([])
        ax0.set_yticklabels([])
        ax0.tick_params(axis='both', which='both', bottom=False,left=False, labelbottom=False)
    else:
        fig, ax = plt.subplots(1,1,figsize=figsize)
        
    ax.imshow(img, cmap="gray", zorder=1, vmin=0, vmax=1)
    fontsize=16
    if traces is not None:
        if len(traces) == 2:
            for tr in traces:
                 ax.plot(tr[:,0], tr[:,1], zorder=3, **trace_kwds)
        else:
            ax.plot(traces[:,0], traces[:,1], zorder=3, **trace_kwds)

    if cmap is not None:
        cmap_data = cmap.copy().astype(np.float64)
        cmap_data -= cmap_data.min()
        cmap_data /= cmap_data.max()
        ax.imshow(cmap_data, alpha=0.5, zorder=2)
    if fname is not None:
        ax.set_title(fname, fontsize=15)

    if fovea is not None:
        ax.scatter(fovea[0], fovea[1], color="r", edgecolors=(0,0,0), marker="X", s=50, linewidth=1)
            
    ax.set_axis_off()
    fig.tight_layout(pad = 0)
    if save_path is not None and fname is not None:
        ax.set_title(None)
        fig.savefig(os.path.join(save_path, f"{fname}.png"), bbox_inches="tight", pad_inches=0)

    if close:
        plt.close()
    else:
        return fig



def generate_imgmask(mask, thresh=None, cmap=0):
    """
    Generate a plottable RGBA mask from a binary or probabilistic mask.

    Parameters:
    -----------
    mask : numpy.ndarray
        Input mask, typically binary or probabilistic.

    thresh : float, optional
        Threshold for binarising the mask. Values below the threshold are set to 0, others to 1.

    cmap : int or None, default=0
        Index of the RGB channel to colorise. If `None`, all channels are used equally.

    Returns:
    --------
    numpy.ndarray
        Plottable RGBA mask with transparency for non-mask regions.

    Notes:
    ------
    - The function creates an RGBA image, where the alpha channel corresponds to the mask's binary values.
    """
    # Threshold
    pred_mask = mask.copy()
    if thresh is not None:
        pred_mask[pred_mask < thresh] = 0
        pred_mask[pred_mask >= thresh] = 1
    max_val = pred_mask.max()
    
    # Compute plottable cmap using transparency RGBA image.
    trans = max_val*((pred_mask > 0).astype(int)[...,np.newaxis])
    if cmap is not None:
        rgbmap = np.zeros((*mask.shape,3))
        rgbmap[...,cmap] = pred_mask
    else:
        rgbmap = np.transpose(3*[pred_mask], (1,2,0))
    pred_mask_plot = np.concatenate([rgbmap,trans], axis=-1)
    
    return pred_mask_plot


def remove_nans(trace):
    """
    Remove NaN values from a trace.

    Parameters:
    -----------
    trace : numpy.ndarray
        Input trace data, either 2D or 3D.

    Returns:
    --------
    numpy.ndarray
        Trace with NaN values removed.
    """
    if trace.ndim > 2:
        return trace[:,~np.isnan(trace[...,1]).any(axis=0)].astype(np.int64)
    else:
        return trace[~np.isnan(trace[:,1])].astype(np.int64)


def extract_bounds(mask):
    """
    Extract the top and bottom boundaries of a binary mask.

    Parameters:
    -----------
    mask : numpy.ndarray
        Binary mask with a connected region of interest.

    Returns:
    --------
    tuple of (numpy.ndarray, numpy.ndarray)
        Top and bottom boundaries of the mask as arrays of coordinates.

    Notes:
    ------
    - Assumes the mask is fully connected and can be sorted along the horizontal axis.
    """
    # Stack of indexes where mask has predicted 1
    where_ones = np.vstack(np.where(mask.T)).T
    
    # Sort along horizontal axis and extract indexes where differences are
    sort_idxs = np.argwhere(np.diff(where_ones[:,0]))
    
    # Top and bottom bounds are either at these indexes or consecutive locations.
    bot_bounds = np.concatenate([where_ones[sort_idxs].squeeze(),
                                 where_ones[-1,np.newaxis]], axis=0)
    top_bounds = np.concatenate([where_ones[0,np.newaxis],
                                 where_ones[sort_idxs+1].squeeze()], axis=0)
    
    return (top_bounds, bot_bounds)


def select_largest_mask(binmask):
    """
    Retain only the largest connected region in a binary mask.

    Parameters:
    -----------
    binmask : numpy.ndarray
        Binary mask with (potentially) multiple connected regions.

    Returns:
    --------
    numpy.ndarray
        Binary mask with only the largest connected region retained.
    """
    # Look at which of the region has the largest area, and set all other regions to 0
    labels_mask = measure.label(binmask)                       
    regions = measure.regionprops(labels_mask)
    regions.sort(key=lambda x: x.area, reverse=True)
    if len(regions) > 1:
        for rg in regions[1:]:
            labels_mask[rg.coords[:,0], rg.coords[:,1]] = 0
    labels_mask[labels_mask!=0] = 1

    return labels_mask


def interp_trace(traces, align=True):
    """
    Interpolate traces to ensure continuity along the x-axis.

    Parameters:
    -----------
    traces : list of numpy.ndarray
        List of traces, each containing x and y coordinates.

    align : bool, default=True
        Whether to align traces to a common x-range.

    Returns:
    --------
    tuple of numpy.ndarray
        Interpolated traces.
    """
    # Interpolate traces
    new_traces = []
    N = len(traces)
    for i in range(N):
        tr = traces[i]  
        min_x, max_x = (tr[:,0].min(), tr[:,0].max())
        x_grid = np.arange(min_x, max_x)
        y_interp = np.interp(x_grid, tr[:,0], tr[:,1]).astype(int)
        interp_trace = np.concatenate([x_grid.reshape(-1,1), y_interp.reshape(-1,1)], axis=1)
        new_traces.append(interp_trace)

    # Crop traces to make sure they are aligned
    if align:
        top, bot = new_traces
        h_idx=0
        top_stx, bot_stx = top[0,h_idx], bot[0,h_idx]
        common_st_idx = max(top[0,h_idx], bot[0,h_idx])
        common_en_idx = min(top[-1,h_idx], bot[-1,h_idx])
        shifted_top = top[common_st_idx-top_stx:common_en_idx-top_stx]
        shifted_bot = bot[common_st_idx-bot_stx:common_en_idx-bot_stx]
        new_traces = (shifted_top, shifted_bot)

    return tuple(new_traces)



def smart_crop(traces, check_idx=20, ythresh=1, align=True):
    """
    Crop traces to remove discontinuities based on local y-value changes at the end of
    traces.

    Parameters:
    -----------
    traces : list of numpy.ndarray
        List of traces, each containing x and y coordinates.

    check_idx : int, default=20
        Number of points to check at the start and end of each trace.

    ythresh : float, default=1
        Threshold for discontinuity detection in y-values.

    align : bool, default=True
        Whether to align cropped traces to a common x-range.

    Returns:
    --------
    tuple of numpy.ndarray
        Cropped and aligned traces.
    """
    cropped_tr = []
    for i in range(2):
        lyr = traces[i]
        ends_l = np.argwhere(np.abs(np.diff(lyr[:check_idx,1])) > ythresh)
        ends_r = np.argwhere(np.abs(np.diff(lyr[-check_idx:,1])) > ythresh)
        if ends_r.shape[0] != 0:
            lyr = lyr[:-(check_idx-ends_r.min())]
        if ends_l.shape[0] != 0:
            lyr = lyr[ends_l.max()+1:]
        cropped_tr.append(lyr)

    return interp_trace(cropped_tr, align=align)



def get_trace(pred_mask, threshold=0.5, align=False):
    """
    Extract top and bottom traces from a prediction mask.

    The function thresholds the mask, selects the largest connected region, extracts boundaries, and crops discontinuities.

    Parameters:
    -----------
    pred_mask : numpy.ndarray
        Prediction mask, typically probabilistic.

    threshold : float, default=0.5
        Threshold for binarizing the mask.

    align : bool, default=False
        Whether to align the traces to a common x-range.

    Returns:
    --------
    tuple of numpy.ndarray
        Top and bottom traces extracted from the mask.
    """
    if threshold is not None:
        binmask = (pred_mask > threshold).astype(int)
    binmask = select_largest_mask(binmask)
    traces = extract_bounds(binmask)
    traces = smart_crop(traces, align=align)
    return traces



def rebuild_mask(traces, img_shape=None):
    """
    Rebuild a binary mask from upper and lower boundaries. The mask is created by filling in the region between the top and bottom traces.

    Parameters:
    -----------
    traces : tuple of numpy.ndarray
        Top and bottom traces.

    img_shape : tuple of (int, int), optional
        Shape of the output mask. If `None`, the mask size is inferred from traces.

    Returns:
    --------
    numpy.ndarray
        Binary mask reconstructed from the traces.
    """
    # Work out extremal coordinates of traces
    top_lyr, bot_lyr = interp_trace(traces)
    common_st_idx = np.maximum(top_lyr[0,0], bot_lyr[0,0])
    common_en_idx = np.minimum(top_lyr[-1,0], bot_lyr[-1,0])
    top_idx = top_lyr[:,1].min()
    bot_idx = bot_lyr[:,1].max()

    # Initialise binary mask
    if img_shape is not None:
        binmask = np.zeros(img_shape)
    else:
        binmask = np.zeros((bot_idx+100, common_en_idx+100))

    # Fill in region between upper and lower boundaries
    for i in range(common_st_idx, common_en_idx):
        top_i = top_lyr[i-common_st_idx,1]
        bot_i = bot_lyr[i-common_st_idx,1]
        binmask[top_i:bot_i,i] = 1

    return binmask



def normalise(img, minmax_val=(0,1), astyp=np.float64):
    """
    Normalise an image to a specified range.

    Parameters:
    -----------
    img : numpy.ndarray
        Input image to be normalised. The image can be of any numeric data type.

    minmax_val : tuple of (float, float), optional, default=(0, 1)
        The minimum and maximum values to which the image will be normalised.

    astyp : numpy.dtype, optional, default=np.float64
        Data type for the normalised image output.

    Returns:
    --------
    numpy.ndarray
        Normalised image scaled to the specified range (`minmax_val`) and converted to the specified data type (`astyp`).

    Example:
    --------
    >>> import numpy as np
    >>> img = np.array([[10, 20, 30], [40, 50, 60]], dtype=np.uint8)
    >>> normalised_img = normalise(img, minmax_val=(0, 255), astyp=np.uint8)
    >>> print(normalised_img)
    [[  0  51 102]
     [153 204 255]]
    """
    # Extract minimum and maximum values
    min_val, max_val = minmax_val

    # Convert to float type to perform [0, 1] normalisation
    img = img.astype(np.float64)

    # Normalise to [0, 1]
    img -= img.min()
    img /= img.max()

    # Rescale to max_val and output as specified data type
    img *= (max_val - min_val)
    img += min_val

    return img.astype(astyp)



def normalise_brightness(img, target_mean=0.2):
    """
    Adjusts the brightness of an image by applying a gamma transformation to reach a target mean brightness.

    Parameters:
    -----------
    img : numpy.ndarray
        The input image array, which can be of any numeric data type.

    target_mean : float, default=0.2
        The desired target mean brightness of the image after adjustment.

    Returns:
    --------
    numpy.ndarray
        The gamma-transformed and normalized image with improved brightness/contrast.

    Notes:
    ------
    - The function calculates a gamma correction factor based on the ratio of the target mean brightness 
      to the current mean brightness of the image.
    - After adjusting the brightness using gamma transformation, the image is normalized to the range [0, 1].
    
    Example:
    --------
    bright_img = normalise_brightness(img, target_mean=0.25)
    """
    # Adjust image to mean brightness of 0.25
    gamma = np.log(target_mean) / np.log(img.mean())
    img = exposure.adjust_gamma(img, gamma=gamma)
    img = normalise(img)

    return img


def shadow_compensate(img, gamma=1, win_size=75, plot=False):
    """
    Compensates for vessel shadowing in an OCT B-scan image by adjusting A-scan pixel intensities. 
    The compensation is achieved by scaling the intensities to average out the drop in signal 
    caused by shadowing, using a moving average filter. The `gamma` parameter enhances the image 
    by adjusting the pixel intensity scaling.

    Parameters:
    -----------
    img : numpy.ndarray
        A 2D or 3D array representing the OCT B-scan image. If the image has more than two dimensions 
        (e.g., RGB), the first channel is assumed to be the grayscale (OCT) image.
        
    gamma : float, optional, default=1
        A scaling factor for intensity adjustment. Values greater than 1 darken the image, 
        while values less than 1 brighten it. This serves as an implicit image enhancement.

    win_size : int, optional, default=75
        The window size for the moving average used to smooth the energy levels across A-scans. 

    plot : bool, optional, default=False
        If True, the function will generate plots showing the energy levels, compensation factors, 
        and comparisons between the original and compensated images.

    Returns:
    --------
    numpy.ndarray
        A 2D array representing the shadow-compensated OCT image, normalized to the same range 
        as the original input image.

    Notes:
    ------
    - The function first crops any black columns on the left and right sides of the image, assuming 
      these columns represent areas with no data.
    - It adjusts pixel intensities using a gamma correction followed by a moving average filtering 
      of the total energy across each A-scan.
    - The compensation is applied by scaling each A-scan energy to match its smoothed energy value.
    - If `plot` is set to True, two sets of plots are generated: one for the energy levels and correction 
      factors, and another comparing the original and compensated images.

    Example:
    --------
    compensated_img = shadow_compensate(img, gamma=1.2, win_size=100, plot=True)
    """
    # If RGB, select first channel on the assumption it is an OCT B-scan 
    # and is therefore grayscale. Channels in last dimension by assumption.
    if img.ndim > 2:
        img = img[...,0]
    
    # Remove any black columns either side of image
    comp_idx_l = img[:,:img.shape[1]//2].mean(axis=0) != 0
    comp_idx_r = img[:,img.shape[1]//2:].mean(axis=0) != 0
    img_crop = img[:,np.concatenate([comp_idx_l, comp_idx_r], axis=0)]

    # Energy of each pixel of the A-line, where gamma > 1 for implicit image enhancement
    # by darkening the image
    E_ij = exposure.adjust_gamma(img_crop, gamma=gamma)

    # Total energy of each A-scan is the sum across their rows
    E_i = E_ij.sum(axis=0)

    # Centred, moving average according to win_size, pad edges of average with original signal
    E_i_smooth = pd.Series(E_i).rolling(win_size, center=True).mean().values
    E_i_smooth[:win_size//2], E_i_smooth[-win_size//2:] = E_i[:win_size//2], E_i[-win_size//2:]

    # Compensation is linear scale made to individual energy levels to match total energy per A-scan
    # with its moving average value
    E_comp = (E_i_smooth / E_i)

    # If plotting energy levels
    if plot:
        fig, (ax, ax1) = plt.subplots(1,2,figsize=(14,7))
        ax.set_xlabel("A-scan (column) index", fontsize=14)
        ax.set_ylabel(rf"Energy level (I$^\gamma$)", fontsize=14)
        ax.plot(np.arange(E_i.shape[0]), E_i, c="b", linewidth=2)
        ax.plot(np.arange(E_i_smooth.shape[0]), E_i_smooth, c="r", linewidth=2)
        ax1.set_xlabel("A-scan (column) index", fontsize=14)
        ax1.set_ylabel(rf"Correction factor (I$^\gamma$ / MA(I$^\gamma$)", fontsize=14)
        ax1.plot(np.arange(E_comp.shape[0]), E_comp, c="r", linewidth=2)
        ax.set_title(rf"Energy per A-scan ($\gamma$ = {gamma})", fontsize=18)
        ax1.set_title(rf"Correction per A-scan ($\gamma$ = {gamma})", fontsize=18)

    # Reshape to apply element-wise to original image and darken
    E_comp_arr = E_comp.reshape(1,-1).repeat(img_crop.shape[0], axis=0)
    output = E_ij*E_comp_arr
    #output = (img_crop*E_comp_arr)**gamma

    # Put back any black columns either side of image
    if (~comp_idx_l).sum() > 0:
        output = np.pad(output, ((0,0),((~comp_idx_l).sum(),0)))
    if (~comp_idx_r).sum() > 0:
        output = np.pad(output, ((0,0),(0,(~comp_idx_r).sum())))

    # Plot original and compensated versions
    if plot:
        fig, (ax, ax1) = plt.subplots(1,2,figsize=(18,7))
        ax.imshow(img, cmap="gray")
        ax1.imshow(output, cmap="gray")
        ax.set_axis_off()
        ax1.set_axis_off()
        fig.tight_layout()

    # Return normalised output
    output = normalise(output)

    return output



def flatten_dict(nested_dict):
    """
    Recursively flattens a nested dictionary where each value can be a dictionary itself. 
    The function traverses the nested structure and constructs a flat dictionary where 
    the keys represent the hierarchical path to the value.

    Parameters:
    -----------
    nested_dict : dict
        A nested dictionary where the values can be either other dictionaries or non-dictionary values.

    Returns:
    --------
    dict
        A flattened dictionary where the keys are tuples representing the path to the value 
        in the original nested structure, and the values are the corresponding leaf values.

    Example:
    --------
    >>> example_dict = {
        'A': {'A1': 1, 'A2': 2},
        'B': {'B1': 3}
    }
    >>> flatten_dict(example_dict)
    {
        ('A', 'A1'): 1,
        ('A', 'A2'): 2,
        ('B', 'B1'): 3
    }
    """
    res = {}
    if isinstance(nested_dict, dict):
        for k in nested_dict:
            flattened_dict = flatten_dict(nested_dict[k])
            for key, val in flattened_dict.items():
                key = list(key)
                key.insert(0, k)
                res[tuple(key)] = val
    else:
        res[()] = nested_dict
    return res


def nested_dict_to_df(values_dict):
    """
    Converts a nested dictionary into a multi-level Pandas DataFrame by first flattening it.
    The resulting DataFrame has a hierarchical column structure, where the dictionary keys 
    define the index and columns.

    Parameters:
    -----------
    values_dict : dict
        A nested dictionary that needs to be converted into a DataFrame. The dictionary is flattened 
        before being transformed into a DataFrame.

    Returns:
    --------
    pandas.DataFrame
        A Pandas DataFrame where the flattened dictionary is represented with a multi-level 
        index and columns. The index reflects the path from the root to the leaf in the nested structure.

    Notes:
    ------
    - This function uses `flatten_dict(...)` to flatten the input dictionary before converting it into a DataFrame.
    - The final DataFrame is "unstacked" to create hierarchical columns, which are formatted based on 
      the second level of the dictionary keys.

    Example:
    --------
    >>> example_dict = {
        'A': {'A1': 1, 'A2': 2},
        'B': {'B1': 3}
    }
    >>> nested_dict_to_df(example_dict)
    A         B
    A1  A2    B1
    1   2     3
    """
    flat_dict = flatten_dict(values_dict)
    df = pd.DataFrame.from_dict(flat_dict, orient="index")
    df.index = pd.MultiIndex.from_tuples(df.index)
    df = df.unstack(level=-1)
    df.columns = df.columns.map("{0[1]}".format)
    return df



def align_peripapillary_data(metadata, fovea_at_slo, slo_acq, slo_avimout, fname, save_path, save=True):
    """
    Align the peripapillary thickness profile with reference to the fovea and optic disc center.

    This function aligns the thickness profile (B-scan) around the optic nerve head by determining the 
    relative position of the fovea and optic disc (OD) and locating the index in the thickness profile
    whose position on the circular acquisition location is co-linear with the fovea and optic disc. This
    index is used to shift the thickness profile of the OCT B-scan to its centre

    Parameters
    ----------
    metadata : dict
        Metadata containing information about the acquisition, including user-specified OD center.
        
    fovea_at_slo : np.ndarray
        (x,y)-coordinates of the fovea in the SLO image.
        
    slo_acq : np.ndarray
        SLO acquisition image with the peripapillary circular acquisition location overlaid.
        
    slo_avimout : np.ndarray
        SLO image with optic disc segmentation and other features.
        
    fname : str
        Filename to use for saving the alignment plot.
        
    save_path : str
        Path where the alignment plot will be saved.
        
    save : bool, default=True
        Whether to save the alignment plot.

    Returns
    -------
    od_mask_center : np.ndarray
        (x,y)-coordinates of the model-predicted OD center.
        
    offset_ratio : float
        Ratio of the distance between user-specified and model-predicted OD centers 
        to the optic disc diameter.
        
    ascan_idx_temp0 : int
        Index of the A-scan corresponding to the temporal midpoint.

    Notes
    -----
    - This function also computes the optic disc overlap index, which is the alignment between 
    user-specified and model-predicted OD centers, helping determine how off-centre the peripapillary
    OCT acquisition is.
    """
    # Extract Optic disc mask and acquisition line boundary, dilate circ mask for plotting
    od_mask = slo_avimout[...,1]
    N = od_mask.shape[1]
    circ_mask = slo_acq[...,1] == 1
    circ_mask_dilate = morphology.dilation(circ_mask, footprint=morphology.disk(radius=2))
    
    # Extract user-specified OD center, and binary mask-based OD center and setup
    # measuring overlap
    od_user_center = np.array([metadata["acquisition_optic_disc_center_x"], 
                               metadata["acquisition_optic_disc_center_y"]]).astype(int)
    od_mask_center = measure.centroid(od_mask.astype(bool))[[1,0]]

    # Overlap index measures distance between user-specified and model-specified (acting as ground
    # truth) and normalises that according to the radius of the mask's optic disc binary mask. The 
    # radius is deduced by the minor axis lenth of the optic disc mask.
    props = measure.regionprops(measure.label(od_mask))[0]
    od_diameter = int((props.axis_minor_length + props.axis_major_length)/2)
    od_centers_both = np.concatenate([od_mask_center[np.newaxis], od_user_center[np.newaxis]], axis=0)
    center_distance = feature_measurement._curve_length(od_centers_both[:,0], od_centers_both[:,1])
    offset_ratio = np.round(center_distance / od_diameter,4)

    # Work out reference line between fovea and user-specified optic disc center. 
    Xy =  np.concatenate([od_user_center[np.newaxis], fovea_at_slo[np.newaxis]], axis=0)
    linmod = LinearRegression().fit(Xy[:,0].reshape(-1,1), Xy[:,1])
    x_grid = np.arange(min(Xy[0,0], Xy[1,0]), max(Xy[0,0], Xy[1,0])).astype(int)
    y_grid = linmod.predict(x_grid.reshape(-1,1)).astype(int)
    
    # Intersection of reference line and circular acquisition line is where the temporal
    # midpoint
    temp_mid_idx = np.argwhere(circ_mask[(y_grid, x_grid)] == 1)[0]
    temporal_mid = np.array((x_grid[temp_mid_idx], y_grid[temp_mid_idx])).reshape(-1)
    
    # Work out where this temporal midpoint is along the peripapillary OCT B-can (A-scan
    # location). We use this value to align our thickness profile to, i.e. shift
    # the peripapillary OCT B-scan in order for the central A-scan to be where the middle
    # of the Temporal subregion is (temporal midpoint)
    od_user_radius = metadata["acquisition_radius_px"]
    circ_coords = np.array([(od_user_radius*np.cos(theta)+od_user_center[0],
                             od_user_radius*np.sin(theta)+od_user_center[1]) 
                            for theta in np.linspace(0, 2*np.pi, N)])
    ascan_idx_temp0 = np.sum(np.square(circ_coords - temporal_mid), axis=1).argmin()

    # Plot the result onto SLO 
    od_user_edge = np.array(metadata["stxy_coord"].split(",")).astype(int)
    fig, ax = plt.subplots(1,1,figsize=(10,10))
    ax.imshow(slo_acq)
    ax.imshow(slo_avimout)
    ax.imshow(generate_imgmask(circ_mask_dilate,cmap=1))
    ax.scatter(od_mask_center[0], od_mask_center[1], zorder=3, s=200, c="green", edgecolors=(0,0,0), marker="X", label="True OD center")
    ax.scatter(od_user_center[0], od_user_center[1], zorder=3, s=200, c="b", marker="X", edgecolors=(0,0,0), label="User-specified OD center")
    ax.scatter(fovea_at_slo[0], fovea_at_slo[1], label="Fovea", c="orange", 
               marker="X", edgecolors=(0,0,0), s=200, zorder=5)
    ax.plot(x_grid, y_grid, c="m", linewidth=3, label="Reference line (Fovea -> User OD Center)")
    ax.scatter(temporal_mid[0], temporal_mid[1], marker="X", c="r", 
               edgecolors=(0,0,0), s=200, label="Temporal center of peripapillary thickness", zorder=5)
    ax.plot([-1],[-1],c="lime",label="Line of acquisition")
    
    legend_loc = "lower right"
    if metadata["eye"] == "Right":
        legend_loc = "lower left" 
    ax.legend(loc=legend_loc, fontsize=16)
    ax.axis([0,N,N,0])
    ax.set_axis_off()
    ax.set_title(f"Overlap(user,true) of OD center: {np.round(100*offset_ratio,2)}% of OD diameter", fontsize=20)
    fig.tight_layout()
    if save:
        fig.savefig(os.path.join(save_path, f"{fname}_peripapillary_alignment.png"), bbox_inches="tight")
    plt.close()

    return od_mask_center, offset_ratio, ascan_idx_temp0


def compute_opticdisc_radius(fovea, od_centre, od_mask):
    """
    Compute the optic disc radius relative to its position with the fovea.

    This function calculates the optic disc radius by determining the intersection of a 
    reference line (from the fovea to the optic disc center) with the optic disc boundary.

    Parameters
    ----------
    fovea : np.ndarray
        (x,y)-coordinates of the fovea in the image.
        
    od_centre : np.ndarray
        (x,y)-coordinates of the optic disc center in the image.
        
    od_mask : np.ndarray
        Binary mask of the optic disc.

    Returns
    -------
    od_radius : int
        Radius of the optic disc in pixels, calculated relative to the fovea.
        
    plot_info : tuple
        Information for plotting, including the intersection coordinates, 
        the reference line grid, and intersection indices.

    Notes
    -----
    - This is deprecated, as a simpler version averages and minor and major axis lengths
    in '_process_opticdisc'.
    """
    # Extract Optic disc mask and acquisition line boundary,
    od_mask_props = measure.regionprops(measure.label(od_mask))[0]
    #od_mask_radius = od_mask_props.axis_minor_length/2 # naive radius, without account for orientation with fovea
    #od_mask_radius = od_mask_props.axis_major_length/2
    od_boundary = segmentation.find_boundaries(od_mask)

    # Work out reference line between fovea and  optic disc center. 
    Xy =  np.concatenate([od_centre[np.newaxis], fovea[np.newaxis]], axis=0)
    linmod = LinearRegression().fit(Xy[:,0].reshape(-1,1), Xy[:,1])
    x_grid = np.arange(min(Xy[0,0], Xy[1,0]), max(Xy[0,0], Xy[1,0])).astype(int)
    y_grid = linmod.predict(x_grid.reshape(-1,1)).astype(int)
    
    # Intersection of reference line and optic disc boundary
    intersection_idx = np.argwhere(od_boundary[(y_grid, x_grid)] == 1)[0]
    od_intersection = np.array((x_grid[intersection_idx], 
                                y_grid[intersection_idx])).reshape(1,-1)

    # Now we can work out the optic disc radius, according to it's position with the fovea
    od_bounds = np.concatenate([od_centre.reshape(1,-1), od_intersection], axis=0)
    od_radius = feature_measurement._curve_length(od_bounds[:,0], od_bounds[:,1])

    plot_info = (od_intersection, (x_grid, y_grid), intersection_idx)

    return np.round(od_radius).astype(int), plot_info



def _process_opticdisc(od_mask):
    """
    Compute the optic disc radius and extract its boundary.

    This function calculates the average optic disc radius using the minor and major 
    axis lengths of the optic disc mask and identifies the boundary of the optic disc.

    Parameters
    ----------
    od_mask : np.ndarray
        Binary mask of the optic disc.

    Returns
    -------
    od_radius : int or None
        Radius of the optic disc in pixels, averaged from the major and minor axes.
        Returns `None` if the optic disc mask is invalid.
        
    od_boundary : np.ndarray
        Binary mask of the optic disc boundary.
    """
    # Extract Optic disc radius and OD boundary if detected
    try:
        od_mask_props = measure.regionprops(measure.label(od_mask))[0]
    except:
        return None, np.zeros_like(od_mask)
    od_radius = int((od_mask_props.axis_minor_length + od_mask_props.axis_major_length)/4)
    od_boundary = segmentation.find_boundaries(od_mask)

    return od_radius, od_boundary


def sort_trace(df, layers=['CHORupper', 'CHORlower']):
    '''
    Load in paired layer segmentations from OCTolyzer's output DataFrame.
    '''
    trace = pd.melt(df, id_vars='layer', value_name='y', var_name='x')
    traces = [trace[trace.layer==lyr].iloc[:,1:].values.astype(int) for lyr in layers]
    traces = tuple([tr[tr[:,1]!=0] for tr in traces])
    return interp_trace(traces)



def load_annotation(path, key=None, raw=False, binary=False):
    """
    Load a `.nii.gz` annotation file and extract region and vessel masks.

    This function reads a `.nii.gz` file containing manual segmentations for an SLO image 
    and extracts artery, vein, optic disc, and overall segmentation masks. It supports 
    returning the raw segmentation map or binary vessel masks.

    Parameters:
    ----------
    path : str or pathlib.Path
        File path to the `.nii.gz` annotation file.

    key : tuple of int, optional
        Custom grayscale intensity values for artery, vein, and optic disc in the 
        annotation file. Defaults to:
        - Artery: 191
        - Vein: 127
        - Optic Disc: 255.

    raw : bool, optional, default=False
        If True, returns the raw segmentation map without further processing.

    binary : bool, optional, default=False
        If True, returns a binary mask corresponding to the optic disc region 
        (grayscale intensity `key[2]` or default value `255`).

    Returns:
    -------
    np.ndarray
        - If `raw=True`: Returns a 2D array representing the raw segmentation map.
        - If `binary=True`: Returns a binary mask for the all-vessel model.
        - Otherwise: Returns a 3D binary mask for the AVOD model:
            - Channel 0: Binary mask for arteries.
            - Channel 1: Binary mask for the optic disc.
            - Channel 2: Binary mask for veins.
            - Channel 3: Binary mask for all segmented regions.

    Notes:
    -----
    - Default grayscale intensity values correspond to the label encoding used in ITK-Snap.
    - The returned array's shape is `(H, W, 4)` when `raw` and `binary` are False.

    Examples:
    --------
    >>> cmap = load_annotation("/path/to/annotation.nii.gz")
    >>> print(cmap.shape)
    (512, 512, 4)

    >>> raw_segmentation = load_annotation("/path/to/annotation.nii.gz", raw=True)
    >>> print(raw_segmentation.shape)
    (512, 512)

    >>> optic_disc_binary = load_annotation("/path/to/annotation.nii.gz", binary=True)
    >>> print(optic_disc_binary.shape)
    (512, 512)
    """    
    # Read the .nii image containing thevsegmentations
    sitk_t1 = sitk.ReadImage(path)

    # Default grayscale intensity encoding from ITK-Snap for 
    # manual SLO segmentation
    if key is None:
        a_i = 191
        od_i = 255
        v_i = 127
    else:
        a_i, v_i, od_i = key
        
    # and access the numpy array, saved as (1, N, N)
    segmentations = sitk.GetArrayFromImage(sitk_t1)[0]

    # returning raw segmentation map
    if raw:
        return segmentations

    # if binary, only return vessels with label 255
    if binary:
        return (segmentations == od_i).astype(int)

    # for artery-vein-optic disc map
    artery = (segmentations == a_i)
    OD = (segmentations == od_i)
    vein = (segmentations == v_i)
    mask = (segmentations > 0)
    cmap = np.concatenate([artery[...,np.newaxis], 
                           OD[...,np.newaxis],
                           vein[...,np.newaxis], 
                           mask[...,np.newaxis]], axis=-1).astype(int)
    
    
    return cmap



def plot_composite_bscans(bscan_data, vmasks, fovea_info, layer_pairwise, reshape_idx, analyse_choroid, fname, save_path, overlay_areas=None, side_by_side=False):
    """
    Create and save a composite high-resolution visualization of all B-scans in an OCT stack.

    This function generates a stitched image of all B-scans from an OCT scan stack, with overlaid 
    segmentations, regions of interest (ROI), and optional vessel maps. It supports volume and 
    H-line/V-line/Radial scan formats and can include fovea-centered landmarks and additional ROI overlays if specified.

    Parameters:
    ----------
    bscan_data : ndarray
        3D array containing the OCT B-scan data with dimensions `(num_scans, height, width)`.

    vmasks : ndarray
        3D array of choroidal vessel masks corresponding to the B-scans. Required if `analyse_choroid` is `True`.

    fovea_info : int or list
        - If an integer, it specifies the index of the fovea-centered B-scan in volume scans.
        - If a list, it contains `(x, y)` coordinates of the fovea for H-line/V-line/Radial scans.

    layer_pairwise : dict
        Dictionary where keys are layer boundary pairs (e.g., "ILM_BM") and values are lists of 
        segmentation traces for each B-scan.

    reshape_idx : tuple
        Tuple specifying how the B-scans should be arranged in the composite (e.g., `(rows, cols)`).

    analyse_choroid : bool
        Flag indicating whether to include choroid-related visualizations (e.g., vessel maps).

    fname : str
        The base name of the output file (excluding the file extension).

    save_path : str or pathlib.Path
        Directory path where the resulting composite image will be saved.

    overlay_areas : dict, optional
        Dictionary containing additional overlays representing the ROIs measured, with the following keys:
        - `areas`: List of 2D arrays specifying regions of interest for the retina and choroid.
        - `thicks`: List of thickness measurements and their corresponding overlays.
        - `macula_rum`: Radius of the macular region of interest in microns.

    side_by_side : bool, optional
        Flag indicating whether to display raw and segmented B-scans side by side.
    Returns:
    -------
    None
        The function saves the generated composite image to the specified directory.

    Outputs:
    -------
    - A high-resolution PNG file showing the stitched B-scans with overlays:
        - For volume scans: `{fname}_volume_octseg.png`
        - For H-line/V-line/Radial scans: `{fname}_linescan_octseg.png`

    Notes:
    -----
    - For volume scans, the fovea-centered B-scan is excluded from the composite image.
    - Overlays include fovea-centered ROIs for the retina (and choroid if `analyse_choroid`
      is True), and thickness lines to define their boundaries.
    - Colours for segmentations are randomized for clear visualisation.
    - The figure dimensions and DPI are optimised for high-resolution output.

    Examples:
    --------
    >>> plot_composite_bscans(
    ...     bscan_data=bscan_array,
    ...     vmasks=choroid_vessel_masks,
    ...     fovea_info=30,  # For volume scans
    ...     layer_pairwise=segmentation_traces,
    ...     reshape_idx=(5, 5),
    ...     analyse_choroid=True,
    ...     fname="example_scan",
    ...     save_path="/path/to/output",
    ...     overlay_areas={
    ...         "areas": retina_choroid_rois,
    ...         "thicks": thickness_overlays,
    ...         "macula_rum": 1000
    ...     }
    ... )
    """
    # Get layer names
    pairwise_keys = list(layer_pairwise.keys())
    layer_keys = list(set(pd.DataFrame(pairwise_keys).reset_index(drop=True)[0].str.split("_", expand=True).values.flatten()))
    
    # Organise B-scan data and choroid vessel maps
    img_shape = bscan_data.shape[-2:]
    M, N = img_shape
    bscan_list = list(bscan_data.copy())
    if isinstance(fovea_info, int):
        bscan_list.pop(fovea_info)
    bscan_arr = np.array(bscan_list)
    bscan_arr = bscan_arr.reshape(*reshape_idx,*img_shape)
    bscan_stacked = np.concatenate(np.concatenate(bscan_arr, axis=-2), axis=-1)

    # Sort out vessel maps if analysing choroid
    if analyse_choroid:
        vmasks_list = list(vmasks.copy())
        if isinstance(fovea_info, int):
            vmasks_list.pop(fovea_info)
        vmasks_arr = np.asarray(vmasks_list)
        vmasks_arr = vmasks_arr.reshape(*reshape_idx,M,N)
        vmask_stacked = np.concatenate(np.concatenate(vmasks_arr, axis=-2), axis=-1)
        all_vcmap = np.concatenate([vmask_stacked[...,np.newaxis]] 
                    + 2*[np.zeros_like(vmask_stacked)[...,np.newaxis]] 
                    + [vmask_stacked[...,np.newaxis] > 0.01], axis=-1)

    # Stack measurement ROIs if overlays are provided
    if overlay_areas is not None:
        areas = overlay_areas['areas']
        retmaps = np.array([arr[0] for arr in areas]).reshape(*reshape_idx,M,N)
        retmaps = np.concatenate(np.concatenate(retmaps, axis=-2), axis=-1)
        if analyse_choroid:
            chormaps = np.array([arr[1] for arr in areas]).reshape(*reshape_idx,M,N)
            chormaps = np.concatenate(np.concatenate(chormaps, axis=-2), axis=-1)

    # Colour scheme for different layer segmentations    
    np.random.seed(0)
    COLORS = {key:np.random.randint(255, size=3)/255 for key in layer_keys}

    # Figure to be saved out at same dimensions as stacked array
    h,w = bscan_stacked.shape
    if side_by_side:
        fig, (ax0, ax) = plt.subplots(1,2,figsize=(2*w/1000, h/1000), dpi=100)
        ax0.imshow(bscan_stacked, cmap='gray')
        ax0.set_axis_off()
    else:
        fig, ax = plt.subplots(1,1,figsize=(w/1000, h/1000), dpi=100)
    ax.set_axis_off()

    # Overlay stacked B-scans and ROI maps of retina and choroid (if provided)
    ax.imshow(bscan_stacked, cmap='gray')
    if overlay_areas is not None:
        ax.imshow(generate_imgmask(retmaps, None, 1), alpha=0.25, zorder=1)
        if analyse_choroid:
            ax.imshow(generate_imgmask(chormaps, None, 1), alpha=0.25, zorder=1)

    # Add all traces and fovea (if provided)
    for (i, j) in np.ndindex(reshape_idx):
        layer_keys_copied = layer_keys.copy()
        for key, traces in layer_pairwise.items():
            tr = traces.copy()

            # If volume scan, remove fovea-centred trace
            if isinstance(fovea_info, int):
                tr.pop(fovea_info)

            # If radial scan, overlay foveas
            else:
                fovea_xy = fovea_info[reshape_idx[1]*i + j]
                ax.scatter(fovea_xy[0]+j*N, fovea_xy[1]+i*M, label='_ignore', color='r', zorder=3, marker='X', edgecolors=(0,0,0), linewidth=0.1, s=2)

            # If radial, overlay ROI area
            if overlay_areas is not None:
                all_thicks = overlay_areas['thicks'][reshape_idx[1]*i + j]
                for thicks in all_thicks:
                    if thicks is not None:
                        for line_pts in thicks:
                            ax.plot(line_pts[:,0]+j*N, line_pts[:,1]+i*M, color='g', linestyle='--', linewidth=0.175, zorder=3, label='_ignore')
                if i == 0 and j == 0 and key == 'ILM_BM':
                    ax.fill_between([-2,-1], [-2,-1], color='g', alpha=0.25, label=f"Region of interest\n({2*overlay_areas['macula_rum']} microns fovea-centred)")

            # Overlay trace
            for (k, t) in zip(key.split("_"), tr[reshape_idx[1]*i + j]):
                if k in layer_keys_copied:
                    c = COLORS[k]
                    ax.plot(t[:,0]+j*N,t[:,1]+i*M, label='_ignore', color=c, zorder=2, linewidth=0.175)
                    layer_keys_copied.remove(k)

    # Add vessel maps  (if analyse_choroid is 1)
    if analyse_choroid:
        ax.imshow(all_vcmap, alpha=0.5)

    # Prepare to save out
    if overlay_areas is not None:
        ax.axis([0, w-1, h-1, 0])
        ax.legend(fontsize=h/1000)
    fig.tight_layout(pad=0)
    if isinstance(fovea_info, int):
        fig.savefig(os.path.join(save_path, f"{fname}_volume_octseg.png"), dpi=1000)
    else:
        fig.savefig(os.path.join(save_path, f"{fname}_linescan_octseg.png"), dpi=1000)
    plt.close()


def print_error(e, verbose=True):
    """
    Generate and print detailed information about an exception, including its traceback.

    This function is used to handle unexpected errors during execution. It provides a detailed 
    explanation of the exception type, message, and traceback information. The output is 
    optionally printed to the console and returned as a log-friendly list of strings.

    Parameters:
    ----------
    e : Exception
        The exception instance to be analysed and logged.
    verbose : bool, optional, default=True
        If True, prints the error message and traceback details to the console.

    Returns:
    -------
    logging_list : list of str
        A list of strings containing the exception message and full traceback details.

    Notes:
    -----
    - This function is typically used in robust execution contexts where errors need 
      to be logged for debugging without halting the program.
    - The traceback information includes the filename, function name, and line number 
      for each level of the traceback.

    Examples:
    --------
    >>> try:
    ...     1 / 0
    ... except Exception as e:
    ...     error_log = print_error(e, verbose=True)
    ...     # Outputs error details to the console and saves them in error_log.
    """
    message = f"\nAn exception of type {type(e).__name__} occurred. Error description:\n{str(e)}\n"
    if verbose:
        print(message)
    trace = ["Full traceback:\n"]
    if verbose:
        print(trace[0])
    tb = e.__traceback__
    tb_i = 1
    while tb is not None:
        tb_fname = tb.tb_frame.f_code.co_filename
        tb_func = tb.tb_frame.f_code.co_name
        tb_lineno = tb.tb_lineno
        tb_str = f"Traceback {tb_i} to filename\n{tb_fname}\nfor function {tb_func}(...) at line {tb_lineno}.\n"
        if verbose:
            print(tb_str)
        trace.append(tb_str)
        tb = tb.tb_next
        tb_i += 1
    logging_list = [message] + trace
    return logging_list


def superimpose_slo_segmentation(slo, slo_vbinmap, slo_avimout, 
                             od_mask, od_centre, 
                             fovea, location, zonal_masks,
                             save_info):
    '''
    Superimpose the segmentation masks onto the SLO image.
    '''
    fname, save_path, dirpath, save_images, collate_segmentations = save_info
    # binary vessel mask - purple
    N = slo_vbinmap.shape[0]
    slo_vcmap = generate_imgmask(slo_vbinmap, None, 1)
    stacked_img = np.hstack(3*[slo/255])

    # Stacks the colour maps together, binary, then artery-vein-optic disc
    slo_av_cmap = slo_avimout.copy()
    slo_av_cmap[slo_av_cmap[...,1]>0,-1] = 0
    slo_av_cmap[...,1] = 0
    stacked_cmap = np.hstack([np.zeros_like(slo_vcmap), slo_vcmap, slo_av_cmap])
    if od_mask.sum() != 0:
        od_coords = avo_inference._fit_ellipse((255*od_mask).astype(np.uint8), get_contours=True)[:,0]
        od_coords = od_coords[(od_coords[:,0] > 0) & (od_coords[:,0] < N-1)]
        od_coords = od_coords[(od_coords[:,1] > 0) & (od_coords[:,1] < N-1)]

    fig, ax = plt.subplots(1,1,figsize=(18,6))
    ax.imshow(stacked_img, cmap="gray")
    ax.imshow(stacked_cmap, alpha=0.5)
    for i in [N, 2*N]:
        ax.scatter(fovea[0]+i, fovea[1], marker="X", s=100, edgecolors=(0,0,0), c="r")
        if i == 2*N:  
            if od_mask.sum() != 0:
                ax.plot(od_coords[:,0]+i, od_coords[:,1], color='lime', linestyle='--', linewidth=3, zorder=4)
                if location == "Optic disc":
                    ax.scatter(od_centre[0]+i, od_centre[1], marker="X", s=100, edgecolors=(0,0,0), c="lime", zorder=4)
        else:
            if od_mask.sum() != 0:
                ax.plot(od_coords[:,0]+i, od_coords[:,1], color='blue', linestyle='--', linewidth=3, zorder=4)
                if location == "Optic disc":
                    ax.scatter(od_centre[0]+i, od_centre[1], marker="X", s=100, edgecolors=(0,0,0), c="blue", zorder=4)
        
            if location == "Optic disc":
                for mask, colour, z in zip(zonal_masks[1:], [0,2], [3,2]):
                    mask_bnds = segmentation.find_boundaries(mask)
                    mask = np.hstack(2*[np.zeros_like(mask)]+[mask])
                    cmap = generate_imgmask(mask, None, colour)
                    mask_bnds = np.hstack(2*[np.zeros_like(mask_bnds)]+[mask_bnds])
                    mask_bnds = morphology.dilation(mask_bnds, footprint=morphology.disk(radius=2))
                    cmap_bnds = generate_imgmask(mask_bnds, None, colour)
                    ax.imshow(cmap, alpha=0.25, zorder=z)
                    ax.imshow(cmap_bnds, alpha=0.75, zorder=z)

    # ax.imshow(od_cmap, alpha=0.5)
    ax.set_axis_off()
    fig.tight_layout(pad = 0)
    if save_images:
        fig.savefig(os.path.join(save_path, f"{fname}_superimposed.png"), bbox_inches="tight")
    if collate_segmentations:
        segmentation_directory = os.path.join(dirpath, "slo_segmentations")
        if not os.path.exists(segmentation_directory):
            os.mkdir(segmentation_directory)
        if save_images:
            shutil.copy(os.path.join(save_path, f"{fname}_superimposed.png"), 
                        os.path.join(segmentation_directory, f"{fname}.png"))
        else:
            fig.savefig(os.path.join(segmentation_directory, f"{fname}.png"), bbox_inches="tight")
    plt.close()



def _create_circular_mask(center, img_shape, radius):
    """
    Given a center, radius and image shape, draw a filled circle
    as a binary mask.
    """
    # Circular mask
    h, w = img_shape
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)
    mask = (dist_from_center <= radius).astype(int)
    
    return mask



def generate_zonal_masks(img_shape, od_radius, od_centre, location='Macular'):

    mask_rois = {}
    if location=='Macula':
        zones = ['whole']
    elif location=='Optic disc':
        zones = ['whole', 'B', 'C']

    for roi_type in zones:
        if roi_type == 'whole':
            mask = np.ones(img_shape)
        else:
            if roi_type == "B":
                od_circ = _create_circular_mask(img_shape=img_shape, 
                                            radius=2*od_radius, 
                                            center=od_centre)
                
                mask  = _create_circular_mask(img_shape=img_shape, 
                                            radius=3*od_radius, 
                                            center=od_centre)
            elif roi_type == "C":
                od_circ = _create_circular_mask(img_shape=img_shape, 
                                            radius=od_radius, 
                                            center=od_centre)
                
                mask = _create_circular_mask(img_shape=img_shape, 
                                                radius=5*od_radius, 
                                                center=od_centre)

            mask -= od_circ
        mask_rois[roi_type] = mask

    return mask_rois
