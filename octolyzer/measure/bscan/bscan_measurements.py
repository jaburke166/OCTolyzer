import numpy as np
from octolyzer import utils
from octolyzer.measure.bscan import utils as bscan_utils

def compute_measurement(reg_mask, 
                       vess_mask=None, 
                       fovea: [tuple, np.ndarray] = None,
                       scale: tuple[int, int, int] = (11.49,3.87),
                       macula_rum: int = 3000,
                       N_measures: int = 'all',
                       N_avgs = 0,
                       offset=15,
                       measure_type: str = "perpendicular",
                       img_shape: tuple = (768,768),
                       plottable: bool = False,
                       force_measurement: bool = False,
                       verbose: [int, bool]=0,
                       logging_list: list = []):
    """
    Computes choroidal measurements (thickness, area, and vascular index) based on the provided 
    segmentation masks and parameters.

    Parameters
    ----------
    reg_mask : np.ndarray or tuple
        Binary segmentation mask of the layer. Can be a 2D numpy array or a tuple containing 
        traces of the boundaries.
        
    vess_mask : np.ndarray, optional
        Binary segmentation mask for identifying choroidal vessels. Required for vascular index (CVI).
        
    fovea : tuple or np.ndarray, optional
        Coordinates of the fovea for defining a fovea-centered region of interest (ROI). If not provided,
        it defaults to the center of the mask.
        
    scale : tuple[int, int], default=(11.49, 3.87)
        Scaling factors in microns per pixel in the x and y directions, respectively. Default values are 
        for Heidelberg OCT scaling for a B-scan with lateral pixel resolution of 768.
        
    macula_rum : int, default=3000
        Radius of the macular region of interest in microns, acting as the distance either side of the fovea
        to define and measure.
        
    N_measures : int or str, default='all'
        Number of thickness measurements across the choroid. If 'all', measures at all points in the ROI are
        taken.
        
    N_avgs : int, default=0
        Number of adjacent thickness values to average at each measurement location for robustness to local
        gradient changes.
        
    offset : int, default=15
        Distance either side of reference point (which defines the tangent line) which determines the
        influence of local gradient changes when drawing perpendicular lines for thickness measurement.
        
    measure_type : str, default="perpendicular"
        Specifies the method of thickness measurement:
        - "perpendicular": Measures perpendicular to the upper boundary.
        - "vertical": Measures column-wise (per A-scan).
        
    img_shape : tuple, default=(768, 768)
        Shape of the original image if the mask needs to be rebuilt, e.g. the input is a set of traces rather
        than a binary mask.
        
    plottable : bool, default=False
        If True, returns boundary points and masks that can be visualised.
        
    force_measurement : bool, default=False
        If True, forces measurements even if the segmentation is shorter than the defined ROI but within a
        threshold of `offset`.
        
    verbose : int or bool, default=0
        Verbosity level for logging and printing messages.
        
    logging_list : list, default=[]
        List to store log messages for debugging or tracking.

    Returns
    -------
    measurements : list
        Contains computed choroidal metrics:
        - `ct` : Thickness at each measurement point or as averages (depending on `N_measures`).
        - `ca` : Area in the specified region of interest.
        - `cvi` : Choroidal vascular index (if `vess_mask` is provided).

    plotinfo : tuple, optional
        If `plottable` is True, includes data for visualization:
        - Boundary points for thickness measurements.
        - Mask of the region of interest.
        - Vessel mask if `vess_mask` is provided.
        
    logging_list : list
        Updated list of log messages.

    Notes
    -----
    - If the segmentation mask is shorter than the specified macular radius, the function may return -1s.
    - Enforces robustness by averaging across adjacent points when `N_avgs > 0`.
    - For accurate results, ensure scaling factors (`scale`) match the imaging system's specifications.
    """
    measurements = []

    # Constants
    micron_pixel_x, micron_pixel_y = scale
    pixels_per_micron  = 1 / micron_pixel_x 
    if N_measures == 'all':
        measure_all = True
        N_measures = int(2*macula_rum*pixels_per_micron)
        N_avgs = 0
    else:
        measure_all = False
        N_avgs = N_avgs+1 if N_avgs%2 != 0 else N_avgs
    N_measures = max(N_measures + 1,3) if N_measures % 2 == 0 else max(N_measures,3)  

    # Organise region mask
    if isinstance(reg_mask, np.ndarray):
        if reg_mask.ndim == 2:
            rmask = reg_mask.copy()
            traces = utils.get_trace(reg_mask, None, align=True)
        else:
            rmask = utils.rebuild_mask(reg_mask, img_shape=img_shape)
            traces = reg_mask.copy()
    elif isinstance(reg_mask, tuple):
        rmask = utils.rebuild_mask(reg_mask, img_shape=img_shape)
        traces = utils.interp_trace(reg_mask)

    # If fovea is known - if not, take as halfway along region
    # segmentation
    if fovea is not None:
        if isinstance(fovea, tuple):
            fovea = np.array(fovea)
        ref_pt = fovea
    else:
        x_N = int(0.5 * (traces[0].shape[0] + traces[1].shape[0]))
        # x_st = int(0.5*(traces[0,0] + traces[1,0]))
        x_st = int(0.5 * (traces[0][0, 0] + traces[1][0, 0]))
        x_half = x_N // 2 + x_st
        y_half = traces[0][:,1].mean()
        ref_pt = np.array([x_half, y_half])

    # Work out reference point along upper boundary closest to fovea
    # and re-adjust reference point on upper boundary to align with indexing
    top_lyr, bot_lyr = traces
    toplyr_stx, botlyr_stx = int(top_lyr[0, 0]), int(bot_lyr[0, 0])
    toplyr_refpt, offset_pts = bscan_utils.nearest_coord(top_lyr, 
                                                         ref_pt, 
                                                         offset, 
                                                         columnwise=False)
    ref_idx = toplyr_refpt[0] - toplyr_stx

    # Set up list of micron distances either side of reference point, dictated by N_measures
    delta_micron = 2 * macula_rum // (N_measures - 1)
    delta_i = [i for i in range((N_measures - 1) // 2 + 1)]
    micron_measures = np.array([i * delta_micron for i in delta_i])

    # Locate coordinates along the upper boundary at equal spaces away from foveal pit until macula_rum
    curve_indexes = [bscan_utils.curve_location(top_lyr, distance=d, ref_idx=ref_idx, scale=scale) for d in micron_measures]

    # Catch if this completely fails, then resort to segmentation failure (either due to fovea/layer detection)
    if (None, np.nan) in curve_indexes or (np.nan, None) in curve_indexes or (None, None) in  curve_indexes:

        msg = f"""        Segmentation failure for {macula_rum} micron ROI. 
        Please check fovea detection or segmentation. 
        Perhaps reducing the region of interest might prevent this from happening.
        Returning -1s."""
        logging_list.append(msg)
        if verbose:
            print(msg)

        # Empty outputs depending on input
        plotinfo = None
        avg_thick = [np.array(N_measures*[-1], dtype=np.int64), -1][measure_all]
        measurements = [(-1, avg_thick, -1, -1, -1), (-1, avg_thick, -1)][vess_mask is None]
        if plottable:
            return measurements, plotinfo, logging_list
        else:
            return measurements, logging_list

    # If we can, make sure that the curve_indexes at each end of the choroid are within offset+N_avgs//2 of the 
    # last index of each end of  the trace. Only relevant if measuring perpendicularly.
    if measure_type == "perpendicular":
        toplyr_endpts = np.array([top_lyr[idx, 0] for idx in curve_indexes[-1]])
        x_endpts = top_lyr[[0,-1],0]
        new_curve_indexes = list(curve_indexes[-1])
        st_diff = (toplyr_endpts[0]-(offset+N_avgs//2)) - x_endpts[0]
        en_diff = (toplyr_endpts[1]+(offset+N_avgs//2)) - x_endpts[1]
        st_flag = 0
        en_flag = 0
        if st_diff <= 0:
            st_flag = 1
            new_curve_indexes[0] = curve_indexes[-1][0] - st_diff
        if en_diff >= 0:
            en_flag = 1
            new_curve_indexes[1] = curve_indexes[-1][1] - en_diff
        curve_indexes[-1] = tuple(new_curve_indexes)
        
        # Logging to user about consequence of forcing measurement if segmentation isn#t long enough
        if st_flag + en_flag > 0 and not force_measurement:
            msg = f"""        Segmentation not long enough for {macula_rum} microns.
        Extend segmentation or reduce region of interest to prevent this from happening.
        Returning -1s."""
            logging_list.append(msg)
            if verbose:
                print(msg)
            
            # Empty outputs depending on input
            plotinfo = None
            avg_thick = [np.array(N_measures*[-1], dtype=np.int64), -1][measure_all]
            measurements = [(-1, avg_thick, -1, -1, -1), (-1, avg_thick, -1)][vess_mask is None]
            if plottable:
                return measurements, plotinfo, logging_list
            else:
                return measurements, logging_list

        elif force_measurement and st_flag==1:
            msg = f"""        Segmentation not segmented long enough for {macula_rum} microns.
        Reducing left-endpoint reference point by {-st_diff} pixels.
        Extend segmentation or reduce region of interest to prevent under-measurement."""
            logging_list.append(msg)
            if verbose:
                print(msg)
        elif force_measurement and en_flag==1:
            msg = f"""        Segmentation not long enough for {macula_rum} microns.
        Reducing right-endpoint reference point by {en_diff} pixels.
        Extend segmentation or reduce region of interest to prevent under-measurement."""
            logging_list.append(msg)
            if verbose:
                print(msg)

    # Collect reference points along upper boundary - N_avgs allows us to make more robust thickness measurements by taking average value of advacent positions
    toplyr_pts = np.array([top_lyr[[idx + np.arange(-N_avgs//2, N_avgs//2+1)]] for loc in curve_indexes for idx in loc]).reshape(-1,2)[1:]
    toplyr_pts = toplyr_pts[toplyr_pts[:,0].argsort()]

    # Collect reference points along lower boundary, given upper boundary reference points - taken A-scan wise for vertical measure_type and 
    # locally perpendicular to upper boundary reference point if perpendicular measure_type
    if measure_type == "perpendicular":
        botlyr_pts, toplyr_pts, perps = bscan_utils.detect_orthogonal_coords(toplyr_pts, 
                                                                                      traces, 
                                                                                      offset, 
                                                                                      preserve_shape=True)
    elif measure_type == "vertical":
        st_Bx = bot_lyr[0,0]
        botlyr_pts = bot_lyr[toplyr_pts[:,0]-st_Bx]

    # Collect reference points along boundaries to make thickness measurement
    botlyr_pts = botlyr_pts.astype(np.float64).reshape(N_measures, N_avgs+1, 2)
    toplyr_pts = toplyr_pts.astype(np.float64).reshape(N_measures, N_avgs+1, 2)
    boundary_pts = np.concatenate([toplyr_pts.reshape(*botlyr_pts.shape), botlyr_pts], axis=-1).reshape(*botlyr_pts.shape, 2)
    
    # Compute choroid thickness at each reference point.
    delta_xy = np.abs(np.diff(boundary_pts, axis=boundary_pts.ndim-2)) * np.array([micron_pixel_x, micron_pixel_y])

    # If measuring at every point across ROI, return average and subfoveal (averaging across 5 adjacent measurements)
    if measure_all:
        choroid_thickness = np.sqrt(np.square(delta_xy).sum(axis=-1))[:,0]
        subf_thick = int(np.nanmean(choroid_thickness[N_measures//2-3:N_measures//2+2]))
        avg_thick = int(np.nanmean(choroid_thickness))
        measurements.append(subf_thick)
        measurements.append(avg_thick)

    # Otherwise, return all thickness values measured at all N_measures reference points
    else:
        choroid_thickness = np.rint(np.nanmean(np.sqrt(np.square(delta_xy).sum(axis=-1)), axis=1)).astype(int)[:,0]
        measurements.append(choroid_thickness)
    
    # Compute choroid area                               
    area_bnds_arr = np.swapaxes(boundary_pts[[0,-1], N_avgs//2], 0, 1).reshape(-1,2)
    if np.any(np.isnan(area_bnds_arr)):
        msg = f"""        Segmentation not long enough for {macula_rum} microns.
        Extend segmentation or reduce region of interest to prevent under-measurement.
        Returning -1s."""
        logging_list.append(msg)
        if verbose:
            print(msg)

        # Empty outputs depending on input
        plotinfo = None
        avg_thick = [np.array(N_measures*[-1], dtype=np.int64), -1][measure_all]
        measurements = [(-1, avg_thick, -1, -1, -1), (-1, avg_thick, -1)][vess_mask is None]
        if plottable:
            return measurements, plotinfo, logging_list
        else:
            return measurements, logging_list

    # Compute choroidal area
    choroid_area, plot_output = compute_area_enclosed(traces, area_bnds_arr.astype(int), scale=scale, plot=True)
    chor_pixels, (x_range, y_range), (left_x, right_x) = plot_output
    measurements.append(choroid_area)

    # CVI
    if vess_mask is not None:

        # Compute CVI --- choroidal vessel area is whatever pixels in keep_pixel 
        # which are flagged as 1 in vess_mask
        vessel_area = 0
        xchor = np.unique(chor_pixels[:,0]).astype(np.int64)
        chor_pixels = chor_pixels.astype(np.int64)
        for x in xchor:
            col = chor_pixels[chor_pixels[:,0]==x]
            bmap_col = vess_mask[col[:,1]-1, x]
            vessel_area += bmap_col.sum()
    
        # Total pixel-choroid area is the number of pixels contained with the ROI
        # CVI is just # vessel pixels divided by # choroid area pixels
        total_chor_area = chor_pixels.shape[0]
        choroid_cvi = np.round(vessel_area/total_chor_area, 5)
        measurements.append(choroid_cvi)

        # Other metrics like EVI, vessel area and intersitial area (in mm2)
        micron_area = micron_pixel_x * micron_pixel_y
        choroid_vessel_area = np.round(1e-6 * micron_area * vessel_area, 6)
        measurements.append(choroid_vessel_area)

    if plottable:
        ca_mask = np.zeros_like(rmask)
        chor_pixels = chor_pixels.astype(int)
        ca_mask[chor_pixels[:,1], chor_pixels[:,0]] = 1
        if vess_mask is not None:
            return measurements, (boundary_pts, ca_mask, vess_mask), logging_list
        else:
            return measurements, (boundary_pts, ca_mask), logging_list
    else:
        return measurements, logging_list



def compute_area_enclosed(traces, 
                          area_bnds_arr, 
                          scale=(11.49,3.87), 
                          plot=False):
    """
    Calculate an enclosed area in square millimeters, given boundary traces 
    and the smallest irregular quadrilateral `area_bnds_arr` containing the region of interest.

    Parameters:
    -----------
    traces : tuple of numpy.ndarray
        A tuple containing two arrays:
        - `top_lyr`: Upper boundary trace (array of x, y coordinates).
        - `bot_lyr`: Lower boundary trace (array of x, y coordinates).

    area_bnds_arr : numpy.ndarray
        An array of shape (4, 2) defining the four vertex pixel coordinates of the 
        smallest irregular quadrilateral enclosing the region of interest.

    scale : tuple of float, default=(11.49, 3.87)
        A tuple specifying the scaling factors for converting pixels to microns:
        - `micron_pixel_x`: Microns per pixel in the x-direction.
        - `micron_pixel_y`: Microns per pixel in the y-direction.

    plot : bool, default=False
        If `True`, additional outputs for visualising the area calculation are returned.

    Returns:
    --------
    float or tuple
        - If `plot=False`: Returns `lyr_mm_area`, the calculated area in square millimeters.
        - If `plot=True`: Returns a tuple containing:
            - `lyr_mm_area`: The calculated area in square millimeters.
            - `plot_outputs`: A list of visualization data:
                - `keep_pixel`: Array of pixel coordinates contributing to the area.
                - `(x_range, y_range)`: The rectangular bounds of the area.
                - `(left_x, right_x)`: Left and right bounds derived from the irregular quadrilateral.

    Notes:
    ------
    - The function computes the smallest rectangular region that fully overlaps the irregular quadrilateral 
      enclosing the area and reduces it based on specified conditions.
    - The area is computed by summing the pixels that satisfy the boundary constraints 
      (upper and lower traces, and left and right bounds).
    - The pixel area is converted to square millimeters using the provided scale.
    """
    # Extract reference points scale and traces
    top_lyr, bot_lyr = traces
    toplyr_stx, botlyr_stx = top_lyr[0, 0], bot_lyr[0, 0]
    toplyr_ref, botlyr_ref = area_bnds_arr[:2], area_bnds_arr[2:]

    # Compute microns-per-pixel and how much micron area a single 1x1 pixel represents.
    micron_pixel_x, micron_pixel_y = scale
    micron_area = micron_pixel_x * micron_pixel_y
    
    # Work out range of x- and y-coordinates bound by the area, building the smallest rectangular
    # region which overlaps the area of interest fully
    x_range = np.arange(area_bnds_arr[:, 0].min(), area_bnds_arr[:, 0].max() + 1)
    y_range = np.arange(min(top_lyr[x_range[0] - toplyr_stx: x_range[-1] - toplyr_stx + 1, 1].min(),
                            area_bnds_arr[:, 1].min()),
                        max(bot_lyr[x_range[0] - botlyr_stx: x_range[-1] - botlyr_stx + 1, 1].max(),
                            area_bnds_arr[:, 1].max()) + 1)
    N_y = y_range.shape[0]

    # This defines the left-most perpendicular line and right-most perpendicular line
    # for comparing with coordinates from rectangular region
    left_m, left_c = bscan_utils.construct_line(toplyr_ref[0], botlyr_ref[0])
    right_m, right_c = bscan_utils.construct_line(toplyr_ref[1], botlyr_ref[1])
    if left_m != np.inf:
        left_x = ((y_range - left_c) / left_m).astype(np.int64)
    else:
        left_x = np.array(N_y * [toplyr_ref[0][0]])
    if right_m != np.inf:
        right_x = ((y_range - right_c) / right_m).astype(np.int64)
    else:
        right_x = np.array(N_y * [toplyr_ref[1][0]])
    # The rectangular region above needs reduced to only containing coordinates which lie
    # above the Chor-Sclera boundary, below the RPE-Choroid boundary, to the right of the
    # left-most perpendicular line and to the left of the right-most perpendicular line.
    keep_pixel = []

    # We vectorise check by looping across x_range and figuring out if each coordinate
    # in the column satisfies the four checks described above
    for x in x_range:
        # Extract column
        col = np.concatenate([x * np.ones(N_y)[:, np.newaxis], y_range[:, np.newaxis]], axis=1)

        # Define upper and lower bounds at this x-position
        top, bot = top_lyr[x - toplyr_stx], bot_lyr[x - botlyr_stx]

        # Check all 4 conditions and make sure they are all satisfied
        cond_t = col[:, 1] >= top[1]
        cond_b = col[:, 1] <= bot[1]
        cond_l = x >= left_x
        cond_r = x < right_x
        col_keep = col[cond_t & cond_b & cond_l & cond_r]
        keep_pixel.append(col_keep)

    # All pixels bound within the area of interest
    keep_pixel = np.concatenate(keep_pixel)

    # Calculate area (in square mm)
    lyr_pixel_area = keep_pixel.shape[0]
    lyr_mm_area = np.round(1e-6 * micron_area * lyr_pixel_area, 6)

    # If plotting, reutrn pixels used to compute  area
    if plot:
        plot_outputs = [keep_pixel, (x_range, y_range), (left_x, right_x)]
        outputs = [lyr_mm_area, plot_outputs]
    else:
        outputs = lyr_mm_area

    return outputs
