import os
import sys

SCRIPT_PATH = os.path.realpath(os.path.dirname(__file__))
MODULE_PATH = os.path.split(SCRIPT_PATH)[0]
PACKAGE_PATH = os.path.split(MODULE_PATH)[0]
sys.path.append(SCRIPT_PATH)
sys.path.append(MODULE_PATH)
sys.path.append(PACKAGE_PATH)

import shutil
import sys
import time
import argparse
import pandas as pd
import pprint
import numpy as np
from tqdm import tqdm
from pathlib import Path, PosixPath, WindowsPath

from octolyzer import analyse, collate_data, config_loader, utils
from octolyzer.segment.octseg import choroidalyzer_inference, deepgpet_inference
from octolyzer.segment.sloseg import slo_inference, avo_inference, fov_inference
from octolyzer.measure.bscan.thickness_maps import grid


# Abbreviated versions of different layers in the OCT
KEY_LAYER_DICT = {"ILM": "Inner Limiting Membrane",
                  "GCL": "Ganglion Cell Layer",
                  "RNFL": "Retinal Nerve Fiber Layer",
                  "IPL": "Inner Plexiform Layer",
                  "INL": "Inner Nuclear Layer",
                  "OPL": "Outer Plexiform Layer",
                  "ELM": "External Limiting Membrane", # Outer nuclear layer
                  "PR1": "Photoreceptor Layer 1",
                  "PR2": "Photoreceptor Layer 2",
                  "RPE": "Retinal Pigment Epithelium",
                  "BM": "Bruch's Membrane Complex", 
                  "CHORupper": "Choroid - Sclera boundary",
                  "CHORlower": "Bruch's Membrane - Choroid boundary"}

def run(args, config_path=None):
    """
    Analyze .vol files from a specified directory using the OCT and SLO analysis pipeline.

    This function processes OCT and SLO data from Heidelberg `.vol` files. It performs segmentation,
    measurement, and data collation based on the configuration provided in the `args` dictionary input.
    Results and logs are saved to the specified output directory.

    Parameters:
    ----------
    args : dict
        Configuration dictionary containing the following keys:
        - `analysis_directory` (str): Path to the directory containing `.vol` files.
        - `output_directory` (str): Path to the directory where results will be saved.
        - `robust_run` (int): 1 to skip files with unexpected errors; 0 to halt and allow debugging.
        - `save_individual_segmentations` (int): Save segmentation masks per individual (1 to save, 0 to skip).
        - `save_individual_images` (int): Save SLO/OCT images for each individual (1 to save, 0 to skip).
        - `preprocess_bscans` (int): Preprocess B-scans to enhance choroid visibility (1 to enable, 0 to disable).
        - `analyse_choroid` (int): Include choroid-related results (1 to include, 0 to exclude).
        - `analyse_slo` (int): Analyze the accompanying SLO image (1 to include, 0 to exclude).
        - `custom_maps` (list[str]): Retinal thickness maps such as `ILM_OPL`.
        - `analyse_all_maps` (int): Analyze all retinal layer thickness maps for Ppole data (1 to enable, 0 to disable).
        - `analyse_square_grid` (int): Analyze the square posterior pole grid (1 to enable, 0 to disable).
        - `choroid_measure_type` (str): Type of choroid measurement ("perpendicular" or "vertical").
        - `linescan_roi_distance` (int): Distance in microns for structure measurement around the fovea.
    config_path : str or os.PathLike, optional
        Configuration file copied to the output directory as `configuration_used.txt`.

    Outputs:
    -------
    - Results saved in the `output_directory`:
        - Individual segmentation masks and raw images, if specified.
        - Segmentation visualisations to demonstrate features measured.
        - Consolidated Excel file with metadata, SLO, and OCT measurements.
        - Configuration file (`configuration_used.txt`) copied for reproducibility.
        - Log files for individual and batch processing.

    Notes:
    -----
    - If `robust_run` is set to 1, unexpected errors for specific files will be logged, and analysis will continue with other files.
    - Choroid analysis can be toggled with `analyse_choroid`.
    - Composite SLO visualisations, if enabled, are saved in the `slo_segmentations` subdirectory.
    - Composite OCT visualisations are saved in the `oct_segmentations` subdirectory.

    Examples:
    --------
    >>> args = {
    ...     "analysis_directory": "/path/to/input",
    ...     "output_directory": "/path/to/output",
    ...     "robust_run": 1,
    ...     "save_individual_segmentations": 1,
    ...     "save_individual_images": 1,
    ...     "preprocess_bscans": 1,
    ...     "analyse_choroid": 1,
    ...     "analyse_slo": 1,
    ...     "custom_maps": [],
    ...     "analyse_all_maps": 0,
    ...     "analyse_square_grid": 0,
    ...     "choroid_measure_type": "perpendicular",
    ...     "linescan_roi_distance": 3000
    ... }
    >>> run(args)
    """

    # analysis directory
    analysis_directory = args["analysis_directory"]
    if not os.path.exists(analysis_directory):
        print("Cannot find directory images/ with files to analyse.")
        print("Please create directory and place images inside. Exiting analysis")
        sys.exit()

    # Detect .vol files from analysis_directory
    vol_paths = sorted(Path(analysis_directory).glob("*.vol"))
    N = len(vol_paths)
    if N > 0:
        print(f"Found {len(vol_paths)} to analyse.")
    else:
        print(f'Cannot find any supported files in {analysis_directory}. Please check directory. Exiting analysis')
        return

    # output directory
    save_directory = args["output_directory"]
    if not os.path.exists(save_directory):
        print("Cannot find directory output/ to store results.")
        print('Creating folder.')
        os.makedirs(save_directory, exist_ok=True)

    # Copy the config file over into the folder to take a screenshot of the configuration used during this batch processing
    if config_path is None:
        config_path = os.path.join(MODULE_PATH, 'config.txt')
    shutil.copy(config_path, os.path.join(save_directory, 'configuration_used.txt'))

    # This is a particularly helpful parameter when running large batches, and ignored
    # any unexpected errors from a particular file. Setting it as 0 will throw up errors
    # for debugging
    robust_run = args["robust_run"]
    analyse_slo_flag = args['analyse_slo'] 
    collate_segs = True
    analyse_choroid = bool(args['analyse_choroid'])

    # create segmentations directory if specified
    if analyse_slo_flag:
        slo_segmentation_directory = os.path.join(save_directory, "slo_segmentations")
        if collate_segs:
            os.makedirs(slo_segmentation_directory, exist_ok=True)
    oct_segmentation_directory = os.path.join(save_directory, "oct_segmentations")
    if collate_segs:
        os.makedirs(oct_segmentation_directory, exist_ok=True)

    # construct param dicts
    param_keys = ["save_individual_segmentations",
                "save_individual_images",
                "preprocess_bscans",
                "custom_maps",
                "analyse_all_maps",
                "analyse_choroid",
                'analyse_slo',
                "analyse_square_grid",
                "choroid_measure_type",
                "linescan_roi_distance"]
    param_dict = {key:args[key] for key in param_keys}

    # Instantiate SLO binary/AVOD/Fovea segmentation model
    print(f"\nLoading SLO and OCT models.")
    slosegmenter = slo_inference.SLOSegmenter()
    avosegmenter = avo_inference.AVOSegmenter()
    fovsegmenter = fov_inference.FOVSegmenter()
    choroidalyzer = choroidalyzer_inference.Choroidalyzer()
    deepgpet = deepgpet_inference.DeepGPET()

    if analyse_choroid:
        print(f"\nRunning choroid and retinal analysis.")
    else:
        print(f"\nRunning OCTolyzer...")

    # Loop through .vol files, segment, measure and save out in analyse()
    st = time.time()
    oct_slo_result_dict = {}
    for path in tqdm(vol_paths, desc='Analysing...', leave=False):
        
        # Initialise results dictionary for .vol file and create paths to results
        fname_type = os.path.split(path)[-1]
        oct_slo_result_dict[fname_type] = {}
        fname = fname_type.split(".")[0]
        fname_path = os.path.join(save_directory, fname)
        output_fname = os.path.join(fname_path, f"{fname}_output.xlsx")
        slo_manual_annotations = list((set(Path(fname_path).glob(f"{fname}*slo*.nii.gz"))).difference(set(Path(fname_path).glob(f"{fname}*slo*_used.nii.gz"))))
        oct_manual_annotations = list((set(Path(fname_path).glob(f"{fname}*oct*.nii.gz"))).difference(set(Path(fname_path).glob(f"{fname}*oct*_used.nii.gz"))))
        param_dict['manual_annotations'] = oct_manual_annotations + slo_manual_annotations
        if os.path.exists(output_fname) and len(param_dict['manual_annotations']) == 0:
            print(f"\nPreviously analysed {fname}.")
            ind_df, slo_dfs, oct_dfs, log = collate_data.load_files(fname_path, logging_list=[], analyse_square=param_dict['analyse_square_grid'])
            oct_slo_result_dict[fname_type]['metadata'] = ind_df
            if analyse_slo_flag:
                oct_slo_result_dict[fname_type]['slo'] = slo_dfs
            else:
                oct_slo_result_dict[fname_type]['slo'] = [pd.DataFrame()]
            oct_slo_result_dict[fname_type]['oct'] = oct_dfs
            oct_slo_result_dict[fname_type]['log'] = log

        # Skip if .vol file is an OCT-A scan
        elif "_ANGIO" in fname:
            print(f"{fname} is an OCT-A scan. Skipping.\n\n")

        # If unprocessed and valid, apply pipeline
        else:

            # For batch processing
            if robust_run:

                # Catch any exceptions 
                try:

                    # Analyse file
                    output = analyse.analyse(path, 
                                    save_directory, 
                                    choroidalyzer, 
                                    slosegmenter, 
                                    avosegmenter,
                                    fovsegmenter,
                                    deepgpet,
                                    param_dict)
                    slo_analysis_output, oct_analysis_output = output

                    # Store results for collating
                    oct_slo_result_dict[fname_type]['metadata'] = oct_analysis_output[0]
                    if slo_analysis_output is not None and analyse_slo_flag:
                        oct_slo_result_dict[fname_type]['slo'] = slo_analysis_output[1]
                    else:
                        oct_slo_result_dict[fname_type]['slo'] = [pd.DataFrame()]
                    oct_slo_result_dict[fname_type]['oct'] = oct_analysis_output[3]
                    oct_slo_result_dict[fname_type]['log'] = oct_analysis_output[-1]

                # Unexpected error
                except Exception as e:
                    
                    # print and log error
                    user_fail = f"\nFailed to analyse {fname}."
                    log = utils.print_error(e)
                    logging_list = [user_fail] + log
                    skip = "Skipping and moving to next file.\nCheck data input and/or set robust_run to 0 to debug code.\n"
                    print(skip)

                    # Try at least save out metadata from loading volfile for failed
                    # file - making sure to mark in FAILED column
                    try:
                        _, metadata, _, _, _ = utils.load_volfile(path, verbose=False)
                        metadata['FAILED'] = True
                        if metadata["bscan_type"] == 'Peripapillary':
                            del metadata['stxy_coord']

                    # Catch any exceptions with failing to even load image and metadata from 
                    # volfile
                    except:
                        metadata = {'Filename':os.path.split(path)[1]}
                        fail_load = "Failed to even load path, check utils.load_volfile"
                        print(fail_load)
                        log.append(fail_load)

                    # Store results for collating
                    oct_slo_result_dict[fname_type]['metadata'] = metadata
                    oct_slo_result_dict[fname_type]['oct'] = logging_list[0]
                    oct_slo_result_dict[fname_type]['log'] = logging_list
                    
            else:
                
                # Analyse file
                output = analyse.analyse(path, 
                                save_directory, 
                                choroidalyzer, 
                                slosegmenter, 
                                avosegmenter,
                                fovsegmenter,
                                deepgpet,
                                param_dict)
                slo_analysis_output, oct_analysis_output = output

                # Store results for collating
                oct_slo_result_dict[fname_type]['metadata'] = oct_analysis_output[0]
                if slo_analysis_output is not None and analyse_slo_flag:
                    oct_slo_result_dict[fname_type]['slo'] = slo_analysis_output[1]
                else:
                    oct_slo_result_dict[fname_type]['slo'] = [pd.DataFrame()]
                oct_slo_result_dict[fname_type]['oct'] = oct_analysis_output[3]
                oct_slo_result_dict[fname_type]['log'] = oct_analysis_output[-1]

    # Collect all measurements together
    collate_data.collate_results(oct_slo_result_dict, save_directory, param_dict['analyse_choroid'], param_dict['analyse_slo'])

    # Complete !
    elapsed = time.time() - st
    print(f"Completed analysis in {elapsed:.03f} seconds.")




def main(argv=None):
    parser = argparse.ArgumentParser(description="Analyse Heidelberg .vol files with OCTolyzer.")
    parser.add_argument(
        "-c",
        "--config",
        default=os.path.join(MODULE_PATH, "config.txt"),
        help="Path to the OCTolyzer configuration file.",
    )
    parsed_args = parser.parse_args(argv)
    config_path = Path(parsed_args.config).expanduser()

    print("Checking configuration file for valid inputs...")
    try:
        config = config_loader.load_config(config_path)
    except config_loader.ConfigError as error:
        parser.error(str(error))

    for warning in config.warnings:
        print(f"WARNING: {warning}")

    run(config.to_args(), config_path=config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
