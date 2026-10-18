"""Build review files for a folder that OCTolyzer has already analysed.

    python -m octolyzer.review <output_directory> [--input <analysis_directory>] [--force]

No models are run: segmentations and measurements are read from the saved
results, and only the pixel data is re-read from the original .vol files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.append(str(PACKAGE_ROOT))

from octolyzer.review import builder  # noqa: E402

CONFIG_NAME = "configuration_used.txt"


def read_run_configuration(output_directory: Path) -> dict[str, str]:
    """Read the ``key: value`` pairs OCTolyzer copied into the output folder."""
    values: dict[str, str] = {}
    config_path = output_directory / CONFIG_NAME
    if not config_path.is_file():
        return values
    for line in config_path.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.strip().startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        values[key.strip()] = value.strip()
    return values


def build_all(output_directory, analysis_directory=None, force=False, verbose=True) -> tuple[int, int, int]:
    """Build missing review files; returns ``(built, skipped, failed)`` counts."""
    output_directory = Path(output_directory).expanduser().resolve()
    config = read_run_configuration(output_directory)
    if analysis_directory is None:
        analysis_directory = config.get("analysis_directory") or ""
    analysis_directory = Path(analysis_directory).expanduser() if analysis_directory else None
    if analysis_directory is None or not analysis_directory.is_dir():
        raise FileNotFoundError(
            "Cannot find the folder containing the .vol files. Pass it with --input <analysis_directory>.")
    preprocess = config.get("preprocess_bscans", "1") != "0" and config.get("analyse_choroid", "1") != "0"

    built = skipped = failed = 0
    for vol_path in sorted(analysis_directory.glob("*.vol")):
        fname = vol_path.stem
        if not (output_directory / fname / f"{fname}_output.xlsx").is_file():
            continue
        if not force and builder.review_path(output_directory, fname).is_file():
            skipped += 1
            continue
        try:
            builder.build_review(vol_path, output_directory, preprocess=preprocess)
            built += 1
            if verbose:
                print(f"Built review file for {fname}.")
        except Exception as error:  # keep going: one bad file must not stop the batch
            failed += 1
            if verbose:
                print(f"Could not build review file for {fname}: {type(error).__name__}: {error}")
    hub = builder.write_hub(output_directory)
    if verbose:
        print(f"{built} built, {skipped} already present, {failed} failed.\nOpen {hub} in a browser to review.")
    return built, skipped, failed


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m octolyzer.review",
        description="Build interactive review files for an analysed OCTolyzer output folder.")
    parser.add_argument("output_directory", type=Path, help="OCTolyzer output folder (contains analysis_output.xlsx).")
    parser.add_argument("--input", type=Path, default=None, dest="analysis_directory",
                        help="Folder with the original .vol files. Defaults to the one recorded in configuration_used.txt.")
    parser.add_argument("--force", action="store_true", help="Rebuild review files that already exist.")
    arguments = parser.parse_args(argv)
    if not arguments.output_directory.is_dir():
        parser.error(f"Output folder does not exist: {arguments.output_directory}")
    try:
        build_all(arguments.output_directory, arguments.analysis_directory, force=arguments.force)
    except FileNotFoundError as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
