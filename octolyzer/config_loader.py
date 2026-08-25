"""Loading, validating, and writing OCTolyzer configuration files."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

LAYER_ORDER = (
    "ILM",
    "GCL",
    "RNFL",
    "IPL",
    "INL",
    "OPL",
    "ELM",
    "PR1",
    "PR2",
    "RPE",
    "BM",
    "CHORupper",
    "CHORlower",
)

BINARY_KEYS = (
    "robust_run",
    "save_individual_segmentations",
    "save_individual_images",
    "preprocess_bscans",
    "analyse_choroid",
    "analyse_slo",
    "analyse_all_maps",
    "analyse_square_grid",
)
DIRECTORY_KEYS = ("analysis_directory", "output_directory")
CONFIG_KEYS = DIRECTORY_KEYS + BINARY_KEYS + (
    "custom_maps",
    "choroid_measure_type",
    "linescan_roi_distance",
)


@dataclass(frozen=True)
class ConfigField:
    """Presentation metadata used by the GUI and configuration editor."""

    key: str
    label: str
    description: str
    kind: str


CONFIG_FIELDS = (
    ConfigField("analysis_directory", "Input folder", "Folder containing Heidelberg .vol files.", "directory"),
    ConfigField("output_directory", "Output folder", "Folder where analysis results will be saved.", "directory"),
    ConfigField("robust_run", "Continue if a file fails", "Skip a failed file and continue with the rest of the batch.", "boolean"),
    ConfigField("save_individual_segmentations", "Save segmentation images", "Save segmentation masks and overview images for each file.", "boolean"),
    ConfigField("save_individual_images", "Save individual images", "Save representative OCT and SLO images for each file.", "boolean"),
    ConfigField("preprocess_bscans", "Preprocess B-scans", "Enhance retinal and choroidal structures before analysis.", "boolean"),
    ConfigField("analyse_choroid", "Analyse choroid", "Include choroid measurements in the results.", "boolean"),
    ConfigField("analyse_slo", "Analyse SLO", "Analyse the accompanying scanning laser ophthalmoscopy image.", "boolean"),
    ConfigField("custom_maps", "Custom retinal maps", "Optional retinal layer pairs for posterior pole thickness maps.", "maps"),
    ConfigField("analyse_all_maps", "Analyse all retinal maps", "Compute maps for all available retinal layer pairs.", "boolean"),
    ConfigField("analyse_square_grid", "Analyse square grid", "Measure the 8 by 8 posterior pole grid.", "boolean"),
    ConfigField("choroid_measure_type", "Choroid measurement", "Measure choroid vertically or perpendicular to its upper boundary.", "choice"),
    ConfigField("linescan_roi_distance", "Line-scan ROI distance", "Distance in microns on either side of the fovea for line scans.", "integer"),
)


class ConfigError(ValueError):
    """Configuration errors suitable for display in a CLI or GUI."""

    def __init__(self, errors: list[str]):
        self.errors = tuple(errors)
        super().__init__("\n".join(self.errors))


@dataclass(frozen=True)
class Config:
    analysis_directory: str
    output_directory: str
    robust_run: int
    save_individual_segmentations: int
    save_individual_images: int
    preprocess_bscans: int
    analyse_choroid: int
    analyse_slo: int
    custom_maps: tuple[str, ...]
    analyse_all_maps: int
    analyse_square_grid: int
    choroid_measure_type: str
    linescan_roi_distance: int
    warnings: tuple[str, ...] = ()

    def to_args(self) -> dict[str, object]:
        """Return the legacy dictionary consumed by ``main.run``."""
        return {
            "analysis_directory": self.analysis_directory,
            "output_directory": self.output_directory,
            "robust_run": self.robust_run,
            "save_individual_segmentations": self.save_individual_segmentations,
            "save_individual_images": self.save_individual_images,
            "preprocess_bscans": self.preprocess_bscans,
            "analyse_choroid": self.analyse_choroid,
            "analyse_slo": self.analyse_slo,
            "custom_maps": list(self.custom_maps),
            "analyse_all_maps": self.analyse_all_maps,
            "analyse_square_grid": self.analyse_square_grid,
            "choroid_measure_type": self.choroid_measure_type,
            "linescan_roi_distance": self.linescan_roi_distance,
        }

    def as_values(self) -> dict[str, str]:
        """Return values in the text representation used by ``config.txt``."""
        values = {
            "analysis_directory": self.analysis_directory,
            "output_directory": self.output_directory,
            "robust_run": str(self.robust_run),
            "save_individual_segmentations": str(self.save_individual_segmentations),
            "save_individual_images": str(self.save_individual_images),
            "preprocess_bscans": str(self.preprocess_bscans),
            "analyse_choroid": str(self.analyse_choroid),
            "analyse_slo": str(self.analyse_slo),
            "custom_maps": format_custom_maps(self.custom_maps),
            "analyse_all_maps": str(self.analyse_all_maps),
            "analyse_square_grid": str(self.analyse_square_grid),
            "choroid_measure_type": self.choroid_measure_type,
            "linescan_roi_distance": str(self.linescan_roi_distance),
        }
        return values


def parse_config(path: str | os.PathLike[str]) -> dict[str, str]:
    """Parse ``key: value`` entries while retaining colons in values."""
    config_path = Path(path)
    values: dict[str, str] = {}
    errors: list[str] = []

    try:
        lines = config_path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ConfigError([f"Unable to read configuration file '{config_path}': {error}"]) from error

    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key not in CONFIG_KEYS:
            errors.append(f"Line {line_number}: unknown configuration key '{key}'.")
        elif key in values:
            errors.append(f"Line {line_number}: duplicate configuration key '{key}'.")
        else:
            values[key] = value

    if errors:
        raise ConfigError(errors)
    return values


def load_config(
    path: str | os.PathLike[str],
    *,
    check_analysis_directory: bool = True,
) -> Config:
    """Load and validate a configuration file."""
    config_path = Path(path).expanduser().resolve()
    values = parse_config(config_path)
    for key in DIRECTORY_KEYS:
        directory = Path(values[key]).expanduser()
        if not directory.is_absolute():
            values[key] = str((config_path.parent / directory).resolve())
    return validate_config(values, check_analysis_directory=check_analysis_directory)


def validate_config(
    values: Mapping[str, object] | Config,
    *,
    check_analysis_directory: bool = True,
    require_directories: bool = True,
) -> Config:
    """Validate raw configuration values and return a typed ``Config``.

    ``require_directories`` can be set to ``False`` to allow ``analysis_directory``
    and ``output_directory`` to be blank, e.g. when building the initial, unset
    state of the configuration editor rather than validating a run-ready config.
    """
    if isinstance(values, Config):
        raw_values = values.as_values()
    else:
        raw_values = {str(key): value for key, value in values.items()}

    errors: list[str] = []
    unknown_keys = sorted(set(raw_values) - set(CONFIG_KEYS))
    errors.extend(f"Unknown configuration key '{key}'." for key in unknown_keys)
    missing_keys = [key for key in CONFIG_KEYS if key not in raw_values]
    errors.extend(f"Missing configuration value for '{key}'." for key in missing_keys)
    if errors:
        raise ConfigError(errors)

    directory_values: dict[str, str] = {}
    for key in DIRECTORY_KEYS:
        value = str(raw_values[key]).strip()
        if not value and require_directories:
            errors.append(f"'{key}' cannot be empty.")
        directory_values[key] = value

    if check_analysis_directory and directory_values.get("analysis_directory"):
        if not Path(directory_values["analysis_directory"]).is_dir():
            errors.append(
                "The specified analysis_directory does not exist or is not a folder: "
                f"{directory_values['analysis_directory']}"
            )

    binary_values: dict[str, int] = {}
    for key in BINARY_KEYS:
        value = str(raw_values[key]).strip()
        if value not in {"0", "1"}:
            errors.append(f"'{key}' must be either 0 or 1, not '{value}'.")
        else:
            binary_values[key] = int(value)

    warnings: list[str] = []
    choroid_measure_type = str(raw_values["choroid_measure_type"]).strip()
    if choroid_measure_type not in {"vertical", "perpendicular"}:
        warnings.append(
            f"Invalid choroid_measure_type '{choroid_measure_type}'; using 'perpendicular'."
        )
        choroid_measure_type = "perpendicular"

    raw_distance = str(raw_values["linescan_roi_distance"]).strip()
    try:
        linescan_roi_distance = int(raw_distance)
        if not 100 <= linescan_roi_distance <= 4000:
            warnings.append(
                f"linescan_roi_distance must be between 100 and 4000; using 1500 instead of '{raw_distance}'."
            )
            linescan_roi_distance = 1500
    except ValueError:
        warnings.append(
            f"linescan_roi_distance must be an integer; using 1500 instead of '{raw_distance}'."
        )
        linescan_roi_distance = 1500

    try:
        custom_maps = tuple(normalise_custom_maps(str(raw_values["custom_maps"])))
    except ValueError as error:
        errors.append(f"'custom_maps': {error}")
        custom_maps = ()

    if errors:
        raise ConfigError(errors)

    return Config(
        analysis_directory=directory_values["analysis_directory"],
        output_directory=directory_values["output_directory"],
        robust_run=binary_values["robust_run"],
        save_individual_segmentations=binary_values["save_individual_segmentations"],
        save_individual_images=binary_values["save_individual_images"],
        preprocess_bscans=binary_values["preprocess_bscans"],
        analyse_choroid=binary_values["analyse_choroid"],
        analyse_slo=binary_values["analyse_slo"],
        custom_maps=custom_maps,
        analyse_all_maps=binary_values["analyse_all_maps"],
        analyse_square_grid=binary_values["analyse_square_grid"],
        choroid_measure_type=choroid_measure_type,
        linescan_roi_distance=linescan_roi_distance,
        warnings=tuple(warnings),
    )


def normalise_custom_maps(value: str) -> list[str]:
    """Validate and order custom retinal layer pairs."""
    compact_value = value.replace(" ", "")
    if compact_value == "0":
        return []
    if not compact_value:
        raise ValueError("'custom_maps' cannot be empty; use 0 to disable custom maps.")

    retinal_layers = set(LAYER_ORDER[:-2])
    layer_indices = {layer: index for index, layer in enumerate(LAYER_ORDER[:-2])}
    maps: list[str] = []
    for map_value in compact_value.split(","):
        layers = map_value.split("_")
        if len(layers) != 2 or not all(layers):
            raise ValueError(
                f"Invalid custom map '{map_value}'. Use a layer pair such as ILM_RPE."
            )
        first_layer, second_layer = layers
        if first_layer not in retinal_layers or second_layer not in retinal_layers:
            raise ValueError(
                f"Invalid custom map '{map_value}'. Custom maps may only use retinal layers: "
                f"{', '.join(LAYER_ORDER[:-2])}."
            )
        if first_layer == second_layer:
            raise ValueError(f"Invalid custom map '{map_value}'. The two layers must differ.")
        if layer_indices[first_layer] > layer_indices[second_layer]:
            first_layer, second_layer = second_layer, first_layer
        maps.append(f"{first_layer}_{second_layer}")
    return maps


def format_custom_maps(maps: tuple[str, ...] | list[str]) -> str:
    """Format normalized custom maps for a configuration file."""
    return ", ".join(maps) if maps else "0"


def write_config(
    config: Config | Mapping[str, object],
    destination: str | os.PathLike[str],
    *,
    template_path: str | os.PathLike[str] | None = None,
) -> None:
    """Write values into a template without discarding its explanatory comments."""
    checked_config = validate_config(config, check_analysis_directory=False)
    destination_path = Path(destination)
    if template_path is None:
        template_path = destination_path if destination_path.exists() else None

    if template_path is None:
        lines = [f"{key}: {value}\n" for key, value in checked_config.as_values().items()]
    else:
        template = Path(template_path)
        try:
            lines = template.read_text(encoding="utf-8").splitlines(keepends=True)
        except OSError as error:
            raise ConfigError([f"Unable to read configuration template '{template}': {error}"]) from error

        values = checked_config.as_values()
        seen: set[str] = set()
        updated_lines: list[str] = []
        for line in lines:
            key_candidate = line.split(":", 1)[0].strip() if ":" in line else ""
            if key_candidate in values and not line.lstrip().startswith("#"):
                updated_lines.append(f"{key_candidate}: {values[key_candidate]}\n")
                seen.add(key_candidate)
            else:
                updated_lines.append(line)
        for key, value in values.items():
            if key not in seen:
                updated_lines.append(f"{key}: {value}\n")
        lines = updated_lines

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    destination_path.write_text("".join(lines), encoding="utf-8")


def config_to_args(config: Config | Mapping[str, object]) -> dict[str, object]:
    """Convert a validated configuration to the existing processing contract."""
    if not isinstance(config, Config):
        config = validate_config(config)
    return config.to_args()
