import tempfile
import unittest
from pathlib import Path

from octolyzer.config_loader import (ConfigError, config_to_args, load_config,
                                     parse_config, validate_config,
                                     write_config)

CONFIG_VALUES = {
    "analysis_directory": "/tmp/octolyzer/input",
    "output_directory": "/tmp/octolyzer/output",
    "robust_run": "1",
    "save_individual_segmentations": "1",
    "save_individual_images": "0",
    "preprocess_bscans": "1",
    "analyse_choroid": "1",
    "analyse_slo": "0",
    "custom_maps": "BM_ILM, RNFL_PR1",
    "analyse_all_maps": "0",
    "analyse_square_grid": "1",
    "choroid_measure_type": "perpendicular",
    "linescan_roi_distance": "2000",
}


class ConfigLoaderTests(unittest.TestCase):
    def write_values(self, directory: Path, values=None) -> Path:
        config_path = directory / "config.txt"
        config_values = values or CONFIG_VALUES
        config_path.write_text(
            "\n".join(f"{key}: {value}" for key, value in config_values.items()) + "\n",
            encoding="utf-8",
        )
        return config_path

    def test_parse_retains_colons_in_path_values(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = self.write_values(
                Path(directory),
                {**CONFIG_VALUES, "analysis_directory": "/tmp/input:with-colon"},
            )
            values = parse_config(config_path)
            self.assertEqual(values["analysis_directory"], "/tmp/input:with-colon")

    def test_load_normalizes_custom_maps_and_converts_args(self):
        with tempfile.TemporaryDirectory() as directory:
            input_directory = Path(directory) / "input"
            input_directory.mkdir()
            config_path = self.write_values(
                Path(directory),
                {**CONFIG_VALUES, "analysis_directory": str(input_directory)},
            )
            config = load_config(config_path)
            self.assertEqual(config.custom_maps, ("ILM_BM", "RNFL_PR1"))
            self.assertEqual(config.linescan_roi_distance, 2000)
            self.assertEqual(config_to_args(config)["custom_maps"], ["ILM_BM", "RNFL_PR1"])

    def test_invalid_flag_and_custom_map_are_reported(self):
        with self.assertRaises(ConfigError) as context:
            validate_config({**CONFIG_VALUES, "robust_run": "yes", "custom_maps": "ILM_CHORupper"}, check_analysis_directory=False)
        message = str(context.exception)
        self.assertIn("robust_run", message)
        self.assertIn("custom_maps", message)

    def test_invalid_numeric_values_keep_legacy_fallbacks_as_warnings(self):
        config = validate_config(
            {**CONFIG_VALUES, "linescan_roi_distance": "5000", "choroid_measure_type": "sideways"},
            check_analysis_directory=False,
        )
        self.assertEqual(config.linescan_roi_distance, 1500)
        self.assertEqual(config.choroid_measure_type, "perpendicular")
        self.assertEqual(len(config.warnings), 2)

    def test_writer_preserves_template_comments(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_directory = root / "input"
            input_directory.mkdir()
            template_path = self.write_values(
                root,
                {**CONFIG_VALUES, "analysis_directory": str(input_directory)},
            )
            template_path.write_text(
                "# Keep this explanation\n" + template_path.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            config = load_config(template_path)
            destination = root / "nested" / "saved.txt"
            write_config(config, destination, template_path=template_path)
            saved = destination.read_text(encoding="utf-8")
            self.assertIn("# Keep this explanation", saved)
            self.assertIn("custom_maps: ILM_BM, RNFL_PR1", saved)

    def test_load_resolves_relative_directories_from_config_location(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_directory = root / "demo" / "input"
            input_directory.mkdir(parents=True)
            config_path = self.write_values(
                root,
                {**CONFIG_VALUES, "analysis_directory": "demo/input", "output_directory": "demo/output"},
            )

            config = load_config(config_path)

            self.assertEqual(config.analysis_directory, str(input_directory.resolve()))
            self.assertEqual(config.output_directory, str((root / "demo" / "output").resolve()))

    def test_missing_and_unknown_values_are_reported(self):
        with self.assertRaises(ConfigError) as context:
            validate_config({"unknown": "value"}, check_analysis_directory=False)
        message = str(context.exception)
        self.assertIn("unknown configuration key", message.lower())
        self.assertIn("Missing configuration value", message)


if __name__ == "__main__":
    unittest.main()
