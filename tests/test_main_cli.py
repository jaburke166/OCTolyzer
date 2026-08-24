import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from octolyzer import main as main_module


CONFIG_TEMPLATE = """analysis_directory: {analysis_directory}
output_directory: {output_directory}
robust_run: 1
save_individual_segmentations: 1
save_individual_images: 1
preprocess_bscans: 1
analyse_choroid: 0
analyse_slo: 0
custom_maps: 0
analyse_all_maps: 0
analyse_square_grid: 0
choroid_measure_type: vertical
linescan_roi_distance: 1500
"""


class MainCliTests(unittest.TestCase):
    def test_selected_config_is_validated_and_forwarded_to_runner(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            input_directory = root / "input"
            input_directory.mkdir()
            config_path = root / "config.txt"
            config_path.write_text(
                CONFIG_TEMPLATE.format(
                    analysis_directory=input_directory,
                    output_directory=root / "output",
                ),
                encoding="utf-8",
            )
            with patch.object(main_module, "run") as run:
                result = main_module.main(["--config", str(config_path)])
            self.assertEqual(result, 0)
            args, keyword_args = run.call_args
            self.assertEqual(args[0]["analyse_choroid"], 0)
            self.assertEqual(keyword_args["config_path"], config_path)


if __name__ == "__main__":
    unittest.main()
