import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    from PySide6.QtCore import QEvent, QEventLoop, QPointF, Qt
    from PySide6.QtGui import QMouseEvent
    from PySide6.QtWidgets import QApplication, QCheckBox

    from gui.app import MainWindow, UiState
    from gui.config_editor import (ConfigEditor, ModernComboBox, ModernSpinBox,
                                   ToggleSwitch)
    from gui.environment import EnvironmentProbe
    from gui.map_selector import MapSelectionDialog, MapSelector
except ImportError:
    QApplication = None


@unittest.skipUnless(QApplication is not None, "PySide6 is required for widget tests")
class GuiWidgetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def test_map_selector_preserves_order_and_selected_count(self):
        selector = MapSelector(["ILM_RPE", "RNFL_PR1", "GCL_RPE"])
        selector.set_selected_maps({"GCL_RPE", "ILM_RPE"})

        self.assertEqual(selector.selected_maps(), ("ILM_RPE", "GCL_RPE"))
        self.assertEqual(selector.select_button.text(), "Select maps (2)")

    def test_map_dialog_filters_and_selects_visible_maps(self):
        dialog = MapSelectionDialog(["ILM_RPE", "RNFL_PR1", "GCL_RPE"], set())
        self.assertEqual(dialog.minimumWidth(), 560)
        self.assertEqual(dialog.map_list.itemWidget(dialog.map_list.item(0)).minimumHeight(), 28)
        self.assertEqual(dialog.map_list.item(0).text(), "")
        dialog.search_edit.setText("ilm")
        dialog._set_visible_checked(True)

        self.assertTrue(dialog.map_list.item(0).isHidden() is False)
        self.assertTrue(dialog.map_list.item(1).isHidden())
        self.assertEqual(dialog.selected_maps(), ("ILM_RPE",))
        first_row = dialog.map_list.itemWidget(dialog.map_list.item(0))
        self.assertIsInstance(first_row.checkbox, QCheckBox)
        self.assertEqual(first_row.checkbox.text(), "ILM_RPE")
        first_row.mousePressEvent(
            QMouseEvent(
                QEvent.Type.MouseButtonPress,
                QPointF(2, 2),
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.LeftButton,
                Qt.KeyboardModifier.NoModifier,
            )
        )
        self.assertEqual(dialog.selected_maps(), ())
        self.assertIn("0 selected", dialog.count_label.text())
        dialog.reject()

    def test_map_dialog_selected_only_and_empty_filter_state(self):
        dialog = MapSelectionDialog(["ILM_RPE", "RNFL_PR1"], {"RNFL_PR1"})

        dialog.selected_only_checkbox.setChecked(True)
        self.assertTrue(dialog.map_list.item(0).isHidden())
        self.assertFalse(dialog.map_list.item(1).isHidden())
        self.assertIn("1 selected", dialog.count_label.text())

        dialog.search_edit.setText("GCL")
        self.assertEqual(dialog._list_layout.currentIndex(), 1)
        self.assertIn("No selected maps match", dialog.empty_label.text())
        dialog.reject()

    def test_boolean_settings_use_switches_and_requested_defaults(self):
        editor = ConfigEditor(Path("config.txt"))

        for key in ("robust_run", "analyse_all_maps"):
            self.assertIsInstance(editor.widgets[key], ToggleSwitch)
            self.assertEqual(editor.widgets[key].size().width(), 48)
            self.assertTrue(editor.widgets[key].isChecked())
            self.assertEqual(editor.values()[key], "1")
        self.assertIsInstance(editor.widgets["choroid_measure_type"], ModernComboBox)
        self.assertIsInstance(editor.widgets["linescan_roi_distance"], ModernSpinBox)
        self.assertEqual(editor.widgets["choroid_measure_type"].width(), 180)
        self.assertEqual(editor.widgets["linescan_roi_distance"].width(), 180)
        self.assertEqual(editor.widgets["choroid_measure_type"].currentText(), "perpendicular")
        editor.deleteLater()

    def test_template_paths_are_empty_by_default(self):
        editor = ConfigEditor(Path("config.txt"))

        self.assertEqual(editor.values()["analysis_directory"], "")
        self.assertEqual(editor.values()["output_directory"], "")
        editor.deleteLater()

    def test_config_editor_uses_compact_map_control_and_inline_path_validation(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            editor = ConfigEditor(Path("config.txt"))
            editor.widgets["analysis_directory"].line_edit.setText(temporary_directory)
            output_directory = str(Path(temporary_directory) / "new-output")
            editor.widgets["output_directory"].line_edit.setText(output_directory)
            editor.widgets["custom_maps"].set_selected_maps({"RNFL_PR1"})
            self.application.processEvents()

            self.assertIsInstance(editor.widgets["custom_maps"], MapSelector)
            self.assertEqual(editor.values()["custom_maps"], "RNFL_PR1")
            self.assertTrue(editor.widgets["analysis_directory"].line_edit.property("invalid") is False)
            self.assertTrue(editor.widgets["output_directory"].line_edit.property("invalid") is False)
            self.assertEqual(editor.path_status_labels["output_directory"].text(), "Will be created")
            self.assertTrue(editor.validation_label.property("valid"))
            editor.deleteLater()

    def test_main_window_discovery_is_async_and_run_checks_on_demand(self):
        window = MainWindow(Path.cwd())
        self.assertEqual(window.ui_state, UiState.DISCOVERING)
        self.assertFalse(window.run_button.isEnabled())

        loop = QEventLoop()
        if window.discovery_thread is not None:
            window.discovery_thread.finished.connect(loop.quit)
            loop.exec()
        self.application.processEvents()

        self.assertEqual(window.discovery_thread, None)
        self.assertGreater(len(window.candidates), 0)
        self.assertNotIn("|", window.environment_combo.currentText())
        # Input/output folders are blank by default, so the run button stays
        # disabled until the user fills in both paths.
        self.assertFalse(window.run_button.isEnabled())
        with tempfile.TemporaryDirectory() as temporary_directory:
            window.config_editor.widgets["analysis_directory"].line_edit.setText(temporary_directory)
            output_directory = str(Path(temporary_directory) / "output")
            window.config_editor.widgets["output_directory"].line_edit.setText(output_directory)
            self.application.processEvents()
            self.assertTrue(window.run_button.isEnabled())
        self.assertEqual(window.main_splitter.orientation(), Qt.Orientation.Horizontal)
        window.resize(1440, 760)
        window.show()
        self.application.processEvents()
        window._set_default_splitter_sizes()
        left_width, right_width = window.main_splitter.sizes()
        self.assertAlmostEqual(left_width / (left_width + right_width), 0.3, delta=0.02)
        self.assertFalse(hasattr(window, "recheck_button"))
        window.current_probe = EnvironmentProbe(
            window.candidates[0].executable,
            python_implementation="CPython",
            ok=True,
        )
        window._config_valid = False
        window._update_ready_state(probe_message="Compatible")
        self.assertEqual(window.ui_state, UiState.SELECTED)
        self.assertFalse(window.run_button.isEnabled())
        window.close()
        self.application.processEvents()

    def test_run_starts_environment_check_when_probe_is_missing(self):
        window = MainWindow(Path.cwd())
        loop = QEventLoop()
        if window.discovery_thread is not None:
            window.discovery_thread.finished.connect(loop.quit)
            loop.exec()
        self.application.processEvents()

        with tempfile.TemporaryDirectory() as temporary_directory:
            window.config_editor.widgets["analysis_directory"].line_edit.setText(temporary_directory)
            output_directory = str(Path(temporary_directory) / "output")
            window.config_editor.widgets["output_directory"].line_edit.setText(output_directory)
            self.application.processEvents()

            window.current_probe = None
            window._config_valid = True
            window._update_ready_state()
            with patch.object(window, "check_environment") as check_environment:
                window.run_analysis()

        check_environment.assert_called_once_with()
        self.assertTrue(window._run_after_environment_check)
        window.close()
        self.application.processEvents()


if __name__ == "__main__":
    unittest.main()
