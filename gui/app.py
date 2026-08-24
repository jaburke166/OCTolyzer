"""Main window for the OCTolyzer desktop launcher."""

from __future__ import annotations

import os
import sys
import tempfile
from enum import Enum
from pathlib import Path

from PySide6.QtCore import (QObject, QSettings, Qt, QThread, QTimer, QUrl,
                            Signal)
from PySide6.QtGui import QDesktopServices, QKeySequence
from PySide6.QtWidgets import (QApplication, QFileDialog, QFrame, QHBoxLayout,
                               QLabel, QMainWindow, QMessageBox,
                               QPlainTextEdit, QProgressBar, QPushButton,
                               QSplitter, QStyle, QVBoxLayout, QWidget)

from gui.config_editor import ConfigEditor, ModernComboBox
from gui.environment import (EnvironmentCandidate, EnvironmentProbe,
                             discover_environments, load_cached_environments,
                             load_cached_probe, probe_environment,
                             save_probe_cache)
from gui.runner import ProcessRunner
from octolyzer.config_loader import ConfigError, write_config

APP_STYLE = """
QMainWindow, QWidget {
    background: #142329;
    color: #e4f0ef;
    font-family: "Aptos", "Noto Sans", sans-serif;
    font-size: 13px;
}
QFrame#hero {
    background: #0b3b45;
    border-radius: 12px;
}
QFrame#panel, QFrame#actionBar {
    background: #20363b;
    border: 1px solid #36545a;
    border-radius: 10px;
}
QLabel {
    background: transparent;
}
QWidget#panelHeader {
    background: transparent;
}
QScrollArea#configScroll, QScrollArea#configScroll QWidget {
    background: #20363b;
}
QScrollArea#configScroll {
    border: none;
}
QLabel#heroTitle {
    color: #ffffff;
    font-size: 27px;
    font-weight: 700;
}
QLabel#heroSubtitle {
    color: #c9e2e1;
    font-size: 13px;
}
QLabel#sectionTitle {
    color: #e7f2f1;
    font-size: 15px;
    font-weight: 700;
}
QLabel#sectionNumber {
    color: #168a88;
    font-size: 11px;
    font-weight: 700;
}
QLabel#mutedLabel, QLabel#fieldStatus {
    color: #a5bdc0;
}
QLabel#fieldStatus {
    font-size: 11px;
}
QLabel#readinessBadge {
    background: #e4f3f0;
    border: 1px solid #a9d8d1;
    border-radius: 14px;
    color: #0d6a66;
    font-size: 11px;
    font-weight: 700;
    padding: 6px 12px;
}
QLabel#readinessBadge[state="checking"], QLabel#readinessBadge[state="discovering"] {
    background: #fff3d8;
    border-color: #efd28c;
    color: #856018;
}
QLabel#readinessBadge[state="incompatible"], QLabel#readinessBadge[state="failed"] {
    background: #fde9e5;
    border-color: #efb5aa;
    color: #a33a2e;
}
QLabel#readinessBadge[state="running"], QLabel#readinessBadge[state="stopping"] {
    background: #dceef5;
    border-color: #a9cfdb;
    color: #1b6074;
}
QComboBox, QLineEdit, QSpinBox {
    background: #14282d;
    border: 1px solid #49656b;
    border-radius: 6px;
    color: #edf7f5;
    min-height: 30px;
    padding: 0 8px;
}
QComboBox::drop-down {
    width: 28px;
    border: none;
    border-left: 1px solid #36545a;
}
QComboBox::down-arrow {
    width: 1px;
    height: 1px;
    border: none;
    image: none;
}
QSpinBox::up-button, QSpinBox::down-button {
    width: 24px;
    border: none;
    background: transparent;
}
QSpinBox::up-button:hover, QSpinBox::down-button:hover {
    background: #36545a;
}
QSpinBox::up-arrow, QSpinBox::down-arrow {
    width: 1px;
    height: 1px;
    border: none;
    image: none;
}
QComboBox:focus, QLineEdit:focus, QSpinBox:focus, QListWidget:focus {
    border: 2px solid #35c3b9;
}
QListWidget {
    background: #14282d;
    border: 1px solid #49656b;
    color: #edf7f5;
    outline: none;
}
QListWidget::item {
    background: #14282d;
    padding: 3px 4px;
}
QListWidget::item:alternate {
    background: #20363b;
}
QWidget#mapRow {
    background: transparent;
    background-color: transparent;
    border-radius: 4px;
}
QWidget#mapRow:hover {
    background: #294a50;
}
QWidget#mapRow[selected="true"] {
    background: #1d3d42;
}
QWidget#mapRow[selected="true"]:hover {
    background: #2b5559;
}
QLabel#mapName {
    color: #edf7f5;
}
QLabel#emptyMapLabel {
    color: #a5bdc0;
    padding: 24px;
}
QCheckBox#mapCheck {
    background: transparent;
    background-color: transparent;
}
QCheckBox {
    spacing: 8px;
}
QCheckBox::indicator {
    width: 16px;
    height: 16px;
    border: 1px solid #66858a;
    border-radius: 4px;
    background: #14282d;
}
QCheckBox::indicator:hover {
    border-color: #35c3b9;
}
QCheckBox::indicator:checked {
    border-color: #1eafa5;
    background: #1eafa5;
}
QCheckBox:focus {
    color: #72e5d9;
}
QLineEdit[invalid="true"] {
    background: #382526;
    border: 1px solid #d66b5d;
}
QLabel#validationLabel[valid="false"] {
    color: #ffb0a2;
    background: #512f2d;
    border-radius: 5px;
    padding: 7px;
}
QLabel#validationLabel[valid="true"] {
    color: #9ce7da;
    background: #1d4948;
    border-radius: 5px;
    padding: 7px;
}
QPushButton {
    background: #36545a;
    border: 1px solid #56757a;
    border-radius: 6px;
    color: #f0f8f7;
    min-height: 30px;
    padding: 0 12px;
}
QPushButton:hover {
    background: #46676d;
    border-color: #35c3b9;
}
QPushButton:focus {
    border: 2px solid #35c3b9;
}
QPushButton:disabled {
    color: #6f8589;
    background: #293a3e;
}
QPushButton#primaryButton {
    background: #1eafa5;
    border-color: #168d86;
    color: #ffffff;
    font-weight: 700;
    min-height: 36px;
    padding: 0 18px;
}
QPushButton#primaryButton:hover {
    background: #2bc1b6;
}
QPushButton#primaryButton:disabled {
    background: #293a3e;
    border-color: #405257;
    color: #6f8589;
}
QPushButton#dangerButton {
    background: #512f2d;
    border-color: #b85d52;
    color: #ffb0a2;
}
QPushButton#dangerButton:disabled {
    background: #293a3e;
    border-color: #405257;
    color: #6f8589;
}
QPushButton#secondaryButton {
    background: transparent;
    border-color: #56757a;
}
QProgressBar {
    background: #355057;
    border: none;
    border-radius: 2px;
    max-height: 4px;
}
QProgressBar::chunk {
    background: #168a88;
    border-radius: 2px;
}
QScrollBar:vertical {
    background: #14282d;
    width: 10px;
    margin: 0;
    border: none;
}
QScrollBar::handle:vertical {
    background: #49656b;
    min-height: 28px;
    border-radius: 5px;
}
QScrollBar::handle:vertical:hover {
    background: #6a8b90;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical,
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
    height: 0;
    background: transparent;
}
QScrollBar:horizontal {
    background: #14282d;
    height: 10px;
    margin: 0;
    border: none;
}
QScrollBar::handle:horizontal {
    background: #49656b;
    min-width: 28px;
    border-radius: 5px;
}
QScrollBar::handle:horizontal:hover {
    background: #6a8b90;
}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal,
QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
    width: 0;
    background: transparent;
}
QPlainTextEdit {
    background: #0b1c20;
    border: none;
    border-radius: 6px;
    color: #d5eded;
    font-family: "Cascadia Mono", "DejaVu Sans Mono", monospace;
    font-size: 12px;
    padding: 8px;
}
QSplitter::handle {
    background: #3b5c62;
    height: 8px;
    width: 8px;
}
QStatusBar {
    background: #102025;
    color: #9ab2b5;
}
"""


class UiState(str, Enum):
    DISCOVERING = "discovering"
    NO_ENVIRONMENT = "no_environment"
    SELECTED = "selected"
    CHECKING = "checking"
    INCOMPATIBLE = "incompatible"
    READY = "ready"
    RUNNING = "running"
    STOPPING = "stopping"
    COMPLETED = "completed"
    FAILED = "failed"


class ProbeWorker(QObject):
    """Run an environment probe away from the Qt event loop."""

    finished = Signal(object)

    def __init__(self, executable: Path, runtime_root: Path):
        super().__init__()
        self.executable = executable
        self.runtime_root = runtime_root

    def run(self) -> None:
        result = probe_environment(self.executable, runtime_root=self.runtime_root)
        save_probe_cache(result, runtime_root=self.runtime_root)
        self.finished.emit(result)


class DiscoveryWorker(QObject):
    """Find environment paths without blocking the Qt event loop."""

    finished = Signal(object)

    def __init__(self, workspace_roots: list[Path]):
        super().__init__()
        self.workspace_roots = workspace_roots

    def run(self) -> None:
        self.finished.emit(discover_environments(self.workspace_roots))


class MainWindow(QMainWindow):
    """Single-window workflow for configuring and starting OCTolyzer."""

    def __init__(self, runtime_root: Path):
        super().__init__()
        self.runtime_root = runtime_root
        self.default_config = runtime_root / "config.txt"
        self.settings = QSettings("OCTolyzer", "OCTolyzer GUI")
        self.candidates: list[EnvironmentCandidate] = []
        self.current_probe: EnvironmentProbe | None = None
        self.discovery_thread: QThread | None = None
        self.discovery_worker: DiscoveryWorker | None = None
        self.probe_thread: QThread | None = None
        self.probe_worker: ProbeWorker | None = None
        self.probing_executable: Path | None = None
        self._run_after_environment_check = False
        self.run_directory: tempfile.TemporaryDirectory[str] | None = None
        configured_workspace = os.environ.get("OCTOLYZER_WORKSPACE")
        workspace_root = Path(configured_workspace).expanduser() if configured_workspace else Path.cwd()
        self.workspace_roots = [workspace_root, runtime_root]
        self.runner = ProcessRunner(self)
        self._build_ui()
        self._restore_settings()
        QApplication.instance().aboutToQuit.connect(self._wait_for_workers)
        self.refresh_environments()

    def _build_ui(self) -> None:
        self.setWindowTitle("OCTolyzer")
        screen = QApplication.primaryScreen()
        if screen is None:
            self.setMinimumSize(900, 680)
            self.resize(1180, 840)
        else:
            available = screen.availableGeometry()
            minimum_width = min(900, available.width())
            minimum_height = min(680, available.height())
            self.setMinimumSize(minimum_width, minimum_height)
            self.resize(
                min(1180, max(minimum_width, int(available.width() * 0.88))),
                min(840, max(minimum_height, int(available.height() * 0.88))),
            )
        self.setStyleSheet(APP_STYLE)
        self.ui_state = UiState.DISCOVERING
        self._config_valid = False
        self._output_directory_available = False

        hero = QFrame()
        hero.setObjectName("hero")
        hero_layout = QHBoxLayout(hero)
        hero_layout.setContentsMargins(24, 15, 24, 15)
        hero_layout.setSpacing(16)
        hero_copy = QVBoxLayout()
        hero_copy.setSpacing(1)
        hero_title = QLabel("OCTolyzer")
        hero_title.setObjectName("heroTitle")
        hero_subtitle = QLabel("Configure, validate, and run retinal OCT image analysis from one focused workspace.")
        hero_subtitle.setObjectName("heroSubtitle")
        hero_subtitle.setWordWrap(True)
        hero_copy.addWidget(hero_title)
        hero_copy.addWidget(hero_subtitle)
        hero_layout.addLayout(hero_copy, stretch=1)
        self.readiness_badge = QLabel()
        self.readiness_badge.setObjectName("readinessBadge")
        self.readiness_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.readiness_badge.setMinimumWidth(126)
        hero_layout.addWidget(self.readiness_badge, alignment=Qt.AlignmentFlag.AlignTop)

        environment_panel = QFrame()
        environment_panel.setObjectName("panel")
        environment_layout = QVBoxLayout(environment_panel)
        environment_layout.setContentsMargins(18, 12, 18, 12)
        environment_layout.setSpacing(6)
        environment_layout.addWidget(self._panel_header(
            "Processing environment", "Choose the Python installation that contains OCTolyzer's scientific dependencies."
        ))
        environment_row = QHBoxLayout()
        environment_row.setSpacing(8)
        self.environment_combo = ModernComboBox()
        self.environment_combo.setMinimumHeight(36)
        self.environment_combo.setToolTip("Select the Python environment used for analysis")
        self.browse_button = QPushButton()
        self.browse_button.setToolTip("Browse for a Python executable")
        self.refresh_button = QPushButton()
        self.refresh_button.setToolTip("Refresh Python environments")
        self.probe_button = QPushButton()
        self.probe_button.setToolTip("Check the selected Python environment")
        for button in (self.browse_button, self.refresh_button, self.probe_button):
            button.setFixedSize(36, 30)
        environment_row.addWidget(self.environment_combo, stretch=1)
        environment_actions = QHBoxLayout()
        environment_actions.setSpacing(6)
        environment_actions.addStretch(1)
        environment_actions.addWidget(self.browse_button)
        environment_actions.addWidget(self.refresh_button)
        environment_actions.addWidget(self.probe_button)
        self.environment_status = QLabel("Choose an environment to check.")
        self.environment_status.setObjectName("statusMessage")
        self.environment_status.setWordWrap(True)
        self.environment_source_label = QLabel()
        self.environment_source_label.setObjectName("sectionNumber")
        self.environment_path_label = QLabel()
        self.environment_path_label.setObjectName("mutedLabel")
        self.environment_path_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.environment_path_label.setWordWrap(True)
        self.environment_progress = QProgressBar()
        self.environment_progress.setRange(0, 0)
        self.environment_progress.setTextVisible(False)
        self.environment_progress.setVisible(False)
        environment_layout.addLayout(environment_row)
        environment_layout.addLayout(environment_actions)
        environment_details = QHBoxLayout()
        environment_details.setSpacing(10)
        environment_details.addWidget(self.environment_source_label)
        environment_details.addWidget(self.environment_path_label, stretch=1)
        environment_layout.addLayout(environment_details)
        environment_layout.addWidget(self.environment_status)
        environment_layout.addWidget(self.environment_progress)

        configuration_panel = QFrame()
        configuration_panel.setObjectName("panel")
        configuration_panel.setMinimumHeight(180)
        configuration_layout = QVBoxLayout(configuration_panel)
        configuration_layout.setContentsMargins(18, 12, 18, 10)
        configuration_layout.setSpacing(8)
        configuration_layout.addWidget(self._panel_header(
            "Analysis configuration", "Set the input, output, and measurement options for this run."
        ))
        self.config_editor = ConfigEditor(self.default_config)
        configuration_layout.addWidget(self.config_editor)

        log_panel = QFrame()
        log_panel.setObjectName("panel")
        log_panel.setMinimumHeight(140)
        log_layout = QVBoxLayout(log_panel)
        log_layout.setContentsMargins(18, 12, 18, 12)
        log_layout.setSpacing(8)
        log_header = QHBoxLayout()
        log_header.addWidget(self._panel_header(
            "Processing log", "Live output from the selected analysis environment."
        ), stretch=1)
        self.clear_log_button = QPushButton("Clear")
        self.copy_log_button = QPushButton("Copy")
        self.export_log_button = QPushButton("Export")
        self.clear_log_button.setToolTip("Clear the processing log")
        self.copy_log_button.setToolTip("Copy the processing log to the clipboard")
        self.export_log_button.setToolTip("Save the processing log to a text file")
        for button in (self.clear_log_button, self.copy_log_button, self.export_log_button):
            button.setObjectName("secondaryButton")
        log_header.addWidget(self.clear_log_button)
        log_header.addWidget(self.copy_log_button)
        log_header.addWidget(self.export_log_button)
        log_layout.addLayout(log_header)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setPlaceholderText("Analysis output will appear here.")
        log_layout.addWidget(self.log_view)

        self.run_button = QPushButton("Run analysis")
        self.run_button.setObjectName("primaryButton")
        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("dangerButton")
        self.open_output_button = QPushButton("Open output folder")
        self.stop_button.setEnabled(False)
        self.open_output_button.setEnabled(False)
        action_bar = QFrame()
        action_bar.setObjectName("actionBar")
        action_layout = QHBoxLayout(action_bar)
        action_layout.setContentsMargins(16, 10, 16, 10)
        self.activity_label = QLabel("Ready when your environment and configuration are validated.")
        self.activity_label.setObjectName("mutedLabel")
        self.activity_label.setWordWrap(True)
        action_layout.addWidget(self.activity_label, stretch=1)
        action_layout.addWidget(self.open_output_button)
        action_layout.addWidget(self.stop_button)
        action_layout.addWidget(self.run_button)

        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.setObjectName("mainSplitter")
        left_column = QWidget()
        left_column_layout = QVBoxLayout(left_column)
        left_column_layout.setContentsMargins(0, 0, 0, 0)
        left_column_layout.setSpacing(12)
        left_column_layout.addWidget(environment_panel)
        left_column_layout.addWidget(configuration_panel, stretch=1)

        self.main_splitter.addWidget(left_column)
        self.main_splitter.addWidget(log_panel)
        self.main_splitter.setStretchFactor(0, 3)
        self.main_splitter.setStretchFactor(1, 7)

        central_widget = QWidget()
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(18, 18, 18, 14)
        layout.setSpacing(12)
        layout.addWidget(hero)
        layout.addWidget(self.main_splitter, stretch=1)
        layout.addWidget(action_bar)
        self.setCentralWidget(central_widget)
        self.statusBar().showMessage("Starting OCTolyzer")

        self.refresh_button.clicked.connect(self.refresh_environments)
        self.browse_button.clicked.connect(self.browse_environment)
        self.environment_combo.currentIndexChanged.connect(self._environment_changed)
        self.probe_button.clicked.connect(self.check_environment)
        self.config_editor.save_button.clicked.connect(self.save_configuration)
        self.config_editor.validation_changed.connect(self._configuration_validity_changed)
        self.run_button.clicked.connect(self.run_analysis)
        self.stop_button.clicked.connect(self.stop_analysis)
        self.open_output_button.clicked.connect(self.open_output_folder)
        self.clear_log_button.clicked.connect(self.log_view.clear)
        self.copy_log_button.clicked.connect(self.log_view.copy)
        self.export_log_button.clicked.connect(self.export_log)
        self.runner.output_received.connect(self._append_log)
        self.runner.state_changed.connect(self._runner_state_changed)
        self.runner.completed.connect(self._analysis_completed)
        self.runner.failed.connect(self._runner_failed)
        self.run_button.setShortcut(QKeySequence("Ctrl+Return"))
        self.stop_button.setShortcut(QKeySequence("Escape"))
        self.refresh_button.setShortcut(QKeySequence("F5"))
        self.config_editor.save_button.setShortcut(QKeySequence("Ctrl+S"))
        for button, accessible_name in (
            (self.browse_button, "Browse for a Python executable"),
            (self.refresh_button, "Refresh Python environments"),
            (self.probe_button, "Check the selected Python environment"),
            (self.run_button, "Run OCTolyzer analysis"),
            (self.stop_button, "Stop OCTolyzer analysis"),
            (self.open_output_button, "Open the output folder"),
            (self.clear_log_button, "Clear the processing log"),
            (self.copy_log_button, "Copy the processing log"),
            (self.export_log_button, "Export the processing log"),
        ):
            button.setAccessibleName(accessible_name)
        self._set_standard_icon(self.browse_button, QStyle.StandardPixmap.SP_DirOpenIcon)
        self._set_standard_icon(self.refresh_button, QStyle.StandardPixmap.SP_BrowserReload)
        self._set_standard_icon(self.probe_button, QStyle.StandardPixmap.SP_DialogApplyButton)
        self._set_standard_icon(self.run_button, QStyle.StandardPixmap.SP_MediaPlay)
        self._set_standard_icon(self.stop_button, QStyle.StandardPixmap.SP_MediaStop)
        self._set_standard_icon(self.open_output_button, QStyle.StandardPixmap.SP_DirOpenIcon)
        self._set_standard_icon(self.clear_log_button, QStyle.StandardPixmap.SP_DialogResetButton)
        try:
            self._config_valid = self.config_editor.get_config().analysis_directory != ""
        except ConfigError:
            self._config_valid = False
        self._update_environment_details()

    def _set_ui_state(self, state: UiState, message: str | None = None) -> None:
        self.ui_state = state
        badge_text = {
            UiState.DISCOVERING: "DISCOVERING",
            UiState.NO_ENVIRONMENT: "CHOOSE ENVIRONMENT",
            UiState.SELECTED: "CHECK REQUIRED",
            UiState.CHECKING: "CHECKING",
            UiState.INCOMPATIBLE: "NOT READY",
            UiState.READY: "READY TO RUN",
            UiState.RUNNING: "RUNNING",
            UiState.STOPPING: "STOPPING",
            UiState.COMPLETED: "COMPLETED",
            UiState.FAILED: "FAILED",
        }
        activity_text = {
            UiState.DISCOVERING: "Searching for installed Python environments...",
            UiState.NO_ENVIRONMENT: "Install Python with OCTolyzer's dependencies, then refresh.",
            UiState.SELECTED: "Check the selected environment before running.",
            UiState.CHECKING: "Checking scientific dependencies in the selected environment...",
            UiState.INCOMPATIBLE: "Resolve the missing dependencies before running analysis.",
            UiState.READY: "Environment and configuration are ready.",
            UiState.RUNNING: "Analysis is running. Live output is shown below.",
            UiState.STOPPING: "Waiting for the analysis process to stop...",
            UiState.COMPLETED: "Analysis completed. Review the output folder or log.",
            UiState.FAILED: "Analysis failed. Review the log for details.",
        }
        self.readiness_badge.setText(badge_text[state])
        self.readiness_badge.setProperty("state", state.value)
        self.readiness_badge.style().unpolish(self.readiness_badge)
        self.readiness_badge.style().polish(self.readiness_badge)
        if message is not None:
            self.environment_status.setText(message)
        self.activity_label.setText(activity_text[state])
        self.statusBar().showMessage(activity_text[state])
        self.environment_progress.setVisible(
            state in {UiState.DISCOVERING, UiState.CHECKING, UiState.RUNNING, UiState.STOPPING}
        )
        self._update_action_state()

    def _update_action_state(self) -> None:
        candidate = self._selected_candidate()
        busy = self.ui_state in {UiState.DISCOVERING, UiState.CHECKING, UiState.RUNNING, UiState.STOPPING}
        running = self.ui_state in {UiState.RUNNING, UiState.STOPPING}
        self.environment_combo.setEnabled(not busy and bool(self.candidates))
        self.browse_button.setEnabled(not busy)
        self.refresh_button.setEnabled(not busy and self.discovery_thread is None)
        self.probe_button.setEnabled(bool(candidate) and not busy)
        self.config_editor.setEnabled(not running)
        self.run_button.setEnabled(
            self.ui_state not in {
                UiState.DISCOVERING,
                UiState.CHECKING,
                UiState.INCOMPATIBLE,
                UiState.RUNNING,
                UiState.STOPPING,
            }
            and candidate is not None
            and self._config_valid
        )
        self.stop_button.setEnabled(self.ui_state in {UiState.RUNNING, UiState.STOPPING})
        self.open_output_button.setEnabled(self._output_directory_available and not running)

    def export_log(self) -> None:
        destination, _ = QFileDialog.getSaveFileName(
            self,
            "Export processing log",
            str(Path.home() / "octolyzer-analysis.log"),
            "Log files (*.log *.txt);;All files (*)",
        )
        if not destination:
            return
        try:
            Path(destination).write_text(self.log_view.toPlainText(), encoding="utf-8")
        except OSError as error:
            QMessageBox.warning(self, "Unable to export log", str(error))
            return
        self.statusBar().showMessage(f"Processing log exported to {destination}", 5000)

    @staticmethod
    def _panel_header(title: str, subtitle: str) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)
        container.setObjectName("panelHeader")
        title_label = QLabel(title)
        title_label.setObjectName("sectionTitle")
        subtitle_label = QLabel(subtitle)
        subtitle_label.setObjectName("mutedLabel")
        subtitle_label.setWordWrap(True)
        layout.addWidget(title_label)
        layout.addWidget(subtitle_label)
        return container

    def _set_standard_icon(self, button: QPushButton, icon: QStyle.StandardPixmap) -> None:
        button.setIcon(self.style().standardIcon(icon))

    def _append_log(self, text: str) -> None:
        scrollbar = self.log_view.verticalScrollBar()
        at_bottom = scrollbar.value() >= scrollbar.maximum() - 2
        self.log_view.insertPlainText(text)
        if at_bottom:
            scrollbar.setValue(scrollbar.maximum())

    def _restore_settings(self) -> None:
        geometry = self.settings.value("window_geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)
        splitter_state = self.settings.value("horizontal_splitter_state_v2")
        if splitter_state is not None:
            self.main_splitter.restoreState(splitter_state)
        else:
            QTimer.singleShot(0, self._set_default_splitter_sizes)

    def _set_default_splitter_sizes(self) -> None:
        total_width = self.main_splitter.width()
        if total_width <= 0:
            return
        left_width = round(total_width * 0.3)
        self.main_splitter.setSizes([left_width, total_width - left_width])

    def refresh_environments(self) -> None:
        if self.discovery_thread is not None:
            return
        selected = self._selected_candidate()
        previous_executable = str(selected.executable) if selected else self.settings.value("environment_executable", "")
        if not self.candidates:
            self._replace_candidates(load_cached_environments(), previous_executable)
        self.current_probe = None
        self.probing_executable = None
        self._set_ui_state(UiState.DISCOVERING)

        discovery_thread = QThread(self)
        self.discovery_thread = discovery_thread
        worker = DiscoveryWorker(self.workspace_roots)
        self.discovery_worker = worker
        worker.moveToThread(discovery_thread)
        discovery_thread.started.connect(worker.run)
        worker.finished.connect(self._discovery_finished)
        worker.finished.connect(discovery_thread.quit)
        worker.finished.connect(worker.deleteLater)
        discovery_thread.finished.connect(self._discovery_thread_finished)
        discovery_thread.finished.connect(discovery_thread.deleteLater)
        discovery_thread.start()

    def _replace_candidates(
        self,
        candidates: list[EnvironmentCandidate],
        preferred_executable: str | os.PathLike[str] = "",
    ) -> None:
        manual_candidates = [candidate for candidate in self.candidates if candidate.source == "manual"]
        combined = manual_candidates + candidates
        unique: list[EnvironmentCandidate] = []
        seen: set[Path] = set()
        for candidate in combined:
            executable = self._candidate_key(candidate.executable)
            if executable in seen:
                continue
            seen.add(executable)
            unique.append(candidate)
        self.candidates = unique
        self.environment_combo.blockSignals(True)
        self.environment_combo.clear()
        for candidate in self.candidates:
            self.environment_combo.addItem(
                candidate.label,
                candidate,
            )
        for index, candidate in enumerate(self.candidates):
            if self._candidate_key(candidate.executable) == self._candidate_key(preferred_executable):
                self.environment_combo.setCurrentIndex(index)
                break
        self.environment_combo.blockSignals(False)
        self._update_environment_details()

    def _discovery_finished(self, candidates: list[EnvironmentCandidate]) -> None:
        selected = self._selected_candidate()
        preferred_executable = str(selected.executable) if selected else self.settings.value("environment_executable", "")
        self.current_probe = None
        self._replace_candidates(candidates, preferred_executable)
        self._set_ui_state(
            UiState.SELECTED if self.candidates else UiState.NO_ENVIRONMENT,
            f"Found {len(self.candidates)} Python environment(s). Select one and check its dependencies."
            if self.candidates
            else "No Python environments found. Browse for a compatible Python installation.",
        )

    def _discovery_thread_finished(self) -> None:
        self.discovery_thread = None
        self.discovery_worker = None
        if self.ui_state == UiState.DISCOVERING:
            self._update_ready_state()
        else:
            self._update_action_state()

    @staticmethod
    def _candidate_key(executable: str | os.PathLike[str]) -> Path:
        try:
            return Path(executable).expanduser().resolve(strict=False)
        except OSError:
            return Path(executable).expanduser()

    def browse_environment(self) -> None:
        executable, _ = QFileDialog.getOpenFileName(
            self,
            "Select Python executable",
            str(Path.home()),
            "Python executable (*)",
        )
        if not executable:
            return
        selected_path = Path(executable).expanduser()
        if not selected_path.is_file():
            QMessageBox.warning(self, "Invalid Python executable", "The selected path is not a file.")
            return
        manual_candidate = EnvironmentCandidate(selected_path, "Manually selected Python", "manual")
        selected_key = self._candidate_key(selected_path)
        self.candidates = [
            candidate for candidate in self.candidates if self._candidate_key(candidate.executable) != selected_key
        ]
        self.candidates.insert(0, manual_candidate)
        self.environment_combo.insertItem(0, manual_candidate.label, manual_candidate)
        self.environment_combo.setCurrentIndex(0)
        self._set_ui_state(UiState.SELECTED, f"Selected {selected_path}. Check this environment.")

    def _selected_candidate(self) -> EnvironmentCandidate | None:
        candidate = self.environment_combo.currentData()
        return candidate if isinstance(candidate, EnvironmentCandidate) else None

    def _environment_changed(self) -> None:
        self.current_probe = None
        self.probing_executable = None
        self._update_environment_details()
        candidate = self._selected_candidate()
        self._set_ui_state(
            UiState.SELECTED if candidate else UiState.NO_ENVIRONMENT,
            f"Selected {candidate.executable}. Check this environment."
            if candidate
            else "Choose an environment to check.",
        )

    def check_environment(self, force: bool = False) -> None:
        candidate = self._selected_candidate()
        if candidate is None or self.probe_thread is not None:
            return
        cached_probe = None if force else load_cached_probe(candidate.executable, runtime_root=self.runtime_root)
        if cached_probe is not None:
            self.current_probe = cached_probe
            self._update_ready_state(probe_message=f"{cached_probe.summary} (cached)")
            self._run_pending_analysis()
            return
        self.probing_executable = self._candidate_key(candidate.executable)
        self._set_ui_state(UiState.CHECKING, f"Checking {candidate.executable}...")
        probe_thread = QThread(self)
        self.probe_thread = probe_thread
        worker = ProbeWorker(candidate.executable, self.runtime_root)
        self.probe_worker = worker
        worker.moveToThread(probe_thread)
        probe_thread.started.connect(worker.run)
        worker.finished.connect(self._probe_finished)
        worker.finished.connect(probe_thread.quit)
        worker.finished.connect(worker.deleteLater)
        probe_thread.finished.connect(self._probe_thread_finished)
        probe_thread.finished.connect(probe_thread.deleteLater)
        probe_thread.start()

    def _probe_finished(self, result: EnvironmentProbe) -> None:
        if self.probing_executable != self._candidate_key(result.executable):
            return
        self.current_probe = result
        self._update_ready_state(probe_message=result.summary)
        self._run_pending_analysis()

    def _probe_thread_finished(self) -> None:
        self.probe_thread = None
        self.probe_worker = None
        self.probing_executable = None
        self._update_action_state()

    def _run_pending_analysis(self) -> None:
        if not self._run_after_environment_check:
            return
        self._run_after_environment_check = False
        if self.current_probe is None or not self.current_probe.ok:
            return
        self.run_analysis()

    def _update_environment_details(self) -> None:
        candidate = self._selected_candidate()
        if candidate is None:
            self.environment_source_label.clear()
            self.environment_path_label.clear()
            return
        self.environment_source_label.setText(candidate.source.upper())
        self.environment_path_label.setText(str(candidate.executable))

    def _update_ready_state(self, probe_message: str | None = None) -> None:
        if not self._selected_candidate():
            self._set_ui_state(UiState.NO_ENVIRONMENT, "Choose a Python environment to continue.")
        elif self.current_probe is not None and not self.current_probe.ok:
            self._set_ui_state(UiState.INCOMPATIBLE, probe_message or self.current_probe.summary)
        elif self.current_probe is not None and self.current_probe.ok and self._config_valid:
            self._set_ui_state(UiState.READY, probe_message or "Environment and configuration are ready to run.")
        elif self.current_probe is not None and self.current_probe.ok:
            self._set_ui_state(
                UiState.SELECTED,
                f"{probe_message or self.current_probe.summary} Set a valid configuration before running.",
            )
        else:
            self._set_ui_state(UiState.SELECTED, "Check this environment before running.")

    def _configuration_validity_changed(self, valid: bool, message: str) -> None:
        self._config_valid = valid
        if message:
            self.statusBar().showMessage(message, 5000)
        if self.ui_state not in {UiState.DISCOVERING, UiState.CHECKING, UiState.RUNNING, UiState.STOPPING}:
            self._update_ready_state()

    def save_configuration(self) -> None:
        try:
            config = self.config_editor.get_config()
        except ConfigError as error:
            self._show_config_error(error)
            return

        destination, _ = QFileDialog.getSaveFileName(
            self,
            "Save OCTolyzer configuration",
            str(Path(config.output_directory) / "config.txt"),
            "Text files (*.txt);;All files (*)",
        )
        if not destination:
            return
        try:
            write_config(config, destination, template_path=self.default_config)
        except ConfigError as error:
            self._show_config_error(error)
            return
        self.statusBar().showMessage(f"Configuration saved to {destination}", 5000)

    def run_analysis(self) -> None:
        candidate = self._selected_candidate()
        if candidate is None:
            QMessageBox.warning(self, "Environment required", "Select a Python environment first.")
            return
        try:
            config = self.config_editor.get_config()
        except ConfigError as error:
            self._show_config_error(error)
            return
        if self.current_probe is None:
            self._run_after_environment_check = True
            self.check_environment()
            return
        if not self.current_probe.ok:
            QMessageBox.warning(self, "Environment incompatible", "The selected environment is missing required dependencies.")
            return

        self.run_directory = tempfile.TemporaryDirectory(prefix="octolyzer-run-")
        run_config = Path(self.run_directory.name) / "config.txt"
        try:
            write_config(config, run_config, template_path=self.default_config)
            self.runner.start(candidate.executable, run_config, self.runtime_root)
        except (ConfigError, RuntimeError, OSError) as error:
            self.run_directory.cleanup()
            self.run_directory = None
            QMessageBox.critical(self, "Unable to start analysis", str(error))
            self._set_ui_state(UiState.FAILED, str(error))
            return

        self.settings.setValue("environment_executable", str(candidate.executable))
        self.settings.setValue("last_output_directory", config.output_directory)
        self.log_view.clear()
        self._output_directory_available = False
        self._set_ui_state(UiState.RUNNING, f"Running analysis with {candidate.executable}.")

    def stop_analysis(self) -> None:
        self.runner.stop()

    def _runner_state_changed(self, state: str) -> None:
        if state in {"starting", "running"}:
            self._set_ui_state(UiState.RUNNING)
        elif state == "stopping":
            self._set_ui_state(UiState.STOPPING)

    def _analysis_completed(self, exit_code: int, was_stopped: bool) -> None:
        output_directory = self.config_editor.values().get("output_directory", "")
        if self.run_directory is not None:
            self.run_directory.cleanup()
            self.run_directory = None
        self._output_directory_available = bool(output_directory and Path(str(output_directory)).is_dir())
        if was_stopped:
            self._set_ui_state(UiState.COMPLETED, "Analysis stopped by the user.")
        elif exit_code == 0:
            self._set_ui_state(UiState.COMPLETED, "Analysis completed successfully.")
        else:
            self._set_ui_state(UiState.FAILED, f"Analysis failed with exit code {exit_code}. See the log above.")
        self.statusBar().showMessage(
            "Analysis stopped" if was_stopped else "Analysis completed successfully" if exit_code == 0 else "Analysis failed",
            5000,
        )

    def _runner_failed(self, message: str) -> None:
        self._set_ui_state(UiState.FAILED, f"Unable to run analysis: {message}")
        self.log_view.appendPlainText(f"\nERROR: {message}\n")
        self.statusBar().showMessage("Unable to start analysis", 5000)

    def _wait_for_workers(self) -> None:
        for thread in (self.discovery_thread, self.probe_thread):
            if thread is not None and thread.isRunning():
                thread.quit()
                thread.wait()

    def open_output_folder(self) -> None:
        output_directory = str(self.config_editor.values().get("output_directory", ""))
        if output_directory:
            QDesktopServices.openUrl(QUrl.fromLocalFile(output_directory))
            self.statusBar().showMessage(f"Opened {output_directory}", 3000)

    def _show_config_error(self, error: ConfigError) -> None:
        QMessageBox.warning(self, "Invalid configuration", str(error))

    def closeEvent(self, event) -> None:
        if self.runner.running:
            answer = QMessageBox.question(
                self,
                "Analysis is running",
                "Stop the current analysis and close OCTolyzer?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.runner.stop()
        self._wait_for_workers()
        self.settings.setValue("window_geometry", self.saveGeometry())
        self.settings.setValue("horizontal_splitter_state_v2", self.main_splitter.saveState())
        super().closeEvent(event)


def find_runtime_root() -> Path:
    """Locate the source/runtime payload in development and Nuitka builds."""
    candidates = []
    configured_root = os.environ.get("OCTOLYZER_RUNTIME")
    if configured_root:
        candidates.append(Path(configured_root).expanduser())
    candidates.append(Path(sys.argv[0]).resolve().parent / "runtime")
    candidates.append(Path(__file__).resolve().parents[1])
    for candidate in candidates:
        if (candidate / "octolyzer").is_dir() and (candidate / "config.txt").is_file():
            return candidate
    raise FileNotFoundError("Unable to locate the OCTolyzer runtime payload.")


def main() -> int:
    application = QApplication(sys.argv)
    try:
        window = MainWindow(find_runtime_root())
    except (FileNotFoundError, RuntimeError) as error:
        QMessageBox.critical(None, "OCTolyzer", str(error))
        return 1
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
