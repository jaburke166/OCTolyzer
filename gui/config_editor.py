"""Qt form for editing every OCTolyzer configuration field."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (QAbstractButton, QComboBox, QFileDialog,
                               QFormLayout, QHBoxLayout, QLabel, QLineEdit,
                               QPushButton, QScrollArea, QSizePolicy, QSpinBox,
                               QVBoxLayout, QWidget)

from gui.map_selector import MapSelector
from octolyzer.config_loader import (CONFIG_FIELDS, LAYER_ORDER, Config,
                                     ConfigError, validate_config)

FIELD_SECTIONS = {
    "analysis_directory": ("Paths", "Choose the scan folder and where results should be written."),
    "robust_run": ("Processing", "Control how OCTolyzer handles errors and prepares scans."),
    "save_individual_segmentations": ("Output", "Choose which intermediate images and masks to keep."),
    "custom_maps": ("Posterior-pole maps", "Choose optional retinal slabs for volume scans."),
}


class ToggleSwitch(QAbstractButton):
    """Small dependency-free switch for binary configuration options."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setFixedSize(48, 28)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleDescription("Toggle configuration option")

    def sizeHint(self) -> QSize:
        return QSize(48, 28)

    def minimumSizeHint(self) -> QSize:
        return self.sizeHint()

    def paintEvent(self, event) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setOpacity(1.0 if self.isEnabled() else 0.5)

        track = self.rect().adjusted(4, 4, -4, -4)
        track_color = QColor("#1eafa5" if self.isChecked() else "#49656b")
        painter.setPen(track_color)
        painter.setBrush(track_color)
        painter.drawRoundedRect(track, track.height() / 2, track.height() / 2)

        thumb_color = QColor("#f5fffd" if self.isChecked() else "#c2d3d4")
        thumb_size = track.height() - 6
        thumb_x = track.right() - thumb_size - 3 if self.isChecked() else track.left() + 3
        thumb = track.adjusted(thumb_x - track.left(), 3, thumb_x - track.right() + thumb_size, -3)
        painter.setPen(thumb_color)
        painter.setBrush(thumb_color)
        painter.drawEllipse(thumb)

        if self.hasFocus():
            painter.setOpacity(1.0)
            painter.setPen(QColor("#72e5d9"))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRoundedRect(self.rect().adjusted(1, 1, -1, -1), 12, 12)


class ModernComboBox(QComboBox):
    """Combo box with a simple painted chevron instead of a native arrow."""

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = QColor("#72e5d9" if self.hasFocus() or self.underMouse() else "#9ab2b5")
        painter.setPen(QPen(color, 1.8, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        center_x = self.width() - 14
        center_y = self.height() // 2
        painter.drawLine(center_x - 4, center_y - 2, center_x, center_y + 2)
        painter.drawLine(center_x, center_y + 2, center_x + 4, center_y - 2)


class ModernSpinBox(QSpinBox):
    """Spin box with compact painted chevrons for its stepper buttons."""

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = QColor("#72e5d9" if self.hasFocus() or self.underMouse() else "#9ab2b5")
        painter.setPen(QPen(color, 1.6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        center_x = self.width() - 12
        center_y = self.height() // 2
        painter.drawLine(center_x - 4, center_y - 1, center_x, center_y - 5)
        painter.drawLine(center_x, center_y - 5, center_x + 4, center_y - 1)
        painter.drawLine(center_x - 4, center_y + 1, center_x, center_y + 5)
        painter.drawLine(center_x, center_y + 5, center_x + 4, center_y + 1)


class ConfigEditor(QWidget):
    """Edit a typed OCTolyzer configuration using native Qt controls."""

    validation_changed = Signal(bool, str)
    configuration_changed = Signal()

    def __init__(self, template_path: Path, parent: QWidget | None = None):
        super().__init__(parent)
        self.template_path = template_path
        self.widgets: dict[str, QWidget] = {}
        self.path_status_labels: dict[str, QLabel] = {}
        self.validation_label: QLabel | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        content_layout = QVBoxLayout()
        content_layout.setSpacing(16)
        for field in CONFIG_FIELDS:
            if field.key in FIELD_SECTIONS:
                title, description = FIELD_SECTIONS[field.key]
                content_layout.addWidget(self._section_header(title, description))
            widget = self._create_widget(field.key, field.kind)
            widget.setToolTip(field.description)
            label = QLabel(field.label)
            label.setToolTip(field.description)
            form = QFormLayout()
            form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
            form.setContentsMargins(0, 0, 0, 0)
            form.addRow(label, widget)
            content_layout.addLayout(form)
            self.widgets[field.key] = widget

        self.validation_label = QLabel()
        self.validation_label.setObjectName("validationLabel")
        self.validation_label.setWordWrap(True)
        content_layout.addWidget(self.validation_label)
        content_layout.addStretch(1)
        content = QWidget()
        content.setLayout(content_layout)
        content.setMinimumWidth(0)
        content.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        scroll = QScrollArea()
        scroll.setObjectName("configScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setWidget(content)

        self.save_button = QPushButton("Save configuration")
        self.save_button.setObjectName("secondaryButton")
        self.save_button.setAccessibleName("Save OCTolyzer configuration")
        self.reset_button = QPushButton("Reset to template")
        self.reset_button.setObjectName("secondaryButton")
        self.reset_button.setAccessibleName("Reset configuration to template")
        self.reset_button.clicked.connect(self.reset)
        self._connect_change_signals()
        action_row = QHBoxLayout()
        action_row.addStretch(1)
        action_row.addWidget(self.reset_button)
        action_row.addWidget(self.save_button)
        layout = QVBoxLayout(self)
        layout.addWidget(scroll)
        layout.addLayout(action_row)
        self.reset()

    @staticmethod
    def _section_header(title: str, description: str) -> QWidget:
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 8, 0, 0)
        title_label = QLabel(title)
        title_label.setObjectName("sectionTitle")
        description_label = QLabel(description)
        description_label.setObjectName("mutedLabel")
        description_label.setWordWrap(True)
        layout.addWidget(title_label)
        layout.addWidget(description_label)
        return container

    def _create_widget(self, key: str, kind: str) -> QWidget:
        if kind == "directory":
            line_edit = QLineEdit()
            line_edit.setClearButtonEnabled(True)
            browse_button = QPushButton("Browse")
            browse_button.setObjectName("secondaryButton")
            browse_button.clicked.connect(lambda: self._browse_directory(line_edit))
            status_label = QLabel()
            status_label.setObjectName("fieldStatus")
            status_label.setMinimumWidth(0)
            status_label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)
            self.path_status_labels[key] = status_label
            line_edit.textChanged.connect(lambda: self._update_path_status(key, line_edit))
            container = QWidget()
            layout = QHBoxLayout(container)
            layout.setContentsMargins(0, 0, 0, 0)
            layout.addWidget(status_label)
            layout.addWidget(line_edit, stretch=1)
            layout.addWidget(browse_button)
            container.line_edit = line_edit
            return container

        if kind == "boolean":
            return ToggleSwitch()

        if kind == "choice":
            combo = ModernComboBox()
            combo.setFixedWidth(180)
            combo.addItems(["perpendicular", "vertical"])
            return combo

        if kind == "integer":
            spin_box = ModernSpinBox()
            spin_box.setFixedWidth(180)
            spin_box.setRange(100, 4000)
            spin_box.setSuffix(" microns")
            return spin_box

        if kind == "maps":
            retinal_layers = LAYER_ORDER[:-2]
            map_names = [
                f"{first_layer}_{second_layer}"
                for first_index, first_layer in enumerate(retinal_layers)
                for second_layer in retinal_layers[first_index + 1:]
            ]
            return MapSelector(map_names)

        raise ValueError(f"Unsupported configuration field type: {kind}")

    def _browse_directory(self, line_edit: QLineEdit) -> None:
        selected_directory = QFileDialog.getExistingDirectory(self, "Choose folder", line_edit.text())
        if selected_directory:
            line_edit.setText(selected_directory)

    def values(self) -> dict[str, object]:
        values: dict[str, object] = {}
        for key, widget in self.widgets.items():
            if key in {"analysis_directory", "output_directory"}:
                values[key] = widget.line_edit.text().strip()
            elif isinstance(widget, ToggleSwitch):
                values[key] = "1" if widget.isChecked() else "0"
            elif isinstance(widget, QComboBox):
                values[key] = widget.currentText()
            elif isinstance(widget, QSpinBox):
                values[key] = str(widget.value())
            elif isinstance(widget, MapSelector):
                values[key] = ", ".join(widget.selected_maps()) or "0"
        return values

    def get_config(self, *, check_analysis_directory: bool = True) -> Config:
        return validate_config(self.values(), check_analysis_directory=check_analysis_directory)

    def set_config(self, config: Config) -> None:
        self.widgets["analysis_directory"].line_edit.setText(config.analysis_directory)
        self.widgets["output_directory"].line_edit.setText(config.output_directory)
        for key in (
            "robust_run",
            "save_individual_segmentations",
            "save_individual_images",
            "preprocess_bscans",
            "analyse_choroid",
            "analyse_slo",
            "analyse_all_maps",
            "analyse_square_grid",
        ):
            self.widgets[key].setChecked(bool(getattr(config, key)))
        self.widgets["choroid_measure_type"].setCurrentText(config.choroid_measure_type)
        self.widgets["linescan_roi_distance"].setValue(config.linescan_roi_distance)
        maps_widget = self.widgets["custom_maps"]
        selected_maps = set(config.custom_maps)
        maps_widget.set_selected_maps(selected_maps)
        self._validate_inline()

    def reset(self) -> None:
        config = validate_config(
            self._template_values(),
            check_analysis_directory=False,
        )
        self.set_config(config)

    def _template_values(self) -> dict[str, str]:
        from octolyzer.config_loader import parse_config

        values = parse_config(self.template_path)
        template_root = self.template_path.expanduser().resolve().parent
        for key in ("analysis_directory", "output_directory"):
            path = Path(values[key]).expanduser()
            if not path.is_absolute():
                values[key] = str((template_root / path).resolve())
        return values

    def _connect_change_signals(self) -> None:
        for widget in self.widgets.values():
            if isinstance(widget, MapSelector):
                widget.selection_changed.connect(self._validate_inline)
            elif isinstance(widget, ToggleSwitch):
                widget.toggled.connect(self._validate_inline)
            elif isinstance(widget, QComboBox):
                widget.currentTextChanged.connect(self._validate_inline)
            elif isinstance(widget, QSpinBox):
                widget.valueChanged.connect(self._validate_inline)

    def _update_path_status(self, key: str, line_edit: QLineEdit) -> None:
        path = Path(line_edit.text().strip()).expanduser()
        has_value = bool(line_edit.text().strip())
        valid = path.is_dir() if key == "analysis_directory" else has_value
        line_edit.setProperty("invalid", not valid)
        line_edit.style().unpolish(line_edit)
        line_edit.style().polish(line_edit)
        if key == "analysis_directory":
            status = "Folder found" if valid else "Choose an existing folder"
        else:
            status = "Folder found" if path.is_dir() else "Will be created"
        self.path_status_labels[key].setText(status)
        self.path_status_labels[key].setProperty("valid", valid)
        self._validate_inline()

    def _validate_inline(self, *_args) -> None:
        if self.validation_label is None:
            return
        try:
            validate_config(self.values(), check_analysis_directory=True)
        except ConfigError as error:
            message = error.errors[0]
            if len(error.errors) > 1:
                message += f" (+{len(error.errors) - 1} more)"
            self.validation_label.setText(message)
            self.validation_label.setProperty("valid", False)
            self.validation_label.style().unpolish(self.validation_label)
            self.validation_label.style().polish(self.validation_label)
            self.validation_changed.emit(False, str(error))
            self.configuration_changed.emit()
            return
        self.validation_label.setText("Configuration ready to run.")
        self.validation_label.setProperty("valid", True)
        self.validation_label.style().unpolish(self.validation_label)
        self.validation_label.style().polish(self.validation_label)
        self.validation_changed.emit(True, "")
        self.configuration_changed.emit()
