"""Searchable custom retinal-map selection for the OCTolyzer GUI."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QCheckBox, QDialog, QDialogButtonBox,
                               QHBoxLayout, QLabel, QLineEdit, QListWidget,
                               QListWidgetItem, QPushButton, QSizePolicy,
                               QStackedLayout, QVBoxLayout, QWidget)


class MapRow(QWidget):
    """A full-width, keyboard-accessible map selection row."""

    toggled = Signal(bool)

    def __init__(self, map_name: str, checked: bool, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("mapRow")
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumHeight(28)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 0, 6, 0)
        layout.setSpacing(8)
        self.checkbox = QCheckBox(map_name)
        self.checkbox.setObjectName("mapCheck")
        self.checkbox.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        self.checkbox.setAccessibleName(f"Select {map_name}")
        self.checkbox.setToolTip(f"Include {map_name}")
        self.checkbox.setChecked(checked)
        layout.addWidget(self.checkbox)
        self._set_checked_style(checked)
        self.checkbox.toggled.connect(self._set_checked_style)
        self.checkbox.toggled.connect(self.toggled)

    def _set_checked_style(self, checked: bool) -> None:
        self.setProperty("selected", checked)
        self.style().unpolish(self)
        self.style().polish(self)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.checkbox.toggle()
        super().mousePressEvent(event)


class MapSelectionDialog(QDialog):
    """Dialog for filtering and selecting retinal layer-pair maps."""

    def __init__(self, map_names: list[str], selected_maps: set[str], parent: QWidget | None = None):
        super().__init__(parent)
        self.setWindowTitle("Choose retinal maps")
        self.setMinimumSize(560, 620)
        self._selected_maps = set(selected_maps)
        self._build_ui(map_names)

    def _build_ui(self, map_names: list[str]) -> None:
        description = QLabel("Select the layer pairs to calculate for posterior-pole scans.")
        description.setWordWrap(True)

        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("Filter maps, for example ILM or RPE")
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.textChanged.connect(self._filter_maps)

        self.map_list = QListWidget()
        self.map_list.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        self.map_list.setAlternatingRowColors(True)
        self.map_list.setSpacing(1)
        self._map_rows: dict[str, MapRow] = {}
        for map_name in map_names:
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, map_name)
            row = MapRow(map_name, map_name in self._selected_maps)
            row.toggled.connect(
                lambda checked, selected_map=map_name: self._item_changed(selected_map, checked)
            )
            item.setSizeHint(row.sizeHint())
            self.map_list.addItem(item)
            self.map_list.setItemWidget(item, row)
            self._map_rows[map_name] = row

        self.empty_label = QLabel("No maps match your search.")
        self.empty_label.setObjectName("emptyMapLabel")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setWordWrap(True)
        list_container = QWidget()
        list_layout = QStackedLayout(list_container)
        list_layout.setContentsMargins(0, 0, 0, 0)
        list_layout.addWidget(self.map_list)
        list_layout.addWidget(self.empty_label)
        self._list_layout = list_layout

        self.count_label = QLabel()
        self.count_label.setObjectName("mutedLabel")
        self.selected_only_checkbox = QCheckBox("Selected only")
        self.selected_only_checkbox.toggled.connect(self._filter_maps)

        select_all_button = QPushButton("Select visible")
        clear_button = QPushButton("Clear visible")
        select_all_button.clicked.connect(lambda: self._set_visible_checked(True))
        clear_button.clicked.connect(lambda: self._set_visible_checked(False))
        selection_row = QHBoxLayout()
        selection_row.addWidget(self.count_label)
        selection_row.addStretch(1)
        selection_row.addWidget(self.selected_only_checkbox)
        selection_row.addWidget(select_all_button)
        selection_row.addWidget(clear_button)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(description)
        layout.addWidget(self.search_edit)
        layout.addWidget(list_container, stretch=1)
        layout.addLayout(selection_row)
        layout.addWidget(buttons)
        self._filter_maps()

    @staticmethod
    def _item_map_name(item: QListWidgetItem) -> str:
        return str(item.data(Qt.ItemDataRole.UserRole))

    def _item_changed(self, map_name: str, checked: bool) -> None:
        if checked:
            self._selected_maps.add(map_name)
        else:
            self._selected_maps.discard(map_name)
        self._filter_maps()

    def _filter_maps(self, _query: str = "") -> None:
        query = self.search_edit.text().strip().casefold()
        visible_count = 0
        for index in range(self.map_list.count()):
            item = self.map_list.item(index)
            map_name = self._item_map_name(item)
            matches_query = query in map_name.casefold()
            matches_selection = not self.selected_only_checkbox.isChecked() or map_name in self._selected_maps
            item.setHidden(not (matches_query and matches_selection))
            visible_count += int(not item.isHidden())
        self.empty_label.setText(
            "No selected maps match your search."
            if self.selected_only_checkbox.isChecked() and query
            else "No selected maps yet."
            if self.selected_only_checkbox.isChecked()
            else "No maps match your search."
        )
        self._list_layout.setCurrentIndex(0 if visible_count else 1)
        self._update_count(visible_count)

    def _set_visible_checked(self, checked: bool) -> None:
        self.map_list.blockSignals(True)
        for index in range(self.map_list.count()):
            item = self.map_list.item(index)
            if not item.isHidden():
                map_name = self._item_map_name(item)
                checkbox = self._map_rows[map_name].checkbox
                checkbox.blockSignals(True)
                checkbox.setChecked(checked)
                checkbox.blockSignals(False)
                self._map_rows[map_name]._set_checked_style(checked)
                if checked:
                    self._selected_maps.add(map_name)
                else:
                    self._selected_maps.discard(map_name)
        self.map_list.blockSignals(False)
        self._filter_maps()

    def _update_count(self, visible_count: int | None = None) -> None:
        if visible_count is None:
            visible_count = sum(
                not self.map_list.item(index).isHidden()
                for index in range(self.map_list.count())
            )
        self.count_label.setText(
            f"{len(self._selected_maps)} selected | {visible_count} shown of {self.map_list.count()}"
        )

    def selected_maps(self) -> tuple[str, ...]:
        return tuple(
            self._item_map_name(self.map_list.item(index))
            for index in range(self.map_list.count())
            if self._map_rows[self._item_map_name(self.map_list.item(index))].checkbox.isChecked()
        )


class MapSelector(QWidget):
    """Compact map-selection control that opens :class:`MapSelectionDialog`."""

    selection_changed = Signal()

    def __init__(self, map_names: list[str], parent: QWidget | None = None):
        super().__init__(parent)
        self.map_names = tuple(map_names)
        self._selected_maps: tuple[str, ...] = ()
        self._build_ui()

    def _build_ui(self) -> None:
        self.select_button = QPushButton()
        self.select_button.setAccessibleName("Select custom retinal maps")
        self.select_button.clicked.connect(self.open_dialog)
        self.summary_label = QLabel()
        self.summary_label.setObjectName("mutedLabel")
        self.summary_label.setWordWrap(True)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.select_button)
        layout.addWidget(self.summary_label, stretch=1)
        self._update_display()

    def open_dialog(self) -> None:
        dialog = MapSelectionDialog(list(self.map_names), set(self._selected_maps), self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._selected_maps = dialog.selected_maps()
        self._update_display()
        self.selection_changed.emit()

    def selected_maps(self) -> tuple[str, ...]:
        return self._selected_maps

    def set_selected_maps(self, selected_maps: set[str] | tuple[str, ...] | list[str]) -> None:
        ordered = set(selected_maps)
        self._selected_maps = tuple(map_name for map_name in self.map_names if map_name in ordered)
        self._update_display()

    def _update_display(self) -> None:
        count = len(self._selected_maps)
        self.select_button.setText("Select maps..." if not count else f"Select maps ({count})")
        self.select_button.setToolTip(", ".join(self._selected_maps) if self._selected_maps else "No custom maps selected")
        self.summary_label.setText(
            "No custom maps selected; ETDRS measurements remain available."
            if not count
            else f"{count} selected: {', '.join(self._selected_maps[:3])}"
            + ("..." if count > 3 else "")
        )
