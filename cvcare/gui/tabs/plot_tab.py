"""
This file is part of CVCaRe.
Copyright (C) 2022-2026 Sebastian Reinke
Licensed under the GNU General Public License v3 or later.

Shared pyqtgraph view for CVs, model curves, mirrored branches, and integration shading.
"""
from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
    QWidget,
)

from cvcare.core.cv import CV
from cvcare.gui.theme import ThemeMode, palette_for


_PLOT_COLORS = [
    "#1f77b4",  # tab:blue
    "#ff7f0e",  # tab:orange
    "#2ca02c",  # tab:green
    "#d62728",  # tab:red
    "#9467bd",  # tab:purple
    "#8c564b",  # tab:brown
    "#e377c2",  # tab:pink
    "#7f7f7f",  # tab:gray
    "#bcbd22",  # tab:olive
    "#17becf",  # tab:cyan
    "#000000",  # black
    "#d3d3d3",  # lightgrey
    "#b22222",  # firebrick
    "#d2691e",  # chocolate
    "#000080",  # navy
    "#4b0082",  # indigo
    "#87ceeb",  # skyblue
    "#9acd32",  # yellowgreen
]


class PlotTab(QWidget):

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)

        pg.setConfigOptions(antialias=True)

        self._x_title_edit = QLineEdit("Potential")
        self._y_title_edit = QLineEdit("Current")
        self._x_unit_edit = QLineEdit("V")
        self._y_unit_edit = QLineEdit("mA")
        self._filter_cb = QCheckBox("Apply Savitzky-Golay filter when plotting")
        self._solo_combo = QComboBox()
        self._solo_combo.addItem("Show all active", None)

        # Plot
        self._plot = pg.PlotWidget()
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        self._plot.addLegend(offset=(10, 10))

        for axis_name in ("bottom", "left"):
            axis = self._plot.getAxis(axis_name)
            axis.enableAutoSIPrefix(False)
            axis.autoSIPrefixScale = 1.0
            axis.labelUnitPrefix = ""

        # Layout: 4 (label, edit) pairs in a single horizontal row above
        # the plot, with the Savitzky-Golay filter toggle directly below.
        for edit in (
            self._x_title_edit,
            self._x_unit_edit,
            self._y_title_edit,
            self._y_unit_edit,
        ):
            edit.setMinimumWidth(90)
            edit.setMaximumWidth(160)

        axis_row = QHBoxLayout()
        axis_row.setContentsMargins(0, 0, 0, 0)
        axis_row.setSpacing(6)
        for label_text, edit in (
            ("X axis title:", self._x_title_edit),
            ("X unit:", self._x_unit_edit),
            ("Y axis title:", self._y_title_edit),
            ("Y unit:", self._y_unit_edit),
        ):
            axis_row.addWidget(QLabel(label_text))
            axis_row.addWidget(edit)
            axis_row.addSpacing(8)
        axis_row.addStretch(1)

        filter_row = QHBoxLayout()
        filter_row.setContentsMargins(0, 0, 0, 0)
        filter_row.addWidget(self._filter_cb)
        filter_row.addSpacing(16)
        filter_row.addWidget(QLabel("Isolate:"))
        filter_row.addWidget(self._solo_combo)
        filter_row.addStretch(1)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(4)
        outer.addLayout(axis_row)
        outer.addLayout(filter_row)
        outer.addWidget(self._plot, 1)

        self._x_title_edit.editingFinished.connect(self._update_axes)
        self._y_title_edit.editingFinished.connect(self._update_axes)
        self._x_unit_edit.editingFinished.connect(self._update_axes)
        self._y_unit_edit.editingFinished.connect(self._update_axes)
        self._filter_cb.toggled.connect(self._replot_current_data)
        self._solo_combo.currentIndexChanged.connect(lambda _i: self._replot_current_data())

        self._current_cvs: list[CV] = []
        self._fit_overlays: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self._integral_shadings: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
        self._mirror_overlays: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}

    # --------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------- #

    def show_cvs(self, cvs: Iterable[CV]) -> None:
        self._current_cvs = list(cvs)
        self._fit_overlays.clear()
        self._integral_shadings.clear()
        self._mirror_overlays.clear()
        self._solo_combo.blockSignals(True)
        self._solo_combo.clear()
        self._solo_combo.addItem("Show all active", None)
        for cv in self._current_cvs:
            try:
                self._solo_combo.addItem(
                    f"#{cv.get_index()} ({type(cv).__name__})", cv.get_index()
                )
            except Exception:  # noqa: BLE001
                pass
        self._solo_combo.setCurrentIndex(0)
        self._solo_combo.blockSignals(False)
        self._replot_current_data()

    def clear(self) -> None:
        self._current_cvs = []
        self._fit_overlays.clear()
        self._integral_shadings.clear()
        self._mirror_overlays.clear()
        self._plot.clear()

    def refresh(self) -> None:
        self._replot_current_data()

    def show_fit_overlay(
        self,
        cv_index: int,
        x_values: np.ndarray,
        y_values_in_amperes: np.ndarray,
    ) -> None:
        x = np.asarray(x_values)
        y = np.asarray(y_values_in_amperes) * 1e3
        self._fit_overlays[int(cv_index)] = (x, y)
        self._replot_current_data()

    def clear_fit_overlay(self, cv_index: Optional[int] = None) -> None:
        if cv_index is None:
            self._fit_overlays.clear()
        else:
            self._fit_overlays.pop(int(cv_index), None)
        self._replot_current_data()

    def show_integral_shading(
        self,
        cv_index: int,
        x_values_in_volts: np.ndarray,
        y_values_in_amperes: np.ndarray,
    ) -> None:
        x = np.asarray(x_values_in_volts, dtype=float)
        y = np.asarray(y_values_in_amperes, dtype=float) * 1e3
        if x.size == 0 or y.size == 0:
            return
        self._integral_shadings[int(cv_index)] = (x, y)
        self._replot_current_data()

    def clear_integral_shading(self, cv_index: Optional[int] = None) -> None:
        if cv_index is None:
            self._integral_shadings.clear()
        else:
            self._integral_shadings.pop(int(cv_index), None)
        self._replot_current_data()

    def show_mirror_overlay(
        self,
        cv_index: int,
        x_values_in_volts: np.ndarray,
        y_values_in_amperes: np.ndarray,
    ) -> None:
        x = np.asarray(x_values_in_volts, dtype=float)
        y = np.asarray(y_values_in_amperes, dtype=float) * 1e3
        if x.size == 0 or y.size == 0:
            return
        self._mirror_overlays[int(cv_index)] = (x, y)
        self._replot_current_data()

    def clear_mirror_overlay(self, cv_index: Optional[int] = None) -> None:
        if cv_index is None:
            self._mirror_overlays.clear()
        else:
            self._mirror_overlays.pop(int(cv_index), None)
        self._replot_current_data()

    def apply_theme(self, mode: ThemeMode) -> None:
        p = palette_for(mode)
        self._plot.setBackground(p["plot_bg"])
        fg = QColor(p["plot_fg"])
        for axis_name in ("bottom", "left"):
            axis = self._plot.getAxis(axis_name)
            axis.setPen(pg.mkPen(fg))
            axis.setTextPen(pg.mkPen(fg))
        self._plot.showGrid(x=True, y=True, alpha=p["plot_grid"])
        # Replot, damit Legende/Titel im neuen Stil rendern.
        self._replot_current_data()

    # --------------------------------------------------------------- #
    # Intern
    # --------------------------------------------------------------- #

    def _update_axes(self) -> None:
        # garantieren ueber autoSIPrefix=False sowie labelUnitPrefix='',
        x_title = self._x_title_edit.text() or "Potential"
        y_title = self._y_title_edit.text() or "Current"
        x_unit = self._x_unit_edit.text().strip()
        y_unit = self._y_unit_edit.text().strip()

        for axis_name, title, unit in (
            ("bottom", x_title, x_unit),
            ("left", y_title, y_unit),
        ):
            axis = self._plot.getAxis(axis_name)
            axis.enableAutoSIPrefix(False)
            axis.autoSIPrefixScale = 1.0
            axis.labelUnitPrefix = ""
            axis.labelUnits = unit
            label = f"{title} ({unit})" if unit else title
            axis.setLabel(text=label)
            axis.picture = None
            axis.update()

    def _color_for_position(self, position: int) -> QColor:
        return QColor(_PLOT_COLORS[position % len(_PLOT_COLORS)])

    def _replot_current_data(self) -> None:
        self._plot.clear()
        self._plot.addLegend(offset=(10, 10))
        self._update_axes()
        use_filter = self._filter_cb.isChecked()
        solo = self._solo_combo.currentData()

        index_to_position = {}
        index_to_cv = {}
        for pos, cv in enumerate(self._current_cvs):
            try:
                index_to_position[int(cv.get_index())] = pos
                index_to_cv[int(cv.get_index())] = cv
            except Exception:  # noqa: BLE001
                pass

        def _is_visible(cv_index: int) -> bool:
            if solo is not None:
                return int(cv_index) == int(solo)
            cv_obj = index_to_cv.get(int(cv_index))
            return cv_obj is None or cv_obj.is_active()

        for pos, cv in enumerate(self._current_cvs):
            if not _is_visible(cv.get_index()):
                continue
            color = self._color_for_position(pos)
            pen = pg.mkPen(color, width=2)

            try:
                if use_filter and hasattr(cv, "get_filtered_dataset"):
                    data = cv.get_filtered_dataset()
                else:
                    data = cv.get_dataset()
            except Exception:  # noqa: BLE001
                data = cv.get_dataset()

            arr = np.asarray(data)
            if arr.size == 0 or arr.shape[1] < 2:
                continue

            x = arr[:, 0]
            y = arr[:, 1] * 1e3  # A -> mA

            name = f"#{cv.get_index()} ({type(cv).__name__})"
            self._plot.plot(x, y, pen=pen, name=name)

        for cv_index, (x, y) in self._fit_overlays.items():
            if not _is_visible(cv_index):
                continue
            pos = index_to_position.get(cv_index)
            if pos is None:
                continue
            color = self._color_for_position(pos)
            pen = pg.mkPen(color, width=2, style=Qt.DashLine)
            self._plot.plot(x, y, pen=pen, name=f"#{cv_index} CaRe fit")

        for cv_index, (x, y) in self._mirror_overlays.items():
            if not _is_visible(cv_index):
                continue
            pos = index_to_position.get(cv_index)
            if pos is None:
                continue
            color = self._color_for_position(pos)
            pen = pg.mkPen(color, width=2, style=Qt.DashDotLine)
            self._plot.plot(x, y, pen=pen, name=f"#{cv_index} mirrored branch")

        for cv_index, (x, y) in self._integral_shadings.items():
            if not _is_visible(cv_index):
                continue
            pos = index_to_position.get(cv_index)
            if pos is None:
                continue
            base_color = self._color_for_position(pos)
            fill_color = QColor(base_color)
            fill_color.setAlpha(110)
            shading_item = pg.PlotDataItem(
                x, y, pen=None, fillLevel=0, brush=pg.mkBrush(fill_color)
            )
            self._plot.addItem(shading_item)
