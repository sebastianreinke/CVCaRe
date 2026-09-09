"""
cvcare.gui.tabs.capacitance_tab
===============================

Kapazitätsberechnung. Spiegelt die Capacitance-Spalte aus ``gui-4.py``
(Methodenwahl, Halbzyklus-Wahl, "Through-zero"-Option, Export).
"""

from __future__ import annotations

from typing import Callable, Optional, TYPE_CHECKING

import numpy as np
import pyqtgraph as pg

from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from cvcare.dataset import Dataset
from cvcare.exceptions import NotEnoughCVsToFitError

if TYPE_CHECKING:  # pragma: no cover
    from cvcare.gui.widgets.sidebar import SidebarWidget


class _CapacitanceFitDialog(QDialog):
    """Diagnostic pop-up: scan-rate vs. current-difference scatter, with the
    linear fit used to extract the capacitance overlaid."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Capacitance fit diagnostic")
        self.resize(560, 420)

        pg.setConfigOptions(antialias=True)
        self._plot = pg.PlotWidget()
        self._plot.showGrid(x=True, y=True, alpha=0.3)
        self._plot.addLegend(offset=(10, 10))

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.addWidget(self._plot)

    def update_data(
        self,
        dataset_to_optimize: np.ndarray,
        capacitance: float,
        offset: float,
        through_zero: bool,
        y_label: str,
        method_label: str,
    ) -> None:
        self._plot.clear()
        self._plot.addLegend(offset=(10, 10))
        self._plot.getAxis("bottom").setLabel("Scan rate", units="mV/s")
        self._plot.getAxis("left").setLabel(y_label, units="mA")
        self.setWindowTitle(f"Capacitance fit diagnostic \u2014 {method_label}")

        data = np.asarray(dataset_to_optimize, dtype=float)
        scanrates = data[:, 1]
        currents = data[:, 2]

        self._plot.plot(
            scanrates, currents, pen=None, symbol="o", symbolSize=9,
            symbolBrush=pg.mkBrush("#1f77b4"), symbolPen=pg.mkPen("#1f77b4"),
            name="Measured CVs",
        )

        x_min, x_max = float(scanrates.min()), float(scanrates.max())
        if x_max == x_min:
            span = abs(x_max) if x_max != 0 else 1.0
            x_min, x_max = x_min - span * 0.1, x_max + span * 0.1
        else:
            pad = (x_max - x_min) * 0.1
            x_min, x_max = x_min - pad, x_max + pad
        if through_zero:
            x_min = min(0.0, x_min)
        x_line = np.linspace(x_min, x_max, 200)
        y_line = capacitance * x_line if through_zero else capacitance * x_line + offset

        self._plot.plot(
            x_line, y_line, pen=pg.mkPen("#d62728", width=2),
            name=f"Linear fit (C = {capacitance:.4g} F)",
        )


class CapacitanceTab(QWidget):
    """Tab für die lineare Kapazitäts-Auswertung."""

    def __init__(
        self,
        get_dataset: Callable[[], Optional[Dataset]],
        sidebar: Optional["SidebarWidget"] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._get_dataset = get_dataset
        self._sidebar = sidebar
        self._fit_dialog: Optional[_CapacitanceFitDialog] = None

        # Methode
        self._method_combo = QComboBox()
        self._method_combo.addItem("minmax_corrected (recommended)", "minmax_corrected")
        self._method_combo.addItem("minmax", "minmax")
        self._method_combo.addItem("at_selected_voltage", "at_selected_voltage")

        # Halbzyklus
        self._half_combo = QComboBox()
        self._half_combo.addItem("full", "full")
        self._half_combo.addItem("anodic", "anodic")
        self._half_combo.addItem("cathodic", "cathodic")

        # Through zero
        self._through_zero_cb = QCheckBox("Force linear fit through zero")
        self._through_zero_cb.setChecked(True)

        # Ergebnis
        self._capacitance_edit = QLineEdit()
        self._capacitance_edit.setReadOnly(True)
        self._offset_edit = QLineEdit()
        self._offset_edit.setReadOnly(True)

        # Buttons
        self._calc_btn = QPushButton("Compute capacitance")
        self._save_btn = QPushButton("Save result …")
        self._save_btn.setEnabled(False)

        # Layout
        method_box = QGroupBox("Computation")
        form = QFormLayout(method_box)
        form.addRow("Method:", self._method_combo)
        form.addRow("Cycle segment:", self._half_combo)
        form.addRow(self._through_zero_cb)
        form.addRow(self._calc_btn)

        result_box = QGroupBox("Result")
        res_form = QFormLayout(result_box)
        res_form.addRow("Capacitance [F]:", self._capacitance_edit)
        res_form.addRow("Offset [A]:", self._offset_edit)
        res_form.addRow(self._save_btn)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)
        outer.addWidget(method_box)
        outer.addWidget(result_box)
        outer.addStretch(1)

        self._calc_btn.clicked.connect(self._on_calculate)
        self._save_btn.clicked.connect(self._on_save)

    def _refresh_dataset_from_sidebar(self, ds: Dataset) -> None:
        if self._sidebar is None:
            return
        try:
            scanrates = self._sidebar.collect_scanrates()
            if scanrates:
                ds.set_scanrates(scanrates)
        except Exception:  # noqa: BLE001
            pass
        try:
            specs = self._sidebar.collect_specs()
        except Exception:  # noqa: BLE001
            specs = []
        for spec in specs:
            cv = ds.get_content_by_index(spec.index)
            if cv is None:
                continue
            if spec.eval_voltage is not None and hasattr(cv, "eval_voltage"):
                try:
                    cv.eval_voltage = float(spec.eval_voltage)
                except Exception:  # noqa: BLE001
                    pass
            if hasattr(cv, "active"):
                try:
                    cv.active = bool(spec.use)
                except Exception:  # noqa: BLE001
                    pass
            if hasattr(cv, "default_filtered"):
                try:
                    cv.default_filtered = bool(spec.filtered)
                except Exception:  # noqa: BLE001
                    pass

    def _on_calculate(self) -> None:
        ds = self._get_dataset()
        if ds is None or ds.count() == 0:
            QMessageBox.warning(
                self, "No data",
                "No CVs loaded. Please load data first.",
            )
            return

        self._refresh_dataset_from_sidebar(ds)

        try:
            capacitance, offset, details = ds.get_capacitance(
                method=self._method_combo.currentData(),
                through_zero=self._through_zero_cb.isChecked(),
                active_only=True,
                half_cycle_select=self._half_combo.currentData(),
            )
        except NotEnoughCVsToFitError as e:
            QMessageBox.warning(self, "Fit not possible", str(e))
            return
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Computation error", str(e))
            return

        self._capacitance_edit.setText(f"{capacitance:.6g}")
        self._offset_edit.setText(f"{offset:.6g}")
        self._save_btn.setEnabled(True)

        self._show_fit_diagnostic(details, capacitance, offset)

    def _show_fit_diagnostic(self, details, capacitance: float, offset: float) -> None:
        try:
            data = np.asarray(details, dtype=float)
            if data.ndim != 2 or data.shape[0] == 0 or data.shape[1] < 3:
                return
        except Exception:  # noqa: BLE001
            return

        half_cycle_select = self._half_combo.currentData()
        y_label = {
            "full": "Half peak-to-peak current",
            "anodic": "Anodic current",
            "cathodic": "Cathodic current",
        }.get(half_cycle_select, "Current")

        if self._fit_dialog is None:
            self._fit_dialog = _CapacitanceFitDialog(self)

        self._fit_dialog.update_data(
            data,
            capacitance,
            offset,
            through_zero=self._through_zero_cb.isChecked(),
            y_label=y_label,
            method_label=self._method_combo.currentText(),
        )
        self._fit_dialog.show()
        self._fit_dialog.raise_()
        self._fit_dialog.activateWindow()

    def _on_save(self) -> None:
        ds = self._get_dataset()
        if ds is None:
            return
        self._refresh_dataset_from_sidebar(ds)
        path, _ = QFileDialog.getSaveFileName(
            self, "Save capacitance", "capacitance.txt",
            "Text files (*.txt);;All files (*)",
        )
        if not path:
            return
        success = ds.write_capacitance_to_file(
            filename=path,
            through_zero=self._through_zero_cb.isChecked(),
            half_cycle_select=self._half_combo.currentData(),
            method=self._method_combo.currentData(),
        )
        if success:
            QMessageBox.information(self, "Saved", f"Written to:\n{path}")
        else:
            QMessageBox.critical(
                self, "Error", "Could not write file."
            )
