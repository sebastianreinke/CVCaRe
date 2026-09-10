"""
This file is part of CVCaRe.
Copyright (C) 2022-2026 Sebastian Reinke
Licensed under the GNU General Public License v3 or later.

User interface for full-CV CaRe analysis.
"""
from __future__ import annotations

from typing import Callable, Optional, TYPE_CHECKING

import numpy as np
import quantities as pq

from PySide6.QtWidgets import (
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from cvcare.core.cv import FullCV, calculate_rc_cv
from cvcare.dataset import Dataset
from cvcare.exceptions import NoScanrateDefinedError
from cvcare.fileio.writers import write_standardized_data_file

if TYPE_CHECKING:  # pragma: no cover - nur fuer Typhinweise
    from cvcare.gui.tabs.plot_tab import PlotTab
    from cvcare.gui.widgets.sidebar import SidebarWidget


class CaReTab(QWidget):

    def __init__(
        self,
        get_dataset: Callable[[], Optional[Dataset]],
        sidebar: Optional["SidebarWidget"] = None,
        plot_tab: Optional["PlotTab"] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._get_dataset = get_dataset
        self._sidebar = sidebar
        self._plot_tab = plot_tab

        self._method_combo = QComboBox()
        self._method_combo.addItem("Analytical", "Analytical")
        self._method_combo.addItem("Optimisation enhanced analytical",
                                   "Optimisation enhanced analytical")

        self._cv_index_spin = QSpinBox()
        self._cv_index_spin.setMinimum(1)
        self._cv_index_spin.setMaximum(999)
        self._cv_index_spin.setValue(1)

        self._resistance_edit = QLineEdit()
        self._resistance_edit.setReadOnly(True)
        self._capacitance_edit = QLineEdit()
        self._capacitance_edit.setReadOnly(True)
        self._distortion_edit = QLineEdit()
        self._distortion_edit.setReadOnly(True)

        self._calc_btn = QPushButton("Compute for selected CV")
        self._save_btn = QPushButton("Save bulk evaluation …")
        self._save_fit_btn = QPushButton("Save selected CV's CaRe fit as V/I …")

        input_box = QGroupBox("Inputs")
        form = QFormLayout(input_box)
        form.addRow("Method:", self._method_combo)
        form.addRow("CV number:", self._cv_index_spin)
        form.addRow(self._calc_btn)

        result_box = QGroupBox("Result")
        res = QFormLayout(result_box)
        res.addRow("Resistance [Ω]:", self._resistance_edit)
        res.addRow("Capacitance [mF]:", self._capacitance_edit)
        res.addRow("Distortion parameter:", self._distortion_edit)
        res.addRow(self._save_fit_btn)
        res.addRow(self._save_btn)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)
        outer.addWidget(input_box)
        outer.addWidget(result_box)
        outer.addStretch(1)

        self._calc_btn.clicked.connect(self._on_calculate)
        self._save_btn.clicked.connect(self._on_save)
        self._save_fit_btn.clicked.connect(self._on_save_fit)

    # --------------------------------------------------------------- #
    # Polling: aktuelle Sidebar-Werte ohne Reload an Dataset reichen
    # --------------------------------------------------------------- #

    def _refresh_dataset_from_sidebar(self, ds: Dataset) -> None:
        if self._sidebar is None:
            return

        # Scanraten
        try:
            scanrates = self._sidebar.collect_scanrates()
            if scanrates:
                ds.set_scanrates(scanrates)
        except Exception:  # noqa: BLE001
            pass

        # Auswertespannungen + Flags pro Zeile
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

    # --------------------------------------------------------------- #
    # Slots
    # --------------------------------------------------------------- #

    def _on_calculate(self) -> None:
        ds = self._get_dataset()
        if ds is None or ds.count() == 0:
            QMessageBox.warning(self, "No data", "Please load CVs first.")
            return

        self._refresh_dataset_from_sidebar(ds)

        cv_index = self._cv_index_spin.value()
        cv = ds.get_content_by_index(cv_index)
        if cv is None:
            QMessageBox.warning(
                self, "Unknown index",
                f"No CV with index {cv_index} is loaded.",
            )
            return
        method = self._method_combo.currentData()
        try:
            if method == "Analytical":
                r, c, vwin, dp, offset = cv.distortion_param_evaluation()
            else:
                r, c, vwin, dp, offset = cv.fit_cv_by_optimisation()
        except NoScanrateDefinedError:
            QMessageBox.warning(
                self, "Scan rate missing",
                "The analysis requires a scan rate in the sidebar.",
            )
            return
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Error", str(e))
            return

        self._resistance_edit.setText(f"{r.magnitude:.6g}")
        self._capacitance_edit.setText(f"{c.magnitude * 1e3:.6g}")  # F -> mF
        self._distortion_edit.setText(f"{dp:.6g}")

        # ------- Overlay: Modell-CV gestrichelt im Plot ------- #
        if self._plot_tab is not None:
            try:
                self._render_fit_overlay(cv, cv_index, r, c, offset, method)
            except Exception as e:  # noqa: BLE001
                # Overlay is an optional convenience -- stay silent on errors
                # and just deliver the result numbers.
                print(f"[CVCaRe] Could not create fit overlay: {e}")

    def _render_fit_overlay(self, cv, cv_index, r, c, offset, method) -> None:
        amplitude = cv.get_amplitude()
        amplitude.units = pq.V
        period = 4 * amplitude / cv.get_scanrate(pq.V / pq.s)
        period.units = pq.s

        fitted = calculate_rc_cv(
            resistance=float(r.magnitude),
            capacitance=float(c.magnitude),
            period=float(period.magnitude),
            amplitude=float(amplitude.magnitude),
        )

        voltage = fitted[:, 0] * pq.V
        voltage = voltage.rescale(cv.unit_voltage)
        v_corrector = min(cv.dataset[:, 0]) * cv.unit_voltage
        voltage = voltage + v_corrector

        current = fitted[:, 1] * pq.A
        current = current.rescale(cv.unit_current)
        if method == "Analytical":
            current_corrector = (
                min(cv.dataset[:, 1]) * cv.unit_current - min(current)
            )
        else:
            current_corrector = offset if offset is not None else 0 * cv.unit_current
        current = current + current_corrector

        v_in_volt = voltage.rescale(pq.V).magnitude
        i_in_ampere = current.rescale(pq.A).magnitude
        self._plot_tab.show_fit_overlay(cv_index, v_in_volt, i_in_ampere)

    def _on_save_fit(self) -> None:
        ds = self._get_dataset()
        if ds is None or ds.count() == 0:
            QMessageBox.warning(self, "No data", "Please load CVs first.")
            return

        # unterscheiden.
        self._refresh_dataset_from_sidebar(ds)

        cv_index = self._cv_index_spin.value()
        cv = ds.get_content_by_index(cv_index)
        if cv is None:
            QMessageBox.warning(
                self, "Unknown index",
                f"No CV with index {cv_index} is loaded.",
            )
            return
        if not isinstance(cv, FullCV):
            QMessageBox.warning(
                self, "Only available for FullCV",
                "CaRe fit export requires a FullCV dataset (full cycles mode).",
            )
            return

        method = self._method_combo.currentData()
        try:
            if method == "Analytical":
                r, c, _vwin, _dp, offset = cv.distortion_param_evaluation()
            else:
                r, c, _vwin, _dp, offset = cv.fit_cv_by_optimisation()
        except NoScanrateDefinedError:
            QMessageBox.warning(
                self, "Scan rate missing",
                "The analysis requires a scan rate in the sidebar.",
            )
            return
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Analysis error", str(e))
            return

        # Reconstruct the fitted CV, matching the overlay logic, so what gets
        # saved is exactly what the user sees plotted.
        try:
            amplitude = cv.get_amplitude()
            amplitude.units = pq.V
            period = 4 * amplitude / cv.get_scanrate(pq.V / pq.s)
            period.units = pq.s

            fitted = calculate_rc_cv(
                resistance=float(r.magnitude),
                capacitance=float(c.magnitude),
                period=float(period.magnitude),
                amplitude=float(amplitude.magnitude),
            )

            voltage_slice = fitted[:, 0] * pq.V
            voltage_slice = voltage_slice.rescale(cv.unit_voltage)
            v_corrector = min(cv.dataset[:, 0]) * cv.unit_voltage
            voltage_slice = voltage_slice + v_corrector

            current_slice = fitted[:, 1] * pq.A
            current_slice = current_slice.rescale(cv.unit_current)
            if method == "Analytical":
                current_corrector = (
                    min(cv.dataset[:, 1]) * cv.unit_current - min(current_slice)
                )
            else:
                current_corrector = (
                    offset if offset is not None else 0 * cv.unit_current
                )
            current_slice = current_slice + current_corrector
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Fit reconstruction error", str(e))
            return

        # Default file name carries the source index for traceability.
        default_name = f"care_fit_cv{cv_index}.txt"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save CaRe fit as V/I", default_name,
            "Text files (*.txt);;All files (*)",
        )
        if not path:
            return

        # Stack into (V, I) rows; prepend a header line with units, identical
        fitted_array = np.stack(
            (voltage_slice.magnitude, current_slice.magnitude), axis=1
        )
        rows = fitted_array.tolist()
        header = [
            f"Voltage [{voltage_slice.units.dimensionality}]",
            f"Current [{current_slice.units.dimensionality}]",
        ]
        rows.insert(0, header)

        ok = write_standardized_data_file(filename=path, data_to_write=rows)
        if ok:
            QMessageBox.information(self, "Saved", f"CaRe fit written to:\n{path}")
        else:
            QMessageBox.critical(self, "Error", "The interpolated CV failed to write.")

    def _on_save(self) -> None:
        ds = self._get_dataset()
        if ds is None:
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save bulk distortion analysis",
            "distortion_results.txt",
            "Text files (*.txt);;All files (*)",
        )
        if not path:
            return
        ok = ds.write_distortion_param_results_to_file(
            filename=path, method=self._method_combo.currentData()
        )
        if ok:
            QMessageBox.information(self, "Saved", f"Written to:\n{path}")
        else:
            QMessageBox.critical(self, "Error", "Saving failed.")
