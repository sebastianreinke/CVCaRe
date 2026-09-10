"""
This file is part of CVCaRe.
Copyright (C) 2022-2026 Sebastian Reinke
Licensed under the GNU General Public License v3 or later.

User interface for HalfCV mirroring and virtual-full-CV CaRe analysis.
"""
from __future__ import annotations

from typing import Callable, Optional, TYPE_CHECKING

import quantities as pq

from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from cvcare.core.cv import FullCV, HalfCV, calculate_rc_cv
from cvcare.dataset import Dataset
from cvcare.exceptions import NoScanrateDefinedError

if TYPE_CHECKING:  # pragma: no cover
    from cvcare.gui.tabs.plot_tab import PlotTab
    from cvcare.gui.widgets.sidebar import SidebarWidget


class HalfCVTab(QWidget):

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

        self._cv_index_spin = QSpinBox()
        self._cv_index_spin.setMinimum(1)
        self._cv_index_spin.setMaximum(999)
        self._cv_index_spin.setValue(1)

        self._method_combo = QComboBox()
        self._method_combo.addItem("Analytical", "Analytical")
        self._method_combo.addItem("Optimisation enhanced analytical", "Optimisation enhanced analytical")

        self._mirror_btn = QPushButton("Show mirrored branch")
        self._care_btn = QPushButton("Compute CaRe on virtual full CV")
        self._clear_btn = QPushButton("Clear overlays")

        info_label = QLabel(
            "For a single recorded half-cycle (anodic or cathodic only), this "
            "constructs a synthetic \u201cmirror\u201d branch by rotating the "
            "curve 180\u00b0 about the midpoint of its own start/end points. "
            "The synthetic branch's endpoints coincide exactly with the real "
            "branch's endpoints, forming a closed virtual full CV.\n\n"
            "\u201cCompute CaRe\u201d runs the same distortion-parameter analysis "
            "used in the CaRe tab on that synthetic CV. The result therefore only "
            "contains information about the selected half-cycle, despite the "
            "appearance of a full CV."
        )
        info_label.setWordWrap(True)

        input_box = QGroupBox("Inputs")
        f = QFormLayout(input_box)
        f.addRow("CV number:", self._cv_index_spin)
        f.addRow("CaRe method:", self._method_combo)
        f.addRow(self._mirror_btn)
        f.addRow(self._care_btn)
        f.addRow(self._clear_btn)

        self._resistance_edit = QLineEdit()
        self._resistance_edit.setReadOnly(True)
        self._capacitance_edit = QLineEdit()
        self._capacitance_edit.setReadOnly(True)
        self._distortion_edit = QLineEdit()
        self._distortion_edit.setReadOnly(True)

        result_box = QGroupBox("CaRe result (virtual full CV)")
        res = QFormLayout(result_box)
        res.addRow("Resistance [\u03a9]:", self._resistance_edit)
        res.addRow("Capacitance [mF]:", self._capacitance_edit)
        res.addRow("Distortion parameter:", self._distortion_edit)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)
        outer.addWidget(info_label)
        outer.addWidget(input_box)
        outer.addWidget(result_box)
        outer.addStretch(1)

        self._mirror_btn.clicked.connect(self._on_show_mirror)
        self._care_btn.clicked.connect(self._on_compute_care)
        self._clear_btn.clicked.connect(self._on_clear_overlays)

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

    def _get_selected_halfcv(self):
        ds = self._get_dataset()
        if ds is None or ds.count() == 0:
            QMessageBox.warning(self, "No data", "Please load CVs first.")
            return None, None
        cv_index = self._cv_index_spin.value()
        cv = ds.get_content_by_index(cv_index)
        if cv is None:
            QMessageBox.warning(
                self, "Unknown index", f"No CV with index {cv_index} is loaded.",
            )
            return None, None
        if not isinstance(cv, HalfCV):
            QMessageBox.warning(
                self, "Only available for half-cycles",
                "This tab is meant for a single recorded half-cycle (anodic "
                "or cathodic only). The selected CV is already a full cycle.",
            )
            return None, None
        return ds, cv

    def _on_show_mirror(self) -> None:
        ds, cv = self._get_selected_halfcv()
        if cv is None:
            return
        cv_index = self._cv_index_spin.value()
        try:
            mirrored = cv.get_mirrored_branch()
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Mirroring error", str(e))
            return
        if self._plot_tab is not None:
            self._plot_tab.show_mirror_overlay(cv_index, mirrored[:, 0], mirrored[:, 1])

    def _on_compute_care(self) -> None:
        ds, cv = self._get_selected_halfcv()
        if cv is None:
            return
        cv_index = self._cv_index_spin.value()

        self._refresh_dataset_from_sidebar(ds)

        try:
            virtual_cv = cv.to_virtual_full_cv()
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Mirroring error", str(e))
            return

        method = self._method_combo.currentData()
        try:
            if method == "Analytical":
                r, c, vwin, dp, offset = virtual_cv.distortion_param_evaluation()
            else:
                r, c, vwin, dp, offset = virtual_cv.fit_cv_by_optimisation()
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
        self._capacitance_edit.setText(f"{c.magnitude * 1e3:.6g}")
        self._distortion_edit.setText(f"{dp:.6g}")

        if self._plot_tab is not None:
            try:
                self._render_care_overlay(virtual_cv, cv_index, r, c, offset, method)
            except Exception as e:  # noqa: BLE001
                print(f"[CVCaRe] Could not create HalfCV CaRe overlay: {e}")

        try:
            mirrored = cv.get_mirrored_branch()
            if self._plot_tab is not None:
                self._plot_tab.show_mirror_overlay(cv_index, mirrored[:, 0], mirrored[:, 1])
        except Exception:  # noqa: BLE001
            pass

    def _render_care_overlay(self, virtual_cv, cv_index, r, c, offset, method) -> None:
        amplitude = virtual_cv.get_amplitude()
        amplitude.units = pq.V
        period = 4 * amplitude / virtual_cv.get_scanrate(pq.V / pq.s)
        period.units = pq.s

        fitted = calculate_rc_cv(
            resistance=float(r.magnitude),
            capacitance=float(c.magnitude),
            period=float(period.magnitude),
            amplitude=float(amplitude.magnitude),
        )

        voltage = fitted[:, 0] * pq.V
        voltage = voltage.rescale(virtual_cv.unit_voltage)
        v_corrector = min(virtual_cv.dataset[:, 0]) * virtual_cv.unit_voltage
        voltage = voltage + v_corrector

        current = fitted[:, 1] * pq.A
        current = current.rescale(virtual_cv.unit_current)
        if method == "Analytical":
            current_corrector = (
                min(virtual_cv.dataset[:, 1]) * virtual_cv.unit_current - min(current)
            )
        else:
            current_corrector = offset if offset is not None else 0 * virtual_cv.unit_current
        current = current + current_corrector

        v_in_volt = voltage.rescale(pq.V).magnitude
        i_in_ampere = current.rescale(pq.A).magnitude
        self._plot_tab.show_fit_overlay(cv_index, v_in_volt, i_in_ampere)

    def _on_clear_overlays(self) -> None:
        if self._plot_tab is None:
            return
        cv_index = self._cv_index_spin.value()
        self._plot_tab.clear_mirror_overlay(cv_index)
        self._plot_tab.clear_fit_overlay(cv_index)
