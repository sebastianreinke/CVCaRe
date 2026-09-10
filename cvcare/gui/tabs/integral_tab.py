"""
This file is part of CVCaRe.
Copyright (C) 2022-2026 Sebastian Reinke
Licensed under the GNU General Public License v3 or later.

User interface for directional CV current integration.
"""
from __future__ import annotations

from typing import Callable, Optional, TYPE_CHECKING

import numpy as np

from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from cvcare.core.cv import FullCV
from cvcare.dataset import Dataset

if TYPE_CHECKING:  # pragma: no cover
    from cvcare.gui.tabs.plot_tab import PlotTab


class IntegralTab(QWidget):

    _DEFAULT_LOWER_V = "-0.2"
    _DEFAULT_UPPER_V = "0.2"

    def __init__(
        self,
        get_dataset: Callable[[], Optional[Dataset]],
        plot_tab: Optional["PlotTab"] = None,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._get_dataset = get_dataset
        self._plot_tab = plot_tab

        self._cv_index_spin = QSpinBox()
        self._cv_index_spin.setMinimum(1)
        self._cv_index_spin.setMaximum(999)
        self._cv_index_spin.setValue(1)

        self._lower_v = QLineEdit(self._DEFAULT_LOWER_V)
        self._lower_v.setToolTip("Lower integration bound in volts. Plain text -- type any number.")
        self._upper_v = QLineEdit(self._DEFAULT_UPPER_V)
        self._upper_v.setToolTip("Upper integration bound in volts. Plain text -- type any number.")

        self._direction = QComboBox()
        self._direction.addItem("anodic (forward)", "forward")
        self._direction.addItem("cathodic (backward)", "backward")

        self._calc_btn = QPushButton("Compute integral")
        self._clear_shading_btn = QPushButton("Clear shading")

        self._integral_edit = QLineEdit()
        self._integral_edit.setReadOnly(True)
        self._integral_err_edit = QLineEdit()
        self._integral_err_edit.setReadOnly(True)

        input_box = QGroupBox("Inputs")
        f = QFormLayout(input_box)
        f.addRow("CV number:", self._cv_index_spin)
        f.addRow("Lower voltage [V]:", self._lower_v)
        f.addRow("Upper voltage [V]:", self._upper_v)
        f.addRow("Direction:", self._direction)
        f.addRow(self._calc_btn)
        f.addRow(self._clear_shading_btn)

        result_box = QGroupBox("Result")
        r = QFormLayout(result_box)
        r.addRow("Integral [V\u00b7A]:", self._integral_edit)
        r.addRow("Estimated error:", self._integral_err_edit)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)
        outer.addWidget(input_box)
        outer.addWidget(result_box)
        outer.addStretch(1)

        self._calc_btn.clicked.connect(self._on_calculate)
        self._clear_shading_btn.clicked.connect(self._on_clear_shading)

    @staticmethod
    def _parse_bound(line_edit: QLineEdit) -> Optional[float]:
        text = line_edit.text().strip()
        if not text:
            return None
        try:
            return float(text.replace(",", "."))
        except ValueError:
            return None

    def _on_calculate(self) -> None:
        ds = self._get_dataset()
        if ds is None or ds.count() == 0:
            QMessageBox.warning(self, "No data", "Please load CVs first.")
            return
        cv = ds.get_content_by_index(self._cv_index_spin.value())
        if cv is None:
            QMessageBox.warning(
                self, "Unknown index",
                f"No CV with index {self._cv_index_spin.value()} is loaded.",
            )
            return
        if not isinstance(cv, FullCV):
            QMessageBox.warning(
                self, "Only available for FullCV",
                "Integral computation requires a FullCV dataset.",
            )
            return

        lower_bound = self._parse_bound(self._lower_v)
        upper_bound = self._parse_bound(self._upper_v)
        if lower_bound is None or upper_bound is None:
            QMessageBox.warning(
                self, "Invalid bound",
                "The lower and upper voltage bounds must both be plain "
                "numbers (e.g. -0.2 or 0.2).",
            )
            return

        try:
            value, err = cv.integrate_one_direction(
                lower_voltage_bound=lower_bound,
                upper_voltage_bound=upper_bound,
                forward_direction=(self._direction.currentData() == "forward"),
            )
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Integration error", str(e))
            return

        try:
            mag = float(value.magnitude)
        except Exception:  # noqa: BLE001
            mag = float(value)
        self._integral_edit.setText(f"{mag:.6g}")
        self._integral_err_edit.setText(f"{err:.6g}")

        if self._plot_tab is not None:
            try:
                self._render_integral_shading(
                    cv,
                    self._cv_index_spin.value(),
                    forward_direction=(self._direction.currentData() == "forward"),
                    lower_bound=lower_bound,
                    upper_bound=upper_bound,
                )
            except Exception as e:  # noqa: BLE001
                QMessageBox.warning(
                    self, "Shading error",
                    "The numeric result is valid, but the integrated area "
                    f"could not be shaded:\n{type(e).__name__}: {e}",
                )

    def _render_integral_shading(
        self, cv, cv_index: int, forward_direction: bool,
        lower_bound: float, upper_bound: float,
    ) -> None:
        segment = cv.get_scan_direction_segment(forward_direction=forward_direction)
        segment = np.asarray(segment, dtype=float)
        if segment.ndim != 2 or segment.shape[0] == 0 or segment.shape[1] < 2:
            raise ValueError(
                "No data available for the selected scan direction; "
                "the CV may not contain a usable half-cycle segment."
            )
        u = segment[:, 0]
        i = segment[:, 1]

        lower, upper = lower_bound, upper_bound
        if upper < lower:
            lower, upper = upper, lower

        mask = (u >= lower) & (u <= upper)
        u_sel = u[mask]
        i_sel = i[mask]

        i_lower = float(np.interp(lower, u, i))
        i_upper = float(np.interp(upper, u, i))
        u_shade = np.concatenate(([lower], u_sel, [upper]))
        i_shade = np.concatenate(([i_lower], i_sel, [i_upper]))

        self._plot_tab.show_integral_shading(cv_index, u_shade, i_shade)

    def _on_clear_shading(self) -> None:
        if self._plot_tab is not None:
            self._plot_tab.clear_integral_shading()
