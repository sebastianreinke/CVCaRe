"""
This file is part of CVCaRe.
Copyright (C) 2022-2026 Sebastian Reinke
Licensed under the GNU General Public License v3 or later.

User interface for experimental CV bias analysis.
"""
from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtWidgets import (
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


class BiasTab(QWidget):

    def __init__(
        self,
        get_dataset: Callable[[], Optional[Dataset]],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._get_dataset = get_dataset

        self._cv_index_spin = QSpinBox()
        self._cv_index_spin.setMinimum(1)
        self._cv_index_spin.setMaximum(999)
        self._cv_index_spin.setValue(1)

        self._first_edit = QLineEdit()
        self._first_edit.setReadOnly(True)
        self._third_edit = QLineEdit()
        self._third_edit.setReadOnly(True)
        self._dp_first_edit = QLineEdit()
        self._dp_first_edit.setReadOnly(True)
        self._dp_third_edit = QLineEdit()
        self._dp_third_edit.setReadOnly(True)

        self._calc_btn = QPushButton("Compute bias")

        input_box = QGroupBox("Inputs")
        f = QFormLayout(input_box)
        f.addRow("CV number:", self._cv_index_spin)
        f.addRow(self._calc_btn)

        result_box = QGroupBox("Result")
        r = QFormLayout(result_box)
        r.addRow("Current ratio, 1st quarter:", self._first_edit)
        r.addRow("Current ratio, 3rd quarter:", self._third_edit)
        r.addRow("Distortion param, 1st quarter:", self._dp_first_edit)
        r.addRow("Distortion param, 3rd quarter:", self._dp_third_edit)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)
        outer.addWidget(input_box)
        outer.addWidget(result_box)
        outer.addStretch(1)

        self._calc_btn.clicked.connect(self._on_calculate)

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
                "Bias analysis requires a FullCV dataset.",
            )
            return
        try:
            result = cv.bias_analysis()
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Bias analysis error", str(e))
            return

        try:
            first_ratio, third_ratio, dp_first, dp_third = result
        except (TypeError, ValueError):
            QMessageBox.information(
                self, "Result", f"Raw output of bias analysis:\n{result}"
            )
            return

        self._first_edit.setText(f"{first_ratio:.6g}")
        self._third_edit.setText(f"{third_ratio:.6g}")
        self._dp_first_edit.setText(f"{dp_first:.6g}")
        self._dp_third_edit.setText(f"{dp_third:.6g}")
