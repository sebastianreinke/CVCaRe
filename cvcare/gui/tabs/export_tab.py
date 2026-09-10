"""
This file is part of CVCaRe.
Copyright (C) 2022-2026 Sebastian Reinke
Licensed under the GNU General Public License v3 or later.

User interface for data and result export actions.
"""
from __future__ import annotations

import os
from typing import Callable, Optional

from PySide6.QtWidgets import (
    QFileDialog,
    QGroupBox,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from cvcare.dataset import Dataset
from cvcare.fileio.writers import write_split_cycles


class ExportTab(QWidget):

    def __init__(
        self,
        get_dataset: Callable[[], Optional[Dataset]],
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self._get_dataset = get_dataset

        # 1) Einzelne CVs in mehrere Dateien
        self._split_btn = QPushButton("Write loaded CVs as separate files …")
        self._split_btn.clicked.connect(self._on_split_files)

        self._single_btn = QPushButton("Write loaded CVs to a single file …")
        self._single_btn.clicked.connect(self._on_single_file)

        self._raw_split_btn = QPushButton(
            "Split original file into cycles (without Dataset) …"
        )
        self._raw_split_btn.clicked.connect(self._on_raw_split)

        cv_box = QGroupBox("Loaded datasets")
        cb = QVBoxLayout(cv_box)
        cb.addWidget(self._split_btn)
        cb.addWidget(self._single_btn)

        raw_box = QGroupBox("Raw data")
        rb = QVBoxLayout(raw_box)
        rb.addWidget(self._raw_split_btn)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)
        outer.addWidget(cv_box)
        outer.addWidget(raw_box)
        outer.addStretch(1)

    # ----- Slots ----- #

    def _on_split_files(self) -> None:
        ds = self._get_dataset()
        if ds is None or ds.count() == 0:
            QMessageBox.warning(self, "No data", "Please load CVs first.")
            return
        directory = QFileDialog.getExistingDirectory(self, "Select target folder")
        if not directory:
            return

        filenames: dict = {}
        filtered: dict = {}
        comment: dict = {}
        for cv in ds.contents:
            base = os.path.basename(cv.get_source()) or f"cv_{cv.get_index()}.txt"
            stem, _ext = os.path.splitext(base)
            out = os.path.join(directory, f"{stem}_CV_{cv.get_index()}.txt")
            filenames[cv.get_index()] = out
            filtered[cv.get_index()] = cv.get_default_filtered()
            comment[cv.get_index()] = None

        ok = ds.write_CVs_to_files(filenames=filenames, filtered=filtered, comment=comment)
        if ok:
            QMessageBox.information(self, "Done", f"Files written to:\n{directory}")
        else:
            QMessageBox.warning(self, "Partial success",
                                "At least one file could not be written. "
                                "See console for details.")

    def _on_single_file(self) -> None:
        ds = self._get_dataset()
        if ds is None or ds.count() == 0:
            QMessageBox.warning(self, "No data", "Please load CVs first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Target file", "all_cvs.txt",
            "Text files (*.txt);;All files (*)",
        )
        if not path:
            return
        filtered = {cv.get_index(): cv.get_default_filtered() for cv in ds.contents}
        comment = {cv.get_index(): None for cv in ds.contents}
        ok = ds.write_CVs_to_single_file(filename=path, filtered=filtered, comment=comment)
        if ok:
            QMessageBox.information(self, "Done", f"Written to:\n{path}")
        else:
            QMessageBox.critical(self, "Error", "Could not write file.")

    def _on_raw_split(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select source file", "",
            "Text files (*.txt *.csv *.dat *.tsv);;All files (*)",
        )
        if not path:
            return
        ok = write_split_cycles(filename=path, assume_standard_csv_format=False)
        if ok:
            QMessageBox.information(
                self, "Done",
                "Cycle-split file was placed in the same directory "
                "(suffix _CVEval_cycle_split).",
            )
        else:
            QMessageBox.critical(self, "Error", "Cycle split failed.")
