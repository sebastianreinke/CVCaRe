"""
cvcare.gui.widgets.cv_row
=========================

Eine einzelne Zeile in der Sidebar für genau eine zu ladende CV-Datei.
Hält Dateipfad, Zyklusnummer, Auswertespannung, Scanrate, Use-Flag und
Filter-Flag und meldet Änderungen per Qt-Signal.

Die Zeile produziert auf Anfrage ein :class:`cvcare.dataset.CVLoadSpec`,
das direkt an :class:`cvcare.dataset.Dataset` gereicht werden kann.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSizePolicy,
    QToolButton,
    QWidget,
)

from cvcare.dataset import CVLoadSpec


class CVRowWidget(QFrame):
    """
    Eine Zeile mit allen Eingabefeldern für eine einzelne CV-Datei.

    Signale
    -------
    changed:
        Wird emittiert, sobald irgendein Eingabewert sich ändert. Trägt den
        Row-Index (1-basiert) als Payload.
    remove_requested:
        Der Nutzer hat den ✕-Knopf gedrückt. Trägt den Row-Index.
    """

    changed = Signal(int)
    remove_requested = Signal(int)
    # Wird emittiert, wenn der Nutzer im Browse-Dialog mehrere Dateien
    # ausgewaehlt hat. Payload: (row_index, [pfad_2, pfad_3, ...]) -- die
    # erste Datei landet bereits in dieser Zeile, die uebrigen muss der
    # Sidebar-Container in neue Zeilen verteilen.
    additional_files_selected = Signal(int, list)
    active_changed = Signal(int, bool)

    def __init__(self, index: int, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self._index = index

        # ------- Widgets aufbauen ------- #
        self._lbl_no = QLabel(f"#{index}")
        self._lbl_no.setFixedWidth(28)
        self._lbl_no.setAlignment(Qt.AlignCenter)

        self._path_edit = QLineEdit()
        self._path_edit.setPlaceholderText("Path to CV file …")
        self._path_edit.setMinimumWidth(120)
        self._path_edit.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        # Tooltip dient als Voll-Pfad-Anzeige, falls die Anzeige zu kurz ist.
        self._path_edit.textChanged.connect(
            lambda txt, w=self._path_edit: w.setToolTip(txt)
        )

        self._browse_btn = QToolButton()
        self._browse_btn.setText("...")
        self._browse_btn.setToolTip("Select file(s)")
        self._browse_btn.setFixedWidth(24)
        self._browse_btn.setStyleSheet("QToolButton { padding: 0px; }")

        self._cycle_edit = QLineEdit()
        self._cycle_edit.setPlaceholderText("Cycle")
        self._cycle_edit.setFixedWidth(56)
        self._cycle_edit.setToolTip("Cycle number to load (1-based).")

        self._voltage_edit = QLineEdit()
        self._voltage_edit.setPlaceholderText("U [V]")
        self._voltage_edit.setFixedWidth(64)
        self._voltage_edit.setToolTip(
            "Evaluation voltage in volts — decimal separator comma or dot."
        )

        self._scanrate_edit = QLineEdit()
        self._scanrate_edit.setPlaceholderText("mV/s")
        self._scanrate_edit.setFixedWidth(64)
        self._scanrate_edit.setToolTip("Scan rate in mV/s.")

        self._use_cb = QCheckBox("")
        self._use_cb.setChecked(True)
        self._use_cb.setToolTip("Include this dataset in fits.")

        self._filter_cb = QCheckBox("")
        self._filter_cb.setChecked(False)
        self._filter_cb.setToolTip("Default Savitzky-Golay filter for this row.")

        self._remove_btn = QToolButton()
        self._remove_btn.setText("X")
        self._remove_btn.setToolTip("Remove row")
        self._remove_btn.setFixedWidth(24)
        self._remove_btn.setStyleSheet(
            "QToolButton { font-weight: 700; color: #c23b3b; padding: 0px; }"
            "QToolButton:hover { color: #ffffff; background-color: #c23b3b; }"
        )

        use_container = QWidget()
        use_layout = QHBoxLayout(use_container)
        use_layout.setContentsMargins(0, 0, 0, 0)
        use_layout.addStretch(1)
        use_layout.addWidget(self._use_cb)
        use_layout.addStretch(1)
        use_container.setFixedWidth(40)

        filter_container = QWidget()
        filter_layout = QHBoxLayout(filter_container)
        filter_layout.setContentsMargins(0, 0, 0, 0)
        filter_layout.addStretch(1)
        filter_layout.addWidget(self._filter_cb)
        filter_layout.addStretch(1)
        filter_container.setFixedWidth(40)

        # ------- Layout ------- #
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(6)
        layout.addWidget(self._lbl_no)
        layout.addWidget(self._path_edit, 1)
        layout.addWidget(self._browse_btn)
        layout.addWidget(self._cycle_edit)
        layout.addWidget(self._voltage_edit)
        layout.addWidget(self._scanrate_edit)
        layout.addWidget(use_container)
        layout.addWidget(filter_container)
        layout.addWidget(self._remove_btn)

        # ------- Signale ------- #
        self._browse_btn.clicked.connect(self._on_browse_clicked)
        self._remove_btn.clicked.connect(
            lambda: self.remove_requested.emit(self._index)
        )
        for w in (
            self._path_edit,
            self._cycle_edit,
            self._voltage_edit,
            self._scanrate_edit,
        ):
            w.textChanged.connect(lambda _txt: self.changed.emit(self._index))
        for cb in (self._use_cb, self._filter_cb):
            cb.toggled.connect(lambda _b: self.changed.emit(self._index))
        self._use_cb.toggled.connect(
            lambda checked: self.active_changed.emit(self._index, checked)
        )

    # --------------------------------------------------------------- #
    # Public API
    # --------------------------------------------------------------- #

    @property
    def index(self) -> int:
        return self._index

    def set_index(self, new_index: int) -> None:
        """Wird vom Sidebar aufgerufen, wenn Zeilen umnummeriert werden."""
        self._index = new_index
        self._lbl_no.setText(f"#{new_index}")

    def filepath(self) -> str:
        return self._path_edit.text().strip()

    def set_filepath(self, path: str) -> None:
        self._path_edit.setText(path)

    def cycle_number(self) -> Optional[int]:
        raw = self._cycle_edit.text().strip()
        if not raw:
            return None
        try:
            return int(raw)
        except ValueError:
            return None

    def set_cycle_number(self, value: Optional[int]) -> None:
        self._cycle_edit.setText("" if value is None else str(value))

    def eval_voltage(self) -> Optional[float]:
        raw = self._voltage_edit.text().strip().replace(",", ".")
        if not raw:
            return None
        try:
            return float(raw)
        except ValueError:
            return None

    def scanrate(self) -> Optional[float]:
        """Scanrate in mV/s; ``None`` wenn ungültig oder leer."""
        raw = self._scanrate_edit.text().strip().replace(",", ".")
        if not raw:
            return None
        try:
            return float(raw)
        except ValueError:
            return None

    def is_active(self) -> bool:
        return self._use_cb.isChecked()

    def set_active(self, value: bool) -> None:
        self._use_cb.setChecked(value)

    def is_filtered(self) -> bool:
        return self._filter_cb.isChecked()

    def set_filtered(self, value: bool) -> None:
        self._filter_cb.setChecked(value)

    def to_load_spec(self) -> CVLoadSpec:
        return CVLoadSpec(
            index=self._index,
            filepath=self.filepath(),
            cycle_number=self.cycle_number(),
            eval_voltage=self.eval_voltage(),
            use=self.is_active(),
            filtered=self.is_filtered(),
        )

    # --------------------------------------------------------------- #
    # Slots
    # --------------------------------------------------------------- #

    def _on_browse_clicked(self) -> None:
        start_dir = ""
        current = self._path_edit.text().strip()
        if current and Path(current).exists():
            start_dir = str(Path(current).parent)

        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select CV file(s)",
            start_dir,
            "Text files (*.txt *.csv *.dat *.tsv);;All files (*)",
        )
        if not paths:
            return
        # Erste Datei in die aktuelle Zeile, weitere an die Sidebar weiterreichen.
        self._path_edit.setText(paths[0])
        if len(paths) > 1:
            self.additional_files_selected.emit(self._index, list(paths[1:]))
