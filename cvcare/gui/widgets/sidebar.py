"""
cvcare.gui.widgets.sidebar
==========================

Linke Spalte: Liste der :class:`CVRowWidget` plus Bedienelemente
("Zeile hinzufügen", "Alle löschen", "Laden", Modusumschalter, CSV-Format).
"""

from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import Signal, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpacerItem,
    QVBoxLayout,
    QWidget,
)

from cvcare.dataset import CVLoadSpec, HalfCycleMode
from cvcare.gui.widgets.cv_row import CVRowWidget


class SidebarWidget(QFrame):
    """
    Container für CV-Zeilen plus Lade-Optionen.

    Signale
    -------
    load_requested:
        Der Nutzer hat "Laden" gedrückt.
    rows_changed:
        Die Anzahl der Zeilen oder ein Eintrag hat sich geändert (für
        Re-Plot-Heuristiken).
    """

    load_requested = Signal()
    rows_changed = Signal()
    active_changed = Signal(int, bool)

    DEFAULT_ROWS = 3

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setFrameShape(QFrame.StyledPanel)
        self.setObjectName("Sidebar")

        self._rows: List[CVRowWidget] = []

        title = QLabel("CV files")
        title.setStyleSheet("font-size: 14pt; font-weight: 600;")

        # Lade-Modus
        self._mode_combo = QComboBox()
        self._mode_combo.addItem("Full cycles", HalfCycleMode.FULL)
        self._mode_combo.addItem("Anodic (forward)", HalfCycleMode.ANODIC)
        self._mode_combo.addItem("Cathodic (backward)", HalfCycleMode.CATHODIC)
        self._mode_combo.setToolTip(
            "How much of each dataset to load."
        )

        self._csv_cb = QCheckBox("Force standard CSV format")
        self._csv_cb.setToolTip(
            "Force the plain standard CSV schema during reading. "
            "Leave this off when your file comes from a potentiostat program; "
            "the auto-detection will handle it."
        )

        # Buttons
        self._add_btn = QPushButton("+ Row")
        self._clear_btn = QPushButton("Clear all")
        self._load_btn = QPushButton("Load")
        self._load_btn.setDefault(True)
        self._load_btn.setStyleSheet("font-weight: 600;")

        btn_bar = QHBoxLayout()
        btn_bar.addWidget(self._add_btn)
        btn_bar.addWidget(self._clear_btn)
        btn_bar.addStretch(1)
        btn_bar.addWidget(self._load_btn)

        # Form für die Lade-Optionen
        form = QFormLayout()
        form.addRow("Mode:", self._mode_combo)
        form.addRow(self._csv_cb)

        # Spaltenüberschriften für die schmalen Felder. Wir bauen sie als
        # nicht-funktionales QLabel-Layout direkt über die Liste.
        header = QHBoxLayout()
        header.setContentsMargins(8, 0, 8, 0)
        header.setSpacing(6)
        for text, width in [
            ("", 28),       # # number
            ("File", -1),   # Stretch
            ("", 24),       # Browse
            ("Cycle", 56),
            ("U [V]", 64),
            ("ν [mV/s]", 64),
            ("active", -1),
            ("Filter", -1),
            ("", 24),       # Remove
        ]:
            lbl = QLabel(text)
            lbl.setStyleSheet("color: palette(mid); font-size: 9pt;")
            if width > 0:
                lbl.setFixedWidth(width)
                header.addWidget(lbl)
            else:
                # Use-/Filter-Header bekommen feste Breite, damit sie über
                # den Checkboxen sitzen; Datei-Header ist Stretcher.
                if text == "File":
                    header.addWidget(lbl, 1)
                else:
                    lbl.setFixedWidth(40)
                    header.addWidget(lbl)

        header_widget = QWidget()
        header_widget.setLayout(header)

        # Scroll-Bereich für die Zeilen
        self._rows_host = QWidget()
        self._rows_layout = QVBoxLayout(self._rows_host)
        self._rows_layout.setContentsMargins(0, 0, 0, 0)
        self._rows_layout.setSpacing(4)
        self._rows_layout.addStretch(1)

        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setWidget(self._rows_host)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)
        outer.addWidget(title)
        outer.addLayout(form)
        outer.addLayout(btn_bar)
        outer.addWidget(header_widget)
        outer.addWidget(self._scroll, 1)

        # Default-Zeilen
        for _ in range(self.DEFAULT_ROWS):
            self.add_row()

        # Signale
        self._add_btn.clicked.connect(self.add_row)
        self._clear_btn.clicked.connect(self.clear_rows)
        self._load_btn.clicked.connect(self.load_requested.emit)

    # --------------------------------------------------------------- #
    # Zeilen-Management
    # --------------------------------------------------------------- #

    def add_row(self) -> CVRowWidget:
        index = len(self._rows) + 1
        row = CVRowWidget(index, parent=self._rows_host)
        # Vor dem stretch-Spacer einfügen
        self._rows_layout.insertWidget(self._rows_layout.count() - 1, row)
        self._rows.append(row)

        row.remove_requested.connect(self._on_row_remove_requested)
        row.changed.connect(lambda _i: self.rows_changed.emit())
        row.additional_files_selected.connect(self._on_additional_files_selected)
        row.active_changed.connect(self.active_changed.emit)
        self.rows_changed.emit()
        return row

    def _on_additional_files_selected(self, _origin_index: int, extra_paths: list) -> None:
        """
        Wird gefeuert, wenn der Browse-Dialog in einer Zeile mehrere Dateien
        liefert. Wir nutzen freie (leere) Zeilen, ansonsten erzeugen wir neue.
        """
        # Erst freie Zeilen ohne Pfad einsammeln (in vorhandener Reihenfolge)
        free_rows = [r for r in self._rows if not r.filepath()]
        # Schreibroutine: pro Pfad eine Zeile bedienen, ggf. neu anlegen
        for path in extra_paths:
            if free_rows:
                target = free_rows.pop(0)
            else:
                target = self.add_row()
            target.set_filepath(path)
        self.rows_changed.emit()

    def clear_rows(self) -> None:
        for row in list(self._rows):
            self._remove_row_widget(row)
        # Default-Zeilen wieder aufbauen
        for _ in range(self.DEFAULT_ROWS):
            self.add_row()

    def _on_row_remove_requested(self, row_index: int) -> None:
        target = next((r for r in self._rows if r.index == row_index), None)
        if target is None:
            return
        # Mindestens eine Zeile soll bleiben — sonst sieht es leer aus.
        if len(self._rows) <= 1:
            target.set_filepath("")
            target.set_cycle_number(None)
            return
        self._remove_row_widget(target)
        self._renumber()
        self.rows_changed.emit()

    def _remove_row_widget(self, row: CVRowWidget) -> None:
        self._rows_layout.removeWidget(row)
        self._rows.remove(row)
        row.deleteLater()

    def _renumber(self) -> None:
        for i, row in enumerate(self._rows, start=1):
            row.set_index(i)

    # --------------------------------------------------------------- #
    # Datenabfrage
    # --------------------------------------------------------------- #

    def collect_specs(self) -> List[CVLoadSpec]:
        """Sammelt die :class:`CVLoadSpec` aller Zeilen mit nicht-leerem Pfad."""
        specs: List[CVLoadSpec] = []
        for row in self._rows:
            if row.filepath():
                specs.append(row.to_load_spec())
        return specs

    def collect_scanrates(self) -> List[list]:
        """
        Liefert eine Liste ``[[index, scanrate_mV_per_s], ...]`` für alle
        Zeilen mit numerischer Scanrate. Direkt an
        :meth:`cvcare.dataset.Dataset.set_scanrates` verfütterbar.
        """
        result: List[list] = []
        for row in self._rows:
            sr = row.scanrate()
            if sr is not None:
                result.append([row.index, float(sr)])
        return result

    def halfcycle_mode(self) -> HalfCycleMode:
        # currentData() liefert bei str-Enums den String — wir koercieren
        # zurück zum Enum, damit downstream-Konsumenten Identitätsvergleiche
        # mit HalfCycleMode.FULL/ANODIC/CATHODIC machen können.
        data = self._mode_combo.currentData()
        if isinstance(data, HalfCycleMode):
            return data
        try:
            return HalfCycleMode(data)
        except ValueError:
            return HalfCycleMode.FULL

    def assume_standard_csv(self) -> bool:
        return self._csv_cb.isChecked()

    def apply_cycle_correction(self, spec_index: int, new_cycle: int) -> None:
        """
        Wird vom Hauptfenster aufgerufen, wenn die Reader-Funktion auf einen
        anderen Zyklus zurückgefallen ist — aktualisiert das entsprechende
        Eingabefeld.
        """
        for row in self._rows:
            if row.index == spec_index:
                row.set_cycle_number(new_cycle)
                break

    def rows(self) -> List[CVRowWidget]:
        return list(self._rows)
