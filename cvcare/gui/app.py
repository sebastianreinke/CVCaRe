"""
This file is part of CVCaRe.
Copyright (C) 2022-2026 Sebastian Reinke
Licensed under the GNU General Public License v3 or later.

Application main window, GUI layout, menu actions, and tab coordination.
"""
from __future__ import annotations

import sys
from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction, QActionGroup, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from cvcare import __version__
from cvcare.dataset import Dataset
from cvcare.gui.tabs import (
    BiasTab,
    CapacitanceTab,
    CaReTab,
    ExportTab,
    HalfCVTab,
    IntegralTab,
    PlotTab,
)
from cvcare.gui.theme import (
    ThemeMode,
    apply_theme,
    get_saved_theme,
    logo_pixmap_for,
    save_theme,
)
from cvcare.gui.widgets import LogoWidget, SidebarWidget


class CVCaReWindow(QMainWindow):

    _SHOW_BIAS_TAB = False

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"CVCaRe {__version__}")
        self.resize(1540, 880)

        # Modell
        self._dataset: Optional[Dataset] = None

        # Theme initial laden
        self._theme: ThemeMode = get_saved_theme()

        # ---- Linke Spalte: Logo + Sidebar ---- #
        self._logo = LogoWidget()
        self._sidebar = SidebarWidget()
        self._sidebar.setMinimumWidth(560)
        self._sidebar.load_requested.connect(self._on_load_requested)
        self._sidebar.active_changed.connect(self._on_row_active_changed)

        left_col = QWidget()
        left_layout = QVBoxLayout(left_col)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)
        left_layout.addWidget(self._logo)
        left_layout.addWidget(self._sidebar, 1)

        # ---- Rechte Spalte: Plot oben, Funktions-Tabs unten ---- #
        self._plot_tab = PlotTab()
        self._cap_tab = CapacitanceTab(
            get_dataset=self._get_dataset, sidebar=self._sidebar
        )
        self._care_tab = CaReTab(
            get_dataset=self._get_dataset,
            sidebar=self._sidebar,
            plot_tab=self._plot_tab,
        )
        self._integral_tab = IntegralTab(
            get_dataset=self._get_dataset, plot_tab=self._plot_tab
        )
        self._halfcv_tab = HalfCVTab(
            get_dataset=self._get_dataset,
            sidebar=self._sidebar,
            plot_tab=self._plot_tab,
        )
        self._bias_tab = BiasTab(get_dataset=self._get_dataset)
        self._export_tab = ExportTab(get_dataset=self._get_dataset)

        self._tabs = QTabWidget()
        self._tabs.setTabPosition(QTabWidget.West)
        self._tabs.addTab(self._cap_tab, "Capacitance")
        self._tabs.addTab(self._care_tab, "CaRe")
        self._tabs.addTab(self._integral_tab, "Integral")
        self._tabs.addTab(self._halfcv_tab, "HalfCV")
        if self._SHOW_BIAS_TAB:
            self._tabs.addTab(self._bias_tab, "Bias")
        self._tabs.addTab(self._export_tab, "Export")

        right_split = QSplitter(Qt.Vertical)
        right_split.addWidget(self._plot_tab)
        right_split.addWidget(self._tabs)
        right_split.setStretchFactor(0, 3)
        right_split.setStretchFactor(1, 2)
        right_split.setSizes([520, 320])

        # ---- Gesamt-Layout ---- #
        main_split = QSplitter(Qt.Horizontal)
        main_split.addWidget(left_col)
        main_split.addWidget(right_split)
        main_split.setStretchFactor(0, 0)
        main_split.setStretchFactor(1, 1)
        main_split.setSizes([620, 920])
        self.setCentralWidget(main_split)

        # Menue
        self._build_menu()

        # Statusleiste
        self.setStatusBar(QStatusBar(self))
        self.statusBar().showMessage(
            "Ready. Select files in the sidebar and click 'Load'."
        )

        # Initiale Theme-Anwendung (auf bereits konstruierte Widgets).
        self._refresh_theme_dependent_widgets()

    # --------------------------------------------------------------- #
    # Modell-Zugriff
    # --------------------------------------------------------------- #

    def _get_dataset(self) -> Optional[Dataset]:
        return self._dataset

    # --------------------------------------------------------------- #
    # Menue
    # --------------------------------------------------------------- #

    def _build_menu(self) -> None:
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")
        view_menu = menubar.addMenu("&View")
        about_menu = menubar.addMenu("&Help")

        act_quit = QAction("&Quit", self)
        act_quit.setShortcut("Ctrl+Q")
        act_quit.triggered.connect(self.close)
        file_menu.addAction(act_quit)

        # Theme selection
        theme_menu = view_menu.addMenu("&Theme")
        self._theme_group = QActionGroup(self)
        self._theme_group.setExclusive(True)
        self._act_light = QAction("&Light", self, checkable=True)
        self._act_dark = QAction("&Dark", self, checkable=True)
        self._theme_group.addAction(self._act_light)
        self._theme_group.addAction(self._act_dark)
        theme_menu.addAction(self._act_light)
        theme_menu.addAction(self._act_dark)
        if self._theme is ThemeMode.DARK:
            self._act_dark.setChecked(True)
        else:
            self._act_light.setChecked(True)
        self._act_light.triggered.connect(lambda: self._switch_theme(ThemeMode.LIGHT))
        self._act_dark.triggered.connect(lambda: self._switch_theme(ThemeMode.DARK))

        act_about = QAction("&About CVCaRe", self)
        act_about.triggered.connect(self._show_about)
        about_menu.addAction(act_about)

    def _switch_theme(self, mode: ThemeMode) -> None:
        self._theme = mode
        save_theme(mode)
        app = QApplication.instance()
        if app is not None:
            apply_theme(app, mode)
        self._refresh_theme_dependent_widgets()

    def _refresh_theme_dependent_widgets(self) -> None:
        self._logo.apply_theme(self._theme)
        self._plot_tab.apply_theme(self._theme)

    def _show_about(self) -> None:
        dlg = QDialog(self)
        dlg.setWindowTitle("About CVCaRe")

        # Logo im About-Dialog -- decoded from embedded base64 data.
        logo_label = QLabel()
        logo_label.setAlignment(Qt.AlignCenter)
        pix = logo_pixmap_for(self._theme)
        if pix is not None and not pix.isNull():
            logo_label.setPixmap(
                pix.scaledToHeight(72, Qt.SmoothTransformation)
            )

        text_label = QLabel(
            f"<h3>CVCaRe {__version__}</h3>"
            "<p>Cyclic voltammogram analysis tool.</p>"
            "<p>Original author: Sebastian Reinke. Released under GPLv3.</p>"
            "<p>GUI: PySide6 (LGPL) and pyqtgraph (MIT).</p>"
        )
        text_label.setTextFormat(Qt.RichText)
        text_label.setWordWrap(True)

        btns = QDialogButtonBox(QDialogButtonBox.Ok)
        btns.accepted.connect(dlg.accept)

        lay = QVBoxLayout(dlg)
        lay.setContentsMargins(16, 16, 16, 12)
        lay.setSpacing(10)
        if not logo_label.pixmap().isNull():
            lay.addWidget(logo_label)
        lay.addWidget(text_label)
        lay.addWidget(btns)
        dlg.resize(420, 260)
        dlg.exec()

    # --------------------------------------------------------------- #
    # Slots
    # --------------------------------------------------------------- #

    def _on_load_requested(self) -> None:
        specs = self._sidebar.collect_specs()
        if not specs:
            QMessageBox.information(
                self, "No files",
                "Please select at least one CV file in the sidebar first.",
            )
            return

        mode = self._sidebar.halfcycle_mode()
        csv_standard = self._sidebar.assume_standard_csv()

        try:
            dataset = Dataset(
                specs=specs,
                halfcycle_mode=mode,
                assume_standard_csv=csv_standard,
            )
        except Exception as e:  # noqa: BLE001
            QMessageBox.critical(self, "Load error", f"{type(e).__name__}: {e}")
            return

        # Push corrected cycle numbers back to the sidebar
        n_ok = 0
        for result in dataset.load_results:
            if result.corrected_cycle_number is not None:
                self._sidebar.apply_cycle_correction(
                    result.spec_index, result.corrected_cycle_number
                )
            if result.success:
                n_ok += 1
            elif result.error is not None:
                print(
                    f"[CVCaRe] Spec {result.spec_index} failed: "
                    f"{type(result.error).__name__}: {result.error}"
                )

        # Apply scan rates from the sidebar
        scanrates = self._sidebar.collect_scanrates()
        if scanrates:
            dataset.set_scanrates(scanrates)

        self._dataset = dataset
        self._plot_tab.show_cvs(dataset.contents)

        n_total = len(dataset.load_results)
        n_corr = sum(
            1 for r in dataset.load_results if r.corrected_cycle_number is not None
        )
        msg = f"Loaded: {n_ok}/{n_total}."
        if n_corr:
            msg += f" {n_corr} cycle entries were auto-corrected."
        self.statusBar().showMessage(msg)

    def _on_row_active_changed(self, index: int, active: bool) -> None:
        if self._dataset is None:
            return
        self._dataset.set_activity_of_element(index, active)
        self._plot_tab.refresh()


# --------------------------------------------------------------------------- #
# Entry-Point
# --------------------------------------------------------------------------- #


def run(argv: Optional[list[str]] = None) -> int:
    if argv is None:
        argv = sys.argv
    app = QApplication.instance() or QApplication(argv)

    initial_theme = get_saved_theme()
    apply_theme(app, initial_theme)

    window = CVCaReWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(run())
