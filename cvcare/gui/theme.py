"""
cvcare.gui.theme
================

Light-/Dark-Theme-Verwaltung fuer die GUI.

Das Theme wirkt auf zwei Ebenen:

1. Die Qt-Anwendung selbst (Palette + Stylesheet) -- so wechseln Fenster,
   Buttons, Eingabefelder ihre Farbe.
2. Den pyqtgraph-Plot. pyqtgraph haelt globale Defaults (foreground /
   background), die wir hier setzen, sowie pro-Achse Einstellungen, die
   :class:`cvcare.gui.tabs.plot_tab.PlotTab` beim Theme-Wechsel mitliest.

Der gewaehlte Modus wird via :class:`QSettings` unter
``CVCaRe/Appearance/Theme`` persistiert.
"""

from __future__ import annotations

from enum import Enum

import pyqtgraph as pg
from PySide6.QtCore import QSettings
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication


class ThemeMode(str, Enum):
    LIGHT = "light"
    DARK = "dark"


# --------------------------------------------------------------------------- #
# Farbtabelle. Die Werte wurden so gewaehlt, dass sie zum Magenta-Akzent des
# Logos passen.
# --------------------------------------------------------------------------- #
_PALETTES = {
    ThemeMode.LIGHT: {
        "bg":          "#fafafa",
        "panel":       "#ffffff",
        "border":      "#dcdcdc",
        "text":        "#1a1a1a",
        "muted":       "#5e5e5e",
        "accent":      "#c8326b",
        "accent_text": "#ffffff",
        "plot_bg":     "#ffffff",
        "plot_fg":     "#1a1a1a",
        "plot_grid":   0.30,
    },
    ThemeMode.DARK: {
        "bg":          "#1f2125",
        "panel":       "#2a2d33",
        "border":      "#3a3d44",
        "text":        "#ececec",
        "muted":       "#a0a4ac",
        "accent":      "#e85c95",
        "accent_text": "#1a1a1a",
        "plot_bg":     "#1f2125",
        "plot_fg":     "#ececec",
        "plot_grid":   0.40,
    },
}


_SETTINGS_KEY = "Appearance/Theme"


def get_saved_theme(default: ThemeMode = ThemeMode.DARK) -> ThemeMode:
    settings = QSettings("CVCaRe", "CVCaRe")
    raw = settings.value(_SETTINGS_KEY, default.value)
    try:
        return ThemeMode(str(raw))
    except ValueError:
        return default


def save_theme(mode: ThemeMode) -> None:
    settings = QSettings("CVCaRe", "CVCaRe")
    settings.setValue(_SETTINGS_KEY, mode.value)


# --------------------------------------------------------------------------- #
# Anwendung des Themes
# --------------------------------------------------------------------------- #


def palette_for(mode: ThemeMode) -> dict:
    return _PALETTES[mode]


def apply_theme(app: QApplication, mode: ThemeMode) -> None:
    """Setzt Palette + Stylesheet + pyqtgraph-Defaults global."""
    p = _PALETTES[mode]

    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(p["bg"]))
    palette.setColor(QPalette.WindowText, QColor(p["text"]))
    palette.setColor(QPalette.Base, QColor(p["panel"]))
    palette.setColor(QPalette.AlternateBase, QColor(p["bg"]))
    palette.setColor(QPalette.Text, QColor(p["text"]))
    palette.setColor(QPalette.Button, QColor(p["panel"]))
    palette.setColor(QPalette.ButtonText, QColor(p["text"]))
    palette.setColor(QPalette.Highlight, QColor(p["accent"]))
    palette.setColor(QPalette.HighlightedText, QColor(p["accent_text"]))
    palette.setColor(QPalette.ToolTipBase, QColor(p["panel"]))
    palette.setColor(QPalette.ToolTipText, QColor(p["text"]))
    palette.setColor(QPalette.PlaceholderText, QColor(p["muted"]))
    app.setPalette(palette)

    # Stylesheet erweitert die Palette dort, wo Qt Defaults nicht reicht.
    app.setStyleSheet(
        f"""
        QMainWindow, QWidget {{
            background-color: {p['bg']};
            color: {p['text']};
        }}
        QFrame#Sidebar, QFrame#LogoFrame {{
            background-color: {p['panel']};
            border: 1px solid {p['border']};
            border-radius: 6px;
        }}
        QGroupBox {{
            background-color: {p['panel']};
            border: 1px solid {p['border']};
            border-radius: 6px;
            margin-top: 12px;
            padding-top: 8px;
        }}
        QGroupBox::title {{
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 6px;
            color: {p['muted']};
        }}
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QTextEdit {{
            background-color: {p['panel']};
            color: {p['text']};
            border: 1px solid {p['border']};
            border-radius: 4px;
            padding: 2px 4px;
            selection-background-color: {p['accent']};
            selection-color: {p['accent_text']};
        }}
        QLineEdit:read-only {{
            background-color: {p['bg']};
            color: {p['muted']};
        }}
        QSpinBox::up-button, QDoubleSpinBox::up-button {{
            subcontrol-origin: border;
            subcontrol-position: top right;
            width: 16px;
            border-left: 1px solid {p['border']};
            border-bottom: 1px solid {p['border']};
            border-top-right-radius: 4px;
            background-color: {p['panel']};
        }}
        QSpinBox::down-button, QDoubleSpinBox::down-button {{
            subcontrol-origin: border;
            subcontrol-position: bottom right;
            width: 16px;
            border-left: 1px solid {p['border']};
            border-bottom-right-radius: 4px;
            background-color: {p['panel']};
        }}
        QSpinBox::up-button:hover, QSpinBox::down-button:hover,
        QDoubleSpinBox::up-button:hover, QDoubleSpinBox::down-button:hover {{
            background-color: {p['accent']};
        }}
        QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
            width: 0px;
            height: 0px;
            border-left: 3px solid transparent;
            border-right: 3px solid transparent;
            border-bottom: 5px solid {p['text']};
        }}
        QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
            width: 0px;
            height: 0px;
            border-left: 3px solid transparent;
            border-right: 3px solid transparent;
            border-top: 5px solid {p['text']};
        }}
        QPushButton, QToolButton {{
            background-color: {p['panel']};
            color: {p['text']};
            border: 1px solid {p['border']};
            border-radius: 4px;
            padding: 4px 12px;
        }}
        QPushButton:hover, QToolButton:hover {{
            border-color: {p['accent']};
        }}
        QPushButton:default {{
            border-color: {p['accent']};
            color: {p['accent']};
            font-weight: 600;
        }}
        QTabWidget::pane {{
            border: 1px solid {p['border']};
            border-radius: 6px;
            background-color: {p['panel']};
        }}
        QTabBar::tab {{
            background: transparent;
            color: {p['muted']};
            padding: 6px 12px;
            border: 1px solid transparent;
            border-radius: 4px;
            margin: 2px;
        }}
        QTabBar::tab:selected {{
            background: {p['panel']};
            color: {p['accent']};
            border: 1px solid {p['border']};
            font-weight: 600;
        }}
        QTabBar::tab:hover:!selected {{
            color: {p['text']};
        }}
        QScrollArea {{
            background-color: {p['bg']};
            border: none;
        }}
        QMenuBar {{
            background-color: {p['bg']};
            color: {p['text']};
        }}
        QMenuBar::item:selected {{
            background: {p['accent']};
            color: {p['accent_text']};
        }}
        QMenu {{
            background-color: {p['panel']};
            color: {p['text']};
            border: 1px solid {p['border']};
        }}
        QMenu::item:selected {{
            background: {p['accent']};
            color: {p['accent_text']};
        }}
        QStatusBar {{
            background-color: {p['panel']};
            color: {p['muted']};
        }}
        QCheckBox {{
            color: {p['text']};
        }}
        QSplitter::handle {{
            background-color: {p['border']};
        }}
        """
    )

    # pyqtgraph-Defaults muessen vor dem Bauen neuer PlotWidgets greifen.
    pg.setConfigOption("background", p["plot_bg"])
    pg.setConfigOption("foreground", p["plot_fg"])


# --------------------------------------------------------------------------- #
# Logo-Pfade
# --------------------------------------------------------------------------- #


def logo_pixmap_for(mode: ThemeMode):
    """Returns a QPixmap decoded from embedded base64 data, or None."""
    from cvcare.gui.embedded_images import get_logo_pixmap
    return get_logo_pixmap(is_dark=(mode is ThemeMode.DARK))
