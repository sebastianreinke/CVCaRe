"""
cvcare.gui.widgets.logo
=======================

Kleines QLabel, das je nach aktivem Theme ``Images/CVCaRe_light.png`` oder
``Images/CVCaRe_dark.png`` anzeigt. Faellt eine Datei aus, wird ein textueller
Platzhalter gezeigt, damit die App auch ohne mitgelieferte Assets startet.
"""

from __future__ import annotations

from typing import Optional

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QWidget

from cvcare.gui.theme import ThemeMode, logo_pixmap_for, palette_for


class LogoWidget(QFrame):
    """Logo-Container. Nutzt :meth:`apply_theme` zum Umschalten."""

    def __init__(self, parent: Optional[QWidget] = None, max_height: int = 56) -> None:
        super().__init__(parent)
        self.setObjectName("LogoFrame")
        self.setFrameShape(QFrame.NoFrame)
        self._max_height = max_height

        self._label = QLabel()
        self._label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        self._label.setMinimumHeight(max_height)
        self._label.setMaximumHeight(max_height + 8)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(0)
        layout.addWidget(self._label)
        layout.addStretch(1)

    def apply_theme(self, mode: ThemeMode) -> None:
        pix = logo_pixmap_for(mode)
        if pix is not None and not pix.isNull():
            scaled = pix.scaledToHeight(self._max_height, Qt.SmoothTransformation)
            self._label.clear()
            self._label.setPixmap(scaled)
            self._label.repaint()
            return
        # Fallback: textual mark in the accent colour.
        p = palette_for(mode)
        self._label.clear()
        self._label.setPixmap(QPixmap())
        self._label.setText(
            f"<span style='font-size:20pt; font-weight:700; color:{p['accent']};'>"
            "CV<span style='color:" + p['text'] + "'>CaRe</span></span>"
        )
        self._label.repaint()
