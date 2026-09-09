"""
cvcare.gui
==========

Grafische Benutzeroberfläche für CVCaRe, basierend auf PySide6 und
pyqtgraph. Diese ersetzt die FreeSimpleGUI-basierte ``gui-4.py`` aus dem
Vorgängerstand und ist sauber von der Domain-Logik in
:mod:`cvcare.core`, :mod:`cvcare.fileio` und :mod:`cvcare.dataset` getrennt.

Einstiegspunkt: ``python -m cvcare`` (siehe :mod:`cvcare.__main__`) oder
direkt ``from cvcare.gui.app import run; run()``.
"""

from cvcare.gui.app import CVCaReWindow, run

__all__ = ["CVCaReWindow", "run"]
