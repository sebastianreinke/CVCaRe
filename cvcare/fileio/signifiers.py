"""
This file is part of CVCaRe.
Copyright (C) 2022-2026 Sebastian Reinke
Licensed under the GNU General Public License v3 or later.

Recognized current and voltage column-header strings.
"""
from __future__ import annotations

# Substrings that, if found in a header cell, mark that column as the current.
current_signifiers: list[str] = [
    "<I>",
    "Current",
    "current",
    "I/",
    "I /",
    "I(A)",
]

# Substrings that, if found in a header cell, mark that column as the voltage.
voltage_signifiers: list[str] = [
    "Ewe",
    "Potential",
    "E/",
    "Voltage 1",
    "E /",
    "Voltage",
    "E(V)",
    "Working Electrode (V)",
]
