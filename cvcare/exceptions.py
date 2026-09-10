"""
This file is part of CVCaRe.
Copyright (C) 2022-2026 Sebastian Reinke
Licensed under the GNU General Public License v3 or later.

Custom exception types used across the CVCaRe package. These exceptions
let the GUI and command-line callers distinguish recoverable problems in
file parsing, cycle selection, and numerical analysis from unexpected
application errors.
"""

from __future__ import annotations

import numpy as np


class ScanrateExistsError(Exception):
    """Raised when a requested scan rate is absent or conflicts with an existing value."""


class NotEnoughCVsToFitError(Exception):
    """Raised when fewer than two CVs are available for a linear fit."""


class InconsistentCapacitanceEstimationError(Exception):
    """Raised when capacitance estimates from different methods disagree."""


class VoltageBoundsOutsideCVError(Exception):
    """Raised when integration bounds lie outside the CV potential window."""


class FunctionNotImplementedError(Exception):
    """Raised when an operation is unavailable for the selected CV type."""


class CycleIndexOutOfBoundsError(Exception):
    """
    Raised when a requested cycle is not present in the source data.

    The exception includes the highest available cycle and its data, allowing
    the caller to fall back gracefully and tell the user what was used.
    """

    def __init__(self, message: str, requested_cycle: int,
                 highest_available_cycle: int, cycle_data: "np.ndarray") -> None:
        super().__init__(message)
        self.requested_cycle = requested_cycle
        self.highest_available_cycle = highest_available_cycle
        self.cycle_data = cycle_data


class NoCycleInformationError(Exception):
    """Raised when an input file contains no explicit cycle-number column."""


class NoScanrateDefinedError(Exception):
    """Raised when a calculation needs a scan rate but none has been supplied."""


class UnknownMethodError(Exception):
    """Raised when an unrecognised calculation-method identifier is passed."""


class NotEnoughDataError(Exception):
    """Raised when a CV or half-cycle has too few points for an operation."""
