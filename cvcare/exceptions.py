"""
This file is part of CVCaRe.
Copyright (C) 2022-2026 Sebastian Reinke
Licensed under the GNU General Public License v3 or later.

Custom exception types used for recoverable CVCaRe loading and analysis errors.
"""
from __future__ import annotations

import numpy as np


class ScanrateExistsError(Exception):
    """Raised when a scan rate is requested but not set, or set in conflict."""


class NotEnoughCVsToFitError(Exception):
    """Raised when fewer than two CVs are available for a linear fit."""


class InconsistentCapacitanceEstimationError(Exception):
    """Raised when capacitance estimates from different methods disagree."""


class VoltageBoundsOutsideCVError(Exception):
    """Raised when integration bounds lie outside the CV's potential window."""


class FunctionNotImplementedError(Exception):
    """Raised when an operation is not available for a given CV type."""


class CycleIndexOutOfBoundsError(Exception):
    """Raised when a requested cycle number exceeds what is available.

    Carries fallback data so callers can recover gracefully and inform the user
    which cycle was used instead.
    """

    def __init__(
        self,
        message: str,
        requested_cycle: int,
        highest_available_cycle: int,
        cycle_data: "np.ndarray",
    ) -> None:
        super().__init__(message)
        self.requested_cycle = requested_cycle
        self.highest_available_cycle = highest_available_cycle
        self.cycle_data = cycle_data


class NoCycleInformationError(Exception):
    """Raised when the input file contains no cycle column."""


class NoScanrateDefinedError(Exception):
    """Raised when a calculation requires a scan rate but none is set."""


class UnknownMethodError(Exception):
    """Raised when an unrecognised method string is passed to a dispatcher."""


class NotEnoughDataError(Exception):
    """Raised when a CV/half-cycle has too few data points for the requested operation."""
