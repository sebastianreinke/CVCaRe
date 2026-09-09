"""
Custom exception types used across the CVCaRe package.

This file is part of CVCaRe. It is a cyclic voltammogram analysis tool that
enables you to calculate capacitance and resistance from capacitive cyclic
voltammograms.

Copyright (C) 2022-2026 Sebastian Reinke

CVCaRe is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

CVCaRe is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
CVCaRe. If not, see <https://www.gnu.org/licenses/>.
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
