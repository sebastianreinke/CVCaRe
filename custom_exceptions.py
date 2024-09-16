"""
This file is part of CVCaRe. It is a cyclic voltammogram analysis tool that enables you to calculate capacitance and resistance from capacitive cyclic voltammograms.
    Copyright (C) 2022-2024 Sebastian Reinke

CVCaRe is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

CVCaRe is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with CVCaRe. If not, see <https://www.gnu.org/licenses/>.

"""

class ScanrateExistsError(Exception):
    def __init__(self, message):
        # Call the base class constructor with the parameters it needs
        super().__init__(message)


class NotEnoughCVsToFitError(Exception):
    def __init__(self, message):
        # Call the base class constructor with the parameters it needs
        super().__init__(message)


class InconsistentCapacitanceEstimationError(Exception):
    def __init__(self, message):
        # Call the base class constructor with the parameters it needs
        super().__init__(message)


class VoltageBoundsOutsideCVError(Exception):
    def __init__(self, message):
        super().__init__(message)


class FunctionNotImplementedError(Exception):
    def __init__(self, message):
        super().__init__(message)


class CycleIndexOutOfBoundsError(Exception):
    def __init__(self, message, requested_cycle, highest_available_cycle, cycle_data):
        super().__init__(message)
        self.requested_cycle = requested_cycle
        self.highest_available_cycle = highest_available_cycle
        self.cycle_data = cycle_data


class NoCycleInformationError(Exception):
    def __init__(self, message):
        # Call the base class constructor with the parameters it needs
        super().__init__(message)


class NoScanrateDefinedError(Exception):
    def __init__(self, message):
        # Call the base class constructor with the parameters it needs
        super().__init__(message)


class UnknownMethodError(Exception):
    def __init__(self, message):
        # Call the base class constructor with the parameters it needs
        super().__init__(message)
