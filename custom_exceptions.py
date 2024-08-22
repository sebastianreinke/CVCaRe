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
