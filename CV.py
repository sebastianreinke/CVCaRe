from functools import partial

import numpy as np
import quantities
import scipy as sc
from quantities.quantity import Quantity
from scipy.optimize import minimize_scalar, minimize
from scipy import signal as sgnl
import quantities as pq
from abc import ABCMeta, abstractmethod
from custom_exceptions import ScanrateExistsError, VoltageBoundsOutsideCVError, NoScanrateDefinedError
from helper_functions import time_this_function


# from numba import jit


def get_positive_and_negative_voltage_peaks(full_dataset):
    direction_upwards = True

    # Mode switch for low-resolution datasets
    is_low_resolution = False
    if len(full_dataset) < 200:
        is_low_resolution = True

    # Truncation indices - choose as necessary. The beginning and end of CV is often noisy, or filled with artefacts,
    # and how far to cut around them is a tradeoff. Empirically, 40 - 10 works well for highly sampled CVs.
    # Note: truncate must always be >= than the lookahead, otherwise it triggers an IndexOutOfBoundsException.
    truncate = (40, 10)

    # How far to check around a peak to detect it
    lookahead = 10
    if is_low_resolution:
        truncate = (5, 5)
        lookahead = 3

    if full_dataset[truncate[0]][0] < full_dataset[0][0]:
        direction_upwards = False
    positive_peaks = []
    negative_peaks = []
    # iterate over all candidate center-elements in a len-3 comparison filter => exclude first and last element
    for index in range(truncate[0], len(full_dataset) - truncate[1]):
        if full_dataset[index - 1, 0] < full_dataset[index, 0] > full_dataset[index + 1, 0] and \
                full_dataset[index - lookahead, 0] < full_dataset[index, 0] > full_dataset[index + lookahead, 0]:
            positive_peaks.append(index)
        if full_dataset[index - 1, 0] > full_dataset[index, 0] < full_dataset[index + 1, 0] and \
                full_dataset[index - lookahead, 0] > full_dataset[index, 0] < full_dataset[index + lookahead, 0]:
            negative_peaks.append(index)
    # positive_peaks = sgnl.find_peaks(full_dataset[:, 0])
    # negative_peaks = sgnl.find_peaks(-1*full_dataset[:, 0])
    # positive_peaks = positive_peaks[0].tolist()
    # negative_peaks = negative_peaks[0].tolist()
    print(direction_upwards)
    print(positive_peaks)
    print(negative_peaks)
    if direction_upwards:
        negative_peaks.insert(0, 0)
        # case: CV has 3 segments
        if len(negative_peaks) == 2:
            positive_peaks.append(len(full_dataset) - 1)
        # case: LSV/1 segment CV
        elif len(negative_peaks) == 1 and len(positive_peaks) == 0:
            positive_peaks.append(len(full_dataset) - 1)
        # case: CV has 2 segments
        else:
            negative_peaks.append(len(full_dataset) - 1)
    else:
        positive_peaks.insert(0, 0)
        # case: 3-segment CV
        if len(positive_peaks) == 2:
            negative_peaks.append(len(full_dataset) - 1)
        # case: LSV/1 segment CV
        elif len(positive_peaks) == 1 and len(negative_peaks) == 0:
            negative_peaks.append(len(full_dataset) - 1)
        # case: CV has 2 segments
        else:
            positive_peaks.append(len(full_dataset) - 1)

    return direction_upwards, negative_peaks, positive_peaks


# the abstract base class for all CV types
class CV(metaclass=ABCMeta):
    index: int
    dataset: np.array
    eval_voltage: float
    source: str
    scanrate: float = None
    active: bool
    default_filtered: bool
    unit_voltage: pq.Quantity
    unit_current: pq.Quantity

    @abstractmethod
    def get_index(self):
        pass

    @abstractmethod
    def get_dataset(self):
        pass

    @abstractmethod
    def convert_quantities(self, unit_voltage, unit_current):
        pass

    @abstractmethod
    def get_current_at_voltage(self, voltage=None, current_dimension=None):
        pass

    @abstractmethod
    def set_active(self, is_active: bool):
        pass

    @abstractmethod
    def is_active(self):
        pass

    def get_cycle_nr(self):
        pass

    @abstractmethod
    def set_scanrate(self, scanrate: float, dimension: quantities.Quantity):
        pass

    @abstractmethod
    def get_scanrate(self, dimension: quantities.Quantity):
        pass

    @abstractmethod
    def get_eval_voltage(self):
        pass

    @abstractmethod
    def collapse_to_writable_list(self, filtered: bool = False, comment: str = None):
        # it is crucial to match the length of the meta-params between half-cycles and full cycles, so they can be
        # meaningfully exported together!
        pass

    @abstractmethod
    def get_type(self):
        return self.__class__

    @abstractmethod
    def set_default_filtered(self, filtered: bool):
        self.default_filtered = filtered

    @abstractmethod
    def get_default_filtered(self):
        return self.default_filtered


# The henceforth defined CV classes handle, if necessary, particular types of CV data. Internal calculations are located
# inside the class and return only results. get_index function must be implemented to allow handling in Dataset class.


# Class that
# - contains a single full CV set,
# - performs internal calculations on the CV, and
# - gives the relevant data and metadata when requested
#
# Note: This is called with a Dataset in the dimensions that are specified with unit_voltage, unit_current at creation
# Unit conversion is performed in-place (and we pray for no conversion glitches).
class FullCV(CV):
    """
    Represents a single full cyclic voltammetry (CV) dataset and provides methods for internal calculations and data retrieval.

    The `FullCV` class is designed to handle and process a full cycle CV dataset. It allows for internal calculations such as
    creating a difference landscape, converting data units, calculating capacitance, and analyzing the CV for distortion
    parameters. The class stores the relevant data and metadata, and provides various methods to access or manipulate these
    data.

    Attributes:
        index (int):
            The index of this CV within a larger dataset.

        dataset (np.array):
            A 2D NumPy array containing the voltage and current data for this CV. Note: This dataset is unit-free.

        eval_voltage (float):
            The voltage at which the current evaluation is performed.

        difference_landscape (list):
            A list of current differences between the forward and reverse scans at corresponding voltage points.

        cycle_nr (int):
            The cycle number associated with this CV.

        source (str):
            The source file name from which this CV was loaded.

        scanrate (float):
            The scan rate at which this CV was measured, in units defined by `unit_scanrate`. Defaults to None if not set.

        active (bool):
            Indicates whether this CV is active for analysis. Defaults to True.

        default_filtered (bool):
            Indicates whether the default filtered dataset should be used for calculations. Defaults to False.

        unit_voltage (quantities.Quantity):
            The unit of voltage used in the dataset, provided during initialization.

        unit_current (quantities.Quantity):
            The unit of current used in the dataset, provided during initialization.

        unit_scanrate (quantities.Quantity or None):
            The unit of scan rate, set when the scan rate is defined.

    Methods:
        __init__(self, source: str, cycle_nr: int, index: int, dataset: list, eval_voltage, unit_voltage, unit_current):
            Initializes the FullCV object with the provided dataset, metadata, and units. Creates the difference landscape.
            Sets the initial values for various attributes like `index`, `dataset`, `eval_voltage`, `cycle_nr`,
            `unit_voltage`, `unit_current`, and others.

        get_eval_voltage(self):
            Returns the evaluation voltage for this CV.

        get_index(self) -> int:
            Returns the index of this CV.

        get_dataset(self):
            Returns the dataset as a NumPy array.

        get_type(self):
            Placeholder method that calls the superclass method.

        get_cycle_nr(self):
            Returns the cycle number of this CV.

        set_active(self, is_active: bool):
            Sets the active status of this CV.

        is_active(self):
            Returns whether this CV is active for analysis.

        get_source(self):
            Returns the source file name of this CV.

        set_scanrate(self, scanrate: float, dimension: quantities.Quantity):
            Sets the scan rate and its unit for this CV.

        get_scanrate(self, dimension: quantities.Quantity):
            Returns the scan rate in the specified units. Raises an error if the scan rate is not set.

        set_default_filtered(self, filtered: bool):
            Sets the default filtering status of this CV.

        get_default_filtered(self):
            Returns the default filtering status of this CV.

        get_amplitude(self):
            Returns the voltage amplitude of this CV, calculated as half the difference between the maximum and minimum voltages.

        get_filtered_dataset(self):
            Returns the dataset after applying a Savitzky-Golay filter for smoothing.

        get_difference_landscape(self):
            Returns the difference landscape calculated for this CV.

        convert_quantities(self, unit_voltage, unit_current):
            Converts the voltage and current data in the dataset to the specified units.

        get_minmax_current(self):
            Returns the difference between the maximum and minimum currents in the dataset.

        get_current_at_voltage_in_halfcycle(self, half_cycle_select, voltage_position=None, current_dimension: pq.Quantity = None):
            Returns the current at the specified voltage within the selected half-cycle (anodic or cathodic).

        get_current_at_voltage(self, voltage_position=None, current_dimension: pq.Quantity = None, query_forward_backward_separately=False):
            Retrieves the current at a specified voltage position, with options to query forward and backward scans separately.

        create_difference_landscape(self):
            Creates and stores the difference landscape between the forward and reverse scans in the dataset.

        integrate_one_direction(self, lower_voltage_bound: float, upper_voltage_bound: float, forward_direction: bool = True, filtered: bool = False, offset: float = None):
            Integrates the current over a specified voltage range in either the forward or reverse direction.

        register_eval_voltage(self):
            Determines and sets the evaluation voltage based on the maximum difference in the difference landscape.

        collapse_to_writable_list(self, filtered: bool = False, comment: str = None):
            Prepares the dataset and metadata as a list suitable for writing to a file.

        true_interfacial_scanrate(self, resistance):
            Calculates the true interfacial scan rate, accounting for the IR drop over the specified resistance.

        fill_factor(self):
            Calculates the fill factor, a measure of the CV's distortion, normalized to the potential window and current range.

        estimate_vertical_current_ratio(self):
            Estimates the vertical current ratio and distortion parameter based on the fill factor.

        distortion_param_evaluation(self):
            Evaluates the CV using an analytical method to determine capacitance, resistance, and distortion parameters.

        bias_analysis(self):
            Analyzes the CV for bias, comparing current ratios at specific points in the potential window with those expected from the fill factor.

        fit_cv_by_optimisation(self):
            Optimizes the resistance, capacitance, and offset parameters using an optimization algorithm based on the RC model.
    """
    index: int
    dataset: np.array
    eval_voltage: float
    difference_landscape: list
    cycle_nr: int
    source: str
    scanrate: float = None
    active: bool
    default_filtered: bool = False

    def __init__(self, source: str, cycle_nr: int, index: int, dataset: list, eval_voltage,
                 unit_voltage, unit_current):
        self.unit_scanrate = None
        self.source = source
        self.dataset = dataset
        self.cycle_nr = cycle_nr
        self.index = index
        self.unit_voltage = unit_voltage
        self.unit_current = unit_current
        self.create_difference_landscape()
        self.active = True

        if eval_voltage is None:
            self.register_eval_voltage()
        else:
            self.eval_voltage = eval_voltage
        # print(
        #     f"Registered Full CV {index}, unit voltage {unit_voltage}, unit current {unit_current}, eval voltage {self.eval_voltage}")

    def get_eval_voltage(self):
        return self.eval_voltage

    def get_index(self) -> int:
        return self.index

    def get_dataset(self):
        x = np.array(self.dataset)
        return x

    def get_type(self):
        super()

    def get_cycle_nr(self):
        return self.cycle_nr

    def set_active(self, is_active: bool):
        self.active = is_active

    def is_active(self):
        return self.active

    def get_source(self):
        return self.source

    def set_scanrate(self, scanrate: float, dimension: quantities.Quantity):
        self.unit_scanrate = dimension
        self.scanrate = scanrate

    def get_scanrate(self, dimension: quantities.Quantity):
        if self.scanrate is not None:
            scanrate = self.scanrate * self.unit_scanrate
            scanrate.units = dimension
            return scanrate
        else:
            raise ScanrateExistsError("This CV has no scanrate.")

    def set_default_filtered(self, filtered: bool):
        self.default_filtered = filtered

    def get_default_filtered(self):
        return self.default_filtered

    # return voltage amplitude, equal to half max(E)-min(E)
    def get_amplitude(self):
        return (max(self.dataset[:, 0]) - min(self.dataset[:, 0])) / 2 * self.unit_voltage

    def get_filtered_dataset(self):
        e_smooth = sgnl.savgol_filter(self.dataset[:, 0], window_length=11, polyorder=3, mode="nearest")
        i_smooth = sgnl.savgol_filter(self.dataset[:, 1], window_length=11, polyorder=3, mode="nearest")
        return np.stack((e_smooth, i_smooth), axis=1)

    def get_difference_landscape(self):
        return self.difference_landscape

    def convert_quantities(self, unit_voltage, unit_current):
        """
        Converts the voltage and current data in the dataset to the specified units.

        This method converts the voltage and current values in the cyclic voltammetry (CV) dataset to the specified units.
        It also updates the `eval_voltage` and the internal unit attributes (`unit_voltage` and `unit_current`) to match
        the new units. The conversion process is necessary when the dataset needs to be analyzed or compared using
        different units than those originally provided.

        Parameters:
            unit_voltage (pq.Quantity):
                The desired unit for the voltage data. This should be a `quantities` unit (e.g., `pq.V` for volts).

            unit_current (pq.Quantity):
                The desired unit for the current data. This should be a `quantities` unit (e.g., `pq.A` for amperes).

        Internal Operations:
            - The method first multiplies the voltage and current values by their respective current units to convert
              them into `quantities` objects.
            - It then assigns the new units to the voltage, `eval_voltage`, and current data.
            - The `.magnitude` attribute is used to extract the numerical values from the `quantities` objects after
              unit conversion.
            - The dataset is then updated with the converted voltage and current values.
            - Finally, the method updates the internal `unit_voltage` and `unit_current` attributes to reflect the new units.

        Notes:
            - This method modifies the state of the object by updating the dataset, `eval_voltage`, and unit attributes.
            - Ensure that the units provided are compatible with the data in the dataset to avoid any unit-related errors.

        Example Usage:
            ```python
            # Convert the dataset to millivolts and microamperes
            convert_quantities(pq.mV, pq.uA)
            ```

        """
        voltage = self.dataset[:, 0] * self.unit_voltage
        current = self.dataset[:, 1] * self.unit_current
        eval_voltage = self.eval_voltage * self.unit_voltage
        voltage.units = unit_voltage
        eval_voltage.units = unit_voltage
        current.units = unit_current
        # voltage /= unit_voltage
        # current /= unit_current
        # eval_voltage /= unit_voltage
        voltage = voltage.magnitude
        current = current.magnitude
        eval_voltage = eval_voltage.magnitude
        self.unit_voltage = unit_voltage
        self.unit_current = unit_current
        self.eval_voltage = eval_voltage
        self.dataset = np.stack((voltage, current), axis=1)
        # print(f"Quantity conversion called with voltage {unit_voltage}, current {unit_current}."
        #       f"eval voltage is {self.eval_voltage}. Internal unit voltage {self.unit_voltage}")

    def get_minmax_current(self):
        if self.default_filtered:
            data = self.get_filtered_dataset()
        else:
            data = self.get_dataset()
        return (max(data[:, 1]) - (min(data[:, 1]))) * self.unit_current

    def get_current_at_voltage_in_halfcycle(self, half_cycle_select, voltage_position=None,
                                            current_dimension: pq.Quantity = None):
        if voltage_position is None:
            voltage_position = self.eval_voltage
        else:
            voltage_position *= pq.V
            voltage_position.units = self.unit_voltage
            voltage_position /= self.unit_voltage
        if self.default_filtered:
            data = self.get_filtered_dataset()
        else:
            data = self.dataset

        # @TODO This here needs to be reworked.
        positive_closest = (np.abs(data[:, 0] - voltage_position)).argmin()
        negative_closest = (np.abs(np.delete(data, positive_closest, 0)[:, 0] - voltage_position)).argmin()
        if half_cycle_select == "anodic":
            current_at_position = min(data[positive_closest, 1], data[negative_closest, 1]) * self.unit_current
            if current_dimension is not None:
                current_at_position.units = current_dimension
            return current_at_position
        elif half_cycle_select == "cathodic":
            current_at_position = max(data[positive_closest, 1], data[negative_closest, 1]) * self.unit_current
            if current_dimension is not None:
                current_at_position.units = current_dimension
            return current_at_position
        else:
            print("Half-cycle-select parameter is invalid.")

    # Value_at_position function
    # Often, only the difference is relevant, which is the default return.
    # Note: It is assumed, that in the dataset, the forward and backward scan are sampled at the same rate, so that
    # excluding the first-closest match to the sought-after voltage, the next-closest match is in the opposite
    # half-cycle of the CV. If this assumption does not hold, faulty near-zero current differences will appear, and
    # the data should be evaluated for ringing and improper sampling.
    # Other options, such as interpolation of the data to prevent such missteps will run into the issue of how to
    # choose a proper interpolation function working reasonably well for the various CV-shapes possible, and verifying
    # these interpolations. This solution should cause the least headache, if the data is sufficiently highly and evenly
    # sampled.
    def get_current_at_voltage(self,
                               voltage_position=None,
                               current_dimension: pq.Quantity = None,
                               query_forward_backward_separately=False):
        """
        Retrieves the current at a specified voltage position in the cyclic voltammetry (CV) dataset.

        This method finds the current value corresponding to a specified voltage in the CV dataset. By default, it returns
        the difference in current between the forward and backward scans at the closest match to the specified voltage.
        The method assumes that the dataset is sampled at a reasonably equidistant rate in both the forward and backward
        scans. If this assumption does not hold, erroneous near-zero current differences may appear, indicating the need
        for further data analysis or interpolation.

        Parameters:
            voltage_position (float or None, optional):
                The voltage at which to find the current. If None, the method uses `self.eval_voltage`.
                The voltage should be provided as a scalar value, and the method will apply the appropriate unit internally.

            current_dimension (pq.Quantity, optional):
                The desired unit for the returned current. If specified, the current is returned in this unit.

            query_forward_backward_separately (bool, optional):
                If True, the method returns the current values separately for the forward and backward scans.
                If False (default), the method returns the absolute difference between these currents.

        Returns:
            current_at_position (pq.Quantity):
                The absolute difference in current between the forward and backward scans at the specified voltage.
                If `query_forward_backward_separately` is True, returns a tuple:
                    - max_curr (pq.Quantity): The maximum current (either forward or backward).
                    - min_curr (pq.Quantity): The minimum current (either forward or backward).

        Raises:
            NoScanrateDefinedError: If the scan rate (`self.scanrate`) is not defined.

        Notes:
            - The method deletes the first-closest match to the desired voltage from the search space and searches again for
              the next-closest match, assuming that the next match is from the opposite scan (forward or backward). This
              assumption holds if the data is sampled at a consistent rate.
            - If the voltage position is exactly between two identical potentials from both scans, the method may produce
              near-zero differences, potentially indicating improper sampling or "ringing" effects.
            - If `query_forward_backward_separately` is True, the method will return the individual current values for the
              forward and backward scans, which may be useful for more detailed analysis.

        """
        if voltage_position is None:
            voltage_position = self.eval_voltage
        else:
            voltage_position *= pq.V
            voltage_position.units = self.unit_voltage
            voltage_position /= self.unit_voltage

        if self.default_filtered:
            data = self.get_filtered_dataset()
        else:
            data = self.dataset

        # Approach: Find the closest match to the desired voltage (least difference). Then, delete the first found
        # match from the searchspace, and search again. The second value will be in the other current branch, if data
        # is reasonably equidistantly sampled. An extremely odd error is possible, if the desired voltage is exactly
        # inbetween two available potentials, that are exactly the same in both branches.
        # Conceivable is a solution that excludes indices exactly adjacent to the initial find from being found, and
        # a repeat of the calculation with a minor shift of the voltage. First solution is implemented here, but carries
        # increased error at the very, very edges of the potential window. Second solution comes with higher
        # computational cost, and just changes the condition under which the problem will occur to one that is less
        # likely in calculated data.
        positive_closest = (np.abs(data[:, 0] - voltage_position)).argmin()
        searchspace = np.delete(data, positive_closest, 0)
        while True:
            negative_closest = (np.abs(searchspace[:, 0] - voltage_position)).argmin()
            if negative_closest in np.arange(positive_closest-2, positive_closest+3):
                searchspace = np.delete(searchspace, negative_closest, 0)
            else:
                break
        current_at_position = np.abs(data[positive_closest, 1] - searchspace[negative_closest, 1]) * self.unit_current

        if not query_forward_backward_separately:
            if current_dimension is not None:
                current_at_position.units = current_dimension
            return current_at_position
        else:
            # return first the forward current, then backward current
            if current_dimension is not None:
                print(f"Pos index {positive_closest}, neg index {negative_closest}"
                      f"pos surround{data[positive_closest -2:positive_closest+2, 1]} neg surround {data[negative_closest -2:negative_closest+2, 1]}"
                      f"pos surround{data[positive_closest -2:positive_closest+2, 0]} neg surround {data[negative_closest -2:negative_closest+2, 0]}"
                      f"Fetch-current_Fn: Positive closest {[data[positive_closest, 1]]}, "
                      f"Negative closest: {data[negative_closest, 1]},"
                      f"min {np.min([data[positive_closest, 1], data[negative_closest, 1]])}")
                max_curr = np.max([data[positive_closest, 1], searchspace[negative_closest, 1]]) * self.unit_current
                max_curr.units = current_dimension
                min_curr = np.min([data[positive_closest, 1], searchspace[negative_closest, 1]]) * self.unit_current
                min_curr.units = current_dimension
                return max_curr, min_curr

            else:
                return (np.max([data[positive_closest, 1], searchspace[negative_closest, 1]]) * self.unit_current,
                        np.min([data[positive_closest, 1], searchspace[negative_closest, 1]]) * self.unit_current)

    # @time_this_function
    def create_difference_landscape(self):
        """
        Creates a difference landscape between the forward and reverse scans in the cyclic voltammetry (CV) dataset.

        This method calculates the absolute difference in current between corresponding voltage points in the forward and
        reverse scans of the CV data. The difference landscape is a list of these current differences, which can be used
        for further analysis of the CV behavior. The method is designed to handle both filtered and unfiltered datasets,
        depending on the `default_filtered` attribute.

        The calculation proceeds as follows:
        1. For each data point in the CV dataset, find the closest matching voltage point in the opposite scan direction.
        2. Calculate the absolute difference in current between the forward and reverse scans at these matched voltage points.
        3. Handle cases where the closest match might be a neighboring point, which could lead to incorrect differences by
           excluding these matches and searching for the next closest point.
        4. If an error occurs during this process (e.g., due to incomplete cycles or other data issues), the method falls
           back to a simpler calculation that may include nearest-neighbor matches, though this approach may compromise
           the accuracy of the difference landscape.

        The resulting difference landscape is stored in the `difference_landscape` attribute.

        :raises Exception:
            If an error occurs during the calculation, a warning is printed, and the method falls back to a less reliable
            calculation method. This may happen if the dataset has incomplete cycles or other issues that prevent accurate
            calculation of the current differences.

        :note:
            - It is assumed that the dataset is sampled at a reasonably equidistant rate in both the forward and reverse
              scans. If this assumption does not hold, the difference landscape may include inaccuracies due to
              nearest-neighbor matches.
            - The method currently does not implement safeguards against next-neighbor selection for all cases. A
              more robust solution might be to incorporate safeguards similar to those in the `get_current_at_voltage`
              method to prevent such issues.

        :warning:
            - If the fallback calculation method is used, the difference landscape should be evaluated with caution, as it
              may include inaccuracies due to nearest-neighbor matches.
        """
        if self.default_filtered:
            data = self.get_filtered_dataset()
        else:
            data = self.dataset

        difference_landscape = []
        try:
            for i in range(len(data)):
                direct_value = data[i, 1]
                searchspace = np.delete(data, i, 0)
                while True:
                    negative_closest = (np.abs(searchspace[:, 0] - data[i, 0])).argmin()
                    if negative_closest in np.arange(i - 1, i + 2):
                        searchspace = np.delete(searchspace, negative_closest, 0)
                    else:
                        break
                current_difference = np.abs(data[i, 1] - searchspace[negative_closest, 1]) # * self.unit_current
                # next_closest = (np.abs(np.delete(data, i, 0)[:, 0] - data[i, 0])).argmin()
                difference_landscape.append(current_difference)
            self.difference_landscape = difference_landscape
        except Exception as e:
            print(f"Warning: The current difference landscape between forward and reverse scan could not be calculated "
                  f"for CV {self.index}."
                  f"This may indicate data with incomplete cycles or other error. Current-difference based calculations"
                  f"may be compromised.")
            # Proceed with old calculation that may also select nearest-neighbour indices. Current difference landscape
            # data of this type should be evaluated with the utmost caution, if at all.
            for i in range(len(data)):
                direct_value = data[i, 1]
                next_closest = (np.abs(np.delete(data, i, 0)[:, 0] - data[i, 0])).argmin()

                # a minor correction, since the next-closest is found in an array that is shortened by deleting the
                # original index, thus all following indices are decreased by one compared to the correct index in the
                # original dataset of the indentical element within.
                if next_closest >= direct_value and next_closest + 1 < len(data):
                    next_closest += 1
                current_difference = np.abs(data[i, 1] - data[next_closest, 1])
                difference_landscape.append(current_difference)
            self.difference_landscape = difference_landscape

    def integrate_one_direction(self, lower_voltage_bound: float, upper_voltage_bound: float,
                                forward_direction: bool = True, filtered: bool = False, offset: float = None):
        def index_exists(list_, index):
            if 0 <= index <= len(list_):
                return True
            else:
                return False

        # obtain the interpolated function in the forward or reverse direction
        # first split the CV into the forward-backward directions, append them to a single array, and use
        # scipy.InterpolateUnivariateSpline to calculate the integral.
        # If desired, this could be changed to other interpolation modes.
        if not filtered:
            this_dataset_as_list = self.dataset
        else:
            this_dataset_as_list = self.get_filtered_dataset().tolist() * pq.dimensionless

        # ensure valid voltage boundaries
        if lower_voltage_bound < min(np.array(this_dataset_as_list)[:, 0]) or \
                upper_voltage_bound > max(np.array(this_dataset_as_list)[:, 0]):
            raise VoltageBoundsOutsideCVError("The integration bounds are outside the dataset!")

        # the peaks include the termination and the beginning of the cv as peaks (in negative_peaks if falling towards
        # the point and/or rising from it, positive_peaks if rising towards and falling from the point.
        # Sketching this out helps.
        begins_upwards, negative_peaks, positive_peaks = get_positive_and_negative_voltage_peaks(
            this_dataset_as_list)
        # print(positive_peaks)
        # print(negative_peaks)
        # split the CV into halfcycles along the voltage peaks
        sub_dataset_single_scan_direction = []

        if forward_direction:
            # TODO iterate over the positive and negative lists, and check if the counterpoint exists. concatenate.
            # TODO this fixes the 2-segment CV case
            if begins_upwards:
                for i in range(len(positive_peaks)):
                    if index_exists(negative_peaks, i):
                        sub_dataset_single_scan_direction = [*sub_dataset_single_scan_direction,
                                                             *this_dataset_as_list[negative_peaks[i]:positive_peaks[i]]]
            else:
                # if the CV begins downwards, the first positive peak is the very beginning.
                for i in range(1, len(positive_peaks)):
                    if index_exists(negative_peaks, i - 1):
                        sub_dataset_single_scan_direction = [*sub_dataset_single_scan_direction,
                                                             *this_dataset_as_list[
                                                              negative_peaks[i - 1]:positive_peaks[i]]]

        else:
            # IF the falling-voltage-cycle-part is desired
            if begins_upwards:
                # if the CV begins upwards, the first negative peak is the very beginning.
                for i in range(1, len(negative_peaks)):
                    if index_exists(positive_peaks, i - 1):
                        sub_dataset_single_scan_direction = [*sub_dataset_single_scan_direction,
                                                             *this_dataset_as_list[
                                                              positive_peaks[i - 1]:negative_peaks[i]]]
            else:
                for i in range(len(negative_peaks)):
                    if index_exists(positive_peaks, i):
                        sub_dataset_single_scan_direction = [*sub_dataset_single_scan_direction,
                                                             *this_dataset_as_list[positive_peaks[i]:negative_peaks[i]]]

        # HERE, sub_dataset_single_scan_direction contains the proper data selection.

        # Split off the voltage and current signals as 1d-lists
        sub_dataset_single_scan_direction = sorted(sub_dataset_single_scan_direction, key=lambda l: l[0], reverse=False)

        data_u = [k[0] for k in sub_dataset_single_scan_direction]
        data_i = [k[1] for k in sub_dataset_single_scan_direction]

        # if offset, apply the current offset
        if offset is not None:
            for i in range(len(data_i)):
                data_i[i] += offset
        # assert equal length. This must be True, if the dataset is not malformed and the code is intact.
        assert len(data_i) == len(data_u)

        # create a spline interpolation of the dataset and integrate
        callable_function_for_integration = sc.interpolate.interp1d(data_u, data_i, kind="nearest-up")

        def get_interp_i(u: float):
            return callable_function_for_integration(u)

        res = sc.integrate.quad(get_interp_i, lower_voltage_bound, upper_voltage_bound, limit=1000)

        unit_integral = self.unit_voltage * self.unit_current
        return res[0] * unit_integral, res[1]

    def register_eval_voltage(self):
        if self.default_filtered:
            data = self.get_filtered_dataset()
        else:
            data = self.dataset

        eval_voltage = data[np.where(self.difference_landscape == max(self.difference_landscape)), 0][0][0]
        if isinstance(eval_voltage, Quantity):
            eval_voltage = eval_voltage.magnitude
        self.eval_voltage = eval_voltage
        # print(f"Eval voltage register called. Determined {eval_voltage}. IsInstance results in {isinstance(eval_voltage, Quantity)}."
        #      f"Self-eval volt is now {self.eval_voltage}")

    def collapse_to_writable_list(self, filtered: bool = False, comment: str = None):
        # it is crucial to match the length of the meta-params between half-cycles and full cycles, so they can be
        # meaningfully exported together!
        if self.scanrate is not None:
            scanrate = str(self.scanrate * self.unit_scanrate)
        else:
            scanrate = "not provided"
        data_to_write = [["Original filename:", self.source],
                         ["Evaluated at Voltage:", str(self.eval_voltage * self.unit_voltage)],
                         ["Filtered:", str(filtered)],
                         ["Scanrate:", scanrate],
                         ["Used cycle:", str(self.cycle_nr)],
                         ["Comment:", comment],
                         [f"Voltage", f"Current"],
                         [f"{str(self.unit_voltage)}", f"{str(self.unit_current)}"]]

        if filtered:
            dataset = self.get_filtered_dataset().tolist()
            data_to_write = [*data_to_write, *dataset]
        else:
            dataset = self.get_dataset().tolist()
            data_to_write = [*data_to_write, *dataset]
        return data_to_write

    # takes the internal scanrate to construct a time dimension from the voltage
    # calculates the IR drop over the resistance for all voltage points in the cycle and subtracts from applied voltage
    # returns the map of dV/dt as the true interfacial scanrate
    # resistance must have pq unit attached
    def true_interfacial_scanrate(self, resistance):
        # Ensure the scanrate is set, else escalate the Error
        if self.scanrate is None:
            raise NoScanrateDefinedError("No scanrate has been set for the CV.")
        if self.default_filtered:
            data = self.get_filtered_dataset()
        else:
            data = self.dataset

        # construct a time dimension from zero, and always increase by the voltage difference, no matter which direction
        # divided by the scanrate
        time_dimension = [0 * pq.s]
        # construct the real interfacial voltage, using the same iteration
        interfacial_voltage = [
            data[0, 0] * self.unit_voltage - data[0, 1] * self.unit_current * resistance]
        for i in range(1, len(data)):
            time_dimension.append(
                time_dimension[i - 1] + np.abs(data[i, 0] - data[i - 1, 0]) * self.unit_voltage / (
                        self.scanrate * self.unit_scanrate))
            interfacial_voltage.append(
                data[i, 0] * self.unit_voltage - data[i, 1] * self.unit_current * resistance)

        # Time and voltage should be equally long
        assert len(time_dimension) == len(interfacial_voltage)

        # calculate the true scanrate at the interface
        volt_diff = np.diff(interfacial_voltage, n=1) * self.unit_voltage

        time_diff = np.diff(time_dimension, n=1) * pq.s
        true_scanrate = volt_diff / time_diff
        true_scanrate.units = self.unit_scanrate

        return true_scanrate

    # returns the area within the CV normalized to a rectangle spanned by the potential window and the maximum and
    # minimum currents. Sane values between 0 and 1. Used to estimate distortion param, which is used to correct the
    # vertical current difference ratio.
    def fill_factor(self):
        if self.default_filtered:
            data = self.get_filtered_dataset()
        else:
            data = self.dataset
        self.create_difference_landscape()
        # print(self.dataset)
        # plt.plot(self.difference_landscape)
        # plt.show()
        potential_window = max(data[:, 0]) - min(data[:, 0])
        points_per_cycle = len(data[:, 0])
        inner_area = np.sum(np.abs(self.difference_landscape)) * (potential_window / points_per_cycle)
        outer_area = ((max(data[:, 0]) - min(data[:, 0])) *
                      (max(data[:, 1]) - min(data[:, 1])))
        #print(f"Fill factor data eval, Potential window: {potential_window}, points_per_cycle: "
        #      f"{points_per_cycle}, inner area: {inner_area}, outer area: {outer_area}")
        return inner_area / outer_area

    # this function estimates the distortion parameter from the fill factor using the known equation for R-C-CVs and
    # via minimization, since the fill factor equation is not invertible.
    # Then, the theoretical vertical ratio is calculated and returned, as well as the distortion parameter.
    def estimate_vertical_current_ratio(self):
        # @jit
        def ff_difference(distortion_param):
            return np.abs(ff - (-2 * distortion_param + np.cosh(1 / (2 * distortion_param)) / np.sinh(
                1 / (2 * distortion_param))))

        ff = np.array([self.fill_factor()])[0]
        # print(f"Fill factor: {ff}")
        param_distortion = minimize_scalar(ff_difference, bounds=(0, 1e6), tol=1e-9)
        # print(f"Distortion param: {param_distortion.x}")
        vertical_ratio = (1 - np.exp(-1 / param_distortion.x)) / (1 + np.exp(-1 / param_distortion.x))
        return vertical_ratio, param_distortion

    def distortion_param_evaluation(self):
        """
        Evaluates the cyclic voltammetry data via the analytical method, by determining the fill factor, calculating
        therefrom the distortion parameter, using this distortion parameter to calculate the ratio of the apparent
        current difference at the potential limits to the true ratio, and finally obtaining capacitance, resistance,
        distortion parameter. It is incapable of determining offset, but will return a None element for compatibility.

        The method performs the following steps:
        1. Checks if a scan rate is defined. If not, raises `NoScanrateDefinedError`.
        2. Selects the appropriate dataset (filtered or unfiltered).
        3. Calculates the potential window and initial capacitance estimate from the current difference at the
            limits of the potential window and scanrate.
        4. Estimates the vertical current ratio and uses it to adjust the capacitance estimate.
        5. Calculates the resistance estimate based on the distortion parameter and other calculated values.
        6. Returns the absolute resistance estimate, capacitance estimate, potential window, and distortion parameter.

        :return: A tuple containing the following:
            - resistance_estimate (Quantity): The estimated resistance in Ohms.
            - capacitance_estimate (Quantity): The estimated capacitance in Farads.
            - potential_window (Quantity): The potential window in Volts.
            - distortion_param (float): The distortion parameter value.
            - None: A placeholder for compatibility with other methods.

        :raises NoScanrateDefinedError: If the scan rate (`self.scanrate`) is not defined.
        """
        if self.scanrate is None:
            raise NoScanrateDefinedError("No scanrate has been set for the CV.")

        if self.default_filtered:
            data = self.get_filtered_dataset()
        else:
            data = self.dataset

        potential_window = (max(data[:, 0]) - min(data[:, 0])) * self.unit_voltage
        potential_window.units = pq.V
        capacitance_estimate = ((max(data[:, 1]) - min(data[:, 1])) * self.unit_current /
                                (2 * self.scanrate * self.unit_scanrate))
        scanrate = self.scanrate * self.unit_scanrate
        scanrate.units = pq.V / pq.s

        # With datasets, where the assumption of equidistant potential sampling is violated, the difference landscape
        # calculation can be thrown severely off. Using the filtered dataset for this calculation suppresses this issue.
        #
        # Update: The solution from get_current_by_voltage has replaced the signal filtering. The latter remains
        # a backup option.
        # default_filtered = self.default_filtered
        # self.default_filtered = True
        vertical_ratio, distortion_param = self.estimate_vertical_current_ratio()
        # self.default_filtered = default_filtered

        capacitance_estimate /= vertical_ratio
        capacitance_estimate.units = pq.F
        # print(f"Vertical ratio {vertical_ratio}, capacitance {capacitance_estimate}")
        resistance_estimate = distortion_param.x * potential_window / (scanrate * capacitance_estimate)
        resistance_estimate.units = pq.Ohm
        # print(f"Resistance estimate: {resistance_estimate.__abs__()}")
        # The final return param is the offset. This method inherently discards the offset in the fit, but it must be
        # returned for compatibility of the interface with methods that do determine it.
        return resistance_estimate.__abs__(), capacitance_estimate, potential_window, distortion_param.x, None

    # Purpose: Determine, if the CV is influenced by underlying currents that shift or distort the capacitive CV.
    # The p_d is determined from the current ratios at 1/4 and 3/4 of the potential window, and compared with the fill-
    # factor based analysis.
    def bias_analysis(self):
        # return the difference between a target current ratio (here: measured) and the current ratio at a given p_d
        # These are the optimisation functions
        # @jit
        def current_ratio_first_quarter(distortion_param, target_current_ratio):
            return np.abs(target_current_ratio - (-1 + 2 /
                                                  (1 + np.sinh(3 / (8 * distortion_param)) /
                                                   np.cosh(1 / (8 * distortion_param)))))

        # @jit
        def current_ratio_third_quarter(distortion_param, target_current_ratio):
            res = np.abs(target_current_ratio - (-1 - 2 /
                                                 (-1 + np.sinh(3 / (8 * distortion_param)) /
                                                  np.cosh(1 / (8 * distortion_param)))))
            # print(f"Optimise: {res}, at distortion {distortion_param}")
            return res

        # @jit
        def get_current_ratio_by_distortion_param_first_quarter(distortion_param):
            return (-1 + 2 /
                    (1 + np.sinh(3 / (8 * distortion_param)) /
                     np.cosh(1 / (8 * distortion_param))))

        # @jit
        def get_current_ratio_by_distortion_param_third_quarter(distortion_param):
            return (-1 - 2 /
                    (-1 + np.sinh(3 / (8 * distortion_param)) /
                     np.cosh(1 / (8 * distortion_param))))

        if self.scanrate is None:
            raise NoScanrateDefinedError("No scanrate has been set for the CV.")

        if self.default_filtered:
            data = self.get_filtered_dataset()
        else:
            data = self.dataset

        potential_window = (max(data[:, 0]) - min(data[:, 0])) * self.unit_voltage
        potential_window.units = pq.V
        min_potential = min(data[:, 0]) * self.unit_voltage

        ################################################################################################################
        # Find the potential at half the potential window width, then query the current
        # correct the current by an offset that causes its mid-potential currents to be symmetric around 0
        high_current, low_current = self.get_current_at_voltage(voltage_position=min_potential.magnitude +
                                                                                 0.5 * potential_window.magnitude,
                                                                current_dimension=pq.A,
                                                                query_forward_backward_separately=True)
        correction = ((high_current + low_current) / 2)

        ################################################################################################################
        # Find the potential at 1/4th the potential window width, then query the current
        high_current, low_current = self.get_current_at_voltage(voltage_position=min_potential.magnitude +
                                                                                 0.25 * potential_window.magnitude,
                                                                current_dimension=pq.A,
                                                                query_forward_backward_separately=True)
        # correct for vertical offset
        high_current -= correction
        low_current -= correction
        print(f"Sanity check function: high current {high_current}, low current {low_current}, "
              f"hc_magnitude {high_current.magnitude}, lc_magnitude {low_current.magnitude}"
              f"Corrected by: {correction}")

        # optimisation target function, difference to measured current ratio, so it can be minimised to 0
        opt_target_fun = partial(current_ratio_first_quarter,
                                 target_current_ratio=high_current.magnitude / low_current.magnitude)
        param_distortion_first_quarter = minimize_scalar(opt_target_fun, bounds=(0, 1e6), tol=1e-6).x
        real_current_ratio_first_quarter = (high_current / low_current).magnitude
        #print(f"Distortion param first quarter: {param_distortion_first_quarter}")

        ################################################################################################################
        # Find the potential at 3/4th the potential window width, then query the current
        high_current, low_current = self.get_current_at_voltage(voltage_position=min_potential.magnitude +
                                                                                 0.75 * potential_window.magnitude,
                                                                current_dimension=pq.A,
                                                                query_forward_backward_separately=True)
        # correct for vertical offset
        high_current -= correction
        low_current -= correction

        # optimisation target function, difference to measured current ratio, so it can be minimised to 0
        # IMPORTANT: This function is somewhat tricky to optimise on, due to a vertical asymptote around pd=0.4102.
        # This is sensible, since the ratio changes signs when the backwards-scan current approaches and then crosses
        # zero at increasing distortion. However, the solver cannot cope with that. For this reason, a section-wise
        # optimisation is used, where first, it is optimised bounded within the negative ratios, then in the positives.
        opt_target_fun = partial(current_ratio_third_quarter,
                                 target_current_ratio=high_current.magnitude / low_current.magnitude)
        #print(f"Third quarter high_current.magnitude / low_current.magnitude "
        #      f"{high_current.magnitude / low_current.magnitude}")
        tolerance = 1e-8
        param_distortion_third_quarter = minimize_scalar(opt_target_fun, bounds=(0, 0.411), tol=tolerance)
        print(param_distortion_third_quarter.success)
        if not param_distortion_third_quarter.fun < tolerance * 100:
            param_distortion_third_quarter = minimize_scalar(opt_target_fun, bounds=(0.410, 1e6), tol=tolerance)
        # print(f"{param_distortion_third_quarter.x}, resulting function {param_distortion_third_quarter.fun}")
        param_distortion_third_quarter = param_distortion_third_quarter.x
        real_current_ratio_third_quarter = (high_current / low_current).magnitude
        # print(f"Distortion param third quarter: {param_distortion_third_quarter}")

        ################################################################################################################
        # Get the fill factor distortion. Calculate the expected current ratios from this p_d at 1/4 and 3/4 of the
        # potential window. At middle current, they are theoretically equal in any case.
        vertical_ratio, distortion_param = self.estimate_vertical_current_ratio()
        distortion_param = distortion_param.x
        target_current_ratio_first_quarter = get_current_ratio_by_distortion_param_first_quarter(distortion_param=
                                                                                                 distortion_param)
        target_current_ratio_third_quarter = get_current_ratio_by_distortion_param_third_quarter(distortion_param=
                                                                                                 distortion_param)

        ################################################################################################################
        # Print the outputs, return a dictionary
        # print(f"Should/is comparison: First quarter, third quarter pd:")
        # print(f"First: {distortion_param}, {param_distortion_first_quarter} "
        #       f"Third: {distortion_param}, {param_distortion_third_quarter}")
        # print(f"Should/is comparison: First quarter, third quarter current ratios:")
        # print(f"First: {target_current_ratio_first_quarter}, {real_current_ratio_first_quarter} "
        #       f"Third: {target_current_ratio_third_quarter}, {real_current_ratio_third_quarter}")

        # construct a return dict for the values
        ret = {"ff_distortion_param": distortion_param,
               "pd_first_quarter_real": param_distortion_first_quarter,
               "pd_third_quarter_real": param_distortion_third_quarter,
               "target_current_ratio_first_quarter": target_current_ratio_first_quarter,
               "target_current_ratio_third_quarter": target_current_ratio_third_quarter,
               "real_current_ratio_first_quarter": real_current_ratio_first_quarter,
               "real_current_ratio_third_quarter": real_current_ratio_third_quarter,
               "correction_applied": correction
               }
        print(ret)

        return ret

    def fit_cv_by_optimisation(self):
        """
        Optimizes the resistance (R), capacitance (C), and offset parameters for a cyclic voltammetry (CV) dataset
        using an optimization algorithm.

        This method enhances the initial distortion parameter evaluation by applying an optimization algorithm to
        refine the estimates of the RC parameters. The process involves splitting the CV dataset into forward and reverse
        scan segments, creating a homogenous forward and backward scan segment, and fitting both to a target function
        that models the expected current response based on the RC parameters.

        The steps involved in this method are:
        1. Verify that the scan rate is defined. If not, raise `NoScanrateDefinedError`.
        2. Convert all quantities to their base SI units (voltage in volts, current in amperes) for consistent calculations.
        3. Split the CV dataset into forward and reverse scan segments based on voltage peaks.
        4. Create a time domain for each segment based on the linear scan rate assumption.
        5. Define the forward and reverse target functions that model the current response over time.
        6. Apply the `scipy.optimize.minimize` function to optimize the RC parameters using the initial estimates provided by
           the analytical distortion parameter evaluation.
        7. Return the optimized resistance, capacitance, potential window, distortion parameter, and offset values.

        :return: A tuple containing the following optimized parameters:
            - optimal_R (Quantity): The optimized resistance value in Ohms.
            - optimal_C (Quantity): The optimized capacitance value in Farads.
            - potential_window (Quantity): The calculated potential window in Volts.
            - distortion_param (float): The optimized distortion parameter.
            - optimal_offset (Quantity): The optimized offset value in Amperes.

        :raises NoScanrateDefinedError: If the scan rate (`self.scanrate`) is not defined.

        The method utilizes the `Nelder-Mead` optimization algorithm and is sensitive to the initial guess provided
        by the `distortion_param_evaluation` method. This guess is critical for convergence, as poor initial values may
        lead to non-converging results or incorrect parameter estimates.
        """
        # Split any CV into a forward and reverse scan part, to be able to apply segment-wise interpolation
        # Then, run optimisation to find RC-parameter.
        if self.scanrate is None:
            raise NoScanrateDefinedError("No scanrate has been set for the CV.")

        def index_exists(list_, index):
            if 0 <= index <= len(list_):
                return True
            else:
                return False
        # Ensure the calculation here is done in SI base Units. If you're brave, figure out what happens to the units
        # downstream otherwise, and rework the function.

        previously_set_voltage = self.unit_voltage
        previously_set_current = self.unit_current
        self.convert_quantities(pq.V, pq.A)

        # obtain the interpolated function in the forward or reverse direction
        # first split the CV into the forward-backward directions, append them to a single array, and use
        # scipy.InterpolateUnivariateSpline to calculate the integral.
        # If desired, this could be changed to other interpolation modes.
        if not self.default_filtered:
            this_dataset_as_list = self.dataset
        else:
            this_dataset_as_list = self.get_filtered_dataset().tolist() * pq.dimensionless

        scanrate = self.scanrate * self.unit_scanrate
        scanrate.units = pq.V / pq.s
        scanrate = scanrate.magnitude
        # Now that the data is retrieved, return the CV to the set units.
        self.convert_quantities(previously_set_voltage, previously_set_current)

        # the peaks include the termination and the beginning of the cv as peaks (in negative_peaks if falling towards
        # the point and/or rising from it, positive_peaks if rising towards and falling from the point.
        # Sketching this out helps.
        begins_upwards, negative_peaks, positive_peaks = get_positive_and_negative_voltage_peaks(
            this_dataset_as_list)
        # print(positive_peaks)
        # print(negative_peaks)
        # split the CV into halfcycles along the voltage peaks
        sub_dataset_forward_direction = []
        sub_dataset_reverse_direction = []


        # FORWARD DIRECTION
        if begins_upwards:
            for i in range(len(positive_peaks)):
                if index_exists(negative_peaks, i):
                    sub_dataset_forward_direction = [*sub_dataset_forward_direction,
                                                         *this_dataset_as_list[negative_peaks[i]:positive_peaks[i]]]
        else:
            # if the CV begins downwards, the first positive peak is the very beginning.
            for i in range(1, len(positive_peaks)):
                if index_exists(negative_peaks, i - 1):
                    sub_dataset_forward_direction = [*sub_dataset_forward_direction,
                                                         *this_dataset_as_list[
                                                          negative_peaks[i - 1]:positive_peaks[i]]]

        # REVERSE DIRECTION
        if begins_upwards:
            # if the CV begins upwards, the first negative peak is the very beginning.
            for i in range(1, len(negative_peaks)):
                if index_exists(positive_peaks, i - 1):
                    sub_dataset_reverse_direction = [*sub_dataset_reverse_direction,
                                                         *this_dataset_as_list[
                                                          positive_peaks[i - 1]:negative_peaks[i]]]
        else:
            for i in range(len(negative_peaks)):
                if index_exists(positive_peaks, i):
                    sub_dataset_reverse_direction = [*sub_dataset_reverse_direction,
                                                         *this_dataset_as_list[positive_peaks[i]:negative_peaks[i]]]


        # Split off the voltage and current signals as 1d-lists
        sub_dataset_forward_direction = sorted(sub_dataset_forward_direction, key=lambda l: l[0], reverse=False)
        sub_dataset_reverse_direction = sorted(sub_dataset_reverse_direction, key=lambda l: l[0], reverse=True)

        # create a timespace starting at 0 by assuming a linear scan rate.
        # Note: For purposes of this calculation, we will pretend that the CV starts at V_min in forward direction at
        # time 0. Because we are calculating assuming steady state, nothing at all changes in the result.
        sub_dataset_forward_direction = np.array(sub_dataset_forward_direction)
        time_domain = (sub_dataset_forward_direction[:, 0] - sub_dataset_forward_direction[0, 0])/scanrate
        time_domain = time_domain.reshape(-1, 1)
        #print(time_domain)
        #print(f"Time domain {len(time_domain)}")
        sub_dataset_forward_direction = np.hstack((time_domain, sub_dataset_forward_direction))
        #print(sub_dataset_forward_direction)

        sub_dataset_reverse_direction = np.array(sub_dataset_reverse_direction)
        v_offset_at_reverse_scan_start = (np.max([sub_dataset_forward_direction[-1, 1],
                                              sub_dataset_reverse_direction[0, 0]]) -
                                          sub_dataset_forward_direction[0, 1])

        #print(f"v_offset {v_offset_at_reverse_scan_start}, end forward at {sub_dataset_forward_direction[-1, 1]},"
        #      f"beginning reverse at {sub_dataset_reverse_direction[0, 0]}, scanrate {scanrate}")
        time_domain = (-(sub_dataset_reverse_direction[:, 0] - (np.max([sub_dataset_forward_direction[-1, 1],
                                              sub_dataset_reverse_direction[0, 0]]))) + v_offset_at_reverse_scan_start)/scanrate
        time_domain= time_domain.reshape(-1, 1)
        #print(f"Time domain reverse len {len(time_domain)}, printed {time_domain}")
        sub_dataset_reverse_direction = np.hstack((time_domain, sub_dataset_reverse_direction))
        #print(sub_dataset_reverse_direction)

        #######################################################################
        # FITTING of the data
        #######################################################################

        # Target functions in forward and reverse section of the CV. Vectorised to allow for efficient application to
        # the entire timespace.
        # Since A and T can be concluded from the dataset, they need not be optimised.
        A = (np.max([np.max(sub_dataset_forward_direction[:, 1]), np.max(sub_dataset_reverse_direction[:, 1])]) -
             np.min([np.min(sub_dataset_forward_direction[:, 1]), np.min(sub_dataset_reverse_direction[:, 1])]))/2
        T = sub_dataset_reverse_direction[-1, 0] - sub_dataset_forward_direction[0, 0]
        # @jit
        def forward_current(t, R, C, offset):
            return ((-8 * A * C / T) * np.exp(-t / (R * C)) / (1 + np.exp(-T / (2 * R * C))) + 4 * A * C / T) + offset
        forward_vectorized = np.vectorize(forward_current)

        def reverse_current(t, R, C, offset):
            return ((8 * A * C / T) * np.exp((-t + T) / (R * C)) / (1 + np.exp(T / (2 * R * C))) - 4 * A * C / T) + offset
        reverse_vectorized = np.vectorize(reverse_current)

        #@time_this_function
        def target_function_optimisation(params):
            R = params[0]
            C = params[1]
            offset = params[2]
            forward_difference = (forward_vectorized(sub_dataset_forward_direction[:, 0], R=R, C=C, offset=offset) -
                                  sub_dataset_forward_direction[:, 2])
            reverse_difference = (reverse_vectorized(sub_dataset_reverse_direction[:, 0], R=R, C=C, offset=offset) -
                                  sub_dataset_reverse_direction[:, 2])
            # print(f"R {R}, C {C}")
            # print(f"forward {np.sum(np.abs(forward_difference))}, reverse {np.sum(np.abs(reverse_difference))}")
            # print(f"tar-fun {np.sqrt(np.sum(np.abs(forward_difference))) + np.sqrt(np.sum(np.abs(reverse_difference)))}")
            return (np.sqrt(np.sum(np.square(forward_difference))/len(forward_difference)) +
                    np.sqrt(np.sum(np.square(reverse_difference))/len(reverse_difference)))

        # The initial guess is critical to convergence of the algorithm. If it is too far off, the target function
        # for one branch or the other may return nan or np.inf, disorienting the search.
        # Here, the analytical distortion param evaluation is called for providing the initial guess. Even though it is
        # sensitive to fill factor-error introduced by noise, it will be in the ballpark. The inital offset guess is 0.

        res_initial, cap_initial, window, p_d, dummy_var = self.distortion_param_evaluation()
        initial_values = np.array([res_initial, cap_initial, 0])
        print(f"Initial pd values: res {res_initial}, cap {cap_initial}, window {window}, pd {p_d}")
        # Optimise with the initial values, bounds are set to exclude negative values.
        opt = minimize(target_function_optimisation, initial_values , bounds=((0, None), (0, None), (None, None)),
                       method='Nelder-Mead')
        # print(opt)
        optimal_R, optimal_C, optimal_offset = opt.x

        distortion_param = scanrate * optimal_C * optimal_R/(2*A)
        print(f"Optimisation complete: R {optimal_R}, C {optimal_C}, W  {2*A}, p_d {distortion_param}, offset {optimal_offset}")
        return optimal_R * pq.Ohm , optimal_C * pq.F, 2*A * pq.V, distortion_param, optimal_offset * pq.A

class HalfCV(CV):
    index: int
    dataset: np.array
    eval_voltage: float
    half_cycle_nr: int
    source: str
    scanrate: float
    active: bool
    default_filtered: bool = False

    def __init__(self, source: str, half_cycle_nr: int, index: int, dataset: list, eval_voltage,
                 unit_voltage, unit_current):
        self.unit_scanrate = None
        self.source = source
        self.dataset = np.array(dataset)
        self.half_cycle_nr = half_cycle_nr
        self.index = index
        self.unit_voltage = unit_voltage
        self.unit_current = unit_current
        self.active = True
        if eval_voltage is None:
            self.register_eval_voltage()
        else:
            self.eval_voltage = eval_voltage

    def get_eval_voltage(self):
        return self.eval_voltage

    def get_index(self) -> int:
        return self.index

    def set_active(self, is_active: bool):
        self.active = is_active

    def is_active(self):
        return self.active

    def get_cycle_nr(self):
        return self.half_cycle_nr

    def get_type(self):
        super()

    def set_default_filtered(self, filtered: bool):
        self.default_filtered = filtered

    def get_default_filtered(self):
        return self.default_filtered

    def get_dataset(self):
        x = np.array(self.dataset)
        return x

    def get_filtered_dataset(self):
        e_smooth = sgnl.savgol_filter(self.dataset[:, 0], window_length=11, polyorder=3, mode="nearest")
        i_smooth = sgnl.savgol_filter(self.dataset[:, 1], window_length=11, polyorder=3, mode="nearest")
        return np.stack((e_smooth, i_smooth), axis=1)

    def get_source(self):
        return self.source

    def set_scanrate(self, scanrate: float, dimension: quantities.Quantity):
        self.unit_scanrate = dimension
        self.scanrate = scanrate

    def get_scanrate(self, dimension: quantities.Quantity):
        if self.scanrate is not None:
            scanrate = self.scanrate * self.unit_scanrate
            scanrate.units = dimension
            return scanrate
        else:
            raise ScanrateExistsError("This CV has no scanrate.")

    def get_current_at_voltage(self, voltage_position=None, current_dimension: pq.Quantity = None):
        if voltage_position is None:
            voltage_position = self.eval_voltage
        else:
            voltage_position *= pq.V
            voltage_position.units = self.unit_voltage
            voltage_position /= self.unit_voltage

        if self.default_filtered:
            data = self.get_filtered_dataset()
        else:
            data = self.dataset

        closest_matching_voltage = (np.abs(data[:, 0] - voltage_position)).argmin()
        current_at_position = data[closest_matching_voltage, 1] * self.unit_current
        if current_dimension is not None:
            current_at_position.units = current_dimension
        return current_at_position

    def convert_quantities(self, unit_voltage, unit_current):
        voltage = self.dataset[:, 0] * self.unit_voltage
        current = self.dataset[:, 1] * self.unit_current
        voltage.units = unit_voltage
        current.units = unit_current
        voltage /= unit_voltage
        current /= unit_current
        self.unit_voltage = unit_voltage
        self.unit_current = unit_current
        self.dataset = np.stack((voltage, current), axis=1)

    def collapse_to_writable_list(self, filtered: bool = False, comment: str = None):
        # it is crucial to match the length of the meta-params between half-cycles and full cycles, so they can be
        # meaningfully exported together!
        if self.scanrate is not None:
            scanrate = str(self.scanrate * self.unit_scanrate)
        else:
            scanrate = "not provided"
        data_to_write = [["Original filename", self.source],
                         ["Evaluated at Voltage [V]", str(self.eval_voltage * self.unit_voltage)],
                         ["Filtered", str(filtered)],
                         ["Scanrate [mV/s]", scanrate],
                         ["Used half-cycle", str(self.half_cycle_nr)],
                         ["Comment:", comment],
                         [f"Voltage {str(self.unit_voltage)}", f"Current {str(self.unit_current)}"]]

        if filtered:
            dataset = self.get_filtered_dataset().tolist()
            data_to_write = [*data_to_write, *dataset]
        else:
            dataset = self.get_dataset().tolist()
            data_to_write = [*data_to_write, *dataset]
        return data_to_write

    def register_eval_voltage(self):
        if self.default_filtered:
            data = self.get_filtered_dataset()
        else:
            data = self.dataset

        self.eval_voltage = data[np.where(data[1, :] == max(data[1, :])), 0][0][0]
