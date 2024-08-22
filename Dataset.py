import os

import numpy as np
import CV
from CV import FullCV, HalfCV
import quantities as pq
from scipy import optimize as opt
import PySimpleGUI as sg
from custom_exceptions import ScanrateExistsError, NotEnoughCVsToFitError, CycleIndexOutOfBoundsError, \
    NoCycleInformationError, NoScanrateDefinedError, UnknownMethodError
from datetime import datetime

# These are the signifying keywords for current and voltage in the CV datasets.
current_signifiers = ["<I>", "Current", "current", "I/", "I /", "I(A)"]
voltage_signifiers = ["Ewe", "Potential", "E/", "Voltage 1", "E /", "Voltage", "E(V)"]


def write_standardized_data_file(filename, data_to_write):
    try:
        with open(filename, "a") as file:
            for i in range(len(data_to_write)):
                writestring = ""
                for element in data_to_write[i]:
                    writestring += str(element) + "\t"
                file.write(writestring + "\n")
        return True
    except FileNotFoundError or FileExistsError:
        print("There was an error opening the file to write on.")
        return False


def read_csv_file(filename, assume_standard_csv_format=False):
    string_list = []
    try:
        with open(filename, "r+") as file:
            for line in file:
                string_list.append(line)
        # print(string_list)
    except FileNotFoundError:
        print("The file was not found.")
        return None

    string_list = np.array(string_list)
    dataset = []
    for element in string_list:
        if element is None or element == "\n":
            continue
        if assume_standard_csv_format:
            element = ";".join(element.split(','))

        cand_line = None


        # parse the separators. High priority to the semicolon, to ensure a default-csv is actually parsed there.
        # Tab-separator is next-common, since it is usually non-ambiguous. Only if no such separator is found,
        # default to space - which may otherwise just be in there gratuitously.
        if element.__contains__(';'):
            cand_line = element.replace('\n', '').split(';')
        elif element.__contains__('\t'):
            cand_line = element.replace('\n', '').split('\t')
        elif element.__contains__(' '):
            # reduce the occurence of n consecutive whitespaces to one
            element = ' '.join(element.split())
            cand_line = element.replace('\n', '').split(' ')

        # if the split has occured, the element should now be list-type. If not, the error is escalated
        # through an AssertionError to be handled at other levels.
        # remove all None-type split results caused by doubled separators
        if cand_line is not None:
            dataset.append([k for k in cand_line if k is not None and k != ' ' and k != ''])
        # print(dataset[-1])

    # Cleanup
    # For every line, check if the line contains both voltage and current signifier. If so, mark this as the header.
    header_line_index = 0
    while header_line_index < len(dataset):
        has_current_signifier = False
        has_voltage_signifier = False
        for substring in dataset[header_line_index]:
            for k in range(len(current_signifiers)):
                has_current_signifier = np.logical_or(has_current_signifier,
                                                      substring.__contains__(current_signifiers[k]))

            for k in range(len(voltage_signifiers)):
                has_voltage_signifier = np.logical_or(has_voltage_signifier,
                                                      substring.__contains__(voltage_signifiers[k]))
        if has_voltage_signifier and has_current_signifier:
            break
        else:
            header_line_index += 1


    return dataset[header_line_index:]


def read_in_full_cv(filename, assume_standard_csv_format):
    dataset = []
    e_column = 0
    i_column = 1

    current_correction_factor = 1
    cycle_column = 2
    # with open(filename, "r+") as file:
    # reader = csv.reader(file, delimiter=' ')
    try:
        read_in = read_csv_file(filename, assume_standard_csv_format)
        has_found_header_line = False
        # if no cycle number is given, return the entire dataset
        for line in read_in:
            try:
                dataset.append([float(line[e_column].replace(",", ".")) * pq.V,
                                float(line[i_column].replace(",", ".")) * current_correction_factor * pq.A])
            except ValueError:
                if not has_found_header_line:
                    e_column, i_column = find_voltage_and_current_column(line)

                    # find current correction factor
                    if line[i_column].__contains__("mA"):
                        current_correction_factor = 0.001
                    elif line[i_column].__contains__("uA"):
                        current_correction_factor = 0.000001
                    elif line[i_column].__contains__("nA"):
                        current_correction_factor = 10 ** -9
                    else:
                        current_correction_factor = 1
                    has_found_header_line = True
                    continue
                else:
                    # Once the header-line is found, all other malformed lines are ignored
                    continue
        dataset = np.array(dataset)
        return dataset
    except AssertionError:
        print("The dataset contains malformed separators that are not yet handled in the program.")


def load_one_cycle(filename, cycle_number, assume_standard_csv_format, throw_full_dataset=False):
    dataset = []
    e_column = 0
    i_column = 1

    current_correction_factor = 1
    cycle_column = 2
    try:
        read_in = read_csv_file(filename, assume_standard_csv_format)
        has_found_header_line = False
        for line in read_in:
            try:
                # if float(line[cycle_column].replace(",", ".")) == cycle_number:
                dataset.append([float(line[e_column].replace(",", ".")) * pq.V,
                                float(line[i_column].replace(",", ".")) * current_correction_factor * pq.A,
                                float(line[cycle_column].replace(",", "."))])
            except ValueError as e:
                if not has_found_header_line:
                    e_column, i_column = find_voltage_and_current_column(line)

                    # find current correction factor
                    if line[i_column].__contains__("mA"):
                        current_correction_factor = 0.001
                    elif line[i_column].__contains__("uA"):
                        current_correction_factor = 0.000001
                    elif line[i_column].__contains__("nA"):
                        current_correction_factor = 10 ** -9
                    else:
                        current_correction_factor = 1
                    cycle_column = [i for i in range(len(line)) if (line[i].__contains__("cycle"))]
                    has_found_header_line = True
                    if not str(*cycle_column).isnumeric():
                        raise NoCycleInformationError("The dataset does not appear to contain cycle data.")
                    else:
                        cycle_column = cycle_column[0]
                        continue
                else:
                    continue

        # If requested, throw full dataset:
        if throw_full_dataset:
            return dataset

        # CONSTRUCT THE RETURN VALUE
        # dataset = np.array(dataset)
        # select cycle and catch invalid cycles, defaulting to the largest possible
        cycle_data = []
        for i in range(len(dataset)):
            if dataset[i][2] == cycle_number:
                cycle_data.append(dataset[i])

        # IF THE RETURN CYCLE DATA IS INEXISTENT, THE CYCLE NUMBER WAS INVALID
        if not cycle_data:
            # TODO this here is the performance.death [line for line in dataset if line[2] == max(dataset[:, 2])]
            max_cycle = max(np.array(dataset)[:, 2])
            max_cycle_data = []
            for i in range(len(dataset)):
                if dataset[i][2] == max_cycle:
                    max_cycle_data.append(dataset[i])
            raise CycleIndexOutOfBoundsError(requested_cycle=cycle_number,
                                             highest_available_cycle=max_cycle,
                                             cycle_data=np.array(max_cycle_data),
                                             message=f"The requested cycle was unavailable. Here's cycle no. "
                                                     f"{max_cycle} instead.")
        return np.array(cycle_data)
    except AssertionError:
        print("The dataset contains malformed separators that are not yet handled in the program.")


def write_split_cycles(filename, assume_standard_csv_format):

    def generate_unique_filename(filename):
        base_name, extension = os.path.splitext(filename)
        new_filename = base_name + "_CVEval_cycle_split" + extension
        if not os.path.exists(new_filename):
            return new_filename

        # If filename exists, append (x) to make it unique
        x = 1
        while True:
            new_filename = f"{base_name}_CVEval_cycle_split({x}){extension}"
            if not os.path.exists(new_filename):
                return new_filename
            x += 1

    def reshape_array(array):
        # Find unique cycle numbers
        cycle_numbers = np.unique(array[:, 2])

        # Initialize list to store cycles
        cycles = []

        # Extract cycles and pad them
        max_length = 0
        for cycle_number in cycle_numbers:
            cycle_data = array[array[:, 2] == cycle_number][:, :2]  # Extract voltage and current for this cycle
            cycle_data_with_header = np.vstack([
                [f"Voltage/ Cycle {cycle_number}", f"Current/ Cycle {cycle_number}"],
                cycle_data
            ])
            cycles.append(cycle_data_with_header)
            max_length = max(max_length, len(cycle_data_with_header))

        # Pad cycles
        for i in range(len(cycles)):
            cycle_length = len(cycles[i])
            if cycle_length < max_length:
                padding = [[" ", " "]] * (max_length - cycle_length)
                cycles[i] = np.vstack([cycles[i], padding])

        # Combine cycles into a new array with headers
        reshaped_array = np.hstack(cycles)

        return reshaped_array

    # Call the cycle detection and request the full dataset with created cycle labels.
    # First, try and refer to the labelled cycles from potentiostat, if that fails, detect them here.
    try:
        dataset = load_one_cycle(filename=filename,
                                 cycle_number=0,
                                 assume_standard_csv_format=assume_standard_csv_format,
                                 throw_full_dataset=True)
        dataset = np.array(dataset)
    except NoCycleInformationError:
        dataset = cycle_detection_parsing(filename=filename,
                                          cycle_number=0,
                                          assume_standard_csv_format=assume_standard_csv_format,
                                          throw_full_dataset=True)

    # apply the reshaping
    data_to_write = reshape_array(dataset)
    success = write_standardized_data_file(generate_unique_filename(filename), data_to_write)
    return success

def cycle_detection_parsing(filename, cycle_number, assume_standard_csv_format, throw_full_dataset=False):
    # default the columns to first, second
    e_column = 0
    i_column = 1
    current_cycle = 0

    # a factor used to norm the read-in dataset to ampere
    current_correction_factor = 1
    dataset = []
    # this value saves the first potential value and is used to mark new cycles
    e_treshold = 0
    # Does the voltage scan start with rising voltage?

    try:
        read_in = read_csv_file(filename, assume_standard_csv_format)
        has_found_header_line = False
        for line in read_in:
            try:
                voltage = float(line[e_column].replace(",", ".")) * pq.V
                current = float(line[i_column].replace(",", ".")) * current_correction_factor * pq.A
                # initialize e_treshold at first value

                dataset.append([voltage, current])
            except ValueError:
                if not has_found_header_line:
                    e_column, i_column = find_voltage_and_current_column(line)

                    # find current correction factor
                    if line[i_column].__contains__("mA"):
                        current_correction_factor = 0.001
                    elif line[i_column].__contains__("uA"):
                        current_correction_factor = 0.000001
                    elif line[i_column].__contains__("nA"):
                        current_correction_factor = 10 ** -9
                    else:
                        current_correction_factor = 1
                    has_found_header_line = True
                    continue
                else:
                    continue

        # print(f"E col{e_column}, I col {i_column}")
        # at the 40th element of the dataset, compare which direction the voltage has gone
        # if nowhere, try at 100, if still the same voltage, enforce a decision at index 250
        direction_upwards = None
        e_treshold = dataset[0][0]

        # some CVs have a period of initially constant voltage. This can produce erroneous cycle assessments if not
        # excluded.
        meaningful_start_of_cv = 0
        if dataset[40][0] != e_treshold:
            direction_upwards = (dataset[40][0] > e_treshold)
            meaningful_start_of_cv = 40
        # if it is still the same, enforce a decision at index 100
        if direction_upwards is None:
            if dataset[100][0] != e_treshold:
                direction_upwards = (dataset[100][0] > e_treshold)
                meaningful_start_of_cv = 100
        if direction_upwards is None:
            direction_upwards = (dataset[100][0] >= e_treshold)
            meaningful_start_of_cv = 250
            # print(f"Voltage begins with an initial rise: {direction_upwards} ")

        for i in range(len(dataset)):
            voltage = dataset[i][0]
            if i == 0:
                # isolate the voltage column. note that the array operator kills the units, so they have to be
                # reestablished
                voltage_column = np.array(dataset)[:, 0] * pq.V
                # print(f"Voltage column is {voltage_column} and min voltage column is {min(voltage_column)} while"
                #       f"{np.where(voltage_column == min(voltage_column))[0][0]}")
                if np.where(voltage_column == min(voltage_column))[0][0] < 200 and direction_upwards:
                    voltage_range = max(voltage_column) - min(voltage_column)
                    e_treshold = voltage + 0.001 * voltage_range
                if np.where(voltage_column == max(voltage_column))[0][0] < 200 and not direction_upwards:
                    voltage_range = max(voltage_column) - min(voltage_column)
                    e_treshold = voltage - 0.001 * voltage_range
                current_cycle = 1

            # depending on whether the scan begins upwards or downwards, cut the cycles accordingly
            # this will cause errors with very low-sample cycle data, in that case, reduce 100 to a convenient number
            if i > 100:
                if direction_upwards and voltage >= e_treshold > dataset[i - 1][0]:
                    current_cycle += 1
                if not direction_upwards and voltage <= e_treshold < dataset[i - 1][0]:
                    current_cycle += 1
            dataset[i].append(int(current_cycle))

        # If the special parameter throw_full_dataset is invoked, hand over the entire labelled dataset.
        if throw_full_dataset:
            return np.array(dataset)
        # CONSTRUCT THE RETURN VALUE
        # select cycle and catch invalid cycle requests, defaulting to the largest possible
        cycle_data = []
        for i in range(len(dataset)):
            if dataset[i][2] == cycle_number:
                cycle_data.append(dataset[i][:2])

        # if the return is empty, the requested cycle was not found. Send the largest instead
        # via Exception to the calling function.
        if not cycle_data:
            for i in range(len(dataset)):
                if dataset[i][2] == current_cycle:
                    cycle_data.append(dataset[i][:2])

            raise CycleIndexOutOfBoundsError(requested_cycle=cycle_number,
                                             highest_available_cycle=current_cycle,
                                             cycle_data=np.array(cycle_data),
                                             message=f"The requested cycle was unavailable. Here's cycle no. "
                                                     f"{current_cycle} instead.")

        # ret_value[:, 0] *= pq.V
        # ret_value[:, 1] *= pq.A
        return np.array(cycle_data)

    except AssertionError:
        print("The dataset contains malformed separators that are not yet handled in the program.")


# This function takes a line from a file (list split along the separators) and returns where the signifying strigns for
# voltage and current are found.
def find_voltage_and_current_column(line):
    e_column = None
    i_column = None
    for i in range(len(line)):
        for k in range(len(current_signifiers)):
            if line[i].__contains__(current_signifiers[k]):
                i_column = i
        # Terminate search at first occurence. This will help parse datasets with multiple additional columns
        # containing relevant signifiers.
        if i_column:
            break
    for i in range(len(line)):
        for k in range(len(voltage_signifiers)):
            if line[i].__contains__(voltage_signifiers[k]):
                e_column = i
        # Terminate search at first occurence. This will help parse datasets with multiple additional columns
        # containing relevant signifiers.
        if e_column:
            break
    return e_column, i_column


def half_cycle_parsing(filename, assume_standard_csv_format, cycle_number=1, mode="forward",
                       window_handle: sg.Window = None, window_input_index=None):
    full_dataset = read_in_full_cv(filename, assume_standard_csv_format)
    # returns a u, i list of lists

    direction_upwards, negative_peaks, positive_peaks = get_positive_and_negative_voltage_peaks(full_dataset)

    if mode == "forward":
        try:
            # forward scans
            if direction_upwards:
                return full_dataset[negative_peaks[cycle_number - 1]:positive_peaks[cycle_number]]
            else:
                return full_dataset[negative_peaks[cycle_number]:positive_peaks[cycle_number]]
        except IndexError:
            # forward scans
            print("The half-cycle provided was not found. Providing the first instead.")
            if window_handle and isinstance(window_input_index, int):
                window_handle[("cycle_nr", window_input_index)].update(f"1")
            if direction_upwards:
                return full_dataset[negative_peaks[0]:positive_peaks[1]]
            else:
                return full_dataset[negative_peaks[1]:positive_peaks[1]]
    else:
        try:
            # reverse scans
            if direction_upwards:
                return full_dataset[positive_peaks[cycle_number]:negative_peaks[cycle_number]]
            else:
                return full_dataset[positive_peaks[cycle_number - 1]:negative_peaks[cycle_number]]
        except IndexError:
            # forward scans
            print("The half-cycle provided was not found. Providing the first instead.")
            if window_handle and isinstance(window_input_index, int):
                window_handle[("cycle_nr", window_input_index)].update(f"1")
            if direction_upwards:
                return full_dataset[positive_peaks[1]:negative_peaks[1]]
            else:
                return full_dataset[positive_peaks[0]:negative_peaks[1]]


def get_positive_and_negative_voltage_peaks(full_dataset):
    direction_upwards = True
    if full_dataset[40][0] < full_dataset[0][0]:
        direction_upwards = False
    positive_peaks = []
    negative_peaks = []
    # iterate over all candidate center-elements in a len-3 comparison filter => exclude first and last element
    for index in range(40, len(full_dataset) - 10):
        if full_dataset[index - 1, 0] < full_dataset[index, 0] > full_dataset[index + 1, 0] and \
                full_dataset[index - 10, 0] < full_dataset[index, 0] > full_dataset[index + 10, 0]:
            positive_peaks.append(index)
        if full_dataset[index - 1, 0] > full_dataset[index, 0] < full_dataset[index + 1, 0] and \
                full_dataset[index - 10, 0] > full_dataset[index, 0] < full_dataset[index + 10, 0]:
            negative_peaks.append(index)
    # positive_peaks = sgnl.find_peaks(full_dataset[:, 0])
    # negative_peaks = sgnl.find_peaks(-1*full_dataset[:, 0])
    # positive_peaks = positive_peaks[0].tolist()
    # negative_peaks = negative_peaks[0].tolist()
    positive_peaks.insert(0, 0)
    negative_peaks.insert(0, 0)
    return direction_upwards, negative_peaks, positive_peaks


# DATASET CLASS
# This is the class handling a read-in dataset. It thereby becomes possible to separate CV and Dataset level calculation
# and locate them precisely in the appropriate classes. A dataset contains CVs. It is self-loading, given the correct
# set of initialization parameters. All static helper functions required are above this class.
class Dataset:
    # The contents are of abstract type CV. It is therefore strictly necessary for compatibility to use only declared
    # methods in the abstract class CV for any type of calculations.
    contents: list[CV]

    # the values dictionary is handed in from the gui. This could be made more efficient/stringent by only selecting
    # the values actually used here, but since they are around 90% of the dictionary, at this time, I am too lazy.
    # TODO extract the window update from this function. It should be done at the level where this function is called.
    def __init__(self, values: dict, count: int, window):
        self.contents = []
        for i in range(1, count + 1):
            # read in only non-empty fields
            if values[("cv", i)] != "":
                voltage = None

                # GET cycle number
                try:
                    if values[("cycle_nr", i)] != "":
                        cycle_number = int(values[("cycle_nr", i)])
                    else:
                        cycle_number = 2
                        window[("cycle_nr", i)].update("2")
                except TypeError:
                    print("Non-readable cycle index in CV " + str(i) + ", Standard 2 is used instead.")
                    cycle_number = 2
                    window[("cycle_nr", i)].update("2")

                # GET eval voltage
                if values[("voltage_eval", i)] != "":
                    try:
                        voltage = float(values[("voltage_eval", i)].replace(",", "."))
                    except ValueError:
                        voltage = None

                # CASE: Full cycles
                if values["halfcycle_mode"] == "full cycles":
                    try:
                        data = load_one_cycle(filename=values[("cv", i)],
                                              cycle_number=cycle_number,
                                              assume_standard_csv_format=values["default_csv_format"])
                    except CycleIndexOutOfBoundsError as e:
                        # this exception is thrown by the readout functions at the point of evaluating their return values
                        requested_cycle = e.requested_cycle
                        highest_available = e.highest_available_cycle
                        data = np.array(e.cycle_data)
                        print(
                            f"Cycle number given {requested_cycle} exceeds the highest available cycle {highest_available} "
                            f"or is negative. Defaulting to highest available.")
                        window[("cycle_nr", i)].update(f"{int(round(highest_available, 0))}")
                        # if values[("f", i)]:
                        #     new_loaded_dataset.append(readout_parsing.savitzky_golay_filtering(data_instead))
                        # else:
                        #     new_loaded_dataset.append(data_instead)
                        #     print(data_instead)

                    except NoCycleInformationError as e:
                        print(e)
                        print("A dataset without cycle data was detected. Move to autodetect.")
                        try:
                            data = cycle_detection_parsing(filename=values[("cv", i)],
                                                           cycle_number=cycle_number,
                                                           assume_standard_csv_format=values["default_csv_format"])

                        except CycleIndexOutOfBoundsError as e:
                            # this exception is thrown by the readout functions at the point of evaluating their return values
                            # TODO same as above
                            requested_cycle = e.requested_cycle
                            highest_available = e.highest_available_cycle
                            data = np.array(e.cycle_data)
                            print(
                                f"Cycle number given {requested_cycle} exceeds the highest available cycle {highest_available} "
                                f"or is negative. Defaulting to highest available.")
                            window[("cycle_nr", i)].update(f"{int(round(highest_available, 0))}")
                        except Exception as e:
                            print(e)
                            print("The dataset " + str(
                                i) + " could not be processed by the auto-cycle-detect, either.")
                            continue
                    except Exception as e:
                        print(e)
                        print("File " + str(i) + " could not be loaded.")
                        continue
                    self.contents.append(FullCV(
                        source=values[("cv", i)],
                        cycle_nr=cycle_number,
                        index=i,
                        dataset=data,
                        eval_voltage=voltage,
                        unit_voltage=pq.V,
                        unit_current=pq.A))

                # CASE: half-cycles
                else:
                    # the case-work from above is not necessary here, since no dataset will provide half-cycle indices.
                    # They have to be assessed here, in any case. The voltage and cycle-nr read-ins from above apply.
                    data = []
                    try:
                        data = half_cycle_parsing(filename=values[("cv", i)],
                                                  assume_standard_csv_format=values["default_csv_format"],
                                                  cycle_number=cycle_number,
                                                  mode=values["halfcycle_mode"],
                                                  window_handle=window,
                                                  window_input_index=i)
                    except Exception as e:
                        print(e)
                        print(f"HalfCV {i} was not successfully loaded.")

                    if data is not []:
                        self.contents.append(HalfCV(
                            source=values[("cv", i)],
                            half_cycle_nr=cycle_number,
                            index=i,
                            dataset=data,
                            eval_voltage=voltage,
                            unit_voltage=pq.V,
                            unit_current=pq.A))

                # At this point, the Dataset is fully loaded.

    # scanrates: list of [index of CV, scanrate]
    def set_scanrates(self, scanrates: list[list]):
        scanrates = np.array(scanrates)
        for i in range(len(scanrates)):
            if self.get_content_by_index(scanrates[i, 0]) is not None:
                self.get_content_by_index(scanrates[i, 0]).set_scanrate(scanrates[i, 1], pq.mV / pq.s)

    def get_content_by_index(self, index):
        for element in self.contents:
            i = element.get_index()
            if index == i:
                return element
        return None

    def count(self):
        return len(self.contents)

    def count_active(self):
        return len([cv for cv in self.contents if cv.is_active()])

    # this is intended as a means to activate and deactivate CVs without deleting them. All indices contained in
    # the list will be set to active. All others to inactive.
    def set_active_for_full_dataset(self, indices_of_active_cv: list[int]):
        for i in range(len(self.contents)):
            if self.contents[i].get_index() in indices_of_active_cv:
                self.contents[i].set_active(True)
            else:
                self.contents[i].set_active(False)

    def set_activity_of_element(self, index: int, activity: bool):
        # This function will not throw errors if the index does not exist. Careful about that.
        for i in range(len(self.contents)):
            if self.contents[i].get_index() == index:
                self.contents[i].set_active(activity)

    def set_default_filtered_of_element(self, index: int, filtered: bool):
        # This function will not throw errors if the index does not exist. Careful about that.
        for i in range(len(self.contents)):
            if self.contents[i].get_index() == index:
                self.contents[i].set_default_filtered(filtered)

    def get_capacitance(self, method="minmax_corrected", through_zero=True, active_only=True, half_cycle_select="full"):
        if method == "at_selected_voltage":
            return self.get_capacitance_at_selected_voltage(through_zero, active_only, half_cycle_select=half_cycle_select)
        if method == "minmax":
            return self.get_capacitance_by_minmax(through_zero, active_only, half_cycle_select=half_cycle_select, corrected=False)
        if method == "minmax_corrected":
            return self.get_capacitance_by_minmax(through_zero, active_only, half_cycle_select=half_cycle_select, corrected=True)

    def get_capacitance_at_selected_voltage(self, through_zero=False, active_only=True,
                                            half_cycle_select: str = "full"):
        def linear(x, m, n):
            return m * x + n

        def linear_through_zero(x, m):
            return m * x

        # select CVs to use
        if active_only:
            subset = [element for element in self.contents if element.is_active()]
        else:
            subset = self.contents

        # get scanrates from CVs
        dataset_to_optimize = []
        for i in range(len(subset)):
            try:
                dataset_to_optimize.append([subset[i].get_index(), subset[i].get_scanrate(pq.mV / pq.s)])
            except ScanrateExistsError as e:
                print(e)

        # at this point, it is assured that at least 2 CVs exist that can be fitted
        if len(dataset_to_optimize) > 1:
            # Praise the Python gods that callables are just objects you can store and pass.
            # Here, it is decided which optimization function to use. Expand at your leisure if other fits are desired.
            if through_zero:
                callable_function = linear_through_zero
            else:
                callable_function = linear

            # scanrates is a list of lists: index, scanrate.
            if half_cycle_select == "full":
                for i in range(len(dataset_to_optimize)):
                    if isinstance(self.get_content_by_index(dataset_to_optimize[i][0]), FullCV):
                        dataset_to_optimize[i].append(
                            self.get_content_by_index(dataset_to_optimize[i][0]).get_current_at_voltage(
                                current_dimension=pq.mA) / 2)
                    else:
                        dataset_to_optimize[i].append(self.get_content_by_index(dataset_to_optimize[i][0]).
                                                      get_current_at_voltage(current_dimension=pq.mA))
            elif half_cycle_select in ["anodic", "cathodic"]:
                for i in range(len(dataset_to_optimize)):
                    if isinstance(self.get_content_by_index(dataset_to_optimize[i][0]), FullCV):
                        dataset_to_optimize[i].append(self.get_content_by_index(dataset_to_optimize[i][0]).
                                                      get_current_at_voltage_in_halfcycle(
                            half_cycle_select=half_cycle_select,
                            current_dimension=pq.mA))
                    else:
                        dataset_to_optimize[i].append(self.get_content_by_index(dataset_to_optimize[i][0]).
                                                      get_current_at_voltage(current_dimension=pq.mA))

            dataset_to_optimize = np.array(dataset_to_optimize)
            popt = opt.curve_fit(callable_function, dataset_to_optimize[:, 1], dataset_to_optimize[:, 2])
            if through_zero:
                capacitance = popt[0][0]
                offset = 0
            else:
                capacitance = popt[0][0]
                offset = popt[0][1]
            # since current is called in mA, scanrate in mV/s, the dimension of C follows.
            # dataset to optimize is now a list of index, scanrate, half-current difference or full anodic/cathodic
            return capacitance, offset, dataset_to_optimize
        else:
            raise NotEnoughCVsToFitError("There are insufficient CVs loaded to perform this calculation.")

    # TODO complete the writing process here. Retrieve the type from the elements themselves.
    # TODO make capacitance calculation available for halfcycles. This should not need to cause any data modification
    # method is in ["at_selected_voltage", "minmax", "minmax_corrected"]
    def write_capacitance_to_file(self, filename: str, through_zero: bool, half_cycle_select: str = "full",
                                  method="minmax_corrected"):
        try:
            capacitance, offset, index_scanrate_current_dataset = self.get_capacitance(
                through_zero=through_zero,
                active_only=True,
                half_cycle_select=half_cycle_select,
                method=method)

        except NotEnoughCVsToFitError as e:
            print(e)
            return False

        index_scanrate_current_dataset = index_scanrate_current_dataset.tolist()
        for k in range(0, len(index_scanrate_current_dataset)):
            local_index = int(index_scanrate_current_dataset[k][0])
            local_cv = self.get_content_by_index(local_index)
            index_scanrate_current_dataset[k].extend([local_cv.get_eval_voltage() * local_cv.unit_voltage,
                                                      local_cv.get_default_filtered(),
                                                      local_cv.get_cycle_nr(),
                                                      type(local_cv).__name__,
                                                      local_cv.source])

        if half_cycle_select == "anodic":
            current_name = f"anodic current {method}"
        elif half_cycle_select == "cathodic":
            current_name = f"cathodic current {method}"
        else:
            current_name = f"half current difference {method}"

        index_scanrate_current_dataset.insert(0, ["Capacitance_Estimate [F]:", capacitance])
        index_scanrate_current_dataset.insert(1, ["Cycle mode selected", half_cycle_select])
        index_scanrate_current_dataset.insert(2, ["Dataset No.", "Scanrate [mV/s]", current_name, "Evaluated at",
                                                  "Filtered",
                                                  "Used cycle", "Data type", "Original Filename"])

        return write_standardized_data_file(filename, index_scanrate_current_dataset)

    def write_CVs_to_files(self, filenames: dict, filtered: dict, comment: dict):
        has_written = True
        for index in filenames.keys():
            write_array = []
            filter_this_cv = filtered.get(index, False)
            comment_this_cv = comment.get(index, None)
            get_cv = self.get_content_by_index(index)
            if get_cv is not None:
                write_array = get_cv.collapse_to_writable_list(
                    filtered=filter_this_cv,
                    comment=comment_this_cv)
            if write_array:
                success = write_standardized_data_file(filenames.get(index), write_array)
                if success:
                    print(f"CV {index} successfully written to file {filenames.get(index)}")
                else:
                    print(f"CV {index} could not be written to file {filenames.get(index)}")
                has_written = has_written and success
        return has_written

    # For convenience in import in Origin, single_file export suppresses the original filenames.
    def write_CVs_to_single_file(self, filename: str, filtered: dict, comment: dict):
        # a custom list-stacking function along the vertical axis, since i could no longer be bothered with np.stack
        def stack_lists(list1: list, list2: list):
            assert len(list1) == len(list2)
            # If this assertion fails, a black hole will emerge and swallow the island of Nauru. This will be rather
            # unpleasant.
            result = []
            for list_index in range(len(list1)):
                result.append([*list1[list_index], *list2[list_index]])
            return result

        # This function has to collect data and metadata from all affected CVs, stack them, and pad datasets of
        # unequal length with neutral characters. Here, tab \t is used.
        data_list = []
        for index in filtered.keys():
            comment_this_cv = comment.get(index, None)
            get_cv = self.get_content_by_index(index)
            if get_cv is not None:
                new_cv_data = self.get_content_by_index(index).collapse_to_writable_list(
                    filtered=filtered.get(index),
                    comment=comment_this_cv)
                data_list.append(new_cv_data)
        # here, we have a list of datasets of even header line number, but uneven length.

        if data_list:
            # determine the length of the longest dataset
            max_len = np.max([len(k) for k in data_list])
            for i in range(len(data_list)):
                # for all datasets, retrieve them
                dataset = data_list[i]
                # while they are shorter than the longest
                while len(dataset) < max_len:
                    # append this padding line
                    dataset.append([" ", " "])
                # finally, replace the shorter dataset with the padded one
                data_list[i] = dataset
            # print(data_list)
            # it's stacking time!
            write_array = data_list[0]
            used_array_counter = 1
            while used_array_counter < len(data_list):
                write_array = stack_lists(write_array, data_list[used_array_counter])
                used_array_counter += 1

            return write_standardized_data_file(filename, write_array)

        else:
            print("There are no datasets loaded that could be written.")
            return False

    def write_distortion_param_results_to_file(self, filename, method):
        # create the data list to write, a 2D array of caption, then results
        write_list = [["CV No.", "Source file", "Resistance [Ohm]", "Capacitance [F]", "Scan rate [mV/s]", "p_d",
                       "Potential window[V]"]]
        for cv in self.contents:
            if cv.is_active():
                try:
                    if method == "Analytical":
                        resistance, capacitance, potential_window, distortion_param = cv.distortion_param_evaluation()
                    elif method == "Optimisation enhanced analytical":
                        resistance, capacitance, potential_window, distortion_param = cv.fit_cv_by_optimisation()
                    if not 'resistance' in locals():
                        raise UnknownMethodError("Method string code was unrecognised.")
                    write_list.append([cv.get_index(), cv.get_source(), resistance.magnitude, capacitance.magnitude,
                                       cv.get_scanrate(dimension=pq.mV/pq.s).magnitude, distortion_param, potential_window])
                except NoScanrateDefinedError:
                    print(f"CV no. {cv.get_index()} failed to evaluate, due to lack of scan rate. "
                          f"Writing process proceeds without it.")

        # Here, the writeable dataset is created, and can be fed to the default-writing routine created above. Return
        # the success boolean that gets returned from the writing function.
        return write_standardized_data_file(filename=filename, data_to_write=write_list)

    def get_capacitance_by_minmax(self, through_zero, active_only, half_cycle_select, corrected):
        def linear(x, m, n):
            return m * x + n

        def linear_through_zero(x, m):
            return m * x

        # select CVs to use
        if active_only:
            subset = [element for element in self.contents if element.is_active()]
        else:
            subset = self.contents

        # get scanrates from CVs
        dataset_to_optimize = []
        for i in range(len(subset)):
            try:
                dataset_to_optimize.append([subset[i].get_index(), subset[i].get_scanrate(pq.mV / pq.s)])
            except ScanrateExistsError as e:
                print(e)

        # at this point, it is assured that at least 2 CVs exist that can be fitted
        if len(dataset_to_optimize) > 1:
            # Praise the Python gods that callables are just objects you can store and pass.
            # Here, it is decided which optimization function to use. Expand at your leisure if other fits are desired.
            if through_zero:
                callable_function = linear_through_zero
            else:
                callable_function = linear
            # TODO Apply units to this calculation properly! Interpolation Minmax does not integrate units.
            # scanrates is a list of lists: index, scanrate.
            if half_cycle_select == "full":
                for i in range(len(dataset_to_optimize)):
                    cv = self.get_content_by_index(dataset_to_optimize[i][0])
                    if isinstance(cv, FullCV):
                        current_to_append = (cv.get_minmax_current() / 2)
                    else:
                        current_to_append = (cv.get_minmax_current())

                    current_to_append.units = pq.mA
                    dataset_to_optimize[i].append(current_to_append)
            elif half_cycle_select == "anodic":
                for i in range(len(dataset_to_optimize)):
                    cv = self.get_content_by_index(dataset_to_optimize[i][0])
                    if isinstance(cv, FullCV):
                        filter_dataset = cv.get_default_filtered()
                        if filter_dataset:
                            current_to_append = max(cv.get_filtered_dataset()[:, 1])
                        else:
                            current_to_append = max(cv.get_dataset()[:, 1])

                        current_to_append *= cv.unit_current
                        current_to_append.units = pq.mA
                        dataset_to_optimize[i].append(current_to_append)
                    # @TODO AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
                    else:
                        dataset_to_optimize[i].append(self.get_content_by_index(dataset_to_optimize[i][0]).
                                                      get_current_at_voltage(current_dimension=pq.mA))
            elif half_cycle_select == "cathodic":
                for i in range(len(dataset_to_optimize)):
                    cv = self.get_content_by_index(dataset_to_optimize[i][0])
                    if isinstance(cv, FullCV):
                        filter_dataset = cv.get_default_filtered()
                        if filter_dataset:
                            current_to_append = min(cv.get_filtered_dataset()[:, 1])
                        else:
                            current_to_append = min(cv.get_dataset()[:, 1])

                        current_to_append *= cv.unit_current
                        current_to_append.units = pq.mA
                        dataset_to_optimize[i].append(current_to_append)
                    # @TODO AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA
                    else:
                        dataset_to_optimize[i].append(self.get_content_by_index(dataset_to_optimize[i][0]).
                                                      get_current_at_voltage(current_dimension=pq.mA))

            # CORRECTION FACTOR APPLICATION
            if corrected:
                for i in range(len(dataset_to_optimize)):
                    cv = self.get_content_by_index(dataset_to_optimize[i][0])
                    vert_ratio, distortion_param = cv.estimate_vertical_current_ratio()
                    print(f"Corrected dataset: Distortion param {distortion_param}, correction by {1 / vert_ratio}")
                    dataset_to_optimize[i][-1] = dataset_to_optimize[i][-1] / vert_ratio

            dataset_to_optimize = np.array(dataset_to_optimize)
            popt = opt.curve_fit(callable_function, dataset_to_optimize[:, 1], dataset_to_optimize[:, 2])
            if through_zero:
                capacitance = popt[0][0]
                offset = 0
            else:
                capacitance = popt[0][0]
                offset = popt[0][1]
            # since current is called in mA, scanrate in mV/s, the dimension of C follows.
            # dataset to optimize is now a list of index, scanrate, half-current difference or full anodic/cathodic
            return capacitance, offset, dataset_to_optimize
        else:
            raise NotEnoughCVsToFitError("There are insufficient CVs loaded to perform this calculation.")
