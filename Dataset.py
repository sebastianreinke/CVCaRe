import os
import numpy as np
import CV
from CV import FullCV, HalfCV
import quantities as pq
from scipy import optimize as opt
import PySimpleGUI as sg
from custom_exceptions import ScanrateExistsError, NotEnoughCVsToFitError, CycleIndexOutOfBoundsError, \
    NoCycleInformationError, NoScanrateDefinedError, UnknownMethodError

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
    """
    Reads and parses a CSV file, returning the dataset starting from the identified header row.

    This function reads the contents of a CSV file and attempts to parse it into a dataset. The function handles various
    common CSV formats, including those with semicolon (`;`), tab (`\t`), or space-separated values. The function also
    identifies the header row based on the presence of voltage and current signifiers, ensuring that the returned dataset
    includes only the data rows. The signifiers are defined at the top of Dataset.py, for easy inclusion of new formats.

    Parameters:
        filename (str):
            The path to the CSV file to be read.

        assume_standard_csv_format (bool, optional):
            If True, the function assumes that the file is in a standard CSV format where values are comma-separated
            and automatically converts commas to semicolons for further processing. If False (default), the function
            attempts to determine the separator used in the file.

    Returns:
        dataset (list of lists):
            A list of lists where each sublist represents a row in the CSV file, starting from the identified header row.
            Each row contains the parsed values as strings.
            If the file is not found, the function returns `None`.

    Raises:
        FileNotFoundError:
            If the specified file does not exist, the function catches this exception and prints an error message,
            returning `None`.

    Notes:
        - The function attempts to detect the separator in the file, prioritizing semicolons (`;`), then tabs (`\t`),
          and finally spaces (` `). This is to accommodate different CSV formats.
        - If the `assume_standard_csv_format` parameter is set to True, the function will first convert all commas
          to semicolons before further processing.
        - The function identifies the header row by searching for common voltage and current signifiers. Data rows
          below this header row are returned.

    Example Usage:
        ```python
        dataset = read_csv_file("data.csv", assume_standard_csv_format=True)
        if dataset is not None:
            for row in dataset:
                print(row)
        ```
    """
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
        if element in (None, "\n", "\t\n", ",\n", ";\n", " \n"):
            continue
        if assume_standard_csv_format:
            element = ";".join(element.split(','))

        # candidate line to be appended, if suitable
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
    potential_correction_factor = 1
    cycle_column = 2
    # with open(filename, "r+") as file:
    # reader = csv.reader(file, delimiter=' ')
    try:
        read_in = read_csv_file(filename, assume_standard_csv_format)
        has_found_header_line = False
        # if no cycle number is given, return the entire dataset
        for line in read_in:
            try:
                dataset.append([float(line[e_column].replace(",", ".")) * potential_correction_factor * pq.V,
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
                    
                    # find potential correction factor
                    if line[e_column].__contains__("mV"):
                        potential_correction_factor = 0.001
                    elif line[e_column].__contains__("uV"):
                        potential_correction_factor = 0.000001
                    elif line[e_column].__contains__("nV"):
                        potential_correction_factor = 10 ** -9
                    else:
                        potential_correction_factor = 1
                        
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
    """
    Loads a specific cycle from a cyclic voltammetry (CV) dataset contained in a CSV file.

    This function reads a CSV file containing cyclic voltammetry data, identifies the relevant voltage, current, and
    cycle columns, and extracts the data corresponding to a specified cycle number. The data is returned as a NumPy
    array, with the option to return the full dataset if required. The function also handles different current units
    by applying a correction factor based on the unit found in the header.

    Parameters:
        filename (str):
            The path to the CSV file containing the CV dataset.

        cycle_number (int):
            The specific cycle number to load from the dataset. The function will extract and return the data
            corresponding to this cycle.

        assume_standard_csv_format (bool):
            If True, the function assumes the CSV file is in a standard format with comma-separated values and converts
            commas to semicolons for parsing. If False, the function attempts to automatically detect the correct separator.

        throw_full_dataset (bool, optional):
            If True, the function returns the entire dataset after parsing. If False (default), only the data
            corresponding to the specified cycle number is returned.

    Returns:
        np.array:
            A NumPy array containing the voltage, current, and cycle number for the specified cycle. Each row represents
            a data point, with the voltage in Volts, current in Amperes, and the cycle number as a float.

        If `throw_full_dataset` is True, the function returns the entire dataset as a list of lists.

    Raises:
        FileNotFoundError:
            If the specified CSV file does not exist.

        ValueError:
            If the CSV file contains malformed data that cannot be parsed into floats.

        NoCycleInformationError:
            If the dataset does not contain cycle information. This exception is then handled in the main loading
            routine, and cycle_detection_parsing is called to create the cycle information from the data.

        CycleIndexOutOfBoundsError:
            If the requested cycle number does not exist in the dataset, this exception is raised with the highest
            available cycle returned instead.

    Notes:
        - The function assumes that the CSV file contains a header row that specifies the units for current (e.g., mA,
          uA, nA). A correction factor is applied to convert the current to Amperes based on the unit.
        - The function identifies the header row based on the presence of voltage, current, and cycle columns. If these
          columns are not found, or if the dataset is malformed, an exception may be raised.
        - The function defaults to the highest available cycle if the requested cycle number is out of bounds.

    Example Usage:
        ```python
        cycle_data = load_one_cycle("cv_data.csv", 5, assume_standard_csv_format=True)
        if cycle_data is not None:
            print("Cycle data loaded successfully.")
        ```
    """
    # TODO Possibly include a voltage_correction_factor, if there are datasets with the potential in mV, for example.
    dataset = []
    e_column = 0
    i_column = 1

    current_correction_factor = 1
    potential_correction_factor = 1
    cycle_column = 2
    try:
        read_in = read_csv_file(filename, assume_standard_csv_format)
        has_found_header_line = False
        for line in read_in:
            try:
                # if float(line[cycle_column].replace(",", ".")) == cycle_number:
                dataset.append([float(line[e_column].replace(",", ".")) * potential_correction_factor * pq.V,
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

                    # find potential correction factor
                    if line[e_column].__contains__("mV"):
                        potential_correction_factor = 0.001
                    elif line[e_column].__contains__("uV"):
                        potential_correction_factor = 0.000001
                    elif line[e_column].__contains__("nV"):
                        potential_correction_factor = 10 ** -9
                    else:
                        potential_correction_factor = 1

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
    """
    Detects and parses cycles from a cyclic voltammetry (CV) dataset in a CSV file, returning the data for a specific cycle.

    This function reads a CSV file containing cyclic voltammetry (CV) data, detects the different cycles within the
    dataset based on voltage thresholds, and returns the data corresponding to a specified cycle number. The function
    can also return the entire dataset with cycle numbers appended to each data point if requested. The function assumes
    that the voltage scan direction (upwards or downwards) can be inferred from the initial part of the dataset.

    Parameters:
        filename (str):
            The path to the CSV file containing the CV dataset.

        cycle_number (int):
            The specific cycle number to extract from the dataset. The function will parse the dataset and return the
            data corresponding to this cycle.

        assume_standard_csv_format (bool):
            If True, the function assumes the CSV file is in a standard format with comma-separated values and converts
            commas to semicolons for parsing. If False, the function attempts to automatically detect the correct separator.

        throw_full_dataset (bool, optional):
            If True, the function returns the entire dataset with cycle numbers appended to each data point.
            If False (default), only the data corresponding to the specified cycle number is returned.

    Returns:
        np.array:
            A NumPy array containing the voltage and current data for the specified cycle. Each row represents a data
            point, with voltage in Volts and current in Amperes.

        If `throw_full_dataset` is True, the function returns the entire dataset with an additional column indicating
        the cycle number for each data point.

    Raises:
        FileNotFoundError:
            If the specified CSV file does not exist.

        ValueError:
            If the CSV file contains malformed data that cannot be parsed into floats.

        CycleIndexOutOfBoundsError:
            If the requested cycle number does not exist in the dataset, this exception is raised with the highest
            available cycle returned instead.

    Notes:
        - The function assumes that the dataset starts with a sufficiently large sample size to determine the initial
          scan direction (upwards or downwards). If this assumption does not hold, the cycle detection may be inaccurate.
        - The function adjusts for various current units (e.g., mA, uA, nA) found in the dataset by applying a correction
          factor to normalize the current values to Amperes.
        - If the requested cycle is not found, the function defaults to the highest available cycle and raises an
          exception to notify the caller.
        - Some manufacturers store an initially constant potential before the actual CV initiates. Because there is an
          inherent tradeoff between the minimum resolution of a CV that can be parsed, and the robustness against this
          behaviour, a very long initial phase may have to be manually deleted from the dataset to ensure compatibility.

    Example Usage:
        ```python
        cycle_data = cycle_detection_parsing("cv_data.csv", 5, assume_standard_csv_format=True)
        if cycle_data is not None:
            print("Cycle data loaded successfully.")
        ```

    """
    # default the columns to first, second
    e_column = 0
    i_column = 1
    current_cycle = 0

    # a factor used to norm the read-in dataset to ampere
    current_correction_factor = 1
    potential_correction_factor = 1
    dataset = []
    # this value saves the first potential value and is used to mark new cycles
    e_treshold = 0
    # Does the voltage scan start with rising voltage?

    try:
        read_in = read_csv_file(filename, assume_standard_csv_format)
        has_found_header_line = False
        for line in read_in:
            try:
                voltage = float(line[e_column].replace(",", ".")) * potential_correction_factor * pq.V
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

                    # find potential correction factor
                    if line[e_column].__contains__("mV"):
                        potential_correction_factor = 0.001
                    elif line[e_column].__contains__("uV"):
                        potential_correction_factor = 0.000001
                    elif line[e_column].__contains__("nV"):
                        potential_correction_factor = 10 ** -9
                    else:
                        potential_correction_factor = 1
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
        if dataset[10][0] != e_treshold:
            direction_upwards = (dataset[10][0] > e_treshold)
            meaningful_start_of_cv = 10
        if direction_upwards is None and dataset[40][0] != e_treshold:
            direction_upwards = (dataset[40][0] > e_treshold)
            meaningful_start_of_cv = 40
        # if it is still the same, enforce a decision at index 100
        if direction_upwards is None and dataset[100][0] != e_treshold:
            direction_upwards = (dataset[100][0] > e_treshold)
            meaningful_start_of_cv = 100
        # TODO Was I on crack? Why 250?
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
            if i > 10:
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
    """
    Identifies positive and negative voltage peaks in a cyclic voltammetry (CV) dataset.

    This function analyzes a cyclic voltammetry dataset to detect the indices of positive and negative voltage peaks.
    A positive peak is defined as a point where the voltage value is greater than both its immediate neighbors and
    those a fixed distance away, indicating a local maximum. Similarly, a negative peak is identified as a local
    minimum where the voltage value is lower than its neighbors. The function also determines the initial scan
    direction based on the dataset.

    Parameters:
        full_dataset (np.array):
            A 2D NumPy array where the first column represents the voltage values and the second column represents
            the current values. The function primarily operates on the voltage column to detect peaks.

    Returns:
        direction_upwards (bool):
            A boolean value indicating the initial scan direction. If True, the scan begins with rising voltage;
            if False, the scan begins with falling voltage.

        negative_peaks (list of int):
            A list of indices in the dataset where negative voltage peaks (local minima) are detected.

        positive_peaks (list of int):
            A list of indices in the dataset where positive voltage peaks (local maxima) are detected.

    Notes:
        - The function assumes that the dataset starts at index 0 and uses a look-ahead/look-back distance of 10 points
          to identify peaks. This helps to ensure that only significant peaks are detected, avoiding noise.
        - The function inserts 0 as the first element in both the `positive_peaks` and `negative_peaks` lists to
          account for the starting point of the dataset, ensuring that the first point is considered a peak if no
          other condition excludes it.
        - This function operates with the assumption that the dataset is sufficiently sampled and that peaks are
          distinguishable within the provided window size (10 points).

    Example Usage:
        ```python
        full_dataset = np.array([[0.1, 0.2], [0.2, 0.4], ...])
        direction_upwards, negative_peaks, positive_peaks = get_positive_and_negative_voltage_peaks(full_dataset)
        if direction_upwards:
            print("The scan begins with a rising voltage.")
        print(f"Positive peaks are at indices: {positive_peaks}")
        print(f"Negative peaks are at indices: {negative_peaks}")
        ```
    """
    # direction_upwards = True
    # if full_dataset[40][0] < full_dataset[0][0]:
    #     direction_upwards = False
    # positive_peaks = []
    # negative_peaks = []
    # # iterate over all candidate center-elements in a len-3 comparison filter => exclude first and last element
    # for index in range(40, len(full_dataset) - 10):
    #     if full_dataset[index - 1, 0] < full_dataset[index, 0] > full_dataset[index + 1, 0] and \
    #             full_dataset[index - 10, 0] < full_dataset[index, 0] > full_dataset[index + 10, 0]:
    #         positive_peaks.append(index)
    #     if full_dataset[index - 1, 0] > full_dataset[index, 0] < full_dataset[index + 1, 0] and \
    #             full_dataset[index - 10, 0] > full_dataset[index, 0] < full_dataset[index + 10, 0]:
    #         negative_peaks.append(index)
    # # positive_peaks = sgnl.find_peaks(full_dataset[:, 0])
    # # negative_peaks = sgnl.find_peaks(-1*full_dataset[:, 0])
    # # positive_peaks = positive_peaks[0].tolist()
    # # negative_peaks = negative_peaks[0].tolist()
    # positive_peaks.insert(0, 0)
    # negative_peaks.insert(0, 0)
    # return direction_upwards, negative_peaks, positive_peaks
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


# DATASET CLASS
# This is the class handling a read-in dataset. It thereby becomes possible to separate CV and Dataset level calculation
# and locate them precisely in the appropriate classes. A dataset contains CVs. It is self-loading, given the correct
# set of initialization parameters. All static helper functions required are above this class.
class Dataset:
    """
    Handles the loading and management of cyclic voltammetry (CV) datasets.

    The `Dataset` class is designed to manage and process multiple cyclic voltammetry (CV) datasets. It separates
    CV-level calculations and dataset-level management, making it easier to handle complex operations on multiple CVs.
    The class can automatically load and parse datasets based on initialization parameters and provides various methods
    to manage, filter, and analyze the CV data.

    Attributes:
        contents (list[CV]):
            A list of CV objects that the dataset contains. Each CV object represents an individual cyclic voltammetry
            experiment, either as a full cycle (`FullCV`) or half cycle (`HalfCV`).

    Methods:
        __init__(self, values: dict, count: int, window):
            Initializes the dataset by loading and parsing the specified number of CVs based on the given values and
            window handle. The values, window parameters are given from the initialised PySimpleGUI window defined in
            gui.py.

        set_scanrates(self, scanrates: list[list]):
            Sets the scan rates for the CVs in the dataset.

        get_content_by_index(self, index):
            Retrieves the CV object at the specified index within the dataset.

        count(self):
            Returns the total number of CVs in the dataset.

        count_active(self):
            Returns the number of active CVs in the dataset.

        set_active_for_full_dataset(self, indices_of_active_cv: list[int]):
            Activates the CVs at the specified indices and deactivates all others.

        set_activity_of_element(self, index: int, activity: bool):
            Sets the active status of the CV at the specified index.

        set_default_filtered_of_element(self, index: int, filtered: bool):
            Sets the default filtering status of the CV at the specified index.

        get_capacitance(self, method="minmax_corrected", through_zero=True, active_only=True, half_cycle_select="full"):
            Calculates the capacitance of the active CVs using the specified method.

        get_capacitance_at_selected_voltage(self, through_zero=False, active_only=True, half_cycle_select: str = "full"):
            Calculates the capacitance based on the current at the selected voltage for the active CVs.

        write_capacitance_to_file(self, filename: str, through_zero: bool, half_cycle_select: str = "full", method="minmax_corrected"):
            Writes the capacitance calculations to a file.

        write_CVs_to_files(self, filenames: dict, filtered: dict, comment: dict):
            Writes the CV data to individual files, one for each CV.

        write_CVs_to_single_file(self, filename: str, filtered: dict, comment: dict):
            Writes the CV data to a single file, stacking the datasets side by side for convenience.

        write_distortion_param_results_to_file(self, filename, method):
            Writes the distortion parameter results to a file using the specified method.

        get_capacitance_by_minmax(self, through_zero, active_only, half_cycle_select, corrected):
            Calculates capacitance by finding the minimum and maximum current values for the active CVs.

    Example Usage:
        ```python
        values = {
            ("cv", 1): "cv_data_1.csv",
            ("cycle_nr", 1): "5",
            ("voltage_eval", 1): "0.5",
            "halfcycle_mode": "full cycles",
            "default_csv_format": True
        }
        window = some_window_handle
        dataset = Dataset(values, count=1, window=window)
        print(f"Number of CVs loaded: {dataset.count()}")
        ```
    """
    # The contents are of abstract type CV. It is therefore strictly necessary for compatibility to use only declared
    # methods in the abstract class CV for any type of calculations.
    contents: list[CV]

    # the values dictionary is handed in from the gui. This could be made more efficient/stringent by only selecting
    # the values actually used here, but since they are around 90% of the dictionary, at this time, I am too lazy.
    # TODO extract the window update from this function. It should be done at the level where this function is called.
    def __init__(self, values: dict, count: int, window):
        """
        Initializes the Dataset object by loading and parsing the specified number of cyclic voltammetry (CV) datasets.

        This constructor reads the provided `values` dictionary and loads the corresponding CV data for each entry. The
        datasets are either full cycles or half-cycles, depending on the specified mode. It also handles various exceptions
        related to dataset loading, such as missing or invalid cycle numbers, and updates the associated GUI window
        elements accordingly.

        Parameters:
            values (dict):
                A dictionary containing the paths to CV files, cycle numbers, evaluation voltages, and other relevant
                settings passed from the GUI. The keys are tuples where the first element is a string indicating the
                type of value (e.g., "cv", "cycle_nr") and the second element is an integer representing the index.

            count (int):
                The number of CV datasets to load. The constructor will attempt to load and process this many datasets.

            window:
                A handle to the GUI window that allows the function to update the displayed cycle number if necessary.

        Raises:
            CycleIndexOutOfBoundsError:
                If the specified cycle number exceeds the highest available cycle in the dataset, or if it is negative.
                In this case, the constructor defaults to the highest available cycle and updates the GUI accordingly.

            NoCycleInformationError:
                If the dataset does not contain cycle information, the constructor attempts to auto-detect cycles.

            Exception:
                Catches any other exceptions that may occur during the loading of datasets and prints an error message.

        Notes:
            - The constructor currently updates the GUI window directly, which could be refactored to improve efficiency
              and separation of concerns.
            - The `values` dictionary contains more data than is used by this constructor, but only the relevant portions
              are processed.
        """
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
                        resistance, capacitance, potential_window, distortion_param, offset = cv.distortion_param_evaluation()
                    elif method == "Optimisation enhanced analytical":
                        resistance, capacitance, potential_window, distortion_param, offset = cv.fit_cv_by_optimisation()
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
