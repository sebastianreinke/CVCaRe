import numpy as np
import csv
import matplotlib
import scipy.signal as sgnl
from matplotlib import pyplot as plt
# from scipy import signal
import quantities as pq

# TODO Option cycle number - DONE
# TODO Option Savitzky-Golay filter - DONE
# TODO detect current difference  - DONE
# TODO GUI => recycle - DONE
# TODO single-CV and multi-CV capacity determination - DONE
# TODO plotting - DONE
# TODO multiple CVs with separate selectable Cycles - DONE
# TODO only anodic/cathodic current plotting or extraction - DONE
# TODO unify format support (if line in ["Ewe", "Potential"]...  +- DONE


def readCSVfile(filename):
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
    # print(dataset)
    dataset = []
    for element in string_list:
        cand_line = None
        if element.__contains__('\t'):
            cand_line = element.replace('\n', '').split('\t')
        elif element.__contains__(' '):
            cand_line = element.replace('\n', '').split(' ')
        elif element.__contains__(';'):
            cand_line = element.replace('\n', '').split(';')
        # if the split has occured, the element should now be list-type. If not, the error is escalated
        # through an AssertionError to be handled at other levels.
        assert type(cand_line) == list
        # remove all None-type split results caused by doubled separators
        dataset.append([k for k in cand_line if k is not None and k != ' ' and k != ''])
        # print(dataset[-1])

    # print(dataset)
    return dataset


# TODO develop file format - which separator? []? {}?
# TODO Export filename, name, scanrate, eval_voltage, filtered, voltage, current (A), current (uA)
# which information:
# raw data set, scanrate, filtered?, used voltage for eval, capacitance, date, filename, chemical setup,
# cycle number used,  comment
# take a well-formed 2D array
def write_standardized_data_file(filename, data_to_write):
    try:
        with open(filename, "a") as file:
            for i in range(len(data_to_write)):
                writestring = ""
                for element in data_to_write[i]:
                    writestring += str(element) + "\t"
                file.write(writestring + "\n")
    except FileNotFoundError or FileExistsError:
        print("There was an error opening the file to write on.")


# PINE helper function
# AfterMath does not currently export cycle number
# this function takes a read dataset of U, I and appends a third column cycle_number
def cycle_detection_parsing(filename, cycle_number):
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
    direction_upwards = True
    # with open(filename, "r+") as file:
    try:
        read_in = readCSVfile(filename)
        #reader = csv.reader(file, delimiter=' ')
        for line in read_in:
            try:
                voltage = float(line[e_column].replace(",", ".")) * pq.V
                current = float(line[i_column].replace(",", ".")) * current_correction_factor * pq.A
                # initialize e_treshold at first value
                if current_cycle == 0:
                    e_treshold = voltage
                    current_cycle = 1

                # at the 40th element of the dataset, compare which direction the voltage has gone
                if len(dataset) == 40:
                    direction_upwards = (dataset[-1][0] > e_treshold)
                    print(f"Voltage begins with an initial rise: {direction_upwards} ")
                # depending on whether the scan begins upwards or downwards, cut the cycles accordingly
                # this will cause errors with very low-sample cycle data, in that case, reduce 40 to a convenient number
                if len(dataset) > 40:
                    if direction_upwards and voltage >= e_treshold > dataset[-1][0]:
                        current_cycle += 1
                    if not direction_upwards and voltage <= e_treshold < dataset[-1][0]:
                        current_cycle += 1

                dataset.append([voltage, current, current_cycle])
            except ValueError:
                e_column = [i for i in range(len(line)) if (line[i].__contains__("Ewe") or
                                                            line[i].__contains__("Potential") or
                                                            line[i].__contains__("E/"))][0]
                i_column = [i for i in range(len(line)) if (line[i].__contains__("<I>") or
                                                            line[i].__contains__("Current") or
                                                            line[i].__contains__("I/"))][0]
                # find current correction factor
                if line[i_column].__contains__("mA"):
                    current_correction_factor = 0.001
                elif line[i_column].__contains__("uA"):
                    current_correction_factor = 0.000001
                elif line[i_column].__contains__("nA"):
                    current_correction_factor = 10**-9
                else:
                    current_correction_factor = 1
                continue
        # select cycle and catch invalid cycles, defaulting to the largest possible
        ret_value = [line for line in dataset if line[2] == cycle_number]
        # dataset = np.array(dataset)

        if not ret_value:
            # print(np.array([line for line in dataset if line[2] == current_cycle]))
            raise IndexError(cycle_number, current_cycle, [line for line in dataset if line[2] == current_cycle])
        return np.array([line for line in dataset if line[2] == cycle_number])
    except AssertionError:
        print("The dataset contains malformed separators that are not yet handled in the program.")


def standard_parsing(filename, cycle_number=0):
    dataset = []
    e_column = 0
    i_column = 1

    current_correction_factor = 1
    cycle_column = 2
    # with open(filename, "r+") as file:
        # reader = csv.reader(file, delimiter=' ')
    try:
        read_in = readCSVfile(filename)
        if cycle_number == 0:
            # if no cycle number is given, return the entire dataset
            for line in read_in:
                try:
                    dataset.append([float(line[e_column].replace(",", ".")) * pq.V,
                                    float(line[i_column].replace(",", ".")) * current_correction_factor * pq.A])
                except ValueError:
                    e_column = [i for i in range(len(line)) if (line[i].__contains__("Ewe") or
                                                                line[i].__contains__("Potential") or
                                                                line[i].__contains__("E/"))][0]
                    i_column = [i for i in range(len(line)) if (line[i].__contains__("<I>") or
                                                                line[i].__contains__("Current") or
                                                                line[i].__contains__("I/"))][0]
                    # find current correction factor
                    if line[i_column].__contains__("mA"):
                        current_correction_factor = 0.001
                    elif line[i_column].__contains__("uA"):
                        current_correction_factor = 0.000001
                    elif line[i_column].__contains__("nA"):
                        current_correction_factor = 10 ** -9
                    else:
                        current_correction_factor = 1
                    continue
        else:
            # determine the column where cycle number is given
            for line in read_in:
                try:
                    #if float(line[cycle_column].replace(",", ".")) == cycle_number:
                    dataset.append([float(line[e_column].replace(",", ".")) * pq.V,
                                    float(line[i_column].replace(",", ".")) * current_correction_factor * pq.A,
                                    float(line[cycle_column].replace(",", "."))])
                except ValueError as e:
                    e_column = [i for i in range(len(line)) if line[i].__contains__("Ewe") or
                                line[i].__contains__("Potential") or
                                line[i].__contains__("E/")][0]
                    i_column = [i for i in range(len(line)) if line[i].__contains__("<I>") or
                                line[i].__contains__("Current") or
                                line[i].__contains__("I/")][0]
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
                    if not str(*cycle_column).isnumeric():
                        raise TypeError("The dataset does not appear to contain cycle data.")
                    else:
                        cycle_column = cycle_column[0]
                        continue
            # select cycle and catch invalid cycles, defaulting to the largest possible
            ret_value = [line for line in dataset if line[2] == cycle_number]
            #print(ret_value)
            # print(dataset)
            dataset = np.array(dataset)
            # print(dataset[:, 2])
            if not ret_value:
                # print(np.array([line for line in dataset if line[2] == max(dataset[:, 2])]))
                raise IndexError(cycle_number, max(dataset[:, 2]),
                                 [line for line in dataset if line[2] == max(dataset[:, 2])])
            return np.array([line for line in dataset if line[2] == cycle_number])

        dataset = np.array(dataset)
        return dataset
    except AssertionError:
        print("The dataset contains malformed separators that are not yet handled in the program.")


def half_cycle_parsing(filename, cycle_number=1, mode="forward"):
    full_dataset = standard_parsing(filename, 0)
    # returns a u, i list of lists

    direction_upwards = True
    if full_dataset[40][0] < full_dataset[0][0]:
        direction_upwards = False

    positive_peaks = []
    negative_peaks = []
    # iterate over all candidate center-elements in a len-3 comparison filter => exclude first and last element
    for index in range(40, len(full_dataset)-10):
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
    # print(f"positive peaks {positive_peaks}")
    # print(f"negative peaks {negative_peaks}")

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
            if direction_upwards:
                return full_dataset[positive_peaks[1]:negative_peaks[1]]
            else:
                return full_dataset[positive_peaks[0]:negative_peaks[1]]


# in: np.array of e, i
# out: smoothed np.array of e, i
def savitzky_golay_filtering(data):
    e_smooth = sgnl.savgol_filter(data[:, 0], window_length=11, polyorder=3, mode="nearest")
    i_smooth = sgnl.savgol_filter(data[:, 1], window_length=11, polyorder=3, mode="nearest")
    # print(e_smooth)
    return np.stack((e_smooth, i_smooth), axis=1)


# with a given voltage or without, find the difference between anodic and cathodic cycle current.
# If no position is given, search the data for the maximum difference, since distortion decreases the current difference
def min_max_current(data, voltage_position=None, half_cycle_select=None):
    if voltage_position is not None:
        # find anodic and cathodic closest indices where the voltage matches most closely voltage_position
        # this assumes that the potentiostat measures in fixed E-intervals, so that once e.g. the cathodic closest
        # voltage is deleted, the anodic scan has the closest match. If for some reason the data is heavily oversampled
        # in one scan, this code will not work and life becomes very sad indeed.
        # print(data[:, 0])

        positive_closest = (np.abs(data[:, 0] - voltage_position)).argmin()
        negative_closest = (np.abs(np.delete(data, positive_closest, 0)[:, 0] - voltage_position)).argmin()
        if half_cycle_select is None:
            return np.abs(data[positive_closest, 1] - data[negative_closest, 1]), positive_closest
        elif half_cycle_select == "anodic":
            print(data[positive_closest, 1])
            print(data[negative_closest, 1])
            return min(data[positive_closest, 1], data[negative_closest, 1]), negative_closest
        elif half_cycle_select == "cathodic":
            return max(data[positive_closest, 1], data[negative_closest, 1]), positive_closest
        else:
            print("Half-cycle-select-Parameter ist unzulässig gesetzt.")
    else:
        difference_landscape = []
        for i in range(len(data[:, 0])):
            direct_value = data[i, 1]
            negative_closest = (np.abs(np.delete(data, i, 0)[:, 0] - data[i, 0])).argmin()
            difference_landscape.append(direct_value - data[negative_closest, 1])
        # plt.plot(difference_landscape)
        # plt.title("Current difference landscape over the cycle")
        # plt.show()
        if half_cycle_select is None:
            max(difference_landscape), data[np.where(difference_landscape == max(difference_landscape)), 0]
        elif half_cycle_select == "anodic":
            i = np.where(difference_landscape == max(difference_landscape))
            direct_value = data[i, 1]
            negative_closest = (np.abs(np.delete(data, i, 0)[:, 0] - data[i, 0])).argmin()
            negative_value = data[negative_closest, 1]
            return min(direct_value, negative_value), negative_closest
        elif half_cycle_select == "cathodic":
            i = np.where(difference_landscape == max(difference_landscape))
            direct_value = data[i, 1]
            negative_closest = (np.abs(np.delete(data, i, 0)[:, 0] - data[i, 0])).argmin()
            negative_value = data[negative_closest, 1]
            return max(direct_value, negative_value), direct_value
        else:
            print("Half-cycle-select-Parameter ist unzulässig gesetzt.")
        return max(difference_landscape), data[np.where(difference_landscape == max(difference_landscape)), 0]


# data = biologic_parsing("Z:/Research/Capacitance-Project/Pt-Micro, 1M H2SO4, repolished, 1M KCl, 0.1M KCl 16.11.22/06_0.1MKCl_OCP10min_10mVs_640mVs_x2_reihe_30secOCV_dazwischen_14_CV_C01.txt",3)
#
# # Z:/Research/Capacitance-Project/Pt-Micro, 1M H2SO4, repolished, 1M KCl, 0.1M KCl 16.11.22/"
# # #                         "05_1MKCl_10mVs_640mVs_x2_reihe_30secOCV_dazwischen_13_CV_C01.txt
# # Z:/Research/Capacitance-Project/Pt-Macro, 1M H2S04, KCl, 14.11.22/19_1MKCl_BW6_bis20mVs_100uA_dann_auto_OCP_CV_reihe_06_CV_C01.txt
# data = savitzky_golay_filtering(data)
#
# max_delta_i, position = min_max_current(data)
# print(max_delta_i)
# print(data[position, 0])
# plt.title("Cycle plotted with detected max_current_difference marked")
# plt.plot(data[:, 0], data[:, 1])
# plt.vlines(data[position, 0], -max(data[:, 1]), max(data[:, 1]), colors="orange")
# plt.show()

