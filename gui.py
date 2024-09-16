"""
This file is part of CVCaRe. It is a cyclic voltammogram analysis tool that enables you to calculate capacitance and resistance from capacitive cyclic voltammograms.
    Copyright (C) 2022-2024 Sebastian Reinke

CVCaRe is free software: you can redistribute it and/or modify it under the terms of the GNU General Public License as published by the Free Software Foundation, either version 3 of the License, or (at your option) any later version.

CVCaRe is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with CVCaRe. If not, see <https://www.gnu.org/licenses/>.

"""

from functools import partial

import PySimpleGUI as sg
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import quantities as pq
import os.path
from CV import FullCV
from custom_exceptions import NotEnoughCVsToFitError, VoltageBoundsOutsideCVError, FunctionNotImplementedError, \
    ScanrateExistsError, NoScanrateDefinedError, UnknownMethodError
import Dataset
# from numba import jit


########################################################################################################################
# GLOBAL SETTINGS
########################################################################################################################

matplotlib.use("TkAgg")
sg.theme('DarkGrey2')

# This is the global setting of Units to use in outputting and writing
setUnits = [pq.A, pq.V, pq.mF]

# This is a global setting declared here for easy adjustability. It is a list in the order of the colors used to plot
# the first, second, etc. CV.
color_cycle = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray',
               'tab:olive', 'tab:cyan', "black", "lightgrey", "firebrick", "chocolate", "navy", "indigo", "skyblue",
               "yellowgreen"]


# TODO create CV.downsampled() method for faster display
# TODO EXAMINE routine to avoid duplicate loading (possibly unnecessary)


########################################################################################################################
# FUNCTION BLOCK
# HELPER FUNCTIONS FOR THE PROPER CREATION OF THE GUI AND CALLING THE DATASET METHODS
########################################################################################################################

def calculate_RC_cv(Res, Cap, T_p, Amp):
    # @jit(nopython=True)
    def forward_current(t, A, T, R, C):
        return (-8 * A * C / T) * np.exp(-t / (R * C)) / (1 + np.exp(-T / (2 * R * C))) + 4 * A * C / T

    # @jit(nopython=True)
    def backward_current(t, A, T, R, C):
        return (8 * A * C / T) * np.exp((-t + T) / (R * C)) / (1 + np.exp(T / (2 * R * C))) - 4 * A * C / T

    def voltage_form(time_array, amplitude, forward=True):
        interval = time_array[-1] - time_array[0]
        time_spacing = time_array[1] - time_array[0]
        rise = 2 * amplitude / interval
        voltage_response = []
        for item in time_array:
            if forward:
                voltage_response.append((item - time_array[0]) * rise)
            else:
                voltage_response.append(2 * amplitude - (item - time_array[0]) * rise)
        return voltage_response

    time = np.array(np.linspace(0, T_p, 10000))
    # vert_ratio = np.vectorize(vertical_ratio)(time)
    # fill_factor = np.vectorize(fill_factor)(time)
    # middle_current = np.vectorize(middle_range_current_diff)(time)
    forward_response = np.vectorize(partial(forward_current, A=Amp, R=Res, C=Cap, T=T_p))
    reverse_response = np.vectorize(partial(backward_current, A=Amp, R=Res, C=Cap, T=T_p))

    forward_current_response = forward_response(time[:5000])
    reverse_current_response = reverse_response(time[5000:])

    current = [*forward_current_response, *reverse_current_response]
    voltage = [*voltage_form(time[:5000], Amp, True), *voltage_form(time[5000:], Amp, False)]

    res = np.stack((voltage, current), axis=1)
    # Write the interpolation to a file
    #if write_to_file != "":
    #    write_success = Dataset.write_standardized_data_file(write_to_file, res)
    #    if not write_success:
    #        sg.PopupError("The interpolated CV failed to write.")
    return res


def validate_and_correct_filename(filename):
    try_filename = filename.replace("\\", "/")
    try_number = 1

    while os.path.exists(try_filename):
        folder_separator = np.max(filename.rfind("/"), filename.rfind("\\"))
        insertion_index = filename.rfind(".")
        if insertion_index == -1 or insertion_index < folder_separator:
            try_filename = filename + "(" + str(try_number) + ")"
        else:
            try_filename = filename[:insertion_index] + "(" + str(try_number) + ")" + filename[
                                                                                      insertion_index:]
        try_number += 1
    return try_filename


# handles the window-popup for writing CVs to standardized files
def popup_write_files(dataset: Dataset):
    def apply_use_checkbox_to_enable_inputs(main_value_dict: dict, local_window_handle: sg.Window):
        for k in range(1, window.metadata + 1):
            disabled = not main_value_dict.get(("use", k))
            # local_window_handle.find_element(("store_folder", j)).update(disabled=disabled)
            local_window_handle.find_element(("store_file", j)).update(disabled=disabled)
            local_window_handle.find_element(("browse", j)).update(disabled=disabled)
            local_window_handle.find_element(("comment", j)).update(disabled=disabled)

    col_layout = [[sg.Button('Store!'), sg.Button('Cancel'),
                   sg.Checkbox('Write to single file', key="write_single_file", enable_events=True, default=False)]]

    layout_popup = [
        [sg.Text("Select file to store in, comment in the field to the right")],
        [sg.Text("Only successfully loaded and used CVs will be stored")],
    ]
    for j in range(1, window.metadata + 1):
        layout_popup.append(write_cv_row(j))
    win = sg.Window("Save Files",
                    [*layout_popup, [sg.Column(col_layout, expand_x=True, element_justification='right')]],
                    grab_anywhere=True, use_default_focus=False, finalize=True, modal=True)
    apply_use_checkbox_to_enable_inputs(values, win)

    # this is the PySimpleGUI event loop, where things done to the GUI menu cause events to occur and values to change
    while True:
        events, values_win = win.read()
        # This is the checkbox to switch write together/separately
        if events == "write_single_file":
            if not values_win["write_single_file"]:
                for j in range(2, window.metadata + 1):
                    win.find_element(("label_a", j)).update(visible=True)
                    # win.find_element(("store_folder", j)).update(visible=True)
                    win.find_element(("browse", j)).update(visible=True)
                    # win.find_element(("label_b", j)).update(visible=True)
                    win.find_element(("store_file", j)).update(visible=True)
                    win.find_element(("label_c", j)).update(visible=True)
                    win.find_element(("comment", j)).update(visible=True)
                apply_use_checkbox_to_enable_inputs(values, win)

            if values_win["write_single_file"]:
                # make all but the first element invisible
                for j in range(2, window.metadata + 1):
                    win.find_element(("label_a", j)).update(visible=False)
                    # win.find_element(("store_folder", j)).update(visible=False)
                    win.find_element(("browse", j)).update(visible=False)
                    # win.find_element(("label_b", j)).update(visible=False)
                    win.find_element(("store_file", j)).update(visible=False)
                    win.find_element(("label_c", j)).update(visible=False)
                    win.find_element(("comment", j)).update(visible=False)

                # ensure the first element is enabled
                # win.find_element(("store_folder", 1)).update(disabled=False)
                win.find_element(("store_file", 1)).update(disabled=False)
                win.find_element(("browse", 1)).update(disabled=False)
                win.find_element(("comment", 1)).update(disabled=False)

        if events == 'Store!':
            if values_win["write_single_file"]:
                # validate that the folder exists
                try:
                    file_directory, isolated_filename = os.path.split(values_win[("store_file", 1)])
                    if os.path.isdir(file_directory) and isolated_filename != "":
                        # construct the full filename by concatenating
                        filename = values_win[("store_file", 1)]
                        filename = validate_and_correct_filename(filename)
                        filtered_subset_candidates = [("f", k) for k in range(1, window.metadata + 1)]
                        filtered_subset_dict = dict((j[1], values[j]) for j in filtered_subset_candidates
                                                    if j in values)
                        print(filtered_subset_dict)
                        comment_dict = dict((k[1], values_win[k]) for k in
                                            [("comment", j) for j in range(1, window.metadata + 1)] if
                                            ("comment", j) in values_win)
                        success = dataset.write_CVs_to_single_file(filename, filtered_subset_dict, comment_dict)
                        if success:
                            sg.PopupOK("The valid data has been written successfully.")
                    # else:
                    #    sg.PopupError("The given folder is not valid or the filename is empty.")
                except OSError as error:
                    sg.popup_error('There was an error with the selected file to store in.')
                    print(error)
            else:
                for j in range(1, window.metadata + 1):
                    try:
                        file_directory, isolated_filename = os.path.split(values_win[("store_file", j)])
                        if os.path.isdir(file_directory) and isolated_filename != "":
                            filename_dict = dict((k, validate_and_correct_filename(values_win[("store_file", k)])) for k
                                                 in range(1, window.metadata + 1))
                            # filename = values_win[("store_folder", 1)] + "/" + values_win[("store_file", 1)]
                            filtered_subset_candidates = [("f", k) for k in range(1, window.metadata + 1)]
                            filtered_subset_dict = dict((j[1], values[j]) for j in filtered_subset_candidates
                                                        if j in values)
                            comment_dict = dict((k[1], values_win[k]) for k in
                                                [("comment", j) for j in range(1, window.metadata + 1)] if
                                                ("comment", j) in values_win)

                    except OSError as error:
                        print(error)
                        sg.popup_error('There was an error with the selected file to store in.')
                success = False
                try:
                    success = dataset.write_CVs_to_files(filename_dict, filtered_subset_dict, comment_dict)
                except Exception as error:
                    print(error)
                    sg.PopupError("An error occurred writing the files. Check that data is loaded and the filenames "
                                  "to save in are proper.")
                if success:
                    sg.PopupOK("The valid data has been written successfully.")
        if events == "Cancel":
            break
        if events == sg.WINDOW_CLOSED:
            break
    win.close()


def pack_figure(graph, figure):
    canvas = FigureCanvasTkAgg(figure, graph.Widget)
    plot_widget_new = canvas.get_tk_widget()
    plot_widget_new.pack(side='top', fill='both', expand=1)
    return plot_widget_new


def apply_active_settings_to_dataset(value_dict, dataset):
    for j in range(1, window.metadata + 1):
        dataset.set_activity_of_element(j, value_dict[("use", j)])


def apply_filter_settings_to_dataset(value_dict, dataset):
    for j in range(1, window.metadata + 1):
        dataset.set_default_filtered_of_element(j, value_dict[("f", j)])


def apply_units_to_dataset(value_dict, dataset):
    for j in range(1, window.metadata + 1):
        cv = dataset.get_content_by_index(j)
        # Note: Using the eval function interprets user inputs as code. Great care needs to be taken using this.
        # Here, the inputs ("-UNIT-", index) are not user-writeable!
        if cv is not None:
            cv.convert_quantities(
                unit_current=eval("pq." + value_dict[("-UNIT-", 0)]),
                unit_voltage=eval("pq." + value_dict[("-UNIT-", 1)])
            )


# handles the main figure in the GUI.
# Arguments: given_data, format like load_data() produces
# special_highlight: [cv_index, lower_voltage, upper voltage, above-zero]: creates highlight in CV between indices
# fit_in_supplementary_data: takes a dict of data, in the shape of cv-index to supplement: data to plot.
# Inputs: takes also the setUnits global parameter
# Returns: Nothing
def plot_figure(given_data, x_title, y_title, special_highlight: list = None, fit_in_supplementary_data=None):
    # figure = plt.gcf()      # Active an existing figure
    figure = plt.figure(1)
    ax = plt.gca()  # Get the current axes
    ax.cla()  # Clear the current axes
    # ax.set_title("CVs")
    ax.set_xlabel(x_title + " [" + str(setUnits[1]) + "]")
    ax.set_ylabel(y_title + " [" + str(setUnits[0]) + "]")
    ax.grid()
    # for all input fields activated in the GUI
    absolute_max_value = 0
    for j in range(1, window.metadata + 1):
        # retrieve the CV object, if existent. Otherwise, returns None
        current_cv = given_data.get_content_by_index(j)
        if current_cv is not None:
            # the active setting is applied from the CV! A read-in from the window has to be called explicitly!
            if current_cv.is_active():
                # decide on whether to retrieve filtered data
                if values[("f", j)]:
                    data = current_cv.get_filtered_dataset()
                else:
                    data = current_cv.get_dataset()
                if max(np.abs(data[:, 1])) > absolute_max_value:
                    absolute_max_value = max(np.abs(data[:, 1]))

                ax.plot(data[:, 0], data[:, 1], color=color_cycle[j - 1], label=f"{j}")
                ax.vlines(current_cv.get_eval_voltage(), -max(data[:, 1]),
                          max(data[:, 1]), color=color_cycle[j - 1])
                if special_highlight:
                    if j == special_highlight[0]:
                        if special_highlight[3]:
                            y_min = 0.5
                            y_max = 1
                        else:
                            y_min = 0
                            y_max = 0.5
                        ax.axvspan(special_highlight[1], special_highlight[2], y_min, y_max,
                                   alpha=0.3, color=color_cycle[j - 1])
                if fit_in_supplementary_data is not None:
                    if j in fit_in_supplementary_data.keys():
                        # print(f"{fit_in_supplementary_data[j][:, 0]}")
                        ax.plot(fit_in_supplementary_data[j][:, 0].magnitude,
                                fit_in_supplementary_data[j][:, 1].magnitude,
                                color=color_cycle[j - 1],
                                label=f"{j}",
                                linestyle=(0, (3, 1, 1, 1)))
    ax.legend()
    plt.ylim(-1.1 * absolute_max_value, 1.1 * absolute_max_value)
    # Render figure into canvas
    figure.canvas.draw()


def cv_row(item_nr):
    row = [sg.InputText(key=("cv", item_nr), default_text=""), sg.FileBrowse("Browse File...", font=("Arial", 10), size=(12, 1)),
           sg.Checkbox(text="Use?", font=("Arial", 10), size=5, default=True, key=("use", item_nr)),
           sg.Checkbox(text="Filter?", font=("Arial", 10), size=5, default=False, key=("f", item_nr))]
    return row


def cycle_row(item_nr):
    row = [sg.InputText(key=("cycle_nr", item_nr), default_text="2", size=5, pad=((5, 5), (0, 10)))]
    return row


def voltage_row(item_nr):
    row = [sg.InputText(key=("voltage_eval", item_nr), default_text="", size=5, pad=((5, 5), (0, 10)))]
    return row


def scanrate_row(item_nr):
    row = [sg.InputText(key=("sr", item_nr), default_text="", size=5, pad=((5, 5), (0, 10)))]
    return row


def write_cv_row(item_nr):
    # row = [sg.Text(f"File No. {item_nr}, Folder:", key=("label_a", item_nr)),
    #        sg.InputText(key=("store_folder", item_nr)),
    #        sg.FolderBrowse('Browse', key=("browse", item_nr)),
    #        sg.Text("Filename: ", key=("label_b", item_nr)),
    #        sg.InputText(key=("store_file", item_nr)),
    #        sg.Text("Optional comment: ", key=("label_c", item_nr)),
    #        sg.InputText(default_text='comment', key=("comment", item_nr))]

    # Transfer filenames of the CVs as defaults
    new_file_path = ""
    if loaded_data.get_content_by_index(item_nr) is not None:
        file_path = loaded_data.get_content_by_index(item_nr).get_source()
        directory_path, original_filename = os.path.split(file_path)
        filename_without_extension, file_extension = os.path.splitext(original_filename)
        file_path = filename_without_extension + "_export" + file_extension

        # Join the directory path and the new filename
        new_file_path = directory_path + "/" + file_path
    row = [sg.Text(f"File No. {item_nr}, File:", key=("label_a", item_nr)),
           sg.InputText(key=("store_file", item_nr), default_text=new_file_path),
           sg.FileSaveAs('Save as...', key=("browse", item_nr)),
           sg.Text("Optional comment: ", key=("label_c", item_nr)),
           sg.InputText(default_text='comment', key=("comment", item_nr))]
    return row


def add_row_to_layout():
    window.metadata += 1
    window.extend_layout(window['-FILE COLUMN-'], [cv_row(window.metadata)])
    window.extend_layout(window['-CYCLE COLUMN-'], [cycle_row(window.metadata)])
    window.extend_layout(window['-VOLTAGE COLUMN-'], [voltage_row(window.metadata)])
    window.extend_layout(window['-SR COLUMN-'], [scanrate_row(window.metadata)])


def save_capacitances_to_file(filename, dataset, window_values, method):
    new_filename = validate_and_correct_filename(filename)
    return dataset.write_capacitance_to_file(filename=new_filename,
                                             through_zero=window_values["origin_fit"],
                                             half_cycle_select=window_values["cycle_select"],
                                             method=method
                                             )


def plot_data_with_voltage(bias_eval_results, voltage_array):
    # Extracting data from the dictionary
    distortion_param_res = bias_eval_results["ff_distortion_param"]
    param_distortion_first_quarter = bias_eval_results["pd_first_quarter_real"]
    param_distortion_third_quarter = bias_eval_results["pd_third_quarter_real"]
    target_current_ratio_first_quarter = bias_eval_results["target_current_ratio_first_quarter"]
    target_current_ratio_third_quarter = bias_eval_results["target_current_ratio_third_quarter"]
    real_current_ratio_first_quarter = bias_eval_results["real_current_ratio_first_quarter"]
    real_current_ratio_third_quarter = bias_eval_results["real_current_ratio_third_quarter"]
    correction = bias_eval_results["correction_applied"]

    # Calculate the minimum and maximum voltage
    min_voltage = np.min(voltage_array)
    max_voltage = np.max(voltage_array)

    # Calculate the first and third quarter points
    first_quarter_point = min_voltage + 0.25 * (max_voltage - min_voltage)
    third_quarter_point = min_voltage + 0.75 * (max_voltage - min_voltage)

    # Plotting
    fig, axs = plt.subplots(2, 2, figsize=(10, 8))

    # Plotting distortion parameters
    axs[0, 0].plot(voltage_array, [distortion_param_res] * len(voltage_array), label='Distortion Parameter', linestyle='--', zorder=5)
    axs[0, 0].scatter([min_voltage, first_quarter_point, third_quarter_point, max_voltage],
                   [distortion_param_res, param_distortion_first_quarter, param_distortion_third_quarter, distortion_param_res],
                   color='red', label='Parameters at Quarter Points')

    axs[0, 0].set_xlabel('Voltage')
    axs[0, 0].set_ylabel('Distortion Parameter')
    axs[0, 0].set_title('Distortion Parameters Plot')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    # Plotting current ratios - first quarter
    axs[1, 0].plot(voltage_array, [target_current_ratio_first_quarter] * len(voltage_array), label='Target Current Ratio - First Quarter', linestyle='-')
    axs[1, 0].scatter(first_quarter_point, real_current_ratio_first_quarter, label='Real Current Ratio - First Quarter', marker='o')
    axs[1, 0].set_xlabel('Voltage')
    axs[1, 0].set_ylabel('Current Ratio')
    axs[1, 0].set_title('Current Ratios - First Quarter Plot')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    # Plotting current ratios - third quarter
    axs[1, 1].plot(voltage_array, [target_current_ratio_third_quarter] * len(voltage_array), label='Target Current Ratio - Third Quarter', linestyle='-')
    axs[1, 1].scatter(third_quarter_point, real_current_ratio_third_quarter, label='Real Current Ratio - Third Quarter', marker='o')
    axs[1, 1].set_xlabel('Voltage')
    axs[1, 1].set_ylabel('Current Ratio')
    axs[1, 1].set_title('Current Ratios - Third Quarter Plot')
    axs[1, 1].legend()
    axs[1, 1].grid(True)

    # Hide the empty subplot in the first row and second column
    axs[0, 1].axis('off')

    # Adjust the position of the textbox
    fig.text(0.5, 0.7, f"Distortion Parameter: {distortion_param_res}\n"
                       f"Param Distortion First Quarter: {param_distortion_first_quarter}\n"
                       f"Param Distortion Third Quarter: {param_distortion_third_quarter}\n"
                       f"Target Current Ratio First Quarter: {target_current_ratio_first_quarter}\n"
                       f"Target Current Ratio Third Quarter: {target_current_ratio_third_quarter}\n"
                       f"Real Current Ratio First Quarter: {real_current_ratio_first_quarter}\n"
                       f"Real Current Ratio Third Quarter: {real_current_ratio_third_quarter}\n"
                       f"Correction Applied: {correction}",
             fontsize=10, ha='left', va='top', bbox=dict(facecolor='white', edgecolor='black', boxstyle='round'))

    plt.tight_layout(pad=4)
    plt.show()


def display_capacitance_in_window(capacitance, offset, index_scanrate_current_values):
    global fig1, ax1, plot_widget

    def linear(x, m, n):
        return x * m + n

    window['cap_opt_out'].update(str(capacitance))
    window['offs_opt_out'].update(str(offset))
    figure = plt.figure(2)  # Activate existing figure
    plt.plot(index_scanrate_current_values[:, 1], index_scanrate_current_values[:, 2], 'b.')
    # plot the interpolation for control purposes
    x_inter = np.linspace(-5, 1.1 * max(index_scanrate_current_values[:, 1]), 10)
    y_inter = linear(x_inter, capacitance, offset)
    plt.plot(x_inter, y_inter, 'y-')
    ax = plt.gca()
    ax.set_xlim([-5, 1.1 * max(index_scanrate_current_values[:, 1])])
    ax.grid()
    ax.set_xlabel("Scanrate [mV/s]")
    ax.set_ylabel("Evaluated current (difference) at potential E [mA]")
    plt.show()
    # reinitialize the current loaded display
    # yeet the previous canvas
    plot_widget.destroy()
    # recreate the figure
    fig1 = plt.figure(1)  # Create a new figure
    ax1 = plt.subplot(111)  # Add a subplot to the current figure.
    # create a new canvas and attach it to the graph
    plot_widget = pack_figure(graph1, fig1)  # Pack figure under graph
    plt.ioff()
    # fill the new canvas with the loaded data
    plot_figure(loaded_data, values["x_axis"], values["y_axis"])
    # put the main window back in front
    window.TKroot.attributes('-topmost', True)
    window.TKroot.attributes('-topmost', False)


def transfer_use_checkbox_to_dataset():
    use_indices = []
    # read in the use-checkboxes and tell the dataset which indices to set active
    for j in range(1, window.metadata + 1):
        if values[("use", j)]:
            use_indices.append(j)
    loaded_data.set_active_for_full_dataset(use_indices)


def transfer_scanrates_to_dataset():
    scanrates_local = []
    for j in range(1, window.metadata + 1):
        try:
            scanrates_local.append([j, float(values[("sr", j)].replace(',', '.'))])
        except ValueError:
            continue
    loaded_data.set_scanrates(scanrates_local)


def round_sensibly(value, precision: int):
    # round to the significant _precision_ digits relevant to the dimension of the result
    correct_by = 1
    if value < 0:
        correct_by = -1
        value = np.abs(value)

    k = -30
    while value > 10 ** k:
        k += 1
    precision += 1
    value = np.round(value, -k + precision)

    return value * correct_by


########################################################################################################################
# LAYOUT DEFINITION
########################################################################################################################

file_column = [
    [sg.Text("Select and load CV data files")],
    [sg.HorizontalSeparator()],
    [sg.Button('Load and preview CVs', font=("Arial", 11)), sg.Button('Write CVs to file', font=("Arial", 11)),
     sg.Push(), sg.Checkbox(text="Assume default CSV format", font=("Arial", 10), size=20, default=False,
                            key="default_csv_format"),
     sg.Button('Add Row', font=("Arial", 11)), sg.Button('Clear all', font=("Arial", 11))],
    [sg.VPush()],
    [sg.HorizontalSeparator()],
    [sg.T('Select preferred units'), sg.Push(),
     sg.DropDown(["A", "mA", "uA", "nA", "pA"], key=("-UNIT-", 0), readonly=True, default_value="A", enable_events=True),
     sg.DropDown(["V", "mV", "uV"], key=("-UNIT-", 1), readonly=True, default_value="V", enable_events=True),
     sg.DropDown(["F", "mF", "uF", "nF", "pF"], key=("-UNIT-", 2), readonly=True, default_value="mF", enable_events=True)],
    [sg.HorizontalSeparator()],
    [sg.InputText(key=("cv", 1),
                  default_text="", enable_events=True),
     sg.FilesBrowse("Browse File(s)...", font=("Arial", 10), size=(12, 1)),
     sg.Checkbox(text="Use?", font=("Arial", 10), size=5, default=True, key=("use", 1)),
     sg.Checkbox(text="Filter?", font=("Arial", 10), size=5, default=False, key=("f", 1))]
]

nc_column = [
    [sg.Text("Cycle number", pad=((5, 5), (90, 7)))],
    [sg.HorizontalSeparator()],
    [sg.InputText(key=("cycle_nr", 1), default_text="2", size=5, pad=((5, 5), (5, 7)))],
]

voltage_column = [
    [sg.Text("Voltage [V]", pad=((5, 5), (90, 7)))],
    [sg.HorizontalSeparator()],
    [sg.InputText(key=("voltage_eval", 1), default_text="", size=5, pad=((5, 5), (5, 7)))],
]

sr_filter_column = [
    [sg.Text("Scanrate [mV/s]", pad=((5, 5), (90, 7)))],
    [sg.HorizontalSeparator()],
    [sg.InputText(key=("sr", 1), default_text="", size=5, pad=((5, 5), (5, 7)))],
]

image_column = [
    [sg.Text("Title of x-axis"), sg.InputText(key='x_axis', default_text='Potential', size=(20, 1)),
     sg.Text("Title of y-axis"), sg.InputText(key='y_axis', default_text='Current', size=(20, 1))],
    [sg.Graph((640, 480), (0, 0), (640, 480), key='Graph1')],

]

capacitance_column = [[sg.Text("Capacitance Calculation", font=("Calibri", 16, "bold"))],
                      [sg.Text("Use anodic/cathodic/full CV?"),
                       sg.Combo(["full", "anodic", "cathodic"], key="cycle_select", default_value="full",
                                enable_events=False), sg.Checkbox('Force fit through origin',
                                                                  key="origin_fit", default=False)],
                      [sg.Text("Calculation mode:"),
                       sg.Combo(["At selected voltage", "MinMax Current", "CaRe analysis"],
                                key="capacitance_calculation_method", default_value="CaRe analysis",
                                enable_events=False), sg.Button('Calculate Capacitance')],
                      [sg.Text("Capacitance by linear fit [F]:"), sg.Push(), sg.InputText(key='cap_opt_out')],
                      [sg.Text("Zero-offset of linear fit [A]:"), sg.Push(), sg.InputText(key='offs_opt_out')],
                      [sg.HorizontalSeparator()],
                      [sg.Text("Export to file:"), sg.InputText(default_text='capacitance.txt', key="cap_save"),
                       sg.FileSaveAs(font=("Arial", 11))],
                      [sg.Button('Save Capacitance', font=("Arial", 11)),
                       sg.Button('Save chosen CV No. CaRe fit', font=("Arial", 11)),
                       sg.Button('Save bulk distortion analysis', font=("Arial", 11))]]

advanced_column = [[sg.Text("Advanced CV calculations", font=("Calibri", 16, "bold"))],
                   [sg.HorizontalSeparator()],
                   [sg.Text("Read in full cycle?"), sg.Combo(["full cycles", "forward", "reverse"],
                                                             key='halfcycle_mode', default_value="full cycles",
                                                             enable_events=False, readonly=True),
                    sg.Text("CaRe calculation method"), sg.Combo(["Analytical", "Optimisation enhanced analytical"],
                                                             key='care_calculation_mode', default_value="Analytical",
                                                             enable_events=False, readonly=True)
                    ],
                   [sg.HorizontalSeparator()],
                   [sg.T("Use CV No.:", font=("Calibri", 14, "bold")),
                    sg.InputText(key=("advanced", "index"), size=5),
                    sg.B("Perform distorted capacitive CV analysis"),
                    sg.B("Show true scanrate plot")],
                   [sg.Text("Resistance [Ohm]:"),
                    sg.InputText(key=("advanced", "resistance"), size=8),
                    sg.Text("Capacitance [mF]:", key=("advanced", "capacitance_label")),
                    sg.InputText(key=("advanced", "capacitance"), size=8),
                    sg.Text("Distortion parameter:"),
                    sg.InputText(key=("advanced", "distortion_param"), size=8)],

                   [sg.Text("Calculate the integral", font=("Calibri", 14, "bold"))],
                   [sg.T("Lower bound [V]:"),
                    sg.InputText(key=("integral", "lower_bound"), size=5), sg.T("Upper bound [V]:"),
                    sg.InputText(key=("integral", "upper_bound"), size=5), sg.T("Current offset [current unit]:"),
                    sg.InputText(key=("integral", "offset"), size=5),
                    sg.Combo(["forward", "reverse"], key=("integral", "scan_direction"), default_value="forward",
                             enable_events=False, readonly=True), sg.B("Calculate integral")],
                   [sg.T("Integral:"), sg.InputText(readonly=True, key=("integral", "out_result"), size=20),
                    sg.T("Uncertainty:"), sg.InputText(readonly=True, key=("integral", "out_uncertainty"), size=20),
                    sg.T("Inferred charge by scan rate:"), sg.InputText(readonly=True, key=("integral", "out_charge"),
                                                                        size=20)]
                   ]


menu_def = (
    ['&CV',
            ['&Load and preview CVs', '&Clear all', '---', 'C&lose']],
    ['&Save',
            ['&Write CVs to file', 'Save capacitance calculation', 'Save chosen CV No. CaRe fit',
             'Save bulk distortion analysis']],
    ['&Edit',
            ['&Add Row', 'Copy first row inputs to all']],
    ['Advanced Settings',
            ['Load partial CVs', 'Bias analysis', 'Cycle split all files']])

layout = [
    [sg.Menu(menu_def, text_color='black', disabled_text_color='gray', font='Arial', pad=(0, 20))],
    [sg.Image("Images/CVCaRe.png")],
    #[sg.Text("CVCaRe - Cyclic voltammogram analysis tool", font=("Calibri", 20), justification='c')],
    [sg.HorizontalSeparator()],
    [sg.Column(file_column,
               vertical_alignment='top',
               key="-FILE COLUMN-"),
     # sg.VSeparator(),
     sg.Column(nc_column, vertical_alignment='top', key="-CYCLE COLUMN-", ),
     sg.Column(voltage_column, vertical_alignment='top', key="-VOLTAGE COLUMN-"),
     sg.Column(sr_filter_column, vertical_alignment='top', key="-SR COLUMN-"),
     sg.VSeparator(),
     sg.Column(image_column)],
    [sg.HorizontalSeparator()],
    [sg.Column(capacitance_column,
               vertical_alignment='top',
               key="-CAPACITANCE COLUMN-"),
     sg.VSeparator(),
     sg.Column(advanced_column, vertical_alignment='top',
               key="-SETTINGS COLUMN-")]

]

window = sg.Window('CV Analysis and Export', layout, finalize=True, font='Helvetica', grab_anywhere=True,
                   resizable=True, icon="Images/cvcare.ico")
# size=(1750, 925)

# this metadata variable is the property of the main window and is used here to hold the number of rows of input
window.metadata = 1
for i in range(5):
    add_row_to_layout()

# This is the main-window Dataset holding variable.
loaded_data = None

# Initial
graph1 = window['Graph1']
fig1 = plt.figure(1)  # Create a new figure
ax1 = plt.subplot(111)  # Add a subplot to the current figure.
plot_widget = pack_figure(graph1, fig1)  # Pack figure under graph
plt.ioff()  # Turn the interactive mode off

########################################################################################################################
# EVENT LOOP
########################################################################################################################


while True:
    event, values = window.read()
    if event == sg.WINDOW_CLOSED:
        break

    if event == "Add Row":
        add_row_to_layout()

    if event == 'Clear all':
        for i in range(1, window.metadata + 1):
            window[('cv', i)].update("")
            window[('sr', i)].update("")

    if event == ("cv", 1):
        list_of_filenames = values[("cv", 1)].split(';')
        insertion_index = 1
        for filename in list_of_filenames:
            if filename is not None:
                # if more files are selected than rows are there, make new rows
                if insertion_index > window.metadata:
                    add_row_to_layout()
                # update the input with the current filename
                window[("cv", insertion_index)].update(filename)
                insertion_index += 1


    if isinstance(event, tuple) and event[0] == "-UNIT-":
        loaded_data = Dataset.Dataset(values, window.metadata, window)
        setUnits[0] = eval("pq" + "." + values[("-UNIT-", 0)])
        setUnits[1] = eval("pq" + "." + values[("-UNIT-", 1)])
        setUnits[2] = eval("pq" + "." + values[("-UNIT-", 2)])
        apply_units_to_dataset(values, loaded_data)

    if event == "Load and preview CVs":
        loaded_data = Dataset.Dataset(values, window.metadata, window)
        transfer_use_checkbox_to_dataset()
        transfer_scanrates_to_dataset()
        apply_units_to_dataset(values, loaded_data)
        apply_active_settings_to_dataset(values, loaded_data)
        apply_filter_settings_to_dataset(value_dict=values, dataset=loaded_data)

        plot_figure(loaded_data, values["x_axis"], values["y_axis"])

    if event == 'Write CVs to file':
        popup_write_files(loaded_data)

    if event == "Save Capacitance":
        loaded_data = Dataset.Dataset(values, window.metadata, window)
        apply_units_to_dataset(value_dict=values, dataset=loaded_data)
        transfer_use_checkbox_to_dataset()
        apply_filter_settings_to_dataset(value_dict=values, dataset=loaded_data)

        plot_figure(given_data=loaded_data, x_title=values["x_axis"], y_title=values["y_axis"])
        transfer_scanrates_to_dataset()

        if values["capacitance_calculation_method"] == "At selected voltage":
            method_name = "at_selected_voltage"
        elif values["capacitance_calculation_method"] == "MinMax Current":
            method_name = "minmax"
        elif values["capacitance_calculation_method"] == "CaRe analysis":
            method_name = "minmax_corrected"
        success = save_capacitances_to_file(filename=values["cap_save"], dataset=loaded_data, window_values=values,
                                            method=method_name)
        if success:
            sg.PopupOK("The capacitance file has been written successfully.")

    if event == "Copy first row inputs to all":
        for i in range(2, window.metadata + 1):
            window[('cycle_nr', i)].update(values[('cycle_nr', 1)])
            window[('sr', i)].update(values[('sr', 1)])
            window[('voltage_eval', i)].update(values[('voltage_eval', 1)])
            window[('f', i)].update(values[('f', 1)])

    if event == "Calculate Capacitance":
        if loaded_data:
            transfer_use_checkbox_to_dataset()
            transfer_scanrates_to_dataset()
            apply_filter_settings_to_dataset(value_dict=values, dataset=loaded_data)

            capacitance = None
            offset = None
            index_scanrate_current_dataset = None
            try:
                # sg.Combo(["At selected voltage", "MinMax Current", "CaRe analysis"],
                #                                 key="capacitance_calculation_method",
                # in case of at least 2 fittable CVs

                if values["capacitance_calculation_method"] == "At selected voltage":
                    capacitance, offset, index_scanrate_current_dataset = loaded_data.get_capacitance_at_selected_voltage(
                        through_zero=values["origin_fit"], active_only=True, half_cycle_select=values["cycle_select"])
                elif values["capacitance_calculation_method"] == "MinMax Current":
                    capacitance, offset, index_scanrate_current_dataset = loaded_data.get_capacitance_by_minmax(
                        through_zero=values["origin_fit"], active_only=True, half_cycle_select=values["cycle_select"],
                        corrected=False)
                elif values["capacitance_calculation_method"] == "CaRe analysis":
                    capacitance, offset, index_scanrate_current_dataset = loaded_data.get_capacitance_by_minmax(
                        through_zero=values["origin_fit"], active_only=True, half_cycle_select=values["cycle_select"],
                        corrected=True)
            except NotEnoughCVsToFitError as e:
                print("INFO: Less than two fittable CVs with scanrates found. Proceeding.")
                # Alternatively, calculate capacitance for the only loaded CV, if there is one
                try:
                    # grab the only CV that is in the dataset, if any
                    cv = None
                    for i in range(1, window.metadata + 1):
                        grab = loaded_data.get_content_by_index(i)
                        if grab is not None and grab.is_active():
                            cv = grab

                    current_difference = 0

                    if values["capacitance_calculation_method"] == "At selected voltage":
                        current_difference = cv.get_current_at_voltage(current_dimension=pq.mA)

                    elif values["capacitance_calculation_method"] == "MinMax Current":
                        if cv.get_default_filtered():
                            data = cv.get_filtered_dataset()
                        else:
                            data = cv.get_dataset()
                        current_difference = (max(data[:, 1]) - min(data[:, 1])) * cv.unit_current

                    elif values["capacitance_calculation_method"] == "CaRe analysis":
                        if cv.get_default_filtered():
                            data = cv.get_filtered_dataset()
                        else:
                            data = cv.get_dataset()
                        current_difference = (max(data[:, 1]) - min(data[:, 1])) * cv.unit_current
                        vertical_ratio, distortion_param = cv.estimate_vertical_current_ratio()
                        current_difference /= vertical_ratio

                    scanrate = cv.get_scanrate(dimension=pq.mV / pq.s)
                    # the current difference includes anodic and cathodic current. Division by two!
                    capacitance = current_difference / (2 * scanrate)
                    capacitance.units = pq.F
                except FunctionNotImplementedError:
                    sg.PopupError("Dataset has no readable CVs.")
                except ScanrateExistsError:
                    sg.PopupError("The single found CV has no scanrate")

            if capacitance is not None and offset is not None:
                try:
                    display_capacitance_in_window(capacitance, offset, index_scanrate_current_dataset)
                except Exception as e:
                    print(e)
                    sg.PopupError('An unexpected error occurred attempting to plot the capacitance.')
            elif capacitance is not None:
                window['cap_opt_out'].update(str(capacitance))
                window['offs_opt_out'].update(str("Not applicable for one CV"))

    if event == "Calculate integral":
        if loaded_data is not None:
            res = 0
            try:
                transfer_scanrates_to_dataset()
                # collect the data from the GUI
                cv_no = int(values[("advanced", "index")])
                cv = loaded_data.get_content_by_index(cv_no)
                if cv is None:
                    raise ValueError("Invalid cycle index!")
                lower_bound = float(values[("integral", "lower_bound")].replace(",", ".")) * pq.V
                upper_bound = float(values[("integral", "upper_bound")].replace(",", ".")) * pq.V
                lower_bound.units = setUnits[1]
                upper_bound.units = setUnits[1]
                lower_bound /= setUnits[1]
                upper_bound /= setUnits[1]
                if values[("integral", "scan_direction")] == "forward":
                    direction_forward = True
                else:
                    direction_forward = False

                offset_integral = None
                if values[("integral", "offset")] != "":
                    try:
                        offset_integral = float(values[("integral", "offset")].replace(',', '.').replace(' ', ''))
                    except ValueError:
                        offset_integral = None

                # here, it is already assured that cv_no is valid, since it was possible to load the cv.
                # this may change in future program versions.
                filtered = values[("f", cv_no)]
                try:
                    scanrate = cv.get_scanrate(setUnits[1] / pq.s)
                except ScanrateExistsError as e:
                    print(e)
                    scanrate = None
                if isinstance(cv, FullCV):
                    res, uncertainty = cv.integrate_one_direction(lower_voltage_bound=lower_bound,
                                                                  upper_voltage_bound=upper_bound,
                                                                  forward_direction=direction_forward,
                                                                  filtered=filtered,
                                                                  offset=offset_integral)
                    res = round_sensibly(res, 5)
                    print(res)
                    uncertainty = round_sensibly(uncertainty, 5)
                    print(uncertainty)
                else:
                    raise FunctionNotImplementedError("This function is not available for half-cycles.")
                window[("integral", "out_result")].update(str(res))
                window[("integral", "out_uncertainty")].update(str(uncertainty))
                if scanrate is not None:
                    window[("integral", "out_charge")].update(str(round_sensibly(res / scanrate, 5)))
                else:
                    window[("integral", "out_charge")].update("not calculated")
                special_highlight = [cv_no, lower_bound, upper_bound, direction_forward]
                plot_figure(loaded_data, values["x_axis"], values["y_axis"], special_highlight=special_highlight)
            except ValueError as e:
                print(e)
                sg.PopupError("The index may not exist, or the inputs are not of the correct data type.")
            except VoltageBoundsOutsideCVError as e:
                print(e)
                sg.PopupError("The integral bounds are outside the maximum values of the CV!")
            except FunctionNotImplementedError as e:
                sg.PopupError(str(e))

    if event == "Show true scanrate plot":
        try:
            transfer_scanrates_to_dataset()
            # Get the CV specified in the index field
            cv_no = int(values[("advanced", "index")])
            cv = loaded_data.get_content_by_index(cv_no)

            # Here, a value error may be raised
            resistance = float(values[("advanced", "resistance")]) * pq.Ohm

            # Decline execution when HalfCVs are given
            if not isinstance(cv, FullCV):
                raise FunctionNotImplementedError("This function is only implemented for full CVs.")

            plt.figure(100)
            plt.plot(cv.true_interfacial_scanrate(resistance))
            plt.show()
            plot_widget.destroy()
            # recreate the figure
            fig1 = plt.figure(1)  # Create a new figure
            ax1 = plt.subplot(111)  # Add a subplot to the current figure.
            # create a new canvas and attach it to the graph
            plot_widget = pack_figure(graph1, fig1)  # Pack figure under graph
            plt.ioff()
            # fill the new canvas with the loaded data
            plot_figure(loaded_data, values["x_axis"], values["y_axis"])
            # put the main window back in front
            window.TKroot.attributes('-topmost', True)
            window.TKroot.attributes('-topmost', False)
        except FunctionNotImplementedError as e:
            sg.PopupError(e)
        except ValueError:
            sg.PopupError("Impermissible electrolyte resistance value.")
        except AttributeError:
            sg.PopupError("No data loaded.")

    if event == "Perform distorted capacitive CV analysis":
        try:
            # First, collect all the user inputs to perform the calculation.
            transfer_scanrates_to_dataset()
            # collect the data from the GUI
            cv_no = int(values[("advanced", "index")])
            cv = loaded_data.get_content_by_index(cv_no)
            if cv is None:
                raise ValueError("Invalid cycle index!")
            if not isinstance(cv, FullCV):
                raise FunctionNotImplementedError("This function is only implemented for full CVs.")

            # Take the dataset and request that it perform the distortion analysis, returning resistance, capacitance,
            # distortion parameter and potential window. Choice of method.

            if values["care_calculation_mode"] == "Analytical":
                resistance, capacitance, potential_window, distortion_param, offset = cv.distortion_param_evaluation()
            elif values["care_calculation_mode"] == "Optimisation enhanced analytical":
                resistance, capacitance, potential_window, distortion_param, offset = cv.fit_cv_by_optimisation()

            # Obtain the voltage amplitude = half the potential window for the theoretical calculation.
            amplitude = cv.get_amplitude()
            amplitude.units = pq.V
            T_p_calc = 4 * amplitude / cv.get_scanrate(pq.V / pq.s)
            T_p_calc.units = pq.s

            print(f"Resistance: {resistance}, Capacitance: {capacitance}, abs test: {cv.get_amplitude().magnitude}")
            # test here
            fitted_cv = calculate_RC_cv(Res=resistance.magnitude,
                                        Cap=capacitance.magnitude,
                                        Amp=amplitude.magnitude,
                                        T_p=T_p_calc.magnitude)
            # correction shift of fitted CV. The theoretical calculation assumes base SI-units. They can therefore
            # be appended here, and then the calculation is adjusted to the selected user-units.

            # The voltage slice is unit-converted, and shifted to coincide with the loaded dataset.
            voltage_slice = fitted_cv[:, 0] * pq.V
            voltage_slice = voltage_slice.rescale(setUnits[1])
            voltage_corrector = min(cv.dataset[:, 0]) * cv.unit_voltage
            voltage_slice += voltage_corrector

            # The current slice is unit-converted, and shifted to coincide with the loaded dataset.
            current_slice = fitted_cv[:, 1] * pq.A
            current_slice = current_slice.rescale(setUnits[0])
            current_corrector = 0
            if values["care_calculation_mode"] == "Analytical":
                current_corrector = (min(cv.dataset[:, 1]) * cv.unit_current - min(current_slice))
            elif values["care_calculation_mode"] == "Optimisation enhanced analytical":
                current_corrector = offset
            current_slice += current_corrector

            # the adjusted calculated CV dataset is reassembled by stacking along the vertical axis, and entered into
            # the dictionary for access via the plotting function.
            fitted_cv = np.stack((voltage_slice, current_slice), axis=1)
            sup_dict = {cv_no: fitted_cv}

            # Replotting the CVs, now with dashed line indicating distortion param fit.
            plot_figure(loaded_data, values["x_axis"], values["y_axis"], fit_in_supplementary_data=sup_dict)

            # Select the correct capacitance dimension as set. Adjust the output texts and the capacitance label.
            capacitance.units = setUnits[2]
            window[("advanced", "resistance")].update(f"{np.array([np.round(resistance.magnitude, 3)])[0]}")
            window[("advanced", "capacitance")].update(f"{np.array([np.round(capacitance.magnitude, 6)])[0]}")
            window[("advanced", "capacitance_label")].update(f"Capacitance [{values[('-UNIT-', 2)]}]: ")
            window[("advanced", "distortion_param")].update(f"{np.array([np.round(distortion_param, 6)])[0]}")
        except NoScanrateDefinedError:
            sg.PopupError('No scanrate is defined for this CV.')
        # except FunctionNotImplementedError as e:
        #     sg.PopupError(e)
        # except ValueError or AttributeError as e:
        #     print(e)
        #     sg.PopupError("The CV index may not exist, or the inputs are not of the correct data type.")

    if event == 'Bias analysis':
        try:
            # First, collect all the user inputs to perform the calculation.
            transfer_scanrates_to_dataset()
            # collect the data from the GUI
            cv_no = int(values[("advanced", "index")])
            cv = loaded_data.get_content_by_index(cv_no)
            if cv is None:
                raise ValueError("Invalid cycle index!")
            if not isinstance(cv, FullCV):
                raise FunctionNotImplementedError("This function is only implemented for full CVs.")
            result_array = cv.bias_analysis()
            voltage = cv.get_dataset()[:, 0]
            plot_data_with_voltage(result_array, voltage)

            # Recreate main figure
            plot_widget.destroy()
            fig1 = plt.figure(1)  # Create a new figure
            ax1 = plt.subplot(111)  # Add a subplot to the current figure.
            # create a new canvas and attach it to the graph
            plot_widget = pack_figure(graph1, fig1)  # Pack figure under graph
            plt.ioff()
            # fill the new canvas with the loaded data
            plot_figure(loaded_data, values["x_axis"], values["y_axis"])
            # put the main window back in front
            window.TKroot.attributes('-topmost', True)
            window.TKroot.attributes('-topmost', False)

        except NoScanrateDefinedError:
            sg.PopupError('No scanrate is defined for this CV.')
        except FunctionNotImplementedError as e:
            sg.PopupError(e)
        except ValueError or AttributeError as e:
            sg.PopupError("The CV index may not exist, or the inputs are not of the correct data type.")

    if event == 'Save bulk distortion analysis':
        if loaded_data:
            try:
                # Load the dataset
                loaded_data = Dataset.Dataset(values, window.metadata, window)
                # transfer all the necessary settings
                apply_units_to_dataset(value_dict=values, dataset=loaded_data)
                transfer_use_checkbox_to_dataset()
                apply_filter_settings_to_dataset(value_dict=values, dataset=loaded_data)
                transfer_scanrates_to_dataset()

                # perform the analysis and write it to file
                success = loaded_data.write_distortion_param_results_to_file(values["cap_save"], values["care_calculation_mode"])
                if success:
                    sg.PopupOK("Successfully written the distortion parameter analysis to the specified file.")
                else:
                    sg.PopupError("The writing process was not successful.")
            except FunctionNotImplementedError as e:
                sg.PopupError(e)
            except UnknownMethodError as e:
                sg.PopupError("Method string was unrecognised in Dataset.write_distortion_param_results_to_file(). "
                      "This should not happen. Check the integrity of the source code.")
                print("Method string was unrecognised in Dataset.write_distortion_param_results_to_file(). "
                      "This should not happen. Check the integrity of the source code.")

    if event == "Cycle split all files":
        for i in range(1, window.metadata + 1):
            try:
                if values[("cv", i)]:
                    Dataset.write_split_cycles(values[("cv", 1)], values["default_csv_format"])
            except Exception as e:
                print(f"Error writing split cycle data: {e}")

    if event == "Save chosen CV No. CaRe fit":
        try:
            # First, collect all the user inputs to perform the calculation.
            transfer_scanrates_to_dataset()
            apply_filter_settings_to_dataset(values, loaded_data)
            # collect the data from the GUI
            cv_no = int(values[("advanced", "index")])
            cv = loaded_data.get_content_by_index(cv_no)
            if cv is None:
                raise ValueError("Invalid cycle index!")
            if not isinstance(cv, FullCV):
                raise FunctionNotImplementedError("This function is only implemented for full CVs.")
            # Take the dataset and request that it perform the distortion analysis, returning resistance, capacitance,
            # distortion parameter and potential window. Choice of method.

            if values["care_calculation_mode"] == "Analytical":
                resistance, capacitance, potential_window, distortion_param, offset = cv.distortion_param_evaluation()
            elif values["care_calculation_mode"] == "Optimisation enhanced analytical":
                resistance, capacitance, potential_window, distortion_param, offset = cv.fit_cv_by_optimisation()

            # Obtain the voltage amplitude = half the potential window for the theoretical calculation.
            amplitude = cv.get_amplitude()
            amplitude.units = pq.V
            T_p_calc = 4 * amplitude / cv.get_scanrate(pq.V / pq.s)
            T_p_calc.units = pq.s

            fitted_cv = calculate_RC_cv(Res=resistance.magnitude,
                                        Cap=capacitance.magnitude,
                                        Amp=amplitude.magnitude,
                                        T_p=T_p_calc.magnitude)
            # correction shift of fitted CV. The theoretical calculation assumes base SI-units. They can therefore
            # be appended here, and then the calculation is adjusted to the selected user-units.

            # The voltage slice is unit-converted, and shifted to coincide with the loaded dataset.
            voltage_slice = fitted_cv[:, 0] * pq.V
            voltage_slice = voltage_slice.rescale(setUnits[1])
            voltage_corrector = min(cv.dataset[:, 0]) * cv.unit_voltage
            voltage_slice += voltage_corrector

            # The current slice is unit-converted, and shifted to coincide with the loaded dataset.
            current_slice = fitted_cv[:, 1] * pq.A
            current_slice = current_slice.rescale(setUnits[0])
            current_corrector = 0
            if values["care_calculation_mode"] == "Analytical":
                current_corrector = (min(cv.dataset[:, 1]) * cv.unit_current - min(current_slice))
            elif values["care_calculation_mode"] == "Optimisation enhanced analytical":
                current_corrector = offset
            current_slice += current_corrector

            # the adjusted calculated CV dataset is reassembled by stacking along the vertical axis, and entered into
            # the dictionary for access via the plotting function.
            fitted_cv = np.stack((voltage_slice.magnitude, current_slice.magnitude), axis=1)
            # Convert the array to a list of lists
            fitted_cv_list = fitted_cv.tolist()
            # Insert the title row
            title_row = [f"Voltage [{voltage_slice.units}]", f"Current [{current_slice.units}]"]
            fitted_cv_list.insert(0, title_row)

            # Write the interpolation to a file
            write_success = Dataset.write_standardized_data_file(values["cap_save"], fitted_cv_list)
            if not write_success:
                sg.PopupError("The interpolated CV failed to write.")
        except FunctionNotImplementedError as e:
            sg.PopupError(e)
        except NoScanrateDefinedError:
            sg.PopupError('No scanrate is defined for this CV.')
        except ValueError or AttributeError as e:
            sg.PopupError("The CV index may not exist, or the inputs are not of the correct data type.")

window.close()
