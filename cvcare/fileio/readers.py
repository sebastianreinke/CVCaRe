"""
File reading routines for cyclic voltammetry data files.

The parsing logic in this module is deliberately tolerant to the many quirks
of common potentiostat exports (Biologic, Gamry, etc.): different separators
(``;``, ``\\t``, space), comma vs. dot decimal marks, units encoded in the
header line, and optional cycle columns.

This module mirrors the behaviour of the original ``Dataset.py`` parsers
verbatim. Only the import paths and a small amount of cosmetic clean-up have
changed; the substantive logic is unchanged so that previously-readable files
remain readable.

This file is part of CVCaRe.
Copyright (C) 2022-2026 Sebastian Reinke
Licensed under the GNU General Public License v3 or later.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import quantities as pq

# FreeSimpleGUI is only used as a type hint for the optional ``window_handle``
# argument of ``half_cycle_parsing``. To keep this module GUI-free, we accept
# any object that supports ``__getitem__`` and ``.update()``.
from cvcare.fileio.signifiers import current_signifiers, voltage_signifiers
from cvcare.core.peaks import get_positive_and_negative_voltage_peaks
from cvcare.exceptions import (
    CycleIndexOutOfBoundsError,
    NoCycleInformationError,
)


def read_csv_file(filename: str, assume_standard_csv_format: bool = False) -> list[list[str]] | None:
    """Read and parse a CV data file into a list of rows starting at the header.

    Handles common CSV formats: semicolon-, tab- or space-separated values.
    Identifies the header row by the presence of voltage and current
    signifiers, so that only data rows below it are returned.

    Parameters
    ----------
    filename : str
        Path to the CSV file to be read.
    assume_standard_csv_format : bool, optional
        If True, the function assumes comma-separated values and converts
        commas to semicolons for further processing. If False (default), the
        separator is auto-detected.

    Returns
    -------
    list of lists of str or None
        Each sublist is a row from the file, beginning at the header row.
        Returns ``None`` if the file is not found.
    """
    string_list: list[str] = []
    try:
        with open(filename, "r+") as file:
            for line in file:
                string_list.append(line)
    except FileNotFoundError:
        print("The file was not found.")
        return None

    string_list = np.array(string_list)
    dataset: list[list[str]] = []
    for element in string_list:
        if element in (None, "\n", "\t\n", ",\n", ";\n", " \n"):
            continue
        if assume_standard_csv_format:
            element = ";".join(element.split(","))

        cand_line: list[str] | None = None

        # Parse separators. Semicolon has top priority so default-CSV is
        # actually parsed there. Tab is next-common and usually unambiguous.
        # Only if no such separator is found do we fall back to space.
        if element.__contains__(";"):
            cand_line = element.replace("\n", "").split(";")
        elif element.__contains__("\t"):
            cand_line = element.replace("\n", "").split("\t")
        elif element.__contains__(" "):
            # collapse runs of whitespace to a single space
            element = " ".join(element.split())
            cand_line = element.replace("\n", "").split(" ")

        # If the split occurred, ``cand_line`` is a list; if not, the error
        # bubbles up as an AssertionError handled at the caller level.
        # Remove empty tokens caused by doubled separators.
        if cand_line is not None:
            dataset.append([k for k in cand_line if k is not None and k != " " and k != ""])

    # For every line, check if it contains both a voltage and a current
    # signifier. If so, mark this line as the header.
    header_line_index = 0
    while header_line_index < len(dataset):
        has_current_signifier = False
        has_voltage_signifier = False
        for substring in dataset[header_line_index]:
            for k in range(len(current_signifiers)):
                has_current_signifier = np.logical_or(
                    has_current_signifier,
                    substring.__contains__(current_signifiers[k]),
                )
            for k in range(len(voltage_signifiers)):
                has_voltage_signifier = np.logical_or(
                    has_voltage_signifier,
                    substring.__contains__(voltage_signifiers[k]),
                )
        if has_voltage_signifier and has_current_signifier:
            break
        else:
            header_line_index += 1

    return dataset[header_line_index:]


def find_voltage_and_current_column(line: list[str]) -> tuple[int | None, int | None]:
    """Return the column indices that hold voltage and current in ``line``.

    Used on a parsed header line. The first matching column wins, which helps
    with files that have multiple columns whose headers contain a signifier
    (e.g. a derivative or smoothed column).
    """
    e_column: int | None = None
    i_column: int | None = None
    for i in range(len(line)):
        for k in range(len(current_signifiers)):
            if line[i].__contains__(current_signifiers[k]):
                i_column = i
        if i_column:
            break
    for i in range(len(line)):
        for k in range(len(voltage_signifiers)):
            if line[i].__contains__(voltage_signifiers[k]):
                e_column = i
        if e_column:
            break
    return e_column, i_column


def _detect_current_correction(line_cell: str) -> float:
    """Return the multiplier that brings the current column to amperes."""
    if line_cell.__contains__("mA"):
        return 0.001
    if line_cell.__contains__("uA"):
        return 0.000001
    if line_cell.__contains__("nA"):
        return 10 ** -9
    return 1


def _detect_potential_correction(line_cell: str) -> float:
    """Return the multiplier that brings the voltage column to volts."""
    if line_cell.__contains__("mV"):
        return 0.001
    if line_cell.__contains__("uV"):
        return 0.000001
    if line_cell.__contains__("nV"):
        return 10 ** -9
    return 1


def read_in_full_cv(filename: str, assume_standard_csv_format: bool) -> "np.ndarray | None":
    """Read an entire CV file and return voltage/current as a NumPy array.

    Detects unit prefixes (mA/uA/nA, mV/uV/nV) in the header and applies the
    appropriate correction so the returned data is in volts and amperes.
    """
    dataset: list[list[Any]] = []
    e_column = 0
    i_column = 1

    current_correction_factor = 1
    potential_correction_factor = 1

    try:
        read_in = read_csv_file(filename, assume_standard_csv_format)
        has_found_header_line = False
        for line in read_in:
            try:
                dataset.append(
                    [
                        float(line[e_column].replace(",", ".")) * potential_correction_factor * pq.V,
                        float(line[i_column].replace(",", ".")) * current_correction_factor * pq.A,
                    ]
                )
            except ValueError:
                if not has_found_header_line:
                    e_column, i_column = find_voltage_and_current_column(line)
                    current_correction_factor = _detect_current_correction(line[i_column])
                    potential_correction_factor = _detect_potential_correction(line[e_column])
                    has_found_header_line = True
                    continue
                else:
                    # Once the header line is found, malformed rows are ignored.
                    continue
        return np.array(dataset)
    except AssertionError:
        print("The dataset contains malformed separators that are not yet handled in the program.")
        return None


def load_one_cycle(
    filename: str,
    cycle_number: int,
    assume_standard_csv_format: bool,
    throw_full_dataset: bool = False,
):
    """Load a specific cycle from a CV file that carries cycle information.

    Raises
    ------
    NoCycleInformationError
        If the file has no cycle column. Callers typically respond by calling
        :func:`cycle_detection_parsing` as a fallback.
    CycleIndexOutOfBoundsError
        If the requested cycle does not exist. The exception carries the
        highest available cycle plus its data so the caller can default to it.
    """
    dataset: list[list[Any]] = []
    e_column = 0
    i_column = 1

    current_correction_factor = 1
    potential_correction_factor = 1
    cycle_column: Any = 2
    try:
        read_in = read_csv_file(filename, assume_standard_csv_format)
        has_found_header_line = False
        for line in read_in:
            try:
                dataset.append(
                    [
                        float(line[e_column].replace(",", ".")) * potential_correction_factor * pq.V,
                        float(line[i_column].replace(",", ".")) * current_correction_factor * pq.A,
                        float(line[cycle_column].replace(",", ".")),
                    ]
                )
            except ValueError:
                if not has_found_header_line:
                    e_column, i_column = find_voltage_and_current_column(line)
                    current_correction_factor = _detect_current_correction(line[i_column])
                    potential_correction_factor = _detect_potential_correction(line[e_column])

                    cycle_column = [i for i in range(len(line)) if (line[i].__contains__("cycle"))]
                    has_found_header_line = True
                    if not str(*cycle_column).isnumeric():
                        raise NoCycleInformationError(
                            "The dataset does not appear to contain cycle data."
                        )
                    else:
                        cycle_column = cycle_column[0]
                        continue
                else:
                    continue

        if throw_full_dataset:
            return dataset

        # Select the requested cycle and catch invalid requests, defaulting to
        # the largest cycle that is available.
        cycle_data: list[list[Any]] = []
        for i in range(len(dataset)):
            if dataset[i][2] == cycle_number:
                cycle_data.append(dataset[i])

        if not cycle_data:
            max_cycle = max(np.array(dataset)[:, 2])
            max_cycle_data: list[list[Any]] = []
            for i in range(len(dataset)):
                if dataset[i][2] == max_cycle:
                    max_cycle_data.append(dataset[i])
            raise CycleIndexOutOfBoundsError(
                requested_cycle=cycle_number,
                highest_available_cycle=max_cycle,
                cycle_data=np.array(max_cycle_data),
                message=(
                    f"The requested cycle was unavailable. Here's cycle no. "
                    f"{max_cycle} instead."
                ),
            )
        return np.array(cycle_data)
    except AssertionError:
        print("The dataset contains malformed separators that are not yet handled in the program.")
        return None


def cycle_detection_parsing(
    filename: str,
    cycle_number: int,
    assume_standard_csv_format: bool,
    throw_full_dataset: bool = False,
):
    """Detect cycles in a CV file that has no cycle column, then return one.

    Used as a fallback when :func:`load_one_cycle` raises
    :class:`NoCycleInformationError`. The detection uses an adaptive voltage
    threshold seeded from the first sample and adjusted based on whether the
    scan starts upwards or downwards.
    """
    e_column = 0
    i_column = 1
    current_cycle = 0

    current_correction_factor = 1
    potential_correction_factor = 1
    dataset: list[list[Any]] = []
    e_treshold = 0

    try:
        read_in = read_csv_file(filename, assume_standard_csv_format)
        has_found_header_line = False
        for line in read_in:
            try:
                voltage = float(line[e_column].replace(",", ".")) * potential_correction_factor * pq.V
                current = float(line[i_column].replace(",", ".")) * current_correction_factor * pq.A
                dataset.append([voltage, current])
            except ValueError:
                if not has_found_header_line:
                    e_column, i_column = find_voltage_and_current_column(line)
                    current_correction_factor = _detect_current_correction(line[i_column])
                    potential_correction_factor = _detect_potential_correction(line[e_column])
                    has_found_header_line = True
                    continue
                else:
                    continue

        # At the 40th element of the dataset, compare which direction the
        # voltage has gone. If nowhere, try at 100; if still the same voltage,
        # enforce a decision at index 250.
        direction_upwards = None
        e_treshold = dataset[0][0]

        # Some CVs have a period of initially constant voltage. This can
        # produce erroneous cycle assessments if not excluded.
        meaningful_start_of_cv = 0
        if dataset[10][0] != e_treshold:
            direction_upwards = (dataset[10][0] > e_treshold)
            meaningful_start_of_cv = 10
        if direction_upwards is None and dataset[40][0] != e_treshold:
            direction_upwards = (dataset[40][0] > e_treshold)
            meaningful_start_of_cv = 40
        if direction_upwards is None and dataset[100][0] != e_treshold:
            direction_upwards = (dataset[100][0] > e_treshold)
            meaningful_start_of_cv = 100
        if direction_upwards is None:
            direction_upwards = (dataset[100][0] >= e_treshold)
            meaningful_start_of_cv = 250

        for i in range(len(dataset)):
            voltage = dataset[i][0]
            if i == 0:
                # Isolate the voltage column. The array operator strips units,
                # so we re-establish them.
                voltage_column = np.array(dataset)[:, 0] * pq.V
                if np.where(voltage_column == min(voltage_column))[0][0] < 200 and direction_upwards:
                    voltage_range = max(voltage_column) - min(voltage_column)
                    e_treshold = voltage + 0.001 * voltage_range
                if np.where(voltage_column == max(voltage_column))[0][0] < 200 and not direction_upwards:
                    voltage_range = max(voltage_column) - min(voltage_column)
                    e_treshold = voltage - 0.001 * voltage_range
                current_cycle = 1

            # Depending on whether the scan begins upwards or downwards, cut
            # the cycles accordingly. With very low-sample data this can fail
            # — reduce the leading offset of 10 if necessary.
            if i > 10:
                if direction_upwards and voltage >= e_treshold > dataset[i - 1][0]:
                    current_cycle += 1
                if not direction_upwards and voltage <= e_treshold < dataset[i - 1][0]:
                    current_cycle += 1
            dataset[i].append(int(current_cycle))

        if throw_full_dataset:
            return np.array(dataset)

        cycle_data: list[list[Any]] = []
        for i in range(len(dataset)):
            if dataset[i][2] == cycle_number:
                cycle_data.append(dataset[i][:2])

        if not cycle_data:
            for i in range(len(dataset)):
                if dataset[i][2] == current_cycle:
                    cycle_data.append(dataset[i][:2])

            raise CycleIndexOutOfBoundsError(
                requested_cycle=cycle_number,
                highest_available_cycle=current_cycle,
                cycle_data=np.array(cycle_data),
                message=(
                    f"The requested cycle was unavailable. Here's cycle no. "
                    f"{current_cycle} instead."
                ),
            )

        return np.array(cycle_data)
    except AssertionError:
        print("The dataset contains malformed separators that are not yet handled in the program.")
        return None


def half_cycle_parsing(
    filename: str,
    assume_standard_csv_format: bool,
    cycle_number: int = 1,
    mode: str = "forward",
    on_cycle_fallback=None,
):
    """Extract a single half-cycle (forward or reverse) from a CV file.

    Parameters
    ----------
    filename, assume_standard_csv_format, cycle_number, mode
        As before.
    on_cycle_fallback : callable or None
        Optional callback ``fn()`` invoked when the requested half-cycle does
        not exist and the function falls back to the first half-cycle. This
        replaces the previous direct GUI update so the I/O layer no longer
        depends on the GUI.
    """
    full_dataset = read_in_full_cv(filename, assume_standard_csv_format)

    direction_upwards, negative_peaks, positive_peaks = get_positive_and_negative_voltage_peaks(full_dataset)

    # ------------------------------------------------------------------
    # FIX (vs. original Dataset-3.py half_cycle_parsing):
    # The original used negative_peaks[cycle-1]:positive_peaks[cycle] for the
    # "forward" / direction_upwards=True branch (and the symmetric variants),
    # which spans ~1.5 half-cycles for cycle >= 2. The intended meaning of
    # "cycle N forward scan" is the N-th ascending segment, bounded by the
    # (N-1)-th valley→peak in a CV that starts ascending and by the N-th
    # valley→peak in a CV that starts descending. After the leading-zero
    # insertion done in get_positive_and_negative_voltage_peaks (Dataset-3.py
    # convention), the correct slice in both cases is
    # ``negative_peaks[cycle-1]:positive_peaks[cycle-1]`` for forward, mirrored
    # for backward. The fallback paths return the first available half-cycle.
    # ------------------------------------------------------------------
    if mode == "forward":
        try:
            if direction_upwards:
                # neg_peaks has a leading 0 inserted; first "true" valley is
                # neg_peaks[1]. Forward scan N spans neg_peaks[N-1] -> pos_peaks[N-1].
                return full_dataset[negative_peaks[cycle_number - 1]:positive_peaks[cycle_number - 1] + 1]
            else:
                # CV starts descending; first ascending scan is bounded by
                # the first valley (neg_peaks[0]) and the first peak (pos_peaks[1]
                # after the leading-zero insertion on pos_peaks).
                return full_dataset[negative_peaks[cycle_number - 1]:positive_peaks[cycle_number] + 1]
        except IndexError:
            print("The half-cycle provided was not found. Providing the first instead.")
            if on_cycle_fallback is not None:
                on_cycle_fallback()
            if direction_upwards:
                return full_dataset[negative_peaks[0]:positive_peaks[0] + 1]
            else:
                return full_dataset[negative_peaks[0]:positive_peaks[1] + 1]
    else:
        try:
            if direction_upwards:
                # pos_peaks has no leading zero (direction_upwards=True branch);
                # backward scan N runs from pos_peaks[N-1] to neg_peaks[N].
                return full_dataset[positive_peaks[cycle_number - 1]:negative_peaks[cycle_number] + 1]
            else:
                # pos_peaks has leading zero; backward scan N runs
                # pos_peaks[N] -> neg_peaks[N-1].
                return full_dataset[positive_peaks[cycle_number]:negative_peaks[cycle_number - 1] + 1]
        except IndexError:
            print("The half-cycle provided was not found. Providing the first instead.")
            if on_cycle_fallback is not None:
                on_cycle_fallback()
            if direction_upwards:
                return full_dataset[positive_peaks[0]:negative_peaks[1] + 1]
            else:
                return full_dataset[positive_peaks[1]:negative_peaks[0] + 1]
