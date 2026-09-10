"""
This file is part of CVCaRe.
Copyright (C) 2022-2026 Sebastian Reinke
Licensed under the GNU General Public License v3 or later.

Standardized output writers for CV datasets and analysis results.
"""
from __future__ import annotations

import os
from typing import Iterable, Sequence

import numpy as np

from cvcare.exceptions import NoCycleInformationError
from cvcare.fileio.readers import load_one_cycle, cycle_detection_parsing


__all__ = [
    "write_standardized_data_file",
    "write_split_cycles",
]


def write_standardized_data_file(filename: str, data_to_write: Sequence[Iterable]) -> bool:
    try:
        with open(filename, "a") as file:
            for row in data_to_write:
                writestring = ""
                for element in row:
                    writestring += str(element) + "\t"
                file.write(writestring + "\n")
        return True
    except (FileNotFoundError, FileExistsError, OSError, PermissionError):
        print("There was an error opening the file to write on.")
        return False


def _generate_unique_filename(filename: str) -> str:
    base_name, extension = os.path.splitext(filename)
    new_filename = base_name + "_CVEval_cycle_split" + extension
    if not os.path.exists(new_filename):
        return new_filename

    x = 1
    while True:
        new_filename = f"{base_name}_CVEval_cycle_split({x}){extension}"
        if not os.path.exists(new_filename):
            return new_filename
        x += 1


def _reshape_array(array: np.ndarray) -> np.ndarray:
    cycle_numbers = np.unique(array[:, 2])

    cycles: list[np.ndarray] = []
    max_length = 0
    for cycle_number in cycle_numbers:
        cycle_data = array[array[:, 2] == cycle_number][:, :2]
        cycle_data_with_header = np.vstack([
            [f"Voltage/ Cycle {cycle_number}", f"Current/ Cycle {cycle_number}"],
            cycle_data,
        ])
        cycles.append(cycle_data_with_header)
        max_length = max(max_length, len(cycle_data_with_header))

    for i in range(len(cycles)):
        cycle_length = len(cycles[i])
        if cycle_length < max_length:
            padding = [[" ", " "]] * (max_length - cycle_length)
            cycles[i] = np.vstack([cycles[i], padding])

    reshaped_array = np.hstack(cycles)
    return reshaped_array


def write_split_cycles(filename: str, assume_standard_csv_format: bool) -> bool:
    try:
        dataset = load_one_cycle(
            filename=filename,
            cycle_number=0,
            assume_standard_csv_format=assume_standard_csv_format,
            throw_full_dataset=True,
        )
        dataset = np.array(dataset)
    except NoCycleInformationError:
        dataset = cycle_detection_parsing(
            filename=filename,
            cycle_number=0,
            assume_standard_csv_format=assume_standard_csv_format,
            throw_full_dataset=True,
        )

    data_to_write = _reshape_array(dataset)
    success = write_standardized_data_file(
        _generate_unique_filename(filename), data_to_write
    )
    return success
