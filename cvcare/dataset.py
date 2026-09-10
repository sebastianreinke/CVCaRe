"""
This file is part of CVCaRe.
Copyright (C) 2022-2026 Sebastian Reinke
Licensed under the GNU General Public License v3 or later.

GUI-independent dataset model for CVCaRe. It loads CV files according to
plain specifications, retains a detailed outcome for each load, and provides
aggregate capacitance, distortion-parameter, and export operations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Iterable, List, Optional

import numpy as np
import quantities as pq
from scipy import optimize as opt

from cvcare.core.cv import CV, FullCV, HalfCV
from cvcare.exceptions import (
    CycleIndexOutOfBoundsError,
    NoCycleInformationError,
    NoScanrateDefinedError,
    NotEnoughCVsToFitError,
    ScanrateExistsError,
    UnknownMethodError,
)
from cvcare.fileio.readers import (
    cycle_detection_parsing,
    half_cycle_parsing,
    load_one_cycle,
)
from cvcare.fileio.writers import write_standardized_data_file


__all__ = [
    "HalfCycleMode",
    "CVLoadSpec",
    "CVLoadResult",
    "Dataset",
]


# --------------------------------------------------------------------------- #
# Data classes and enumerations
# --------------------------------------------------------------------------- #


class HalfCycleMode(str, Enum):
    """Loading mode: complete cycles, ascending/anodic scans, or descending/cathodic scans."""

    FULL = "full cycles"
    ANODIC = "forward"      # Reader mode for ascending/anodic scans
    CATHODIC = "backward"   # Reader mode for descending/cathodic scans

    @property
    def is_halfcycle(self) -> bool:
        return self is not HalfCycleMode.FULL


@dataclass
class CVLoadSpec:
    """Settings required to load one CV file: its sidebar index, path, optional cycle and evaluation voltage, and active/filter flags."""

    index: int
    filepath: str
    cycle_number: Optional[int] = None
    eval_voltage: Optional[float] = None
    use: bool = True
    filtered: bool = False


@dataclass
class CVLoadResult:
    """Result of loading one specification: a CV object on success, or an error and optional cycle correction on failure."""

    spec_index: int
    cv: Optional[CV] = None
    corrected_cycle_number: Optional[int] = None
    error: Optional[BaseException] = None
    info: List[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.cv is not None


# --------------------------------------------------------------------------- #
# Dataset container
# --------------------------------------------------------------------------- #


# Typ für den Half-Cycle-Fallback-Callback. Der Reader fragt nach der
# Zyklusnummer für den Halbzyklus-Fall, falls keine Cycle-Information da ist;
# die GUI kann hier z. B. simpledialog.askinteger() einspielen. Der Default
# ist eine deterministische Funktion, die einfach 1 zurückgibt
OnCycleFallback = Callable[[int], int]
"""Signatur: (spec_index) -> Zyklusnummer."""


def _default_on_cycle_fallback(spec_index: int) -> int:  # noqa: D401
    """Default-Fallback: nimm Zyklus 1, wie im Original ohne GUI-Input."""
    return 1


class Dataset:
    """Collection of successfully loaded CVs and operations performed across them.

    ``contents`` holds usable CV objects. ``load_results`` retains a result for
    every requested specification, including errors and safe cycle fallbacks.
    """

    contents: list[CV]
    load_results: list[CVLoadResult]

    def __init__(
        self,
        specs: Iterable[CVLoadSpec],
        halfcycle_mode: HalfCycleMode = HalfCycleMode.FULL,
        assume_standard_csv: bool = False,
        on_cycle_fallback: OnCycleFallback = _default_on_cycle_fallback,
    ):
        self.contents = []
        self.load_results = []
        # Defensiv koercieren — Aufrufer könnten einen rohen String
        # übergeben (z. B. wenn ein str-Enum durch QComboBox.currentData()
        # geht und dort seinen Enum-Typ verliert).
        if not isinstance(halfcycle_mode, HalfCycleMode):
            try:
                halfcycle_mode = HalfCycleMode(halfcycle_mode)
            except (ValueError, TypeError):
                halfcycle_mode = HalfCycleMode.FULL
        self._halfcycle_mode = halfcycle_mode
        self._assume_standard_csv = assume_standard_csv
        self._on_cycle_fallback = on_cycle_fallback

        for spec in specs:
            result = self._load_one_spec(spec)
            self.load_results.append(result)
            if result.cv is not None:
                self.contents.append(result.cv)

    # ----- File loading logic ----- #

    def _load_one_spec(self, spec: CVLoadSpec) -> CVLoadResult:
        """Load one specification, normalize optional inputs, and return a detailed load result."""
        # An empty file path is ignored without reporting an error.
        if not spec.filepath:
            return CVLoadResult(spec_index=spec.index)

        # ----- 1. Normalize the requested cycle number ----- #
        # darf der Fallback-Callback eingreifen.
        cycle_number = spec.cycle_number
        corrected_cycle_number: Optional[int] = None

        if cycle_number is None:
            cycle_number = 2
            corrected_cycle_number = 2
        else:
            try:
                cycle_number = int(cycle_number)
            except (TypeError, ValueError):
                cycle_number = 2
                corrected_cycle_number = 2

        # ----- 2. Normalize the optional evaluation voltage ----- #
        voltage: Optional[float] = None
        if spec.eval_voltage is not None:
            try:
                if isinstance(spec.eval_voltage, str):
                    voltage = float(spec.eval_voltage.replace(",", "."))
                else:
                    voltage = float(spec.eval_voltage)
            except (TypeError, ValueError):
                voltage = None

        infos: list[str] = []

        # ----- 3. Read the requested data ----- #
        if self._halfcycle_mode is HalfCycleMode.FULL:
            data, corrected_via_load, err = self._load_full_cycle(
                spec=spec, cycle_number=cycle_number
            )
            if corrected_via_load is not None:
                corrected_cycle_number = corrected_via_load
            if err is not None:
                return CVLoadResult(
                    spec_index=spec.index,
                    error=err,
                    corrected_cycle_number=corrected_cycle_number,
                )
            cv: CV = FullCV(
                source=spec.filepath,
                cycle_nr=cycle_number,
                index=spec.index,
                dataset=data,
                eval_voltage=voltage,
                unit_voltage=pq.V,
                unit_current=pq.A,
            )
        else:
            try:
                data = half_cycle_parsing(
                    filename=spec.filepath,
                    assume_standard_csv_format=self._assume_standard_csv,
                    cycle_number=cycle_number,
                    mode=self._halfcycle_mode.value,
                    on_cycle_fallback=lambda: self._on_cycle_fallback(spec.index),
                )
            except Exception as e:  # noqa: BLE001  – Original schluckt auch alles
                return CVLoadResult(
                    spec_index=spec.index,
                    error=e,
                    corrected_cycle_number=corrected_cycle_number,
                )

            if data is None or (isinstance(data, list) and len(data) == 0):
                return CVLoadResult(
                    spec_index=spec.index,
                    error=RuntimeError("HalfCV-Parsing lieferte einen leeren Datensatz."),
                    corrected_cycle_number=corrected_cycle_number,
                )

            cv = HalfCV(
                source=spec.filepath,
                half_cycle_nr=cycle_number,
                index=spec.index,
                dataset=data,
                eval_voltage=voltage,
                unit_voltage=pq.V,
                unit_current=pq.A,
            )

        # ----- 4. Apply active and filtering settings ----- #
        cv.set_active(spec.use)
        cv.set_default_filtered(spec.filtered)

        return CVLoadResult(
            spec_index=spec.index,
            cv=cv,
            corrected_cycle_number=corrected_cycle_number,
            info=infos,
        )

    def _load_full_cycle(self, spec: CVLoadSpec, cycle_number: int):
        """Load one complete cycle, falling back to cycle detection when no cycle column exists."""
        try:
            data = load_one_cycle(
                filename=spec.filepath,
                cycle_number=cycle_number,
                assume_standard_csv_format=self._assume_standard_csv,
            )
            return data, None, None

        except CycleIndexOutOfBoundsError as e:
            highest = int(round(e.highest_available_cycle, 0))
            return np.array(e.cycle_data), highest, None

        except NoCycleInformationError:
            # Fallback: detect cycle boundaries from the voltage trace.
            try:
                data = cycle_detection_parsing(
                    filename=spec.filepath,
                    cycle_number=cycle_number,
                    assume_standard_csv_format=self._assume_standard_csv,
                )
                return data, None, None
            except CycleIndexOutOfBoundsError as e:
                highest = int(round(e.highest_available_cycle, 0))
                return np.array(e.cycle_data), highest, None
            except Exception as e:  # noqa: BLE001
                return None, None, e

        except Exception as e:  # noqa: BLE001
            return None, None, e

    # ----- Convenience constructor for sidebar-style values ----- #

    @classmethod
    def from_gui_values(
        cls,
        values: dict,
        count: int,
        on_cycle_fallback: OnCycleFallback = _default_on_cycle_fallback,
    ) -> "Dataset":
        """Create a dataset from the sidebar-style values dictionary without calling GUI methods."""
        specs: list[CVLoadSpec] = []
        for i in range(1, count + 1):
            filepath = values.get(("cv", i), "")
            if not filepath:
                continue

            cycle_number_raw = values.get(("cycle_nr", i), "")
            cycle_number: Optional[int]
            if cycle_number_raw in ("", None):
                cycle_number = None
            else:
                try:
                    cycle_number = int(cycle_number_raw)
                except (TypeError, ValueError):
                    cycle_number = None  # Spec-Auflösung setzt dann auf 2

            eval_voltage_raw = values.get(("voltage_eval", i), "")
            specs.append(
                CVLoadSpec(
                    index=i,
                    filepath=filepath,
                    cycle_number=cycle_number,
                    eval_voltage=eval_voltage_raw if eval_voltage_raw != "" else None,
                    use=True,
                    filtered=False,
                )
            )

        halfcycle_mode_str = values.get("halfcycle_mode", "full cycles")
        try:
            halfcycle_mode = HalfCycleMode(halfcycle_mode_str)
        except ValueError:
            halfcycle_mode = HalfCycleMode.FULL

        return cls(
            specs=specs,
            halfcycle_mode=halfcycle_mode,
            assume_standard_csv=bool(values.get("default_csv_format", False)),
            on_cycle_fallback=on_cycle_fallback,
        )

    # --------------------------------------------------------------------- #
    # --------------------------------------------------------------------- #

    # Scan rates are supplied as [CV index, scan rate] pairs.
    def set_scanrates(self, scanrates: list[list]):
        scanrates = np.array(scanrates)
        for i in range(len(scanrates)):
            if self.get_content_by_index(scanrates[i, 0]) is not None:
                self.get_content_by_index(scanrates[i, 0]).set_scanrate(
                    scanrates[i, 1], pq.mV / pq.s
                )

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

    def set_active_for_full_dataset(self, indices_of_active_cv: list[int]):
        for i in range(len(self.contents)):
            if self.contents[i].get_index() in indices_of_active_cv:
                self.contents[i].set_active(True)
            else:
                self.contents[i].set_active(False)

    def set_activity_of_element(self, index: int, activity: bool):
        for i in range(len(self.contents)):
            if self.contents[i].get_index() == index:
                self.contents[i].set_active(activity)

    def set_default_filtered_of_element(self, index: int, filtered: bool):
        for i in range(len(self.contents)):
            if self.contents[i].get_index() == index:
                self.contents[i].set_default_filtered(filtered)

    def get_capacitance(
        self,
        method="minmax_corrected",
        through_zero=True,
        active_only=True,
        half_cycle_select="full",
    ):
        if method == "at_selected_voltage":
            return self.get_capacitance_at_selected_voltage(
                through_zero, active_only, half_cycle_select=half_cycle_select
            )
        if method == "minmax":
            return self.get_capacitance_by_minmax(
                through_zero,
                active_only,
                half_cycle_select=half_cycle_select,
                corrected=False,
            )
        if method == "minmax_corrected":
            return self.get_capacitance_by_minmax(
                through_zero,
                active_only,
                half_cycle_select=half_cycle_select,
                corrected=True,
            )

    def get_capacitance_at_selected_voltage(
        self,
        through_zero=False,
        active_only=True,
        half_cycle_select: str = "full",
    ):
        def linear(x, m, n):
            return m * x + n

        def linear_through_zero(x, m):
            return m * x

        if active_only:
            subset = [element for element in self.contents if element.is_active()]
        else:
            subset = self.contents

        dataset_to_optimize = []
        for i in range(len(subset)):
            try:
                dataset_to_optimize.append(
                    [subset[i].get_index(), subset[i].get_scanrate(pq.mV / pq.s)]
                )
            except ScanrateExistsError as e:
                print(e)

        if len(dataset_to_optimize) > 1:
            if through_zero:
                callable_function = linear_through_zero
            else:
                callable_function = linear

            if half_cycle_select == "full":
                for i in range(len(dataset_to_optimize)):
                    if isinstance(self.get_content_by_index(dataset_to_optimize[i][0]), FullCV):
                        dataset_to_optimize[i].append(
                            self.get_content_by_index(dataset_to_optimize[i][0]).get_current_at_voltage(
                                current_dimension=pq.mA
                            )
                            / 2
                        )
                    else:
                        dataset_to_optimize[i].append(
                            self.get_content_by_index(
                                dataset_to_optimize[i][0]
                            ).get_current_at_voltage(current_dimension=pq.mA)
                        )
            elif half_cycle_select in ["anodic", "cathodic"]:
                for i in range(len(dataset_to_optimize)):
                    if isinstance(self.get_content_by_index(dataset_to_optimize[i][0]), FullCV):
                        dataset_to_optimize[i].append(
                            self.get_content_by_index(
                                dataset_to_optimize[i][0]
                            ).get_current_at_voltage_in_halfcycle(
                                half_cycle_select=half_cycle_select,
                                current_dimension=pq.mA,
                            )
                        )
                    else:
                        dataset_to_optimize[i].append(
                            self.get_content_by_index(
                                dataset_to_optimize[i][0]
                            ).get_current_at_voltage(current_dimension=pq.mA)
                        )

            dataset_to_optimize = np.array(dataset_to_optimize)
            popt = opt.curve_fit(
                callable_function, dataset_to_optimize[:, 1], dataset_to_optimize[:, 2]
            )
            if through_zero:
                capacitance = popt[0][0]
                offset = 0
            else:
                capacitance = popt[0][0]
                offset = popt[0][1]
            return capacitance, offset, dataset_to_optimize
        else:
            raise NotEnoughCVsToFitError(
                "There are insufficient CVs loaded to perform this calculation."
            )

    def write_capacitance_to_file(
        self,
        filename: str,
        through_zero: bool,
        half_cycle_select: str = "full",
        method="minmax_corrected",
    ):
        try:
            capacitance, offset, index_scanrate_current_dataset = self.get_capacitance(
                through_zero=through_zero,
                active_only=True,
                half_cycle_select=half_cycle_select,
                method=method,
            )
        except NotEnoughCVsToFitError as e:
            print(e)
            return False

        index_scanrate_current_dataset = index_scanrate_current_dataset.tolist()
        for k in range(0, len(index_scanrate_current_dataset)):
            local_index = int(index_scanrate_current_dataset[k][0])
            local_cv = self.get_content_by_index(local_index)
            index_scanrate_current_dataset[k].extend(
                [
                    local_cv.get_eval_voltage() * local_cv.unit_voltage,
                    local_cv.get_default_filtered(),
                    local_cv.get_cycle_nr(),
                    type(local_cv).__name__,
                    local_cv.source,
                ]
            )

        if half_cycle_select == "anodic":
            current_name = f"anodic current {method}"
        elif half_cycle_select == "cathodic":
            current_name = f"cathodic current {method}"
        else:
            current_name = f"half current difference {method}"

        index_scanrate_current_dataset.insert(0, ["Capacitance_Estimate [F]:", capacitance])
        index_scanrate_current_dataset.insert(1, ["Cycle mode selected", half_cycle_select])
        index_scanrate_current_dataset.insert(
            2,
            [
                "Dataset No.",
                "Scanrate [mV/s]",
                current_name,
                "Evaluated at",
                "Filtered",
                "Used cycle",
                "Data type",
                "Original Filename",
            ],
        )

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
                    filtered=filter_this_cv, comment=comment_this_cv
                )
            if write_array:
                success = write_standardized_data_file(filenames.get(index), write_array)
                if success:
                    print(f"CV {index} successfully written to file {filenames.get(index)}")
                else:
                    print(f"CV {index} could not be written to file {filenames.get(index)}")
                has_written = has_written and success
        return has_written

    def write_CVs_to_single_file(self, filename: str, filtered: dict, comment: dict):
        def stack_lists(list1: list, list2: list):
            assert len(list1) == len(list2)
            result = []
            for list_index in range(len(list1)):
                result.append([*list1[list_index], *list2[list_index]])
            return result

        data_list = []
        for index in filtered.keys():
            comment_this_cv = comment.get(index, None)
            get_cv = self.get_content_by_index(index)
            if get_cv is not None:
                new_cv_data = self.get_content_by_index(index).collapse_to_writable_list(
                    filtered=filtered.get(index), comment=comment_this_cv
                )
                data_list.append(new_cv_data)

        if data_list:
            max_len = np.max([len(k) for k in data_list])
            for i in range(len(data_list)):
                dataset = data_list[i]
                while len(dataset) < max_len:
                    dataset.append([" ", " "])
                data_list[i] = dataset

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
        write_list = [
            [
                "CV No.",
                "Source file",
                "Resistance [Ohm]",
                "Capacitance [F]",
                "Scan rate [mV/s]",
                "p_d",
                "Potential window[V]",
            ]
        ]
        for cv in self.contents:
            if cv.is_active():
                try:
                    if method == "Analytical":
                        (
                            resistance,
                            capacitance,
                            potential_window,
                            distortion_param,
                            offset,
                        ) = cv.distortion_param_evaluation()
                    elif method == "Optimisation enhanced analytical":
                        (
                            resistance,
                            capacitance,
                            potential_window,
                            distortion_param,
                            offset,
                        ) = cv.fit_cv_by_optimisation()
                    if "resistance" not in locals():
                        raise UnknownMethodError("Method string code was unrecognised.")
                    write_list.append(
                        [
                            cv.get_index(),
                            cv.get_source(),
                            resistance.magnitude,
                            capacitance.magnitude,
                            cv.get_scanrate(dimension=pq.mV / pq.s).magnitude,
                            distortion_param,
                            potential_window,
                        ]
                    )
                except NoScanrateDefinedError:
                    print(
                        f"CV no. {cv.get_index()} failed to evaluate, due to lack of scan rate. "
                        f"Writing process proceeds without it."
                    )

        return write_standardized_data_file(filename=filename, data_to_write=write_list)

    def get_capacitance_by_minmax(
        self, through_zero, active_only, half_cycle_select, corrected
    ):
        def linear(x, m, n):
            return m * x + n

        def linear_through_zero(x, m):
            return m * x

        if active_only:
            subset = [element for element in self.contents if element.is_active()]
        else:
            subset = self.contents

        dataset_to_optimize = []
        for i in range(len(subset)):
            try:
                dataset_to_optimize.append(
                    [subset[i].get_index(), subset[i].get_scanrate(pq.mV / pq.s)]
                )
            except ScanrateExistsError as e:
                print(e)

        if len(dataset_to_optimize) > 1:
            if through_zero:
                callable_function = linear_through_zero
            else:
                callable_function = linear

            if half_cycle_select == "full":
                for i in range(len(dataset_to_optimize)):
                    cv = self.get_content_by_index(dataset_to_optimize[i][0])
                    if isinstance(cv, FullCV):
                        current_to_append = cv.get_minmax_current() / 2
                    else:
                        current_to_append = cv.get_minmax_current()

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
                    else:
                        dataset_to_optimize[i].append(
                            self.get_content_by_index(
                                dataset_to_optimize[i][0]
                            ).get_current_at_voltage(current_dimension=pq.mA)
                        )
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
                    else:
                        dataset_to_optimize[i].append(
                            self.get_content_by_index(
                                dataset_to_optimize[i][0]
                            ).get_current_at_voltage(current_dimension=pq.mA)
                        )

            if corrected:
                for i in range(len(dataset_to_optimize)):
                    cv = self.get_content_by_index(dataset_to_optimize[i][0])
                    vert_ratio, distortion_param = cv.estimate_vertical_current_ratio()
                    print(
                        f"Corrected dataset: Distortion param {distortion_param}, correction by {1 / vert_ratio}"
                    )
                    dataset_to_optimize[i][-1] = dataset_to_optimize[i][-1] / vert_ratio

            dataset_to_optimize = np.array(dataset_to_optimize)
            popt = opt.curve_fit(
                callable_function, dataset_to_optimize[:, 1], dataset_to_optimize[:, 2]
            )
            if through_zero:
                capacitance = popt[0][0]
                offset = 0
            else:
                capacitance = popt[0][0]
                offset = popt[0][1]
            return capacitance, offset, dataset_to_optimize
        else:
            raise NotEnoughCVsToFitError(
                "There are insufficient CVs loaded to perform this calculation."
            )
