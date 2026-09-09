"""
cvcare.dataset
==============

GUI-unabhängige Dataset-Klasse für CVCaRe.

Im Original (``Dataset-3.py``) war ``Dataset.__init__`` direkt an die
FreeSimpleGUI-Window-Instanz gekoppelt: der Konstruktor las aus einem
``values``-Dictionary (das so nur aus der GUI fällt), schrieb bei
auto-korrigierten Zyklusnummern ``window[("cycle_nr", i)].update(...)`` und
reichte das Window-Handle an :func:`half_cycle_parsing` weiter.

Diese Datei entkoppelt das. Die Lade-Schnittstelle besteht aus drei
Datenklassen:

* :class:`CVLoadSpec`  – eine zu ladende CV-Datei + Parameter.
* :class:`HalfCycleMode` – die drei zulässigen Modi (``full``, ``anodic``,
  ``cathodic``).
* :class:`CVLoadResult` – das Lade-Ergebnis pro Spec: erfolgreich (mit CV)
  oder fehlgeschlagen (mit Fehlerinformation).

Die GUI baut die Liste der :class:`CVLoadSpec`, ruft :class:`Dataset` auf
und liest am Ende ``dataset.load_results`` aus, um ihre Widgets — z. B. das
``cycle_nr``-Eingabefeld — zu aktualisieren. So bleibt die Domain-Logik
vollständig ohne GUI-Aufrufe und damit testbar.

Die Berechnungs- und Schreibmethoden (Kapazität, Verzerrungsparameter,
Export) sind 1:1 aus ``Dataset-3.py`` übernommen.
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
# Datenklassen / Enums
# --------------------------------------------------------------------------- #


class HalfCycleMode(str, Enum):
    """
    Lade-Modus für einen Datensatz.

    Wert-Strings sind so gewählt, dass sie zu den Strings passen, die der
    Original-Reader (:func:`cvcare.fileio.readers.half_cycle_parsing`) im
    Parameter ``mode`` erwartet (``"forward"``/``"backward"``) bzw. die GUI
    im Original verwendet hat (``"full cycles"``).
    """

    FULL = "full cycles"
    ANODIC = "forward"      # entspricht dem Reader-Parameter mode="forward"
    CATHODIC = "backward"   # entspricht dem Reader-Parameter mode="backward"

    @property
    def is_halfcycle(self) -> bool:
        return self is not HalfCycleMode.FULL


@dataclass
class CVLoadSpec:
    """
    Eine zu ladende CV-Datei mit allen Parametern, die für das Lesen und
    Konstruieren eines :class:`~cvcare.core.cv.CV`-Objekts notwendig sind.

    Felder
    ------
    index:
        Logischer 1-basierter Index in der GUI-Sidebar bzw. im Dataset. Wird
        auch dem ``CV``-Objekt mitgegeben und ist die ID, über die die GUI
        später Widgets wiederfindet.
    filepath:
        Pfad zur CV-Quelldatei.
    cycle_number:
        Gewünschte Zyklusnummer (1-basiert nach Konvention der Reader).
        ``None`` → Default 2 wie im Original.
    eval_voltage:
        Optionale Auswertungsspannung in V (für ``get_current_at_voltage``).
        ``None`` ist erlaubt.
    use:
        Aktiv-Flag für den späteren Fit. Mapping auf
        :meth:`cvcare.core.cv.CV.set_active`.
    filtered:
        Default-Filter-Flag (Savitzky-Golay) für Plot und Auswertung.
        Mapping auf :meth:`cvcare.core.cv.CV.set_default_filtered`.
    """

    index: int
    filepath: str
    cycle_number: Optional[int] = None
    eval_voltage: Optional[float] = None
    use: bool = True
    filtered: bool = False


@dataclass
class CVLoadResult:
    """
    Lade-Ergebnis für eine :class:`CVLoadSpec`.

    Felder
    ------
    spec_index:
        Der ``index`` der zugehörigen :class:`CVLoadSpec` (1-basiert).
    cv:
        Das erzeugte :class:`~cvcare.core.cv.CV`-Objekt oder ``None``, wenn
        der Ladevorgang fehlschlug.
    corrected_cycle_number:
        Falls die Reader-Funktion eine Zyklus-Out-of-Bounds-Situation
        festgestellt und auf den höchsten verfügbaren Zyklus zurückgefallen
        ist, steht hier die *tatsächlich verwendete* Zyklusnummer; die GUI
        kann ihr Eingabefeld entsprechend aktualisieren. ``None``, wenn der
        ursprünglich angeforderte Zyklus verwendet wurde.
    error:
        Falls ``cv is None``: die zugehörige Exception (oder None falls
        leere Spec — z. B. leerer Dateiname).
    info:
        Menschenlesbare Hinweise, die in der GUI-Statusleiste angezeigt
        werden können (z. B. „Auto-Cycle-Detection wurde verwendet“).
    """

    spec_index: int
    cv: Optional[CV] = None
    corrected_cycle_number: Optional[int] = None
    error: Optional[BaseException] = None
    info: List[str] = field(default_factory=list)

    @property
    def success(self) -> bool:
        return self.cv is not None


# --------------------------------------------------------------------------- #
# Dataset
# --------------------------------------------------------------------------- #


# Typ für den Half-Cycle-Fallback-Callback. Der Reader fragt nach der
# Zyklusnummer für den Halbzyklus-Fall, falls keine Cycle-Information da ist;
# die GUI kann hier z. B. simpledialog.askinteger() einspielen. Der Default
# ist eine deterministische Funktion, die einfach 1 zurückgibt — das war im
# Original der "OK ohne Eingabe"-Fallback in half_cycle_parsing.
OnCycleFallback = Callable[[int], int]
"""Signatur: (spec_index) -> Zyklusnummer."""


def _default_on_cycle_fallback(spec_index: int) -> int:  # noqa: D401
    """Default-Fallback: nimm Zyklus 1, wie im Original ohne GUI-Input."""
    return 1


class Dataset:
    """
    Sammelt mehrere geladene CVs und stellt Aggregations-/Export-Methoden
    bereit.

    Im Gegensatz zum Original (das ein ``values: dict`` aus der GUI plus ein
    ``window: sg.Window`` entgegennahm) bekommt der Konstruktor hier eine
    Liste von :class:`CVLoadSpec`-Objekten. Nach der Konstruktion stehen
    zwei Listen zur Verfügung:

    * ``contents`` — die erfolgreich geladenen ``CV``-Objekte (wie zuvor).
    * ``load_results`` — pro Spec ein :class:`CVLoadResult` mit
      Erfolgs-/Fehlerinformation und ggf. korrigierter Zyklusnummer. Die
      GUI iteriert darüber, um ihre Widgets zu aktualisieren — das ersetzt
      die direkten ``window[...].update(...)``-Aufrufe des Originals.

    Parameters
    ----------
    specs:
        Liste der zu ladenden CVs.
    halfcycle_mode:
        :class:`HalfCycleMode` (Full / Anodic / Cathodic).
    assume_standard_csv:
        Reicht den Original-Parameter ``assume_standard_csv_format`` an die
        Reader durch.
    on_cycle_fallback:
        Callback für den Fall, dass im Halbzyklus-Modus keine
        Cycle-Information vorhanden ist. Signatur ``(spec_index) -> int``.
        Default: gibt immer 1 zurück (verhält sich wie der Original-Pfad
        ohne GUI-Input).
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

    # ----- Ladelogik (entkoppelt aus dem Original-__init__) ----- #

    def _load_one_spec(self, spec: CVLoadSpec) -> CVLoadResult:
        """
        Lädt eine einzelne :class:`CVLoadSpec` und gibt ein
        :class:`CVLoadResult` zurück.

        Verhalten ist 1:1 aus ``Dataset-3.py`` übernommen — inklusive der
        Fallback-Kette (Cycle-Out-Of-Bounds → höchster verfügbarer Zyklus;
        keine Cycle-Information → Auto-Detect). Geändert wurde nur, *wie*
        Out-of-Bounds-Korrekturen kommuniziert werden: nicht mehr direkt
        an ein GUI-Window, sondern als ``corrected_cycle_number`` im
        Ergebnis.
        """
        # leerer Spec-Eintrag → kein Ladevorgang, kein Fehler
        if not spec.filepath:
            return CVLoadResult(spec_index=spec.index)

        # ----- 1. Zyklusnummer aufbereiten ----- #
        # Originalverhalten: leer → 2; nicht-int → 2; im Halbzyklus-Modus
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

        # ----- 2. Eval-Voltage aufbereiten ----- #
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

        # ----- 3. Tatsächliches Lesen ----- #
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

        # ----- 4. Aktivität / Filter aus Spec übernehmen ----- #
        cv.set_active(spec.use)
        cv.set_default_filtered(spec.filtered)

        return CVLoadResult(
            spec_index=spec.index,
            cv=cv,
            corrected_cycle_number=corrected_cycle_number,
            info=infos,
        )

    def _load_full_cycle(self, spec: CVLoadSpec, cycle_number: int):
        """
        Versucht ``load_one_cycle``; bei Out-of-Bounds nimmt den höchsten
        verfügbaren Zyklus; bei fehlender Cycle-Information fällt auf
        ``cycle_detection_parsing`` zurück. Verhalten 1:1 aus dem Original.

        Returns
        -------
        (data, corrected_cycle_number, error)
            ``data`` ist ein NumPy-Array oder ``None`` bei Fehler. Im
            Fehlerfall enthält ``error`` die Exception, sonst ``None``.
            ``corrected_cycle_number`` ist gesetzt, wenn die Reader-Routine
            auf den höchsten verfügbaren Zyklus zurückgefallen ist.
        """
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
            # Auto-Detect-Fallback
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

    # ----- Klassenmethode für die GUI-Brücke ----- #

    @classmethod
    def from_gui_values(
        cls,
        values: dict,
        count: int,
        on_cycle_fallback: OnCycleFallback = _default_on_cycle_fallback,
    ) -> "Dataset":
        """
        Komfort-Konstruktor, der das alte ``values``-Dict-Schema aus der
        FreeSimpleGUI-Welt akzeptiert. So kann die *neue* GUI schrittweise
        migriert werden und die Original-Aufruflogik bleibt 1:1 testbar.

        Erwartetes Schema (wie im Original):

        ``values[("cv", i)]``            – Dateipfad (string), leere
                                            Strings werden übersprungen.
        ``values[("cycle_nr", i)]``       – Zyklusnummer (string oder int).
        ``values[("voltage_eval", i)]``   – Auswertespannung (string mit
                                            ``,`` als Dezimaltrenner ok).
        ``values["halfcycle_mode"]``      – ``"full cycles"`` /
                                            ``"forward"`` / ``"backward"``.
        ``values["default_csv_format"]``  – bool.

        Out-of-Bounds-Korrekturen werden über ``load_results`` reportet —
        die aufrufende GUI iteriert darüber und aktualisiert ihre
        ``cycle_nr``-Eingabefelder.
        """
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
    # Die übrigen Methoden sind 1:1 aus Dataset-3.py übernommen.
    # --------------------------------------------------------------------- #

    # scanrates: list of [index of CV, scanrate]
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
