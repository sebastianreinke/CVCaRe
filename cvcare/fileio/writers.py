"""
cvcare.fileio.writers
=================

Funktionen zum Schreiben von CV-Datensätzen in standardisierte Dateien.

Diese Funktionen sind 1:1 aus ``Dataset-3.py`` übernommen, mit folgenden
minimalen Korrekturen:

* ``except FileNotFoundError or FileExistsError`` ist in Python ein Bug — der
  ``or``-Operator wertet zur ersten wahrheitsgemäßen Exception aus und fängt
  damit *nur* ``FileNotFoundError``. Korrigiert zu einem Tupel
  ``(FileNotFoundError, FileExistsError, OSError, PermissionError)``.
* Importe auf die neue Paketstruktur ``cvcare.*`` umgestellt.
* Keine GUI-Aufrufe; Fehler werden über Rückgabewerte signalisiert
  (``True``/``False``), damit die Funktionen testbar bleiben.

Die Lese-/Parse-Logik selbst bleibt vollständig unverändert.
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
    """
    Schreibt einen 2D-Datensatz tab-separiert in ``filename`` (Anhang-Modus).

    Parameters
    ----------
    filename:
        Zieldatei. Wird im Anhang-Modus (``"a"``) geöffnet — bestehende Inhalte
        bleiben erhalten. Verhalten ist identisch zur Original-Implementierung
        in ``Dataset-3.py``.
    data_to_write:
        Sequenz von Zeilen; jede Zeile ist selbst eine iterierbare Sequenz von
        Werten (Strings/Zahlen). Werte werden via ``str(...)`` konvertiert und
        durch Tab getrennt geschrieben.

    Returns
    -------
    bool
        ``True`` bei Erfolg, ``False`` wenn die Datei nicht geöffnet werden
        konnte (Datei-/Berechtigungsfehler).
    """
    try:
        with open(filename, "a") as file:
            for row in data_to_write:
                writestring = ""
                for element in row:
                    writestring += str(element) + "\t"
                file.write(writestring + "\n")
        return True
    except (FileNotFoundError, FileExistsError, OSError, PermissionError):
        # Im Original wurde hier per print kommuniziert; das behalten wir bei,
        # damit aufrufender Code (GUI/CLI) sich nicht ändern muss.
        print("There was an error opening the file to write on.")
        return False


def _generate_unique_filename(filename: str) -> str:
    """
    Erzeugt einen eindeutigen Dateinamen mit Suffix ``_CVEval_cycle_split``.

    Falls die Zieldatei bereits existiert, wird ``(1)``, ``(2)``, ... angehängt,
    bis ein noch nicht existierender Pfad gefunden ist. Verhalten identisch
    zur lokalen Hilfsfunktion in ``write_split_cycles`` im Original.
    """
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
    """
    Stellt einen ``(N, 3)``-Datensatz ``[U, I, cycle]`` als nebeneinander
    gelegte Zyklus-Spalten dar.

    Jeder Zyklus wird zu zwei Spalten (Voltage/Current) mit Header
    ``"Voltage/ Cycle <n>"`` bzw. ``"Current/ Cycle <n>"``. Kürzere Zyklen
    werden mit Leerzeichen-Strings auf die Länge des längsten Zyklus aufgefüllt.

    Logik 1:1 aus ``Dataset-3.py``.
    """
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
    """
    Liest einen CV-Datensatz, splittet ihn in Einzelzyklen und schreibt das
    Ergebnis in eine neue Datei (Suffix ``_CVEval_cycle_split``).

    Ablauf identisch zum Original:

    1. Zuerst wird versucht, die vom Potentiostaten gelabelten Zyklen via
       :func:`cvcare.fileio.readers.load_one_cycle` zu lesen
       (``throw_full_dataset=True``).
    2. Schlägt das mit :class:`NoCycleInformationError` fehl, wird die
       Zyklus-Erkennung über :func:`cvcare.fileio.readers.cycle_detection_parsing`
       angestoßen.
    3. Das resultierende ``(N, 3)``-Array wird per :func:`_reshape_array` in
       eine spaltenweise Darstellung umgewandelt.
    4. Geschrieben wird per :func:`write_standardized_data_file` in eine
       eindeutige neue Datei (siehe :func:`_generate_unique_filename`).

    Parameters
    ----------
    filename:
        Pfad zur Quell-CV-Datei.
    assume_standard_csv_format:
        Wird unverändert an die Reader-Funktionen weitergereicht.

    Returns
    -------
    bool
        Erfolg des Schreibvorgangs.
    """
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
