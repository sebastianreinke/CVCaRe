"""
This file is part of CVCaRe.
Copyright (C) 2022-2026 Sebastian Reinke
Licensed under the GNU General Public License v3 or later.

Voltage turning-point detection for splitting CV recordings into cycles and half-cycles.
"""
from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np


__all__ = ["get_positive_and_negative_voltage_peaks"]


_log = logging.getLogger(__name__)


def get_positive_and_negative_voltage_peaks(
    full_dataset: np.ndarray,
) -> Tuple[bool, List[int], List[int]]:
    """
    Identifies positive and negative voltage peaks in a cyclic voltammetry (CV) dataset.

    This function analyses a cyclic voltammetry dataset to detect the indices
    of positive and negative voltage peaks. A positive peak is defined as a
    point where the voltage value is greater than both its immediate
    neighbours and those a fixed distance away, indicating a local maximum.
    Similarly, a negative peak is identified as a local minimum where the
    voltage value is lower than its neighbours.

    The initial scan direction is determined via a multi-stage heuristic.
    Many potentiostats prepend a phase of constant voltage to the recording;
    a naive comparison between ``full_dataset[0]`` and ``full_dataset[40]``
    would then misclassify the direction. The implementation therefore
    probes indices 3, 10, 50 and 150 in order — the first index whose
    voltage differs from ``full_dataset[0][0]`` defines both the direction
    and the truncation/look-ahead window used for the actual peak search.

    Parameters
    ----------
    full_dataset:
        A 2D NumPy array where the first column represents voltage values
        and the second column represents current values. Only the voltage
        column is used for peak detection.

    Returns
    -------
    direction_upwards : bool
        True if the scan begins with rising voltage, False otherwise.
    negative_peaks : list of int
        Indices in the dataset where negative voltage peaks (local minima)
        are detected. The list always contains the start or end index of
        the dataset, depending on direction, so that downstream cycle
        slicing produces complete segments.
    positive_peaks : list of int
        Indices in the dataset where positive voltage peaks (local maxima)
        are detected. Same boundary-padding rules as ``negative_peaks``.

    Notes
    -----
    * For very short datasets (``< 200`` samples) a low-resolution mode is
      activated with smaller truncation and look-ahead distances. This is
      necessary for hand-recorded or down-sampled CVs that would otherwise
      not yield enough comparison samples.
    * Truncation must always be ``>=`` look-ahead, otherwise the
      neighbour-comparison would index out of bounds.
    """
    is_low_resolution = len(full_dataset) < 200

    direction_upwards: bool | None = None
    truncate: Tuple[int, int]
    lookahead: int

    e_threshold = full_dataset[0][0]

    if full_dataset[3][0] != e_threshold:
        direction_upwards = full_dataset[3][0] > e_threshold
        truncate = (3, 3)
        lookahead = 3
    elif full_dataset[10][0] != e_threshold:
        direction_upwards = full_dataset[10][0] > e_threshold
        truncate = (10, 5)
        lookahead = 5
    elif full_dataset[50][0] != e_threshold:
        direction_upwards = full_dataset[50][0] > e_threshold
        truncate = (40, 10)
        lookahead = 10
    else:
        direction_upwards = full_dataset[150][0] >= e_threshold
        truncate = (150, 30)
        lookahead = 10

    if is_low_resolution:
        truncate = (5, 5)
        lookahead = 3

    positive_peaks_raw: List[int] = []
    negative_peaks_raw: List[int] = []
    for index in range(truncate[0], len(full_dataset) - truncate[1]):
        if (
            full_dataset[index - 1, 0] <= full_dataset[index, 0] >= full_dataset[index + 1, 0]
            and full_dataset[index - lookahead, 0]
            < full_dataset[index, 0]
            > full_dataset[index + lookahead, 0]
        ):
            positive_peaks_raw.append(index)
        if (
            full_dataset[index - 1, 0] >= full_dataset[index, 0] <= full_dataset[index + 1, 0]
            and full_dataset[index - lookahead, 0]
            > full_dataset[index, 0]
            < full_dataset[index + lookahead, 0]
        ):
            negative_peaks_raw.append(index)

    def _collapse_plateaus(indices: List[int]) -> List[int]:
        if not indices:
            return []
        collapsed: List[int] = []
        run = [indices[0]]
        for idx in indices[1:]:
            if idx == run[-1] + 1:
                run.append(idx)
            else:
                collapsed.append(run[len(run) // 2])
                run = [idx]
        collapsed.append(run[len(run) // 2])
        return collapsed

    positive_peaks: List[int] = _collapse_plateaus(positive_peaks_raw)
    negative_peaks: List[int] = _collapse_plateaus(negative_peaks_raw)

    _log.debug("direction_upwards=%s", direction_upwards)
    _log.debug("positive_peaks=%s", positive_peaks)
    _log.debug("negative_peaks=%s", negative_peaks)

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

    _log.debug(
        "Corrected peak lists / positive=%s, negative=%s, direction_upwards=%s",
        positive_peaks,
        negative_peaks,
        direction_upwards,
    )
    return direction_upwards, negative_peaks, positive_peaks
