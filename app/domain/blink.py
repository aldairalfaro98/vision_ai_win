# app/domain/blink.py
from __future__ import annotations

from statistics import median
from typing import Sequence


def count_blinks_from_ear_series(
    ears: Sequence[float],
    *,
    baseline_frames: int = 3,
    close_ratio: float = 0.75,
    open_ratio: float = 0.90,
    min_closed_frames: int = 1,
) -> int:
    if not ears:
        return 0

    baseline_frames = int(baseline_frames)
    if baseline_frames <= 0:
        return 0

    if len(ears) <= baseline_frames + 2:
        return 0

    baseline = _safe_median(ears[:baseline_frames])
    if baseline <= 0:
        return 0

    close_th = baseline * float(close_ratio)
    open_th = baseline * float(open_ratio)

    state = "OPEN"
    closed_run = 0
    blink_count = 0

    for ear in ears[baseline_frames:]:
        ear_val = float(ear)

        if state == "OPEN":
            if ear_val < close_th:
                state = "CLOSED"
                closed_run = 1
            continue

        if state == "CLOSED":
            if ear_val < close_th:
                closed_run += 1
                continue

            if ear_val > open_th:
                if closed_run >= int(min_closed_frames):
                    blink_count += 1
                state = "OPEN"
                closed_run = 0

    return int(blink_count)


def _safe_median(values: Sequence[float]) -> float:
    clean = [float(v) for v in values if v is not None]
    return float(median(clean)) if clean else 0.0
