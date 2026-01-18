#app/domain/blink.py
from statistics import median
from typing import Sequence

def detect_blink_from_ear_series(
    ears: Sequence[float],
    *,
    baseline_frames: int = 3,
    close_ratio: float = 0.70,   # cerrado si EAR < baseline * 0.70
    open_ratio: float = 0.90,    # abierto si EAR > baseline * 0.90
    min_closed_frames: int = 1,  # para no exigir demasiado
) -> bool:
    if len(ears) < baseline_frames + 2:
        return False

    base = median(ears[:baseline_frames])
    if base <= 0:
        return False

    close_th = base * close_ratio
    open_th = base * open_ratio

    state = "OPEN"
    closed_run = 0

    for ear in ears[baseline_frames:]:
        if state == "OPEN":
            if ear < close_th:
                state = "CLOSED"
                closed_run = 1
        elif state == "CLOSED":
            if ear < close_th:
                closed_run += 1
            elif ear > open_th:
                if closed_run >= min_closed_frames:
                    return True
                state = "OPEN"
                closed_run = 0

    return False
