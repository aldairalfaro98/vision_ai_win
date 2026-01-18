#app/domain/head_movement.py
from statistics import median
from typing import Sequence, Tuple

Angles = Tuple[float, float, float]  # yaw, pitch, roll (en grados)

def _baseline(angles: Sequence[Angles], n: int) -> Angles:
    ys = [a[0] for a in angles[:n]]
    ps = [a[1] for a in angles[:n]]
    rs = [a[2] for a in angles[:n]]
    return (float(median(ys)), float(median(ps)), float(median(rs)))

def _max_delta(angles: Sequence[Angles], base: Angles) -> float:
    by, bp, br = base
    deltas = []
    for y, p, r in angles:
        deltas.append(max(abs(y - by), abs(p - bp), abs(r - br)))
    return float(max(deltas)) if deltas else 0.0

def detect_head_movement_from_angles(
    angles: Sequence[Angles],
    *,
    baseline_frames: int = 3,
    delta_deg: float = 12.0,
) -> bool:
    if len(angles) < baseline_frames + 1:
        return False
    base = _baseline(angles, baseline_frames)
    return _max_delta(angles[baseline_frames:], base) >= float(delta_deg)
