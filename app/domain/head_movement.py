# app/domain/head_movement.py
from __future__ import annotations

from statistics import median
from typing import Sequence, Tuple

Angles = Tuple[float, float, float]  # yaw, pitch, roll


def detect_head_movement_from_angles(
    angles_series: Sequence[Angles],
    *,
    baseline_frames: int = 3,
    yaw_delta_deg: float = 15.0,
) -> bool:
    return yaw_range_from_angles(
        angles_series,
        baseline_frames=baseline_frames,
    ) >= float(yaw_delta_deg)


def yaw_range_from_angles(
    angles_series: Sequence[Angles],
    *,
    baseline_frames: int = 3,
) -> float:
    if not angles_series:
        return 0.0

    baseline_frames = int(baseline_frames)
    if baseline_frames <= 0:
        return 0.0

    if len(angles_series) <= baseline_frames:
        return 0.0

    yaw_values = [float(a[0]) for a in angles_series]
    baseline = float(median(yaw_values[:baseline_frames]))
    deltas = [abs(y - baseline) for y in yaw_values[baseline_frames:]]
    return float(max(deltas)) if deltas else 0.0
