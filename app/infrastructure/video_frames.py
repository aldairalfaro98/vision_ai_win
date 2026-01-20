# app/infrastructure/video_frames.py
from __future__ import annotations

import os
import tempfile
from typing import List

import cv2
import numpy as np


def extract_evenly_spaced_frames_bgr(video_bytes: bytes, *, max_frames: int) -> List[np.ndarray]:
    if not video_bytes:
        return []

    max_frames = int(max_frames)
    if max_frames <= 0:
        return []

    fd, path = tempfile.mkstemp(suffix=".avi")
    os.close(fd)

    try:
        with open(path, "wb") as f:
            f.write(video_bytes)

        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            return []

        try:
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            indices = _build_indices(total_frames=total, max_frames=max_frames)

            frames: List[np.ndarray] = []
            want = set(indices)
            i = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if (not want) or (i in want):
                    if isinstance(frame, np.ndarray):
                        frames.append(frame)
                    if len(frames) >= max_frames:
                        break
                i += 1

            return frames
        finally:
            cap.release()
    finally:
        try:
            os.remove(path)
        except OSError:
            pass

def _build_indices(*, total_frames: int, max_frames: int) -> List[int]:
    if total_frames <= 0:
        return list(range(max_frames))

    if total_frames <= max_frames:
        return list(range(total_frames))

    lin = np.linspace(0, total_frames - 1, num=max_frames)
    indices = [int(x) for x in lin]

    out: List[int] = []
    prev = -1
    for idx in indices:
        if idx <= prev:
            continue
        out.append(idx)
        prev = idx
    return out
