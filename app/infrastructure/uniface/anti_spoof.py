# app/infrastructure/uniface/anti_spoof.py
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np

from uniface import RetinaFace
from uniface.spoofing import MiniFASNet


@dataclass(frozen=True)
class AntiSpoofResult:
    is_real: bool
    confidence: float
    bbox: Tuple[float, float, float, float]  # <- nuevo


class UniFaceAntiSpoofPredictor:
    def __init__(self) -> None:
        self.detector = RetinaFace()
        self.spoofer = MiniFASNet()

    def predict_from_bgr(self, image_bgr: np.ndarray) -> Optional[AntiSpoofResult]:
        faces = self.detector.detect(image_bgr)
        if not faces:
            return None

        face = faces[0]
        result = self.spoofer.predict(image_bgr, face.bbox)

        return AntiSpoofResult(
            is_real=bool(result.is_real),
            confidence=float(result.confidence),
            bbox=tuple(face.bbox),  # <- nuevo
        )


def decode_image_bytes_to_bgr(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image bytes (cv2.imdecode failed).")
    return img
