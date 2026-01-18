# app/infrastructure/uniface/anti_spoof.py
from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

from uniface import RetinaFace
from uniface.spoofing import MiniFASNet


@dataclass(frozen=True)
class AntiSpoofResult:
    is_real: bool
    confidence: float


class UniFaceAntiSpoofPredictor:
    def __init__(self) -> None:
        # Nota UniFace: modelos pueden auto-descargarse en primer uso :contentReference[oaicite:4]{index=4}
        self.detector = RetinaFace()
        self.spoofer = MiniFASNet()  # V2 por defecto (recomendado en docs) :contentReference[oaicite:5]{index=5}

    def predict_from_bgr(self, image_bgr: np.ndarray) -> Optional[AntiSpoofResult]:
        faces = self.detector.detect(image_bgr)
        if not faces:
            return None

        face = faces[0]  # MVP: primera cara
        result = self.spoofer.predict(image_bgr, face.bbox)  # :contentReference[oaicite:6]{index=6}

        return AntiSpoofResult(
            is_real=bool(result.is_real),
            confidence=float(result.confidence),
        )


def decode_image_bytes_to_bgr(image_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Invalid image bytes (cv2.imdecode failed).")
    return img
