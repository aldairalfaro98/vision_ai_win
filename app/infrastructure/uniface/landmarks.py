# app/infrastructure/uniface/landmarks.py
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import cv2

from uniface import Landmark106


@dataclass(frozen=True)
class LandmarkResult:
    landmarks: np.ndarray  # shape (106, 2)
    ear_left: float
    ear_right: float
    rvec: np.ndarray
    tvec: np.ndarray


class UniFaceLandmarks:
    def __init__(self) -> None:
        self.landmarker = Landmark106()

    @staticmethod
    def _eye_aspect_ratio(eye_landmarks: np.ndarray) -> float:
        # contentReference[oaicite:3]{index=3}
        v1 = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
        v2 = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
        h = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
        return float((v1 + v2) / (2.0 * h)) if h != 0 else 0.0

    @staticmethod
    def _estimate_head_pose(landmarks: np.ndarray, image_shape) -> Tuple[np.ndarray, np.ndarray]:
        # (solvePnP) :contentReference[oaicite:4]{index=4}
        model_points = np.array([
            (0.0, 0.0, 0.0),          # Nose tip
            (0.0, -330.0, -65.0),     # Chin
            (-225.0, 170.0, -135.0),  # Left eye corner
            (225.0, 170.0, -135.0),   # Right eye corner
            (-150.0, -150.0, -125.0), # Left mouth corner
            (150.0, -150.0, -125.0),  # Right mouth corner
        ], dtype=np.float64)

        image_points = np.array([
            landmarks[51],  # Nose tip
            landmarks[16],  # Chin
            landmarks[63],  # Left eye corner
            landmarks[76],  # Right eye corner
            landmarks[87],  # Left mouth corner
            landmarks[93],  # Right mouth corner
        ], dtype=np.float64)

        h, w = image_shape[:2]
        focal_length = w
        center = (w / 2, h / 2)
        camera_matrix = np.array([
            [focal_length, 0, center[0]],
            [0, focal_length, center[1]],
            [0, 0, 1],
        ], dtype=np.float64)

        dist_coeffs = np.zeros((4, 1))
        success, rvec, tvec = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
        if not success:
            raise ValueError("solvePnP failed")
        return rvec, tvec

    def analyze(self, image_bgr: np.ndarray, face_bbox) -> Optional[LandmarkResult]:
        landmarks = self.landmarker.get_landmarks(image_bgr, face_bbox)  # (106,2) :contentReference[oaicite:5]{index=5}
        if landmarks is None or len(landmarks) == 0:
            return None

        # Ojos (rangos de la doc) :contentReference[oaicite:6]{index=6}
        left_eye = landmarks[63:72]
        right_eye = landmarks[76:84]

        ear_left = self._eye_aspect_ratio(left_eye)
        ear_right = self._eye_aspect_ratio(right_eye)

        rvec, tvec = self._estimate_head_pose(landmarks, image_bgr.shape)
        return LandmarkResult(landmarks=landmarks, ear_left=ear_left, ear_right=ear_right, rvec=rvec, tvec=tvec)
