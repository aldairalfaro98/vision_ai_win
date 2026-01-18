#app/infrastructure/uniface/pose.py
import numpy as np
import cv2
import math
from typing import Tuple

def rvec_to_euler_degrees(rvec: np.ndarray) -> Tuple[float, float, float]:
    rmat, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(rmat[0, 0] * rmat[0, 0] + rmat[1, 0] * rmat[1, 0])
    singular = sy < 1e-6

    if not singular:
        x = math.atan2(rmat[2, 1], rmat[2, 2])
        y = math.atan2(-rmat[2, 0], sy)
        z = math.atan2(rmat[1, 0], rmat[0, 0])
    else:
        x = math.atan2(-rmat[1, 2], rmat[1, 1])
        y = math.atan2(-rmat[2, 0], sy)
        z = 0.0

    # (yaw, pitch, roll) en grados (ajuste consistente)
    yaw = math.degrees(y)
    pitch = math.degrees(x)
    roll = math.degrees(z)
    return (float(yaw), float(pitch), float(roll))
