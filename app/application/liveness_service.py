# app/application/liveness_service.py
from typing import Optional

from app.domain.models import LivenessResponse, LivenessChecks
from app.infrastructure.uniface.anti_spoof import (
    UniFaceAntiSpoofPredictor,
    decode_image_bytes_to_bgr,
)

_predictor = UniFaceAntiSpoofPredictor()


class LivenessService:
    def check(self, image_bytes: Optional[bytes]) -> LivenessResponse:
        if image_bytes is None:
            # Mantiene contrato si pegan sin archivo (modo demo)
            return LivenessResponse(
                liveness=False,
                confidence=0.0,
                checks=LivenessChecks(blink_detected=False, head_movement=False),
            )

        image_bgr = decode_image_bytes_to_bgr(image_bytes)

        anti = _predictor.predict_from_bgr(image_bgr)
        if anti is None:
            return LivenessResponse(
                liveness=False,
                confidence=0.0,
                checks=LivenessChecks(blink_detected=False, head_movement=False),
            )

        return LivenessResponse(
            liveness=anti.is_real,
            confidence=anti.confidence,
            checks=LivenessChecks(blink_detected=False, head_movement=False),
        )
