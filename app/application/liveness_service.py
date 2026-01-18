from typing import Optional
from app.domain.models import LivenessResponse, LivenessChecks

class LivenessService:
    def check(self, image_bytes: Optional[bytes]) -> LivenessResponse:
        if image_bytes is None:
            # Modo “sin archivo”: mantiene contrato.
            return LivenessResponse(
                liveness=False,
                confidence=0.0,
                checks=LivenessChecks(blink_detected=False, head_movement=False),
            )
