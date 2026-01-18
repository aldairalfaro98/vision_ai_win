from app.domain.models import LivenessResponse, LivenessChecks


class LivenessService:
    """Use case (por ahora MOCK). Luego se conecta a UniFace."""
    def check(self) -> LivenessResponse:
        # MOCK estable: sirve para fijar contrato, tests y cliente.
        return LivenessResponse(
            liveness=False,
            confidence=0.0,
            checks=LivenessChecks(
                blink_detected=False,
                head_movement=False,
            ),
        )
