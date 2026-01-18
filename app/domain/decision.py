# app/domain/decision.py
def decide_liveness(*, anti_spoof_passed: bool, blink_detected: bool, head_movement: bool) -> bool:
    """Regla híbrida: PAD pasivo + señal activa (blink o movimiento)."""
    return bool(anti_spoof_passed and (blink_detected or head_movement))

