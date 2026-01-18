# app/tests/test_decision.py
from app.domain.decision import decide_liveness

def test_decide_liveness_truth_table():
    assert decide_liveness(anti_spoof_passed=False, blink_detected=False, head_movement=False) is False
    assert decide_liveness(anti_spoof_passed=False, blink_detected=True, head_movement=False) is False
    assert decide_liveness(anti_spoof_passed=True, blink_detected=False, head_movement=False) is False
    assert decide_liveness(anti_spoof_passed=True, blink_detected=True, head_movement=False) is True
    assert decide_liveness(anti_spoof_passed=True, blink_detected=False, head_movement=True) is True

