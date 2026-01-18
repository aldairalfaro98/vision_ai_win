#app/tests/test_blink_detector.py
from app.domain.blink import detect_blink_from_ear_series

def test_detect_blink_true():
    ears = [0.30, 0.29, 0.30, 0.12, 0.11, 0.28, 0.30]
    assert detect_blink_from_ear_series(ears) is True

def test_detect_blink_false():
    ears = [0.30, 0.29, 0.28, 0.27, 0.29, 0.30]
    assert detect_blink_from_ear_series(ears) is False
