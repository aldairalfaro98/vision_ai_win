#app/tests/test_head_movement_detector.py
from app.domain.head_movement import detect_head_movement_from_angles

def test_head_movement_false_small_changes():
    angles = [(0, 0, 0), (1, 0.5, 0), (0.8, 0.2, 0), (2, 1, 0)]
    assert detect_head_movement_from_angles(angles, delta_deg=12.0) is False

def test_head_movement_true_big_change():
    angles = [(0, 0, 0), (1, 0.5, 0), (0.8, 0.2, 0), (20, 0, 0)]
    assert detect_head_movement_from_angles(angles, delta_deg=12.0) is True
