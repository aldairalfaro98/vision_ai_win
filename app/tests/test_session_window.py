# app/tests/test_session_window.py
from app.domain.session_window import SessionWindow

def test_session_window_blink_affects_decision():
    win = SessionWindow(window_size=8, min_real=3, threshold=0.6)

    for _ in range(3):
        win.add(is_real=True, confidence=0.99, ear_avg=0.30, angles=(0.0, 0.0, 0.0))

    win.add(is_real=True, confidence=0.99, ear_avg=0.12, angles=(0.0, 0.0, 0.0))
    win.add(is_real=True, confidence=0.99, ear_avg=0.30, angles=(0.0, 0.0, 0.0))

    live, _, blink, _ = win.compute()
    assert blink is True
    assert live is True
