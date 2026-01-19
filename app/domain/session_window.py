# app/domain/session_window.py
from dataclasses import dataclass, field
from statistics import median
from typing import List, Optional, Tuple

from app.domain.blink import detect_blink_from_ear_series
from app.domain.decision import decide_liveness
from app.domain.head_movement import detect_head_movement_from_angles

Angles = Tuple[float, float, float]


@dataclass
class SessionWindow:
    window_size: int = 8
    min_real: int = 3
    threshold: float = 0.6

    _is_real: List[bool] = field(default_factory=list)
    _conf: List[float] = field(default_factory=list)
    _ears: List[float] = field(default_factory=list)
    _angles: List[Angles] = field(default_factory=list)

    def add(self, *, is_real: bool, confidence: float,
            ear_avg: Optional[float], angles: Optional[Angles]) -> None:
        self._push(self._is_real, bool(is_real))
        self._push(self._conf, float(confidence))
        if ear_avg is not None:
            self._push(self._ears, float(ear_avg))
        if angles is not None:
            self._push(self._angles, angles)

    def compute(self) -> tuple[bool, float, bool, bool]:
        conf_med = self._median_conf()
        anti_ok = self._anti_spoof_passed(conf_med)
        blink = detect_blink_from_ear_series(self._ears)
        head = detect_head_movement_from_angles(self._angles)
        live = decide_liveness(anti_spoof_passed=anti_ok,
                              blink_detected=blink,
                              head_movement=head)
        return live, conf_med, blink, head

    def _push(self, arr: list, value) -> None:
        arr.append(value)
        if len(arr) > self.window_size:
            del arr[0]

    def _median_conf(self) -> float:
        return float(median(self._conf)) if self._conf else 0.0

    def _anti_spoof_passed(self, conf_med: float) -> bool:
        if not self._conf:
            return False
        real_count = sum(1 for x in self._is_real if x)
        return bool(real_count >= self.min_real and conf_med >= self.threshold)
