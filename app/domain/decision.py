# app/domain/decision.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Protocol


@dataclass(frozen=True)
class LivenessSignals:
    anti_spoof_passed: bool
    blink_count: int
    head_movement: bool


class DecisionStrategy(Protocol):
    def decide(self, signals: LivenessSignals) -> bool: ...


class BalancedDecision:
    def __init__(self, *, min_blinks: int = 1) -> None:
        self._min_blinks = int(min_blinks)

    def decide(self, signals: LivenessSignals) -> bool:
        if not signals.anti_spoof_passed:
            return False
        blink_ok = signals.blink_count >= self._min_blinks
        return bool(blink_ok or signals.head_movement)


class StrictDecision:
    def __init__(self, *, min_blinks: int = 2) -> None:
        self._min_blinks = int(min_blinks)

    def decide(self, signals: LivenessSignals) -> bool:
        if not signals.anti_spoof_passed:
            return False
        blink_ok = signals.blink_count >= self._min_blinks
        return bool(blink_ok and signals.head_movement)


def build_decision_strategy(*, mode: str, min_blinks: int) -> DecisionStrategy:
    key = (mode or "").strip().lower()
    strategies: Dict[str, DecisionStrategy] = {
        "balanced": BalancedDecision(min_blinks=min_blinks),
        "strict": StrictDecision(min_blinks=min_blinks),
    }
    return strategies.get(key, strategies["strict"])


def decide_liveness(*, strategy: DecisionStrategy, signals: LivenessSignals) -> bool:
    return bool(strategy.decide(signals))
