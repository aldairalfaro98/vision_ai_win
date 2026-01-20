# app/domain/models.py
from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field

AntiSpoofLabel = Literal["REAL", "FAKE", "NO_FACE"]


class AntiSpoofCheck(BaseModel):
    label: AntiSpoofLabel = "NO_FACE"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    passed: bool = False


class LivenessChecks(BaseModel):
    # Backward compatible
    blink_detected: bool = False
    head_movement: bool = False

    # New (para demo + depuración)
    blink_count: int = Field(default=0, ge=0, le=50)
    anti_spoof: AntiSpoofCheck = AntiSpoofCheck()


class LivenessResponse(BaseModel):
    liveness: bool
    confidence: float = Field(ge=0.0, le=1.0)
    checks: LivenessChecks
    message: Optional[str] = None
