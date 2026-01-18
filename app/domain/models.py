from pydantic import BaseModel, Field


class LivenessChecks(BaseModel):
    blink_detected: bool = False
    head_movement: bool = False


class LivenessResponse(BaseModel):
    liveness: bool
    confidence: float = Field(ge=0.0, le=1.0)
    checks: LivenessChecks
