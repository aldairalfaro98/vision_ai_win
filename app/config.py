# app/config.py
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="LIVENESS_", extra="ignore")

    # --- Batch/video ingestion ---
    max_frames: int = Field(default=90, ge=1, le=300)

    # Preprocesamiento “congelado” en backend
    preproc_max_side_px: int = Field(default=640, ge=160, le=2048)

    # --- Anti-spoofing (PAD) ---
    anti_min_real_frames: int = Field(default=10, ge=1, le=300)
    anti_min_confidence: float = Field(default=0.70, ge=0.0, le=1.0)

    # --- Blink ---
    blink_baseline_frames: int = Field(default=5, ge=1, le=30)
    blink_close_ratio: float = Field(default=0.80, ge=0.1, le=1.0)
    blink_open_ratio: float = Field(default=0.90, ge=0.1, le=2.0)
    blink_min_closed_frames: int = Field(default=1, ge=1, le=30)
    blink_min_count: int = Field(default=2, ge=1, le=10)

    # --- Head movement ---
    head_baseline_frames: int = Field(default=3, ge=1, le=30)
    head_yaw_delta_deg: float = Field(default=15.0, ge=1.0, le=60.0)

    # --- Decision ---
    decision_mode: str = Field(default="balanced")  # "strict" | "balanced"


settings = Settings()
