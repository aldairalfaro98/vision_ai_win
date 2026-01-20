# app/api/routes.py
from fastapi import APIRouter, File, HTTPException, UploadFile
from typing import List, Optional

from app.application.liveness_service import LivenessService
from app.config import settings
from app.domain.models import LivenessResponse

router = APIRouter()
service = LivenessService()


@router.post("/liveness/check", response_model=LivenessResponse)
async def liveness_check(file: Optional[UploadFile] = File(None)) -> LivenessResponse:
    image_bytes = await file.read() if file is not None else None
    return service.check(image_bytes)


@router.post("/liveness/check-video", response_model=LivenessResponse)
async def liveness_check_video(file: UploadFile = File(...)) -> LivenessResponse:
    video_bytes = await file.read()
    if not video_bytes:
        raise HTTPException(status_code=400, detail="Video vacío.")

    return service.check_video(
        video_bytes,
        max_frames=settings.max_frames,
        preproc_max_side_px=settings.preproc_max_side_px,
        anti_min_real_frames=settings.anti_min_real_frames,
        anti_min_confidence=settings.anti_min_confidence,
        blink_baseline_frames=settings.blink_baseline_frames,
        blink_close_ratio=settings.blink_close_ratio,
        blink_open_ratio=settings.blink_open_ratio,
        blink_min_closed_frames=settings.blink_min_closed_frames,
        blink_min_count=settings.blink_min_count,
        head_baseline_frames=settings.head_baseline_frames,
        head_yaw_delta_deg=settings.head_yaw_delta_deg,
        decision_mode=settings.decision_mode,
    )


@router.post("/liveness/check-frames", response_model=LivenessResponse)
async def liveness_check_frames(files: List[UploadFile] = File(...)) -> LivenessResponse:
    if not files:
        raise HTTPException(status_code=400, detail="Debe enviar al menos 1 frame.")

    if len(files) > settings.max_frames:
        raise HTTPException(
            status_code=413,
            detail=f"Demasiados frames: {len(files)}. Max permitido={settings.max_frames}.",
        )

    frames_bytes = [await f.read() for f in files]
    return service.check_frames(
        frames_bytes,
        preproc_max_side_px=settings.preproc_max_side_px,
        anti_min_real_frames=settings.anti_min_real_frames,
        anti_min_confidence=settings.anti_min_confidence,
        blink_baseline_frames=settings.blink_baseline_frames,
        blink_close_ratio=settings.blink_close_ratio,
        blink_open_ratio=settings.blink_open_ratio,
        blink_min_closed_frames=settings.blink_min_closed_frames,
        blink_min_count=settings.blink_min_count,
        head_baseline_frames=settings.head_baseline_frames,
        head_yaw_delta_deg=settings.head_yaw_delta_deg,
        decision_mode=settings.decision_mode,
    )
