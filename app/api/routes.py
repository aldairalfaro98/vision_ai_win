# app/api/routes.py
from fastapi import APIRouter, UploadFile, File
from typing import Optional, List

from app.application.liveness_service import LivenessService
from app.domain.models import LivenessResponse

router = APIRouter()
service = LivenessService()

# Endpoint para verificación de liveness con una imagen
@router.post("/liveness/check", response_model=LivenessResponse)
async def liveness_check(file: Optional[UploadFile] = File(None)) -> LivenessResponse:
    image_bytes = await file.read() if file is not None else None
    return service.check(image_bytes)

# Endpoint para verificación de liveness con múltiples imágenes
@router.post("/liveness/check-frames", response_model=LivenessResponse)
async def liveness_check_frames(files: List[UploadFile] = File(...)) -> LivenessResponse:
    frames_bytes = [await f.read() for f in files]
    return service.check_frames(frames_bytes)

