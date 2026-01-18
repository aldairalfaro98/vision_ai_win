# app/api/routes.py
from fastapi import APIRouter, UploadFile, File
from typing import Optional

from app.application.liveness_service import LivenessService
from app.domain.models import LivenessResponse

router = APIRouter()
service = LivenessService()

@router.post("/liveness/check", response_model=LivenessResponse)
async def liveness_check(file: Optional[UploadFile] = File(None)) -> LivenessResponse:
    image_bytes = await file.read() if file is not None else None
    return service.check(image_bytes)
