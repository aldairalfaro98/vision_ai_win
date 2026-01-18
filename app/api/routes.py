from fastapi import APIRouter, UploadFile, File
from typing import Optional
from app.application.liveness_service import LivenessService
from app.domain.models import LivenessResponse

router = APIRouter()
service = LivenessService()


@router.post("/liveness/check", response_model=LivenessResponse)
async def liveness_check(file: Optional[UploadFile] = File(None)) -> LivenessResponse:
    return service.check()
