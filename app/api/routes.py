from fastapi import APIRouter
from app.application.liveness_service import LivenessService
from app.domain.models import LivenessResponse

router = APIRouter()
service = LivenessService()


@router.post("/liveness/check", response_model=LivenessResponse)
def liveness_check() -> LivenessResponse:
    return service.check()
