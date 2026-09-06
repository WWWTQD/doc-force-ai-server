from fastapi import APIRouter
from app.core.config import settings
from app.schemas.response import HealthResponse

router = APIRouter()


@router.get("/", response_model=HealthResponse, summary="Health check", description="Returns service health")
async def health_check():
    return HealthResponse(status="ok", service=settings.APP_NAME, version=settings.APP_VERSION)

