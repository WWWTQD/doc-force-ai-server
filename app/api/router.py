from fastapi import APIRouter
from app.api.endpoints import health, scan, analyze, ocr, process, defects

api_router = APIRouter()

api_router.include_router(health.router, prefix="/health", tags=["health"])
api_router.include_router(scan.router, prefix="/scan", tags=["scan"])
api_router.include_router(analyze.router, prefix="/analyze", tags=["analyze"])
api_router.include_router(ocr.router, prefix="/ocr", tags=["ocr"])
api_router.include_router(process.router, prefix="/process", tags=["process"])
api_router.include_router(defects.router, prefix="/defects", tags=["defects"])

