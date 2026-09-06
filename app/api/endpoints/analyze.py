import io
from fastapi import APIRouter, UploadFile, File
from app.services.quality_service import QualityService
from app.services.document_service import DocumentService
from app.core.exceptions import UnsupportedFormatError

router = APIRouter()
quality_service = QualityService()
document_service = DocumentService()


@router.post("/", summary="Analyze image quality", description="Run document detection and quality analysis on uploaded image")
async def analyze_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise UnsupportedFormatError("Uploaded file is not an image")
    data = await file.read()
    doc_info = document_service.detect_document_from_bytes(data)
    quality = quality_service.analyze_bytes(data, doc_info)
    return {"document": doc_info, "quality": quality}

