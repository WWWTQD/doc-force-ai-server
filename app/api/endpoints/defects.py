from fastapi import APIRouter, UploadFile, File
from app.services.defect_service import DefectService
from app.services.document_service import DocumentService
from app.services.quality_service import QualityService
from app.core.exceptions import UnsupportedFormatError

router = APIRouter()
defect_service = DefectService()
document_service = DocumentService()
quality_service = QualityService()


@router.post("/", summary="Detect defects", description="Return list of defects for uploaded image")
async def detect_defects(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise UnsupportedFormatError("Uploaded file is not an image")
    data = await file.read()
    doc_info = document_service.detect_document_from_bytes(data)
    quality = quality_service.analyze_bytes(data, doc_info)
    defects = defect_service.detect_defects(data, doc_info, quality)
    return {"defects": defects}

