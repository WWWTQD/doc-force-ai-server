from fastapi import APIRouter, UploadFile, File, Form
from app.services.ocr_service import OCRService
from app.core.exceptions import UnsupportedFormatError

router = APIRouter()
ocr_service = OCRService()


@router.post("/", summary="Run OCR on image", description="Extract text from an uploaded image using OCR provider")
async def run_ocr(file: UploadFile = File(...), lang: str = Form(None)):
    if not file.content_type.startswith("image/"):
        raise UnsupportedFormatError("Uploaded file is not an image")
    data = await file.read()
    res = ocr_service.extract_text_from_bytes(data, lang=lang)
    return res

