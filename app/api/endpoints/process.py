import os
from fastapi import APIRouter, UploadFile, File
from app.services.document_service import DocumentService
from app.core.exceptions import UnsupportedFormatError

router = APIRouter()
document_service = DocumentService()


@router.post("/", summary="Process image", description="Apply document extraction and preprocessing and return saved processed file path")
async def process_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise UnsupportedFormatError("Uploaded file is not an image")
    data = await file.read()
    doc_info = document_service.detect_document_from_bytes(data)
    processed = document_service.extract_document_bytes(data, doc_info)
    out_dir = os.path.join("data", "output")
    os.makedirs(out_dir, exist_ok=True)
    fname = f"processed_{os.urandom(6).hex()}.jpg"
    path = os.path.join(out_dir, fname)
    with open(path, "wb") as fh:
        fh.write(processed)
    return {"processed_path": path, "document": doc_info}

