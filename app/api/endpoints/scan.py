import uuid
from fastapi import APIRouter, UploadFile, File
from app.core.exceptions import AppException
from fastapi.responses import JSONResponse
from app.schemas.scan import ScanResult
from app.services.scan_service import ScanService
from app.core.logging import setup_logging
from app.core.exceptions import ProcessingError, UnsupportedFormatError
import os
import json

router = APIRouter()
setup_logging()


@router.post("/", response_model=ScanResult, summary="Upload and scan document", description="Upload an image to run the scan pipeline")
async def upload_scan(file: UploadFile = File(...)):
    # Basic content-type check
    if not file.content_type.startswith("image/"):
        # let ImageService also validate; return structured error
        raise UnsupportedFormatError("Uploaded file is not an image")

    scan_id = str(uuid.uuid4())
    service = ScanService()
    # let exceptions bubble to global exception handler
    result = await service.process_scan(scan_id=scan_id, upload_file=file)
    return result



@router.get("/{scan_id}", summary="Get scan result", description="Retrieve stored scan JSON result by scan_id")
async def get_scan(scan_id: str):
    path = os.path.join("data", "output", f"{scan_id}.json")
    if not os.path.exists(path):
        raise AppException("NOT_FOUND", "Scan result not found", status_code=404)
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data


