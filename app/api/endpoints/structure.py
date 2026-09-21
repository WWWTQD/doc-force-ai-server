import os
import json
import time
import uuid
import logging
from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.schemas.structure import AnalyzeResponse, HealthResponse, ErrorResponse, ErrorDetail
from app.services.structure_engine import StructureEngine
from app.services.structure_parser import StructureParser

logger = logging.getLogger("doc-force-ai-server")

router = APIRouter()

SUPPORTED_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
SUPPORTED_PDF_EXTS = {".pdf"}
SUPPORTED_EXTS = SUPPORTED_IMAGE_EXTS | SUPPORTED_PDF_EXTS


@router.get("/health", response_model=HealthResponse)
def get_health():
    """Health check endpoint returning service name and model info."""
    engine = StructureEngine.get_instance()
    model_name = settings.AI_MODEL_NAME if engine.initialized else f"{settings.AI_MODEL_NAME} (loading)"
    return HealthResponse(
        status="ok",
        service=settings.AI_SERVICE_NAME,
        model=model_name,
    )


@router.post(
    "/analyze",
    response_model=AnalyzeResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
        500: {"model": ErrorResponse},
    },
)
async def analyze_document(file: UploadFile = File(...)):
    """
    Accepts an uploaded image or PDF document and runs PP-StructureV3.
    Returns structured PBL4 JSON containing layout, text OCR, tables, and page metadata.
    """
    start_time = time.time()
    req_id = uuid.uuid4().hex

    if not file or not file.filename:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(
                success=False,
                error=ErrorDetail(code="INVALID_FILE", message="No file was provided in the request."),
            ).model_dump(),
        )

    orig_filename = file.filename
    ext = os.path.splitext(orig_filename)[1].lower()

    if ext not in SUPPORTED_EXTS:
        return JSONResponse(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            content=ErrorResponse(
                success=False,
                error=ErrorDetail(
                    code="UNSUPPORTED_FORMAT",
                    message=f"File extension '{ext}' is not supported. Allowed formats: {', '.join(sorted(SUPPORTED_EXTS))}",
                ),
            ).model_dump(),
        )

    # Read content to check file size
    contents = await file.read()
    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    if len(contents) > max_bytes:
        return JSONResponse(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            content=ErrorResponse(
                success=False,
                error=ErrorDetail(
                    code="FILE_TOO_LARGE",
                    message=f"File size exceeds limit of {settings.MAX_FILE_SIZE_MB}MB.",
                ),
            ).model_dump(),
        )

    if len(contents) == 0:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content=ErrorResponse(
                success=False,
                error=ErrorDetail(code="EMPTY_FILE", message="Uploaded file is empty (0 bytes)."),
            ).model_dump(),
        )

    # Save uploaded file safely with UUID filename
    input_dir = os.path.join("data", "input")
    os.makedirs(input_dir, exist_ok=True)
    saved_path = os.path.join(input_dir, f"{req_id}{ext}")

    with open(saved_path, "wb") as f:
        f.write(contents)

    logger.info(f"[AI] request_id={req_id} filename={orig_filename} ext={ext} size_bytes={len(contents)}")

    try:
        engine = StructureEngine.get_instance()
        if not engine.initialized:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content=ErrorResponse(
                    success=False,
                    error=ErrorDetail(code="MODEL_NOT_READY", message="PP-StructureV3 model is initializing. Please retry shortly."),
                ).model_dump(),
            )

        # Run prediction
        if ext in SUPPORTED_PDF_EXTS:
            raw_res = engine.predict_pdf_path(saved_path, temp_dir=input_dir)
        else:
            raw_res = engine.predict_image_path(saved_path)

        elapsed_ms = (time.time() - start_time) * 1000.0

        # Parse output into PBL4 JSON schema
        parsed_response = StructureParser.parse(
            raw_results=raw_res,
            file_id=req_id,
            filename=orig_filename,
            processing_time_ms=elapsed_ms,
        )

        logger.info(f"[AI] request_id={req_id} success=True processing_time_ms={elapsed_ms:.2f}ms pages={len(parsed_response.pages)}")

        # Save outputs for debugging / audit
        output_dir = settings.OUTPUT_DIR
        raw_output_dir = os.path.join(output_dir, "raw")
        parsed_output_dir = os.path.join(output_dir, "parsed")
        os.makedirs(raw_output_dir, exist_ok=True)
        os.makedirs(parsed_output_dir, exist_ok=True)

        if settings.AI_SAVE_RAW_OUTPUT:
            try:
                with open(os.path.join(raw_output_dir, f"{req_id}.json"), "w", encoding="utf-8") as f:
                    json.dump(raw_res, f, ensure_ascii=False, indent=2, default=str)
            except Exception as ex:
                logger.warning(f"Failed saving raw output: {ex}")

        try:
            with open(os.path.join(parsed_output_dir, f"{req_id}.json"), "w", encoding="utf-8") as f:
                json.dump(parsed_response.model_dump(), f, ensure_ascii=False, indent=2)
        except Exception as ex:
            logger.warning(f"Failed saving parsed output: {ex}")

        return parsed_response

    except Exception as e:
        logger.error(f"[AI] request_id={req_id} error: {e}", exc_info=True)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                success=False,
                error=ErrorDetail(
                    code="AI_PROCESSING_ERROR",
                    message="Failed to process document with PP-StructureV3.",
                ),
            ).model_dump(),
        )
