import logging
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core.config import settings
from app.core.logging import setup_logging
from app.api.router import api_router
from app.api.endpoints import structure
from app.services.structure_engine import StructureEngine
from app.core.exceptions import AppException

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: Load PP-StructureV3 engine
    logger.info("[App] Starting up: initializing PP-StructureV3 engine...")
    try:
        engine = StructureEngine.get_instance()
        engine.initialize(
            device=settings.AI_DEVICE,
            use_doc_orientation_classify=settings.AI_USE_DOC_ORIENTATION,
            use_doc_unwarping=settings.AI_USE_DOC_UNWARPING,
            use_textline_orientation=settings.AI_USE_TEXTLINE_ORIENTATION,
            use_seal_recognition=settings.AI_USE_SEAL_RECOGNITION,
            use_table_recognition=settings.AI_USE_TABLE_RECOGNITION,
            use_formula_recognition=settings.AI_USE_FORMULA_RECOGNITION,
        )
    except Exception as e:
        logger.error(f"[App] Failed to load PP-StructureV3 on startup: {e}")

    # Startup: Preload DocumentRestorationUNet
    try:
        from app.services.restoration_service import RestorationService
        restoration = RestorationService.get_instance()
        restoration.initialize(device_name=settings.AI_DEVICE)
    except Exception as e:
        logger.warning(f"[App] Failed to preload RestorationService: {e}")

    # Startup: Preload YOLOv8 document detector
    try:
        from app.services.yolo_service import YOLOService
        yolo = YOLOService()
        yolo.load_model()
    except Exception as e:
        logger.warning(f"[App] Failed to preload YOLOService: {e}")

    yield

    # Shutdown: Cleanup if needed
    logger.info("[App] Shutting down AI service...")


app = FastAPI(title=settings.APP_NAME, version=settings.APP_VERSION, lifespan=lifespan)

# CORS Middleware
origins = settings.CORS_ORIGINS
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Top-level endpoints: GET /health, POST /analyze
app.include_router(structure.router)

# Versioned API routes: /api/v1/...
app.include_router(api_router, prefix="/api/v1")


@app.exception_handler(AppException)
async def app_exception_handler(request: Request, exc: AppException):
    logger.warning("AppException: %s %s", exc.code, exc.message)
    content = {
        "detail": {"code": exc.code, "message": exc.message},
        "success": False,
        "error": {"code": exc.code, "message": exc.message},
    }
    return JSONResponse(status_code=exc.status_code, content=content)


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception("Unhandled exception: %s", exc)
    return JSONResponse(
        status_code=500,
        content={"success": False, "error": {"code": "INTERNAL_ERROR", "message": "Internal server error"}},
    )
