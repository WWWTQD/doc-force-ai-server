from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field, field_validator
from typing import List, Union, Any


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    APP_NAME: str = "doc-force-ai-server"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = True
    MAX_FILE_SIZE_MB: int = 20
    BLUR_THRESHOLD: float = 100.0
    LOW_BRIGHTNESS_THRESHOLD: int = 70
    HIGH_BRIGHTNESS_THRESHOLD: int = 200
    OCR_LANGUAGE: str = "eng"
    OUTPUT_DIR: str = "data/output"
    CORS_ORIGINS: Any = Field(default=["http://localhost:3000"])

    # AI Device & General Structure Engine
    AI_DEVICE: str = "cpu"
    AI_SERVICE_NAME: str = "doc-force-ai-server"
    AI_MODEL_NAME: str = "PP-StructureV3"
    AI_USE_DOC_ORIENTATION: bool = False
    AI_USE_DOC_UNWARPING: bool = False
    AI_USE_TEXTLINE_ORIENTATION: bool = False
    AI_USE_SEAL_RECOGNITION: bool = False
    AI_USE_TABLE_RECOGNITION: bool = True
    AI_USE_FORMULA_RECOGNITION: bool = False
    AI_SAVE_RAW_OUTPUT: bool = True

    # YOLO Document Detection
    YOLO_MODEL_PATH: str = "worker/Model/best.pt"
    YOLO_CONF_THRESHOLD: float = 0.25

    # Document Restoration U-Net
    ENABLE_IMAGE_RESTORATION: bool = True
    RESTORATION_MODEL_PATH: str = "models/restoration_best.pth"
    RESTORATION_TILE_SIZE: int = 512
    RESTORATION_OVERLAP: int = 64
    RESTORATION_PRESERVE_COLOR: bool = True

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Any) -> List[str]:
        if isinstance(v, str):
            if not v.startswith("["):
                return [i.strip() for i in v.split(",") if i.strip()]
            try:
                import json
                parsed = json.loads(v)
                if isinstance(parsed, list):
                    return parsed
            except Exception:
                pass
            return [v]
        elif isinstance(v, list):
            return [str(i) for i in v]
        return ["http://localhost:3000"]


settings = Settings()
