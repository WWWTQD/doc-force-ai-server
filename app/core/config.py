from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import Field
from typing import List


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
    CORS_ORIGINS: List[str] = Field(default=["http://localhost:3000", "http://localhost:8080"])

    # AI Service Config
    AI_SERVICE_NAME: str = "pbl4-ai-service"
    AI_MODEL_NAME: str = "PP-StructureV3"
    AI_DEVICE: str = "cpu"
    AI_USE_DOC_ORIENTATION: bool = False
    AI_USE_DOC_UNWARPING: bool = False
    AI_USE_TEXTLINE_ORIENTATION: bool = False
    AI_USE_SEAL_RECOGNITION: bool = False
    AI_USE_TABLE_RECOGNITION: bool = True
    AI_USE_FORMULA_RECOGNITION: bool = False
    AI_SAVE_RAW_OUTPUT: bool = True


settings = Settings()


