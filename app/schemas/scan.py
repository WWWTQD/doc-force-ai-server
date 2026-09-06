from pydantic import BaseModel
from typing import List, Optional
from app.schemas.defect import DefectModel


class QualityModel(BaseModel):
    score: float
    level: str


class DocumentModel(BaseModel):
    detected: bool
    confidence: float
    rotation: int
    skew_angle: float


class OCRModel(BaseModel):
    text: str
    confidence: float
    language: Optional[str] = None


class FilesModel(BaseModel):
    original: str
    processed: Optional[str]


class ScanResult(BaseModel):
    scan_id: str
    status: str
    quality: QualityModel
    document: DocumentModel
    defects: List[DefectModel]
    ocr: OCRModel
    files: FilesModel

