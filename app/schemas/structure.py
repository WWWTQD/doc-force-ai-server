from pydantic import BaseModel, Field
from typing import List, Optional, Any, Dict


class DocumentElement(BaseModel):
    id: str = Field(..., description="Unique element ID within document")
    type: str = Field(..., description="Normalized element type: text, title, paragraph, table, image, header, footer, formula, chart, seal, or unknown")
    raw_type: Optional[str] = Field(None, description="Original raw category label from PaddleOCR PP-StructureV3")
    text: Optional[str] = Field("", description="Extracted textual content")
    bbox: List[float] = Field(default_factory=list, description="Bounding box [xmin, ymin, xmax, ymax]")
    confidence: float = Field(0.0, description="Detection / OCR confidence score (0.0 to 1.0)")
    html: Optional[str] = Field(None, description="HTML representation for table elements")
    table_data: Optional[Dict[str, Any]] = Field(None, description="Structured table metadata or cell matrix if applicable")
    attributes: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Additional custom attributes for fine-tuning")


class DocumentPage(BaseModel):
    page_number: int = Field(1, description="1-indexed page number")
    width: int = Field(0, description="Page width in pixels")
    height: int = Field(0, description="Page height in pixels")
    elements: List[DocumentElement] = Field(default_factory=list, description="Structured document elements on this page")


class DocumentContent(BaseModel):
    file_id: str = Field(..., description="Unique internal UUID for file")
    filename: str = Field(..., description="Original filename")
    pages: List[DocumentPage] = Field(default_factory=list, description="Array of pages in the document")


class AnalyzeResponse(BaseModel):
    success: bool = True
    file_id: str
    filename: str
    pages: List[DocumentPage] = Field(default_factory=list, description="Top-level pages shortcut")
    document: DocumentContent
    processing_time_ms: float
    model: str = "PP-StructureV3"


class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "pbl4-ai-service"
    model: str = "PP-StructureV3"


class ErrorDetail(BaseModel):
    code: str
    message: str


class ErrorResponse(BaseModel):
    success: bool = False
    error: ErrorDetail
