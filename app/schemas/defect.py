from pydantic import BaseModel
from typing import Optional


class DefectModel(BaseModel):
    code: str
    name: str
    severity: str
    confidence: float
    message: str
    recommendation: Optional[str]

