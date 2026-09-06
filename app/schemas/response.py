from pydantic import BaseModel


class HealthResponse(BaseModel):
    status: str
    service: str
    version: str


class ErrorResponse(BaseModel):
    success: bool = False
    error: dict

