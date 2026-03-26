from __future__ import annotations

from pydantic import BaseModel


class StageInfo(BaseModel):
    stage_name: str
    duration_ms: float
    input_shape: list[int]
    output_shape: list[int]


class ExtractionResponse(BaseModel):
    job_id: str
    filename: str
    stages_completed: list[StageInfo]
    processing_time_ms: float


class ErrorResponse(BaseModel):
    error: str
    detail: str
    stage: str | None = None


class HealthResponse(BaseModel):
    status: str
    device: str
    gpu_available: bool
    loaded_models: list[str]
