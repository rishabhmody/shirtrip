from __future__ import annotations

import logging
import time
import uuid
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, Depends, UploadFile
from fastapi.responses import FileResponse

from shirtrip.api.dependencies import get_settings, validate_upload
from shirtrip.api.schemas import ExtractionResponse, HealthResponse, StageInfo
from shirtrip.config.settings import Settings
from shirtrip.models.model_registry import ModelRegistry
from shirtrip.pipeline.image_utils import resize_if_needed, to_pipeline_image
from shirtrip.pipeline.orchestrator import run_pipeline

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1")


@router.post("/extract", response_model=ExtractionResponse)
async def extract_graphic(
    file: UploadFile,
    settings: Settings = Depends(get_settings),
) -> ExtractionResponse:
    """Extract graphic design from a t-shirt photograph."""
    file = await validate_upload(file, settings)
    start = time.perf_counter()

    # Read and decode the image
    content = await file.read()
    img_array = np.frombuffer(content, dtype=np.uint8)
    bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if bgr is None:
        from shirtrip.pipeline.errors import InvalidInputError

        raise InvalidInputError("Could not decode image file")

    # Resize if needed
    bgr = resize_if_needed(bgr, settings.max_image_dimension)

    # Run pipeline
    pipeline_image = to_pipeline_image(bgr)
    result = run_pipeline(pipeline_image, settings)

    # Save output as RGBA PNG
    job_id = str(uuid.uuid4())[:8]
    settings.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = settings.output_dir / f"{job_id}.png"
    bgra = result.image.rgba
    cv2.imwrite(str(output_path), bgra)

    processing_time_ms = (time.perf_counter() - start) * 1000

    return ExtractionResponse(
        job_id=job_id,
        filename=file.filename or "unknown",
        stages_completed=[
            StageInfo(
                stage_name=m.stage_name,
                duration_ms=m.duration_ms,
                input_shape=list(m.input_shape),
                output_shape=list(m.output_shape),
            )
            for m in result.metadata
        ],
        processing_time_ms=processing_time_ms,
    )


@router.get("/output/{job_id}")
async def get_output(
    job_id: str,
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    """Download the extracted graphic PNG."""
    output_path = settings.output_dir / f"{job_id}.png"
    if not output_path.exists():
        from fastapi import HTTPException

        raise HTTPException(status_code=404, detail=f"Output not found: {job_id}")
    return FileResponse(
        path=str(output_path),
        media_type="image/png",
        filename=f"{job_id}.png",
    )


@router.get("/health", response_model=HealthResponse)
async def health(settings: Settings = Depends(get_settings)) -> HealthResponse:
    """Health check with model status."""
    import torch

    registry = ModelRegistry.get()
    return HealthResponse(
        status="ok",
        device=settings.device,
        gpu_available=torch.cuda.is_available(),
        loaded_models=registry.loaded_models,
    )
