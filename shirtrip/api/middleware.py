from __future__ import annotations

import logging
import uuid

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from shirtrip.pipeline.errors import (
    GarmentNotFoundError,
    InvalidInputError,
    PipelineError,
    StageError,
)

logger = logging.getLogger(__name__)


def add_middleware(app: FastAPI) -> None:
    """Add all middleware to the FastAPI app."""

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request_id = str(uuid.uuid4())[:8]
        request.state.request_id = request_id
        response = await call_next(request)
        response.headers["X-Request-ID"] = request_id
        return response

    @app.exception_handler(GarmentNotFoundError)
    async def garment_not_found_handler(request: Request, exc: GarmentNotFoundError):
        return JSONResponse(
            status_code=404,
            content={"error": "garment_not_found", "detail": str(exc), "stage": exc.stage_name},
        )

    @app.exception_handler(InvalidInputError)
    async def invalid_input_handler(request: Request, exc: InvalidInputError):
        return JSONResponse(
            status_code=422,
            content={"error": "invalid_input", "detail": str(exc), "stage": None},
        )

    @app.exception_handler(StageError)
    async def stage_error_handler(request: Request, exc: StageError):
        logger.error("Stage error in %s: %s", exc.stage_name, exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "stage_error", "detail": str(exc), "stage": exc.stage_name},
        )

    @app.exception_handler(PipelineError)
    async def pipeline_error_handler(request: Request, exc: PipelineError):
        logger.error("Pipeline error: %s", exc, exc_info=True)
        return JSONResponse(
            status_code=500,
            content={"error": "pipeline_error", "detail": str(exc), "stage": None},
        )
