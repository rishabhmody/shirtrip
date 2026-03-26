from __future__ import annotations

import logging
from functools import lru_cache
from pathlib import Path

from fastapi import HTTPException, UploadFile

from shirtrip.config.settings import Settings

logger = logging.getLogger(__name__)


@lru_cache
def get_settings() -> Settings:
    """Return cached application settings."""
    return Settings()


async def validate_upload(file: UploadFile, settings: Settings | None = None) -> UploadFile:
    """Validate uploaded file type and size."""
    if settings is None:
        settings = get_settings()

    # Check filename
    if not file.filename:
        raise HTTPException(status_code=422, detail="No filename provided")

    ext = Path(file.filename).suffix.lower()
    if ext not in settings.allowed_extensions:
        raise HTTPException(
            status_code=422,
            detail=f"Unsupported file type '{ext}'. Allowed: {sorted(settings.allowed_extensions)}",
        )

    # Check file size by reading content
    content = await file.read()
    max_bytes = settings.max_upload_mb * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({len(content)} bytes). Maximum: {max_bytes} bytes",
        )

    # Reset file position for downstream use
    await file.seek(0)
    return file
