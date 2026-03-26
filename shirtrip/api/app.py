from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from shirtrip.api.middleware import add_middleware
from shirtrip.api.routes import router
from shirtrip.config.settings import Settings
from shirtrip.models.model_registry import ModelRegistry

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage app lifecycle — cleanup models on shutdown."""
    logger.info("ShirtRip starting up")
    yield
    logger.info("ShirtRip shutting down — unloading models")
    ModelRegistry.get().unload_all()


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the FastAPI application."""
    if settings is None:
        settings = Settings()

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    app = FastAPI(
        title="ShirtRip",
        description="T-shirt graphic extraction pipeline",
        version="0.1.0",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Exception handlers
    add_middleware(app)

    # API routes
    app.include_router(router)

    # Ensure output dir exists
    settings.output_dir.mkdir(parents=True, exist_ok=True)

    # Mount frontend
    frontend_dir = Path(__file__).parent.parent / "frontend"
    if frontend_dir.exists():
        app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")

    return app
