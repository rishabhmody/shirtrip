from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import pytest
from httpx import ASGITransport, AsyncClient

from shirtrip.api.app import create_app
from shirtrip.config.settings import Settings
from shirtrip.pipeline.types import PipelineImage, PipelineResult, StageMetadata


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    return Settings(
        device="cpu",
        model_cache_dir=tmp_path / "models",
        upload_dir=tmp_path / "uploads",
        output_dir=tmp_path / "outputs",
        use_sam2=False,
    )


@pytest.fixture
def app(test_settings: Settings):
    return create_app(test_settings)


@pytest.fixture
async def client(app):
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as c:
        yield c


def _make_test_jpeg() -> bytes:
    """Create a small valid JPEG in memory."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[20:80, 20:80] = (200, 150, 100)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def _mock_pipeline_result() -> PipelineResult:
    """Create a mock PipelineResult for testing."""
    bgr = np.full((50, 50, 3), 128, dtype=np.uint8)
    alpha = np.full((50, 50), 255, dtype=np.uint8)
    return PipelineResult(
        image=PipelineImage(bgr=bgr, alpha=alpha),
        metadata=[
            StageMetadata(
                stage_name="garment_parse",
                duration_ms=42.0,
                input_shape=(100, 100, 3),
                output_shape=(50, 50, 3),
            )
        ],
        masks={"garment": alpha},
    )


class TestHealthEndpoint:
    @pytest.mark.asyncio
    async def test_health_returns_ok(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "device" in data
        assert "gpu_available" in data


class TestExtractEndpoint:
    @pytest.mark.asyncio
    @patch("shirtrip.api.routes.run_pipeline")
    async def test_extract_valid_image(self, mock_run, client: AsyncClient) -> None:
        mock_run.return_value = _mock_pipeline_result()
        jpeg_bytes = _make_test_jpeg()

        resp = await client.post(
            "/api/v1/extract",
            files={"file": ("test.jpg", io.BytesIO(jpeg_bytes), "image/jpeg")},
        )

        assert resp.status_code == 200
        data = resp.json()
        assert "job_id" in data
        assert data["filename"] == "test.jpg"
        assert len(data["stages_completed"]) == 1
        assert data["processing_time_ms"] > 0

    @pytest.mark.asyncio
    async def test_extract_invalid_extension(self, client: AsyncClient) -> None:
        resp = await client.post(
            "/api/v1/extract",
            files={"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")},
        )
        assert resp.status_code == 422

    @pytest.mark.asyncio
    @patch("shirtrip.api.routes.run_pipeline")
    async def test_get_output(self, mock_run, client: AsyncClient, test_settings) -> None:
        mock_run.return_value = _mock_pipeline_result()
        jpeg_bytes = _make_test_jpeg()

        # First extract
        resp = await client.post(
            "/api/v1/extract",
            files={"file": ("test.jpg", io.BytesIO(jpeg_bytes), "image/jpeg")},
        )
        job_id = resp.json()["job_id"]

        # Then download
        resp = await client.get(f"/api/v1/output/{job_id}")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"

    @pytest.mark.asyncio
    async def test_get_output_not_found(self, client: AsyncClient) -> None:
        resp = await client.get("/api/v1/output/nonexistent")
        assert resp.status_code == 404
