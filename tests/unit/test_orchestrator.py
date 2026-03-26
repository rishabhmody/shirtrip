from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from shirtrip.config.settings import Settings
from shirtrip.pipeline.errors import GarmentNotFoundError, StageError
from shirtrip.pipeline.orchestrator import run_pipeline
from shirtrip.pipeline.types import PipelineImage, PipelineResult, StageMetadata


@pytest.fixture
def cpu_settings(tmp_path) -> Settings:
    return Settings(
        device="cpu",
        model_cache_dir=tmp_path / "models",
        upload_dir=tmp_path / "uploads",
        output_dir=tmp_path / "outputs",
        use_sam2=False,
    )


def _make_mock_result(input_image: PipelineImage) -> PipelineResult:
    """Create a mock PipelineResult for testing."""
    cropped = input_image.bgr[10:50, 10:50]
    mask = np.ones(cropped.shape[:2], dtype=np.uint8) * 255
    return PipelineResult(
        image=PipelineImage(bgr=cropped, alpha=mask),
        metadata=[
            StageMetadata(
                stage_name="garment_parse",
                duration_ms=10.0,
                input_shape=input_image.bgr.shape,
                output_shape=cropped.shape,
            )
        ],
        masks={"garment": mask},
    )


class TestOrchestrator:
    @patch("shirtrip.pipeline.orchestrator.STAGE_REGISTRY")
    def test_runs_stages_in_order(self, mock_registry, cpu_settings) -> None:
        """Orchestrator calls stages sequentially and accumulates metadata."""
        bgr = np.zeros((100, 100, 3), dtype=np.uint8)
        image = PipelineImage(bgr=bgr)

        call_order: list[str] = []

        def stage_a(img, settings):
            call_order.append("a")
            return PipelineResult(
                image=img,
                metadata=[StageMetadata("a", 5.0, img.bgr.shape, img.bgr.shape)],
            )

        def stage_b(img, settings):
            call_order.append("b")
            return PipelineResult(
                image=img,
                metadata=[StageMetadata("b", 3.0, img.bgr.shape, img.bgr.shape)],
            )

        mock_registry.__contains__ = lambda self, key: key in {"a", "b"}
        mock_registry.__getitem__ = lambda self, key: {"a": stage_a, "b": stage_b}[key]

        result = run_pipeline(image, cpu_settings, stages=["a", "b"])

        assert call_order == ["a", "b"]
        assert len(result.metadata) == 2
        assert result.metadata[0].stage_name == "a"
        assert result.metadata[1].stage_name == "b"

    @patch("shirtrip.pipeline.orchestrator.STAGE_REGISTRY")
    def test_propagates_garment_not_found(self, mock_registry, cpu_settings) -> None:
        """GarmentNotFoundError propagates without wrapping."""
        bgr = np.zeros((100, 100, 3), dtype=np.uint8)
        image = PipelineImage(bgr=bgr)

        def failing_stage(img, settings):
            raise GarmentNotFoundError()

        mock_registry.__contains__ = lambda self, key: key == "garment_parse"
        mock_registry.__getitem__ = lambda self, key: failing_stage

        with pytest.raises(GarmentNotFoundError):
            run_pipeline(image, cpu_settings, stages=["garment_parse"])

    @patch("shirtrip.pipeline.orchestrator.STAGE_REGISTRY")
    def test_wraps_unexpected_errors(self, mock_registry, cpu_settings) -> None:
        """Unexpected errors are wrapped in StageError."""
        bgr = np.zeros((100, 100, 3), dtype=np.uint8)
        image = PipelineImage(bgr=bgr)

        def broken_stage(img, settings):
            raise RuntimeError("boom")

        mock_registry.__contains__ = lambda self, key: key == "broken"
        mock_registry.__getitem__ = lambda self, key: broken_stage

        with pytest.raises(StageError) as exc_info:
            run_pipeline(image, cpu_settings, stages=["broken"])
        assert "boom" in str(exc_info.value)

    @patch("shirtrip.pipeline.orchestrator.STAGE_REGISTRY")
    def test_default_stages_used_when_none(self, mock_registry, cpu_settings) -> None:
        """When stages=None, the default pipeline is used."""
        bgr = np.zeros((100, 100, 3), dtype=np.uint8)
        image = PipelineImage(bgr=bgr)

        called = []

        def mock_garment(img, settings):
            called.append("garment_parse")
            return PipelineResult(
                image=img,
                metadata=[StageMetadata("garment_parse", 1.0, img.bgr.shape, img.bgr.shape)],
            )

        mock_registry.__contains__ = lambda self, key: key == "garment_parse"
        mock_registry.__getitem__ = lambda self, key: mock_garment

        with patch("shirtrip.pipeline.orchestrator.DEFAULT_PIPELINE", ["garment_parse"]):
            result = run_pipeline(image, cpu_settings)

        assert called == ["garment_parse"]

    @patch("shirtrip.pipeline.orchestrator.STAGE_REGISTRY")
    def test_passes_output_to_next_stage(self, mock_registry, cpu_settings) -> None:
        """Each stage receives the output image from the previous stage."""
        bgr = np.zeros((100, 100, 3), dtype=np.uint8)
        image = PipelineImage(bgr=bgr)

        received_shapes: list[tuple] = []

        def stage_a(img, settings):
            received_shapes.append(img.bgr.shape)
            small = PipelineImage(bgr=np.zeros((50, 50, 3), dtype=np.uint8))
            return PipelineResult(
                image=small,
                metadata=[StageMetadata("a", 1.0, img.bgr.shape, small.bgr.shape)],
            )

        def stage_b(img, settings):
            received_shapes.append(img.bgr.shape)
            return PipelineResult(
                image=img,
                metadata=[StageMetadata("b", 1.0, img.bgr.shape, img.bgr.shape)],
            )

        mock_registry.__contains__ = lambda self, key: key in {"a", "b"}
        mock_registry.__getitem__ = lambda self, key: {"a": stage_a, "b": stage_b}[key]

        run_pipeline(image, cpu_settings, stages=["a", "b"])

        assert received_shapes[0] == (100, 100, 3)  # stage_a gets original
        assert received_shapes[1] == (50, 50, 3)  # stage_b gets stage_a output
