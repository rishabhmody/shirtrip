from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from shirtrip.config.settings import Settings
from shirtrip.pipeline.stage_depth_estimate import depth_estimate
from shirtrip.pipeline.types import PipelineImage, PipelineResult


@pytest.fixture
def cpu_settings(tmp_path) -> Settings:
    return Settings(
        device="cpu",
        model_cache_dir=tmp_path / "models",
        upload_dir=tmp_path / "uploads",
        output_dir=tmp_path / "outputs",
        use_sam2=False,
    )


def _make_fake_depth_output(h: int, w: int) -> MagicMock:
    """Create a fake depth model output with a gradient depth map."""
    depth = torch.linspace(0, 1, h).unsqueeze(1).expand(h, w).unsqueeze(0).unsqueeze(0)
    return MagicMock(predicted_depth=depth)


class TestDepthEstimate:
    @patch("shirtrip.pipeline.stage_depth_estimate._get_depth_model")
    def test_depth_map_shape_matches_input(self, mock_get_model, cpu_settings) -> None:
        h, w = 60, 80
        bgr = np.full((h, w, 3), 128, dtype=np.uint8)
        alpha = np.full((h, w), 255, dtype=np.uint8)
        image = PipelineImage(bgr=bgr, alpha=alpha)

        mock_processor = MagicMock()
        mock_processor.return_value = {"pixel_values": torch.zeros(1, 3, h, w)}
        mock_model = MagicMock()
        mock_model.return_value = _make_fake_depth_output(h, w)
        mock_get_model.return_value = (mock_processor, mock_model)

        result = depth_estimate(image, cpu_settings)

        assert "depth" in result.masks
        depth_map = result.masks["depth"]
        assert depth_map.shape == (h, w)

    @patch("shirtrip.pipeline.stage_depth_estimate._get_depth_model")
    def test_depth_values_normalized(self, mock_get_model, cpu_settings) -> None:
        h, w = 60, 80
        bgr = np.full((h, w, 3), 128, dtype=np.uint8)
        alpha = np.full((h, w), 255, dtype=np.uint8)
        image = PipelineImage(bgr=bgr, alpha=alpha)

        mock_processor = MagicMock()
        mock_processor.return_value = {"pixel_values": torch.zeros(1, 3, h, w)}
        mock_model = MagicMock()
        mock_model.return_value = _make_fake_depth_output(h, w)
        mock_get_model.return_value = (mock_processor, mock_model)

        result = depth_estimate(image, cpu_settings)
        depth_map = result.masks["depth"]

        assert depth_map.dtype == np.float32
        assert depth_map.min() >= 0.0
        assert depth_map.max() <= 1.0

    @patch("shirtrip.pipeline.stage_depth_estimate._get_depth_model")
    def test_depth_varies_on_gradient_input(self, mock_get_model, cpu_settings) -> None:
        h, w = 60, 80
        bgr = np.full((h, w, 3), 128, dtype=np.uint8)
        alpha = np.full((h, w), 255, dtype=np.uint8)
        image = PipelineImage(bgr=bgr, alpha=alpha)

        mock_processor = MagicMock()
        mock_processor.return_value = {"pixel_values": torch.zeros(1, 3, h, w)}
        mock_model = MagicMock()
        mock_model.return_value = _make_fake_depth_output(h, w)
        mock_get_model.return_value = (mock_processor, mock_model)

        result = depth_estimate(image, cpu_settings)
        depth_map = result.masks["depth"]

        assert np.std(depth_map) > 0.1

    @patch("shirtrip.pipeline.stage_depth_estimate._get_depth_model")
    def test_image_unchanged(self, mock_get_model, cpu_settings) -> None:
        """Depth estimation does not modify the image itself."""
        h, w = 60, 80
        bgr = np.random.RandomState(42).randint(0, 256, (h, w, 3), dtype=np.uint8)
        alpha = np.full((h, w), 255, dtype=np.uint8)
        image = PipelineImage(bgr=bgr, alpha=alpha)

        mock_processor = MagicMock()
        mock_processor.return_value = {"pixel_values": torch.zeros(1, 3, h, w)}
        mock_model = MagicMock()
        mock_model.return_value = _make_fake_depth_output(h, w)
        mock_get_model.return_value = (mock_processor, mock_model)

        result = depth_estimate(image, cpu_settings)

        np.testing.assert_array_equal(result.image.bgr, bgr)

    @patch("shirtrip.pipeline.stage_depth_estimate._get_depth_model")
    def test_metadata_populated(self, mock_get_model, cpu_settings) -> None:
        h, w = 60, 80
        bgr = np.full((h, w, 3), 128, dtype=np.uint8)
        alpha = np.full((h, w), 255, dtype=np.uint8)
        image = PipelineImage(bgr=bgr, alpha=alpha)

        mock_processor = MagicMock()
        mock_processor.return_value = {"pixel_values": torch.zeros(1, 3, h, w)}
        mock_model = MagicMock()
        mock_model.return_value = _make_fake_depth_output(h, w)
        mock_get_model.return_value = (mock_processor, mock_model)

        result = depth_estimate(image, cpu_settings)

        assert len(result.metadata) == 1
        assert result.metadata[0].stage_name == "depth_estimate"
        assert result.metadata[0].duration_ms >= 0
