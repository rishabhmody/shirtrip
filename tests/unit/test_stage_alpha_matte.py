from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from shirtrip.config.settings import Settings
from shirtrip.pipeline.stage_alpha_matte import _generate_trimap, alpha_matte
from shirtrip.pipeline.types import PipelineImage


@pytest.fixture
def cpu_settings(tmp_path) -> Settings:
    return Settings(
        device="cpu",
        model_cache_dir=tmp_path / "models",
        upload_dir=tmp_path / "uploads",
        output_dir=tmp_path / "outputs",
    )


class TestTrimapGeneration:
    def test_trimap_has_three_regions(self) -> None:
        """Trimap should have exactly 3 values: 0, 128, 255."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 255

        trimap = _generate_trimap(mask, erode_px=5, dilate_px=10)

        unique = set(np.unique(trimap))
        assert unique.issubset({0, 128, 255})
        assert 0 in unique  # background
        assert 255 in unique  # definite foreground
        assert 128 in unique  # unknown band

    def test_trimap_foreground_inside_mask(self) -> None:
        """Definite foreground should be a strict subset of the mask."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 255

        trimap = _generate_trimap(mask, erode_px=5, dilate_px=10)

        fg_pixels = trimap == 255
        assert np.all(mask[fg_pixels] == 255)

    def test_trimap_unknown_band_at_boundary(self) -> None:
        """Unknown region should be at the boundary of the mask."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:80, 20:80] = 255

        trimap = _generate_trimap(mask, erode_px=5, dilate_px=10)

        unknown = trimap == 128
        # Unknown should exist at the boundary
        assert np.sum(unknown) > 0


class TestAlphaMatte:
    @patch("shirtrip.pipeline.stage_alpha_matte._get_vitmatte")
    def test_alpha_is_soft_at_boundaries(self, mock_get_model, cpu_settings) -> None:
        """Output alpha should have soft values (not just 0 and 255)."""
        h, w = 100, 100
        bgr = np.full((h, w, 3), 128, dtype=np.uint8)
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[20:80, 20:80] = 255
        image = PipelineImage(bgr=bgr, alpha=mask)

        # Mock ViTMatte to return a gradient alpha
        mock_processor = MagicMock()
        mock_processor.return_value = {"pixel_values": torch.zeros(1, 4, h, w)}
        mock_model = MagicMock()
        # Create a soft alpha with boundary transitions
        soft_alpha = torch.zeros(1, 1, h, w)
        soft_alpha[:, :, 25:75, 25:75] = 1.0
        # Add soft boundary
        soft_alpha[:, :, 20:25, 20:80] = 0.5
        soft_alpha[:, :, 75:80, 20:80] = 0.5
        soft_alpha[:, :, 25:75, 20:25] = 0.5
        soft_alpha[:, :, 25:75, 75:80] = 0.5
        mock_model.return_value = MagicMock(alphas=soft_alpha)
        mock_get_model.return_value = (mock_processor, mock_model)

        result = alpha_matte(image, cpu_settings)

        alpha = result.image.alpha
        assert alpha is not None
        unique = np.unique(alpha)
        # Should have values between 0 and 255 (soft edges)
        assert len(unique) > 2

    @patch("shirtrip.pipeline.stage_alpha_matte._get_vitmatte")
    def test_interior_alpha_is_opaque(self, mock_get_model, cpu_settings) -> None:
        h, w = 100, 100
        bgr = np.full((h, w, 3), 128, dtype=np.uint8)
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[20:80, 20:80] = 255
        image = PipelineImage(bgr=bgr, alpha=mask)

        mock_processor = MagicMock()
        mock_processor.return_value = {"pixel_values": torch.zeros(1, 4, h, w)}
        mock_model = MagicMock()
        soft_alpha = torch.zeros(1, 1, h, w)
        soft_alpha[:, :, 25:75, 25:75] = 1.0
        mock_model.return_value = MagicMock(alphas=soft_alpha)
        mock_get_model.return_value = (mock_processor, mock_model)

        result = alpha_matte(image, cpu_settings)

        # Well inside the mask should be opaque
        assert np.all(result.image.alpha[35:65, 35:65] == 255)

    @patch("shirtrip.pipeline.stage_alpha_matte._get_vitmatte")
    def test_bgr_channels_unchanged(self, mock_get_model, cpu_settings) -> None:
        """Alpha matte should not modify RGB pixel values."""
        h, w = 100, 100
        bgr = np.random.RandomState(42).randint(0, 256, (h, w, 3), dtype=np.uint8)
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[20:80, 20:80] = 255
        image = PipelineImage(bgr=bgr, alpha=mask)

        mock_processor = MagicMock()
        mock_processor.return_value = {"pixel_values": torch.zeros(1, 4, h, w)}
        mock_model = MagicMock()
        soft_alpha = torch.ones(1, 1, h, w)
        mock_model.return_value = MagicMock(alphas=soft_alpha)
        mock_get_model.return_value = (mock_processor, mock_model)

        result = alpha_matte(image, cpu_settings)

        np.testing.assert_array_equal(result.image.bgr, bgr)

    @patch("shirtrip.pipeline.stage_alpha_matte._get_vitmatte")
    def test_metadata_populated(self, mock_get_model, cpu_settings) -> None:
        h, w = 100, 100
        bgr = np.full((h, w, 3), 128, dtype=np.uint8)
        mask = np.ones((h, w), dtype=np.uint8) * 255
        image = PipelineImage(bgr=bgr, alpha=mask)

        mock_processor = MagicMock()
        mock_processor.return_value = {"pixel_values": torch.zeros(1, 4, h, w)}
        mock_model = MagicMock()
        mock_model.return_value = MagicMock(alphas=torch.ones(1, 1, h, w))
        mock_get_model.return_value = (mock_processor, mock_model)

        result = alpha_matte(image, cpu_settings)

        assert len(result.metadata) == 1
        assert result.metadata[0].stage_name == "alpha_matte"
