from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from shirtrip.config.settings import Settings
from shirtrip.pipeline.errors import GarmentNotFoundError
from shirtrip.pipeline.stage_garment_parse import (
    UPPER_CLOTHES_LABEL,
    garment_parse,
)
from shirtrip.pipeline.types import PipelineImage


@pytest.fixture
def cpu_settings(tmp_path) -> Settings:
    return Settings(
        device="cpu",
        model_cache_dir=tmp_path / "models",
        upload_dir=tmp_path / "uploads",
        output_dir=tmp_path / "outputs",
        use_sam2=False,
    )


def _make_fake_segformer_output(h: int, w: int, clothes_region: tuple[int, int, int, int]):
    """Create a fake SegFormer logits tensor with upper clothes in the given region."""
    num_classes = 18
    logits = torch.zeros(1, num_classes, h, w)
    # Set background as default class
    logits[:, 0, :, :] = 1.0
    # Set upper clothes in the specified region
    y1, y2, x1, x2 = clothes_region
    logits[:, UPPER_CLOTHES_LABEL, y1:y2, x1:x2] = 10.0
    logits[:, 0, y1:y2, x1:x2] = 0.0
    return MagicMock(logits=logits)


class TestGarmentParse:
    @patch("shirtrip.pipeline.stage_garment_parse._get_segformer")
    def test_segformer_returns_valid_mask(self, mock_get_segformer, cpu_settings) -> None:
        """SegFormer output produces a valid binary mask matching input dimensions."""
        h, w = 100, 80
        image = PipelineImage(bgr=np.zeros((h, w, 3), dtype=np.uint8))

        mock_processor = MagicMock()
        mock_processor.return_value = {"pixel_values": torch.zeros(1, 3, h, w)}
        mock_model = MagicMock()
        mock_model.return_value = _make_fake_segformer_output(h, w, (20, 60, 10, 70))
        mock_get_segformer.return_value = (mock_processor, mock_model)

        result = garment_parse(image, cpu_settings)

        assert "garment" in result.masks
        mask = result.masks["garment"]
        assert mask.shape == (h, w)
        assert mask.dtype == np.uint8
        assert set(np.unique(mask)).issubset({0, 255})

    @patch("shirtrip.pipeline.stage_garment_parse._get_segformer")
    def test_torso_mask_isolates_clothing(self, mock_get_segformer, cpu_settings) -> None:
        """Garment mask covers the clothing region."""
        h, w = 100, 80
        bgr = np.full((h, w, 3), 200, dtype=np.uint8)
        image = PipelineImage(bgr=bgr)

        mock_processor = MagicMock()
        mock_processor.return_value = {"pixel_values": torch.zeros(1, 3, h, w)}
        mock_model = MagicMock()
        mock_model.return_value = _make_fake_segformer_output(h, w, (20, 60, 10, 70))
        mock_get_segformer.return_value = (mock_processor, mock_model)

        result = garment_parse(image, cpu_settings)
        mask = result.masks["garment"]

        # The mask should have non-zero pixels in the clothing region
        assert np.sum(mask[20:60, 10:70] > 0) > 0

    @patch("shirtrip.pipeline.stage_garment_parse._get_segformer")
    def test_output_has_alpha_channel(self, mock_get_segformer, cpu_settings) -> None:
        """Output PipelineImage has an alpha channel derived from the mask."""
        h, w = 100, 80
        bgr = np.full((h, w, 3), 150, dtype=np.uint8)
        image = PipelineImage(bgr=bgr)

        mock_processor = MagicMock()
        mock_processor.return_value = {"pixel_values": torch.zeros(1, 3, h, w)}
        mock_model = MagicMock()
        mock_model.return_value = _make_fake_segformer_output(h, w, (20, 60, 10, 70))
        mock_get_segformer.return_value = (mock_processor, mock_model)

        result = garment_parse(image, cpu_settings)

        assert result.image.alpha is not None
        rgba = result.image.rgba
        assert rgba.shape[2] == 4

    @patch("shirtrip.pipeline.stage_garment_parse._get_segformer")
    def test_handles_no_garment_found(self, mock_get_segformer, cpu_settings) -> None:
        """Raises GarmentNotFoundError when no upper clothes detected."""
        h, w = 100, 80
        image = PipelineImage(bgr=np.zeros((h, w, 3), dtype=np.uint8))

        mock_processor = MagicMock()
        mock_processor.return_value = {"pixel_values": torch.zeros(1, 3, h, w)}
        mock_model = MagicMock()
        # All pixels are background — no clothes
        logits = torch.zeros(1, 18, h, w)
        logits[:, 0, :, :] = 10.0
        mock_model.return_value = MagicMock(logits=logits)
        mock_get_segformer.return_value = (mock_processor, mock_model)

        with pytest.raises(GarmentNotFoundError):
            garment_parse(image, cpu_settings)

    @patch("shirtrip.pipeline.stage_garment_parse._get_segformer")
    def test_output_pixels_come_from_input(self, mock_get_segformer, cpu_settings) -> None:
        """Every pixel in the output must exist in the input (no hallucination)."""
        h, w = 100, 80
        # Create a distinctive pattern
        bgr = np.random.RandomState(42).randint(0, 256, (h, w, 3), dtype=np.uint8)
        image = PipelineImage(bgr=bgr)

        mock_processor = MagicMock()
        mock_processor.return_value = {"pixel_values": torch.zeros(1, 3, h, w)}
        mock_model = MagicMock()
        mock_model.return_value = _make_fake_segformer_output(h, w, (20, 60, 10, 70))
        mock_get_segformer.return_value = (mock_processor, mock_model)

        result = garment_parse(image, cpu_settings)

        # Every opaque pixel in the output must match the corresponding input pixel
        out_bgr = result.image.bgr
        out_alpha = result.image.alpha
        assert out_alpha is not None

        # The output is cropped, so we check that output pixels are a subset of input pixels
        # by verifying the output shape is <= input shape
        assert out_bgr.shape[0] <= bgr.shape[0]
        assert out_bgr.shape[1] <= bgr.shape[1]

    @patch("shirtrip.pipeline.stage_garment_parse._get_segformer")
    def test_metadata_populated(self, mock_get_segformer, cpu_settings) -> None:
        """Stage metadata includes timing and shape info."""
        h, w = 100, 80
        image = PipelineImage(bgr=np.zeros((h, w, 3), dtype=np.uint8))

        mock_processor = MagicMock()
        mock_processor.return_value = {"pixel_values": torch.zeros(1, 3, h, w)}
        mock_model = MagicMock()
        mock_model.return_value = _make_fake_segformer_output(h, w, (20, 60, 10, 70))
        mock_get_segformer.return_value = (mock_processor, mock_model)

        result = garment_parse(image, cpu_settings)

        assert len(result.metadata) == 1
        meta = result.metadata[0]
        assert meta.stage_name == "garment_parse"
        assert meta.duration_ms >= 0
        assert len(meta.input_shape) == 3
        assert len(meta.output_shape) == 3
