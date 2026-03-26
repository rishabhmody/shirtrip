from __future__ import annotations

from unittest.mock import MagicMock, patch

import cv2
import numpy as np
import pytest
import torch

from shirtrip.config.settings import Settings
from shirtrip.pipeline.errors import GarmentNotFoundError
from shirtrip.pipeline.stage_garment_parse import (
    UPPER_CLOTHES_LABEL,
    _estimate_fabric_color,
    _isolate_graphic,
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
    logits[:, 0, :, :] = 1.0
    y1, y2, x1, x2 = clothes_region
    logits[:, UPPER_CLOTHES_LABEL, y1:y2, x1:x2] = 10.0
    logits[:, 0, y1:y2, x1:x2] = 0.0
    return MagicMock(logits=logits)


def _make_tshirt_with_graphic(h: int = 200, w: int = 160) -> np.ndarray:
    """Create a synthetic image: gray bg, white shirt, colored graphic."""
    img = np.full((h, w, 3), 128, dtype=np.uint8)  # Gray background
    cv2.rectangle(img, (20, 20), (140, 180), (240, 240, 240), -1)  # White shirt
    # Colorful graphic in center of shirt
    cv2.rectangle(img, (50, 50), (110, 130), (0, 0, 200), -1)  # Red block
    cv2.rectangle(img, (60, 60), (100, 120), (200, 50, 0), -1)  # Blue block
    return img


class TestEstimateFabricColor:
    def test_detects_white_fabric(self) -> None:
        img = _make_tshirt_with_graphic()
        garment_mask = np.zeros((200, 160), dtype=np.uint8)
        cv2.rectangle(garment_mask, (20, 20), (140, 180), 255, -1)

        color = _estimate_fabric_color(img, garment_mask)
        # Should detect the white/light color of the shirt edges, not the graphic
        assert np.mean(color) > 200  # Should be close to white (240)

    def test_handles_small_mask(self) -> None:
        img = np.full((50, 50, 3), 100, dtype=np.uint8)
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[20:25, 20:25] = 255  # Very small mask
        color = _estimate_fabric_color(img, mask)
        assert color.shape == (3,)


class TestIsolateGraphic:
    def test_finds_graphic_region(self) -> None:
        img = _make_tshirt_with_graphic()
        garment_mask = np.zeros((200, 160), dtype=np.uint8)
        cv2.rectangle(garment_mask, (20, 20), (140, 180), 255, -1)
        fabric_color = np.array([240, 240, 240], dtype=np.uint8)

        graphic_mask = _isolate_graphic(img, garment_mask, fabric_color)

        # Graphic should be detected in the center
        assert np.sum(graphic_mask[50:130, 50:110] > 0) > 0
        # Shirt edges (plain fabric) should NOT be in graphic mask
        assert np.sum(graphic_mask[20:30, 20:40] > 0) == 0

    def test_excludes_background(self) -> None:
        img = _make_tshirt_with_graphic()
        garment_mask = np.zeros((200, 160), dtype=np.uint8)
        cv2.rectangle(garment_mask, (20, 20), (140, 180), 255, -1)
        fabric_color = np.array([240, 240, 240], dtype=np.uint8)

        graphic_mask = _isolate_graphic(img, garment_mask, fabric_color)

        # Nothing outside the garment
        assert np.sum(graphic_mask[0:15, :] > 0) == 0
        assert np.sum(graphic_mask[:, 145:] > 0) == 0


class TestGarmentParse:
    @patch("shirtrip.pipeline.stage_garment_parse._get_segformer")
    def test_returns_graphic_mask(self, mock_get_segformer, cpu_settings) -> None:
        h, w = 200, 160
        image = PipelineImage(bgr=_make_tshirt_with_graphic(h, w))

        mock_processor = MagicMock()
        mock_processor.return_value = {"pixel_values": torch.zeros(1, 3, h, w)}
        mock_model = MagicMock()
        mock_model.return_value = _make_fake_segformer_output(h, w, (20, 180, 20, 140))
        mock_get_segformer.return_value = (mock_processor, mock_model)

        result = garment_parse(image, cpu_settings)

        assert "graphic" in result.masks
        assert "garment" in result.masks
        # Graphic mask should be smaller than garment mask
        assert np.sum(result.masks["graphic"] > 0) < np.sum(result.masks["garment"] > 0)

    @patch("shirtrip.pipeline.stage_garment_parse._get_segformer")
    def test_output_has_alpha_channel(self, mock_get_segformer, cpu_settings) -> None:
        h, w = 200, 160
        image = PipelineImage(bgr=_make_tshirt_with_graphic(h, w))

        mock_processor = MagicMock()
        mock_processor.return_value = {"pixel_values": torch.zeros(1, 3, h, w)}
        mock_model = MagicMock()
        mock_model.return_value = _make_fake_segformer_output(h, w, (20, 180, 20, 140))
        mock_get_segformer.return_value = (mock_processor, mock_model)

        result = garment_parse(image, cpu_settings)

        assert result.image.alpha is not None
        assert result.image.rgba.shape[2] == 4

    @patch("shirtrip.pipeline.stage_garment_parse._get_segformer")
    def test_handles_no_garment_found(self, mock_get_segformer, cpu_settings) -> None:
        h, w = 100, 80
        image = PipelineImage(bgr=np.zeros((h, w, 3), dtype=np.uint8))

        mock_processor = MagicMock()
        mock_processor.return_value = {"pixel_values": torch.zeros(1, 3, h, w)}
        mock_model = MagicMock()
        logits = torch.zeros(1, 18, h, w)
        logits[:, 0, :, :] = 10.0
        mock_model.return_value = MagicMock(logits=logits)
        mock_get_segformer.return_value = (mock_processor, mock_model)

        with pytest.raises(GarmentNotFoundError):
            garment_parse(image, cpu_settings)

    @patch("shirtrip.pipeline.stage_garment_parse._get_segformer")
    def test_output_pixels_come_from_input(self, mock_get_segformer, cpu_settings) -> None:
        h, w = 200, 160
        bgr = _make_tshirt_with_graphic(h, w)
        image = PipelineImage(bgr=bgr)

        mock_processor = MagicMock()
        mock_processor.return_value = {"pixel_values": torch.zeros(1, 3, h, w)}
        mock_model = MagicMock()
        mock_model.return_value = _make_fake_segformer_output(h, w, (20, 180, 20, 140))
        mock_get_segformer.return_value = (mock_processor, mock_model)

        result = garment_parse(image, cpu_settings)

        out_bgr = result.image.bgr
        assert out_bgr.shape[0] <= bgr.shape[0]
        assert out_bgr.shape[1] <= bgr.shape[1]

    @patch("shirtrip.pipeline.stage_garment_parse._get_segformer")
    def test_metadata_populated(self, mock_get_segformer, cpu_settings) -> None:
        h, w = 200, 160
        image = PipelineImage(bgr=_make_tshirt_with_graphic(h, w))

        mock_processor = MagicMock()
        mock_processor.return_value = {"pixel_values": torch.zeros(1, 3, h, w)}
        mock_model = MagicMock()
        mock_model.return_value = _make_fake_segformer_output(h, w, (20, 180, 20, 140))
        mock_get_segformer.return_value = (mock_processor, mock_model)

        result = garment_parse(image, cpu_settings)

        assert len(result.metadata) == 1
        meta = result.metadata[0]
        assert meta.stage_name == "garment_parse"
        assert "graphic_area" in meta.extra
        assert "fabric_color_bgr" in meta.extra
