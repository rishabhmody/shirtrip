from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from shirtrip.config.settings import Settings
from shirtrip.pipeline.types import PipelineImage


@pytest.fixture
def test_settings(tmp_path: Path) -> Settings:
    """Settings configured for testing (CPU, temp dirs)."""
    return Settings(
        device="cpu",
        model_cache_dir=tmp_path / "models",
        upload_dir=tmp_path / "uploads",
        output_dir=tmp_path / "outputs",
        use_sam2=False,
    )


@pytest.fixture
def synthetic_tshirt_image() -> np.ndarray:
    """Generate a synthetic t-shirt image (BGR).

    Creates a 400x300 image with:
    - Gray background (128, 128, 128)
    - White shirt region (rectangular)
    - Colored graphic design (checkerboard pattern) on the shirt
    """
    img = np.full((400, 300, 3), 128, dtype=np.uint8)

    # Shirt region (white rectangle)
    cv2.rectangle(img, (50, 50), (250, 350), (255, 255, 255), -1)

    # Graphic design: colored checkerboard in the center of the shirt
    for row in range(4):
        for col in range(4):
            x = 100 + col * 30
            y = 120 + row * 30
            color = (0, 0, 255) if (row + col) % 2 == 0 else (255, 0, 0)
            cv2.rectangle(img, (x, y), (x + 30, y + 30), color, -1)

    return img


@pytest.fixture
def synthetic_shirt_mask() -> np.ndarray:
    """Binary mask matching the shirt region in synthetic_tshirt_image."""
    mask = np.zeros((400, 300), dtype=np.uint8)
    cv2.rectangle(mask, (50, 50), (250, 350), 255, -1)
    return mask


@pytest.fixture
def synthetic_graphic_mask() -> np.ndarray:
    """Binary mask matching the graphic region in synthetic_tshirt_image."""
    mask = np.zeros((400, 300), dtype=np.uint8)
    cv2.rectangle(mask, (100, 120), (220, 240), 255, -1)
    return mask


@pytest.fixture
def pipeline_image(synthetic_tshirt_image: np.ndarray) -> PipelineImage:
    """PipelineImage wrapping the synthetic t-shirt image."""
    return PipelineImage(bgr=synthetic_tshirt_image)
