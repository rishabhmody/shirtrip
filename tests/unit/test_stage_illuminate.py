from __future__ import annotations

import cv2
import numpy as np
import pytest

from shirtrip.config.settings import Settings
from shirtrip.pipeline.stage_illuminate import illuminate
from shirtrip.pipeline.types import PipelineImage


@pytest.fixture
def cpu_settings(tmp_path) -> Settings:
    return Settings(
        device="cpu",
        model_cache_dir=tmp_path / "models",
        upload_dir=tmp_path / "uploads",
        output_dir=tmp_path / "outputs",
    )


def _make_shadowed_image(h: int = 100, w: int = 100) -> np.ndarray:
    """Create a BGR image with artificial shadow (left half darker)."""
    img = np.full((h, w, 3), 200, dtype=np.uint8)
    img[:, : w // 2] = 80  # Left half is dark (shadow)
    return img


class TestIlluminate:
    def test_clahe_modifies_shadowed_image(self, cpu_settings) -> None:
        """CLAHE should modify the brightness of a shadowed image."""
        bgr = _make_shadowed_image()
        alpha = np.ones((100, 100), dtype=np.uint8) * 255
        image = PipelineImage(bgr=bgr, alpha=alpha)

        result = illuminate(image, cpu_settings)

        # The dark region should be brighter after CLAHE
        lab_before = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        lab_after = cv2.cvtColor(result.image.bgr, cv2.COLOR_BGR2LAB)

        dark_region_before = lab_before[:, :50, 0].mean()
        dark_region_after = lab_after[:, :50, 0].mean()

        assert dark_region_after > dark_region_before

    def test_no_clipping(self, cpu_settings) -> None:
        """No pixels should be hard-clipped to extreme values."""
        bgr = _make_shadowed_image()
        alpha = np.ones((100, 100), dtype=np.uint8) * 255
        image = PipelineImage(bgr=bgr, alpha=alpha)

        result = illuminate(image, cpu_settings)

        # Check that not all values are at 0 or 255
        out = result.image.bgr
        assert not np.all(out == 0)
        assert not np.all(out == 255)

    def test_output_is_valid_uint8(self, cpu_settings) -> None:
        """Output should be valid uint8 with no overflow."""
        bgr = np.full((100, 100, 3), 150, dtype=np.uint8)
        alpha = np.ones((100, 100), dtype=np.uint8) * 255
        image = PipelineImage(bgr=bgr, alpha=alpha)

        result = illuminate(image, cpu_settings)

        assert result.image.bgr.dtype == np.uint8
        # Uniform input should remain uniform after CLAHE (CLAHE preserves uniformity)
        assert np.std(result.image.bgr.astype(float)) < 5.0

    def test_output_shape_matches_input(self, cpu_settings) -> None:
        bgr = _make_shadowed_image(80, 120)
        alpha = np.ones((80, 120), dtype=np.uint8) * 255
        image = PipelineImage(bgr=bgr, alpha=alpha)

        result = illuminate(image, cpu_settings)

        assert result.image.bgr.shape == bgr.shape
        assert result.image.bgr.dtype == np.uint8

    def test_metadata_populated(self, cpu_settings) -> None:
        bgr = _make_shadowed_image()
        alpha = np.ones((100, 100), dtype=np.uint8) * 255
        image = PipelineImage(bgr=bgr, alpha=alpha)

        result = illuminate(image, cpu_settings)

        assert len(result.metadata) == 1
        assert result.metadata[0].stage_name == "illuminate"
        assert result.metadata[0].duration_ms >= 0

    def test_handles_no_alpha(self, cpu_settings) -> None:
        """Should work even without an alpha channel."""
        bgr = _make_shadowed_image()
        image = PipelineImage(bgr=bgr)

        result = illuminate(image, cpu_settings)

        assert result.image.bgr.shape == bgr.shape
