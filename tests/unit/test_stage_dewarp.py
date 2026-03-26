from __future__ import annotations

import numpy as np
import pytest

from shirtrip.config.settings import Settings
from shirtrip.pipeline.stage_dewarp import (
    _compute_displacement_from_depth,
    dewarp,
)
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


class TestDisplacementField:
    def test_displacement_field_shape(self) -> None:
        """Displacement field has shape (H, W) for both map_x and map_y."""
        h, w = 100, 80
        depth = np.random.rand(h, w).astype(np.float32)
        mask = np.ones((h, w), dtype=np.uint8) * 255
        map_x, map_y = _compute_displacement_from_depth(depth, mask, strength=1.0)
        assert map_x.shape == (h, w)
        assert map_y.shape == (h, w)
        assert map_x.dtype == np.float32
        assert map_y.dtype == np.float32

    def test_identity_displacement_on_flat_depth(self) -> None:
        """Flat depth (no variation) should produce near-identity displacement."""
        h, w = 50, 50
        depth = np.full((h, w), 0.5, dtype=np.float32)
        mask = np.ones((h, w), dtype=np.uint8) * 255
        map_x, map_y = _compute_displacement_from_depth(depth, mask, strength=1.0)

        # For flat depth, gradients are zero, so maps should be near identity
        identity_x = np.arange(w, dtype=np.float32)[np.newaxis, :].repeat(h, axis=0)
        identity_y = np.arange(h, dtype=np.float32)[:, np.newaxis].repeat(w, axis=1)

        np.testing.assert_allclose(map_x, identity_x, atol=0.5)
        np.testing.assert_allclose(map_y, identity_y, atol=0.5)


class TestDewarp:
    def test_remap_preserves_pixel_values_on_identity(self, cpu_settings) -> None:
        """With flat depth (identity remap), output pixels match input."""
        h, w = 60, 80
        bgr = np.random.RandomState(42).randint(0, 256, (h, w, 3), dtype=np.uint8)
        alpha = np.full((h, w), 255, dtype=np.uint8)
        # Flat depth = no displacement
        depth = np.full((h, w), 0.5, dtype=np.float32)
        image = PipelineImage(bgr=bgr, alpha=alpha)

        result = dewarp(image, cpu_settings, depth_map=depth)

        # Output should be very close to input (interpolation may cause minor rounding)
        diff = np.abs(result.image.bgr.astype(float) - bgr.astype(float))
        assert np.mean(diff) < 2.0  # Allow minor interpolation artifacts

    def test_output_shape_matches_input(self, cpu_settings) -> None:
        """Output dimensions match input."""
        h, w = 60, 80
        bgr = np.full((h, w, 3), 128, dtype=np.uint8)
        alpha = np.full((h, w), 255, dtype=np.uint8)
        depth = np.random.rand(h, w).astype(np.float32)
        image = PipelineImage(bgr=bgr, alpha=alpha)

        result = dewarp(image, cpu_settings, depth_map=depth)

        assert result.image.bgr.shape == (h, w, 3)

    def test_handles_missing_depth(self, cpu_settings) -> None:
        """When no depth map is provided, passes through unchanged."""
        h, w = 60, 80
        bgr = np.full((h, w, 3), 128, dtype=np.uint8)
        alpha = np.full((h, w), 255, dtype=np.uint8)
        image = PipelineImage(bgr=bgr, alpha=alpha)

        result = dewarp(image, cpu_settings, depth_map=None)

        np.testing.assert_array_equal(result.image.bgr, bgr)

    def test_strength_zero_is_identity(self, cpu_settings) -> None:
        """Zero strength produces identity mapping."""
        h, w = 60, 80
        bgr = np.random.RandomState(7).randint(0, 256, (h, w, 3), dtype=np.uint8)
        alpha = np.full((h, w), 255, dtype=np.uint8)
        depth = np.random.rand(h, w).astype(np.float32)
        image = PipelineImage(bgr=bgr, alpha=alpha)

        # Override strength to 0
        result = dewarp(image, cpu_settings, depth_map=depth, strength=0.0)

        diff = np.abs(result.image.bgr.astype(float) - bgr.astype(float))
        assert np.mean(diff) < 1.0

    def test_metadata_populated(self, cpu_settings) -> None:
        h, w = 60, 80
        bgr = np.full((h, w, 3), 128, dtype=np.uint8)
        alpha = np.full((h, w), 255, dtype=np.uint8)
        depth = np.random.rand(h, w).astype(np.float32)
        image = PipelineImage(bgr=bgr, alpha=alpha)

        result = dewarp(image, cpu_settings, depth_map=depth)

        assert len(result.metadata) == 1
        assert result.metadata[0].stage_name == "dewarp"
