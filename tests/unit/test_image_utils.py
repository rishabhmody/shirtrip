from __future__ import annotations

import numpy as np
from PIL import Image

from shirtrip.pipeline.image_utils import (
    apply_mask,
    bgr_to_pil,
    bgr_to_rgb,
    crop_to_content,
    pil_to_bgr,
    resize_if_needed,
    rgb_to_bgr,
    to_pipeline_image,
)


class TestColorConversions:
    def test_rgb_to_bgr_roundtrip(self) -> None:
        rgb = np.array([[[255, 0, 0]]], dtype=np.uint8)  # red pixel
        bgr = rgb_to_bgr(rgb)
        assert bgr[0, 0, 0] == 0  # B
        assert bgr[0, 0, 2] == 255  # R
        back = bgr_to_rgb(bgr)
        np.testing.assert_array_equal(rgb, back)

    def test_pil_to_bgr(self) -> None:
        pil_img = Image.new("RGB", (10, 10), (255, 0, 0))
        bgr = pil_to_bgr(pil_img)
        assert bgr.shape == (10, 10, 3)
        assert bgr[0, 0, 2] == 255  # R channel in BGR position
        assert bgr[0, 0, 0] == 0  # B channel

    def test_bgr_to_pil(self) -> None:
        bgr = np.zeros((10, 10, 3), dtype=np.uint8)
        bgr[:, :, 2] = 255  # Red in BGR
        pil_img = bgr_to_pil(bgr)
        r, g, b = pil_img.getpixel((0, 0))
        assert r == 255
        assert g == 0
        assert b == 0

    def test_pil_bgr_roundtrip(self) -> None:
        original = Image.new("RGB", (20, 20), (100, 150, 200))
        bgr = pil_to_bgr(original)
        recovered = bgr_to_pil(bgr)
        assert original.getpixel((0, 0)) == recovered.getpixel((0, 0))


class TestApplyMask:
    def test_output_has_four_channels(self) -> None:
        bgr = np.zeros((10, 10, 3), dtype=np.uint8)
        mask = np.ones((10, 10), dtype=np.uint8) * 255
        result = apply_mask(bgr, mask)
        assert result.shape == (10, 10, 4)

    def test_masked_pixels_are_opaque(self) -> None:
        bgr = np.zeros((10, 10, 3), dtype=np.uint8)
        mask = np.ones((10, 10), dtype=np.uint8) * 255
        result = apply_mask(bgr, mask)
        assert np.all(result[:, :, 3] == 255)

    def test_unmasked_pixels_are_transparent(self) -> None:
        bgr = np.zeros((10, 10, 3), dtype=np.uint8)
        mask = np.zeros((10, 10), dtype=np.uint8)
        result = apply_mask(bgr, mask)
        assert np.all(result[:, :, 3] == 0)

    def test_partial_mask(self) -> None:
        bgr = np.zeros((10, 10, 3), dtype=np.uint8)
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[2:8, 2:8] = 255
        result = apply_mask(bgr, mask)
        assert np.all(result[2:8, 2:8, 3] == 255)
        assert np.all(result[0:2, :, 3] == 0)


class TestCropToContent:
    def test_crops_to_mask(self) -> None:
        bgr = np.zeros((100, 100, 3), dtype=np.uint8)
        bgr[20:40, 30:60] = 128
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:40, 30:60] = 255
        cropped_bgr, cropped_mask, bbox = crop_to_content(bgr, mask)
        assert cropped_bgr.shape == (20, 30, 3)
        assert cropped_mask.shape == (20, 30)
        assert bbox.x == 30
        assert bbox.y == 20
        assert bbox.w == 30
        assert bbox.h == 20

    def test_empty_mask_returns_full_image(self) -> None:
        bgr = np.zeros((50, 50, 3), dtype=np.uint8)
        mask = np.zeros((50, 50), dtype=np.uint8)
        cropped_bgr, _, bbox = crop_to_content(bgr, mask)
        assert cropped_bgr.shape == bgr.shape
        assert bbox.w == 50
        assert bbox.h == 50


class TestResizeIfNeeded:
    def test_no_resize_when_within_limit(self) -> None:
        img = np.zeros((100, 200, 3), dtype=np.uint8)
        result = resize_if_needed(img, 300)
        assert result.shape == (100, 200, 3)

    def test_resize_when_exceeds_limit(self) -> None:
        img = np.zeros((1000, 2000, 3), dtype=np.uint8)
        result = resize_if_needed(img, 500)
        assert max(result.shape[:2]) <= 500

    def test_preserves_aspect_ratio(self) -> None:
        img = np.zeros((200, 400, 3), dtype=np.uint8)
        result = resize_if_needed(img, 200)
        h, w = result.shape[:2]
        assert abs(w / h - 2.0) < 0.1


class TestToPipelineImage:
    def test_creates_pipeline_image(self) -> None:
        bgr = np.zeros((10, 10, 3), dtype=np.uint8)
        img = to_pipeline_image(bgr)
        assert img.bgr is bgr
        assert img.alpha is None

    def test_with_alpha(self) -> None:
        bgr = np.zeros((10, 10, 3), dtype=np.uint8)
        alpha = np.full((10, 10), 200, dtype=np.uint8)
        img = to_pipeline_image(bgr, alpha)
        assert img.alpha is alpha
