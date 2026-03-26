from __future__ import annotations

import numpy as np

from eval.metrics import compute_mask_iou, compute_psnr, compute_ssim, pixel_origin_check


class TestSSIM:
    def test_identical_images(self) -> None:
        img = np.random.RandomState(0).randint(0, 256, (50, 50, 3), dtype=np.uint8)
        assert compute_ssim(img, img) == 1.0

    def test_different_images(self) -> None:
        img1 = np.zeros((50, 50, 3), dtype=np.uint8)
        img2 = np.full((50, 50, 3), 255, dtype=np.uint8)
        ssim = compute_ssim(img1, img2)
        assert ssim < 0.1


class TestPSNR:
    def test_identical_images(self) -> None:
        img = np.full((50, 50, 3), 128, dtype=np.uint8)
        psnr = compute_psnr(img, img)
        assert psnr == float("inf")

    def test_different_images(self) -> None:
        img1 = np.zeros((50, 50, 3), dtype=np.uint8)
        img2 = np.full((50, 50, 3), 10, dtype=np.uint8)
        psnr = compute_psnr(img1, img2)
        assert psnr > 0
        assert psnr < 100


class TestMaskIoU:
    def test_identical_masks(self) -> None:
        mask = np.zeros((50, 50), dtype=np.uint8)
        mask[10:40, 10:40] = 255
        assert compute_mask_iou(mask, mask) == 1.0

    def test_disjoint_masks(self) -> None:
        m1 = np.zeros((50, 50), dtype=np.uint8)
        m1[0:20, 0:20] = 255
        m2 = np.zeros((50, 50), dtype=np.uint8)
        m2[30:50, 30:50] = 255
        assert compute_mask_iou(m1, m2) == 0.0

    def test_partial_overlap(self) -> None:
        m1 = np.zeros((50, 50), dtype=np.uint8)
        m1[0:30, 0:30] = 255
        m2 = np.zeros((50, 50), dtype=np.uint8)
        m2[15:45, 15:45] = 255
        iou = compute_mask_iou(m1, m2)
        assert 0 < iou < 1

    def test_both_empty(self) -> None:
        m = np.zeros((50, 50), dtype=np.uint8)
        assert compute_mask_iou(m, m) == 1.0


class TestPixelOriginCheck:
    def test_valid_pixels(self) -> None:
        """All output pixels exist in input."""
        input_img = np.array([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8)
        output_img = np.array([[[0, 0, 0]]], dtype=np.uint8)
        mask = np.array([[255]], dtype=np.uint8)
        valid, pct = pixel_origin_check(input_img, output_img, mask)
        assert valid is True
        assert pct == 100.0

    def test_invalid_pixels(self) -> None:
        """Output has a pixel not in input."""
        input_img = np.array([[[0, 0, 0], [255, 255, 255]]], dtype=np.uint8)
        output_img = np.array([[[128, 128, 128]]], dtype=np.uint8)
        mask = np.array([[255]], dtype=np.uint8)
        valid, pct = pixel_origin_check(input_img, output_img, mask)
        assert valid is False
        assert pct == 0.0

    def test_empty_mask(self) -> None:
        """No opaque pixels = valid."""
        input_img = np.zeros((10, 10, 3), dtype=np.uint8)
        output_img = np.full((10, 10, 3), 128, dtype=np.uint8)
        mask = np.zeros((10, 10), dtype=np.uint8)
        valid, pct = pixel_origin_check(input_img, output_img, mask)
        assert valid is True

    def test_none_mask(self) -> None:
        input_img = np.zeros((10, 10, 3), dtype=np.uint8)
        output_img = np.zeros((10, 10, 3), dtype=np.uint8)
        valid, pct = pixel_origin_check(input_img, output_img, None)
        assert valid is True
