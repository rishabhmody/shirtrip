from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from shirtrip.config.settings import Settings
from shirtrip.pipeline.image_utils import to_pipeline_image
from shirtrip.pipeline.orchestrator import run_pipeline

from .metrics import compute_mask_iou, compute_psnr, compute_ssim, pixel_origin_check

logger = logging.getLogger(__name__)


def run_benchmark(
    image_dir: Path,
    settings: Settings,
    ground_truth_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Run the pipeline on all images in a directory and compute metrics.

    Returns a list of dicts with per-image metrics.
    """
    results: list[dict[str, Any]] = []
    image_extensions = {".jpg", ".jpeg", ".png", ".webp"}

    image_files = sorted(
        f for f in image_dir.iterdir()
        if f.suffix.lower() in image_extensions
    )

    if not image_files:
        logger.warning("No images found in %s", image_dir)
        return results

    for img_path in image_files:
        logger.info("Benchmarking: %s", img_path.name)
        row: dict[str, Any] = {"fixture_name": img_path.stem}

        try:
            bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
            if bgr is None:
                row["error"] = "Could not read image"
                results.append(row)
                continue

            pipeline_image = to_pipeline_image(bgr)

            start = time.perf_counter()
            result = run_pipeline(pipeline_image, settings)
            total_ms = (time.perf_counter() - start) * 1000

            row["processing_time_ms"] = total_ms
            row["stages"] = [m.stage_name for m in result.metadata]

            # Per-stage timing
            for meta in result.metadata:
                row[f"{meta.stage_name}_ms"] = meta.duration_ms

            # Ground truth comparison
            if ground_truth_dir is not None:
                gt_path = ground_truth_dir / f"{img_path.stem}.png"
                if gt_path.exists():
                    gt = cv2.imread(str(gt_path), cv2.IMREAD_COLOR)
                    if gt is not None and gt.shape[:2] == result.image.bgr.shape[:2]:
                        row["ssim"] = compute_ssim(result.image.bgr, gt)
                        row["psnr"] = compute_psnr(result.image.bgr, gt)

                gt_mask_path = ground_truth_dir / f"{img_path.stem}_mask.png"
                if gt_mask_path.exists():
                    gt_mask = cv2.imread(str(gt_mask_path), cv2.IMREAD_GRAYSCALE)
                    if gt_mask is not None and "garment" in result.masks:
                        pred_mask = result.masks["garment"]
                        if gt_mask.shape == pred_mask.shape:
                            row["mask_iou"] = compute_mask_iou(pred_mask, gt_mask)

            # Pixel origin check
            alpha = result.image.alpha
            if alpha is not None:
                all_valid, pct = pixel_origin_check(bgr, result.image.bgr, alpha)
                row["pixel_origin_valid"] = all_valid
                row["pixel_origin_pct"] = pct

        except Exception as e:
            row["error"] = str(e)
            logger.error("Benchmark failed for %s: %s", img_path.name, e)

        results.append(row)

    return results
