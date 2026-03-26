from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from shirtrip.config.settings import Settings
from shirtrip.pipeline.errors import StageError
from shirtrip.pipeline.types import PipelineImage, PipelineResult, StageMetadata

logger = logging.getLogger(__name__)


def _clahe_normalize(image_bgr: np.ndarray, clip_limit: float = 2.0) -> np.ndarray:
    """Apply CLAHE to the L channel in LAB color space.

    This normalizes local contrast without destroying color information.
    """
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    l_corrected = clahe.apply(l_channel)

    lab_corrected = cv2.merge([l_corrected, a_channel, b_channel])
    return cv2.cvtColor(lab_corrected, cv2.COLOR_LAB2BGR)


def _depth_shadow_correction(
    image_bgr: np.ndarray,
    depth_map: np.ndarray,
    mask: np.ndarray,
    strength: float = 0.5,
) -> np.ndarray:
    """Correct shadows using depth information.

    Deeper areas (higher depth values) tend to be in shadow.
    Apply a brightness correction proportional to depth.
    """
    if depth_map is None:
        return image_bgr

    h, w = image_bgr.shape[:2]
    if depth_map.shape != (h, w):
        depth_map = cv2.resize(depth_map, (w, h), interpolation=cv2.INTER_LINEAR)

    mask_float = (mask > 0).astype(np.float32) if mask is not None else np.ones((h, w), np.float32)

    # Compute surface normals from depth gradients
    grad_x = cv2.Sobel(depth_map, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(depth_map, cv2.CV_32F, 0, 1, ksize=3)

    # Surface facing away from light (assumed top-down) = more shadow
    # Using gradient magnitude as shadow indicator
    shadow_likelihood = np.sqrt(grad_x**2 + grad_y**2)
    shadow_likelihood = shadow_likelihood * mask_float

    # Normalize and create correction factor
    max_val = shadow_likelihood.max()
    if max_val > 1e-6:
        shadow_likelihood /= max_val

    # Correction: brighten shadowed areas
    correction = 1.0 + shadow_likelihood * strength
    correction = correction[:, :, np.newaxis]  # Broadcast to 3 channels

    corrected = (image_bgr.astype(np.float32) * correction).clip(0, 255).astype(np.uint8)
    return corrected


def illuminate(
    image: PipelineImage,
    settings: Settings,
    depth_map: np.ndarray | None = None,
    clip_limit: float = 2.0,
) -> PipelineResult:
    """Stage 4: Normalize lighting and remove shadows.

    Applies CLAHE for local contrast normalization and optional
    depth-based shadow correction. All operations are mathematical
    transforms on original pixel values.
    """
    start = time.perf_counter()

    try:
        result_bgr = image.bgr.copy()
        mask = image.alpha if image.alpha is not None else np.ones(
            (image.height, image.width), dtype=np.uint8
        ) * 255

        # Step 1: Depth-based shadow correction (if depth available)
        if depth_map is not None:
            result_bgr = _depth_shadow_correction(result_bgr, depth_map, mask)

        # Step 2: CLAHE on L channel
        result_bgr = _clahe_normalize(result_bgr, clip_limit)

        duration_ms = (time.perf_counter() - start) * 1000
        logger.info("illuminate completed in %.1f ms", duration_ms)

        return PipelineResult(
            image=PipelineImage(bgr=result_bgr, alpha=image.alpha),
            metadata=[
                StageMetadata(
                    stage_name="illuminate",
                    duration_ms=duration_ms,
                    input_shape=image.bgr.shape,
                    output_shape=result_bgr.shape,
                    extra={"clip_limit": clip_limit, "used_depth": depth_map is not None},
                )
            ],
        )

    except Exception as e:
        raise StageError("illuminate", str(e), cause=e) from e
