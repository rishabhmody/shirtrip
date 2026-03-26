from __future__ import annotations

import logging
import time

import cv2
import numpy as np

from shirtrip.config.settings import Settings
from shirtrip.pipeline.errors import StageError
from shirtrip.pipeline.types import PipelineImage, PipelineResult, StageMetadata

logger = logging.getLogger(__name__)


def _compute_displacement_from_depth(
    depth: np.ndarray,
    mask: np.ndarray,
    strength: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute displacement maps from a depth map.

    Uses depth gradients to estimate surface curvature and generate
    displacement fields that flatten the surface.

    Returns (map_x, map_y) suitable for cv2.remap().
    """
    h, w = depth.shape

    # Create identity maps
    map_x = np.arange(w, dtype=np.float32)[np.newaxis, :].repeat(h, axis=0)
    map_y = np.arange(h, dtype=np.float32)[:, np.newaxis].repeat(w, axis=1)

    if strength < 1e-6:
        return map_x, map_y

    # Smooth depth to reduce noise before computing gradients
    smoothed = cv2.GaussianBlur(depth, (7, 7), sigmaX=2.0)

    # Compute gradients (surface slope)
    grad_x = np.gradient(smoothed, axis=1).astype(np.float32)
    grad_y = np.gradient(smoothed, axis=0).astype(np.float32)

    # Apply mask
    mask_float = (mask > 0).astype(np.float32)
    grad_x *= mask_float
    grad_y *= mask_float

    # Scale gradients to displacement
    # The displacement corrects for surface curvature by moving pixels
    # opposite to the depth gradient direction
    scale = strength * min(h, w) * 0.1
    map_x -= grad_x * scale
    map_y -= grad_y * scale

    # Clamp to valid image coordinates
    map_x = np.clip(map_x, 0, w - 1)
    map_y = np.clip(map_y, 0, h - 1)

    return map_x, map_y


def dewarp(
    image: PipelineImage,
    settings: Settings,
    depth_map: np.ndarray | None = None,
    strength: float = 1.0,
) -> PipelineResult:
    """Stage 3: Flatten garment graphic using depth-derived displacement.

    Uses cv2.remap() to move ORIGINAL pixels — no pixel synthesis.
    If no depth map is provided, passes through unchanged.
    """
    start = time.perf_counter()
    h, w = image.height, image.width

    try:
        if depth_map is None:
            logger.warning("No depth map provided — skipping dewarp")
            duration_ms = (time.perf_counter() - start) * 1000
            return PipelineResult(
                image=image,
                metadata=[
                    StageMetadata(
                        stage_name="dewarp",
                        duration_ms=duration_ms,
                        input_shape=image.bgr.shape,
                        output_shape=image.bgr.shape,
                        extra={"skipped": True},
                    )
                ],
            )

        mask = image.alpha if image.alpha is not None else np.ones((h, w), dtype=np.uint8) * 255

        # Compute displacement field from depth
        map_x, map_y = _compute_displacement_from_depth(depth_map, mask, strength)

        # Apply remap to ORIGINAL pixels — this is the key operation
        dewarped_bgr = cv2.remap(
            image.bgr, map_x, map_y,
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        # Also remap the alpha channel
        dewarped_alpha = None
        if image.alpha is not None:
            dewarped_alpha = cv2.remap(
                image.alpha, map_x, map_y,
                interpolation=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

        duration_ms = (time.perf_counter() - start) * 1000
        logger.info("dewarp completed in %.1f ms", duration_ms)

        displacement_magnitude = float(np.mean(
            np.sqrt(
                (map_x - np.arange(w, dtype=np.float32)[np.newaxis, :].repeat(h, axis=0)) ** 2
                + (map_y - np.arange(h, dtype=np.float32)[:, np.newaxis].repeat(w, axis=1)) ** 2
            )
        ))

        return PipelineResult(
            image=PipelineImage(bgr=dewarped_bgr, alpha=dewarped_alpha),
            metadata=[
                StageMetadata(
                    stage_name="dewarp",
                    duration_ms=duration_ms,
                    input_shape=image.bgr.shape,
                    output_shape=dewarped_bgr.shape,
                    extra={"displacement_magnitude": displacement_magnitude},
                )
            ],
            masks={"displacement_x": map_x, "displacement_y": map_y},
        )

    except Exception as e:
        raise StageError("dewarp", str(e), cause=e) from e
