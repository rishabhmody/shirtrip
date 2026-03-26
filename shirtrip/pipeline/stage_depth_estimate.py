from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import torch

from shirtrip.config.settings import Settings
from shirtrip.models.model_registry import ModelRegistry
from shirtrip.pipeline.errors import StageError
from shirtrip.pipeline.image_utils import bgr_to_rgb
from shirtrip.pipeline.types import PipelineImage, PipelineResult, StageMetadata

logger = logging.getLogger(__name__)

DEPTH_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


def _get_depth_model(settings: Settings) -> tuple[Any, Any]:
    """Load Depth Anything V2 processor and model via the registry."""
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    registry = ModelRegistry.get()

    def _load() -> tuple[Any, Any]:
        processor = AutoImageProcessor.from_pretrained(
            DEPTH_MODEL_ID, cache_dir=str(settings.model_cache_dir)
        )
        model = AutoModelForDepthEstimation.from_pretrained(
            DEPTH_MODEL_ID, cache_dir=str(settings.model_cache_dir)
        )
        model.to(settings.device)
        model.eval()
        return processor, model

    return registry.load("depth_anything", _load)


def depth_estimate(image: PipelineImage, settings: Settings) -> PipelineResult:
    """Stage 2: Estimate depth map of the garment region.

    Produces a normalized [0, 1] float32 depth map stored in result.masks['depth'].
    The image itself is passed through unchanged.
    """
    start = time.perf_counter()
    h, w = image.height, image.width

    try:
        processor, model = _get_depth_model(settings)

        # Convert BGR to RGB for the model
        rgb = bgr_to_rgb(image.bgr)

        with torch.inference_mode():
            inputs = processor(images=rgb, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(settings.device)
            outputs = model(pixel_values=pixel_values)

        # Extract and normalize depth map
        predicted_depth = outputs.predicted_depth.squeeze().cpu().numpy()

        # Resize to match input dimensions
        if predicted_depth.shape != (h, w):
            import cv2

            predicted_depth = cv2.resize(
                predicted_depth, (w, h), interpolation=cv2.INTER_LINEAR
            )

        # Normalize to [0, 1]
        d_min = predicted_depth.min()
        d_max = predicted_depth.max()
        if d_max - d_min > 1e-6:
            depth_map = ((predicted_depth - d_min) / (d_max - d_min)).astype(np.float32)
        else:
            depth_map = np.zeros((h, w), dtype=np.float32)

        # Apply mask if available (only compute depth within garment region)
        if image.alpha is not None:
            depth_map = depth_map * (image.alpha > 0).astype(np.float32)

        duration_ms = (time.perf_counter() - start) * 1000
        logger.info("depth_estimate completed in %.1f ms", duration_ms)

        return PipelineResult(
            image=image,  # Image passes through unchanged
            metadata=[
                StageMetadata(
                    stage_name="depth_estimate",
                    duration_ms=duration_ms,
                    input_shape=image.bgr.shape,
                    output_shape=image.bgr.shape,
                    extra={
                        "depth_min": float(depth_map.min()),
                        "depth_max": float(depth_map.max()),
                        "depth_std": float(np.std(depth_map)),
                    },
                )
            ],
            masks={"depth": depth_map},
        )

    except Exception as e:
        raise StageError("depth_estimate", str(e), cause=e) from e
