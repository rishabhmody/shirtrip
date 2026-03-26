from __future__ import annotations

import logging
import time
from typing import Any

import cv2
import numpy as np
import torch

from shirtrip.config.settings import Settings
from shirtrip.models.model_registry import ModelRegistry
from shirtrip.pipeline.errors import StageError
from shirtrip.pipeline.image_utils import bgr_to_rgb
from shirtrip.pipeline.types import PipelineImage, PipelineResult, StageMetadata

logger = logging.getLogger(__name__)

VITMATTE_MODEL_ID = "hustvl/vitmatte-small-composition-1k"


def _get_vitmatte(settings: Settings) -> tuple[Any, Any]:
    """Load ViTMatte processor and model via the registry."""
    from transformers import VitMatteForImageMatting, VitMatteImageProcessor

    registry = ModelRegistry.get()

    def _load() -> tuple[Any, Any]:
        processor = VitMatteImageProcessor.from_pretrained(
            VITMATTE_MODEL_ID, cache_dir=str(settings.model_cache_dir)
        )
        model = VitMatteForImageMatting.from_pretrained(
            VITMATTE_MODEL_ID, cache_dir=str(settings.model_cache_dir)
        )
        model.to(settings.device)
        model.eval()
        return processor, model

    return registry.load("vitmatte", _load)


def _generate_trimap(
    mask: np.ndarray, erode_px: int = 10, dilate_px: int = 20
) -> np.ndarray:
    """Generate a trimap from a binary mask.

    Returns uint8 array: 0=background, 128=unknown, 255=foreground.
    """
    kernel_erode = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px, erode_px))
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dilate_px, dilate_px))

    # Definite foreground: eroded mask
    foreground = cv2.erode(mask, kernel_erode, iterations=1)
    # Definite background: outside dilated mask
    dilated = cv2.dilate(mask, kernel_dilate, iterations=1)

    trimap = np.zeros_like(mask, dtype=np.uint8)
    trimap[foreground > 0] = 255  # Definite foreground
    trimap[(dilated > 0) & (foreground == 0)] = 128  # Unknown region
    # Everything else stays 0 (background)

    return trimap


def alpha_matte(image: PipelineImage, settings: Settings) -> PipelineResult:
    """Stage 5: Generate soft alpha matte for clean edges.

    Uses ViTMatte to refine the binary mask into a soft alpha matte.
    Only the alpha channel is modified — BGR pixel values are unchanged.
    """
    start = time.perf_counter()
    h, w = image.height, image.width

    try:
        if image.alpha is None:
            logger.warning("No alpha channel — skipping alpha matte refinement")
            duration_ms = (time.perf_counter() - start) * 1000
            return PipelineResult(
                image=image,
                metadata=[
                    StageMetadata(
                        stage_name="alpha_matte",
                        duration_ms=duration_ms,
                        input_shape=image.bgr.shape,
                        output_shape=image.bgr.shape,
                        extra={"skipped": True},
                    )
                ],
            )

        processor, model = _get_vitmatte(settings)

        # Generate trimap from the binary mask
        trimap = _generate_trimap(image.alpha)

        # Convert image for the model
        from PIL import Image

        rgb = bgr_to_rgb(image.bgr)
        pil_image = Image.fromarray(rgb)
        pil_trimap = Image.fromarray(trimap)

        with torch.inference_mode():
            inputs = processor(images=pil_image, trimaps=pil_trimap, return_tensors="pt")
            inputs = {k: v.to(settings.device) for k, v in inputs.items()}
            outputs = model(**inputs)

        # Extract alpha matte
        alpha_tensor = outputs.alphas.squeeze().cpu().numpy()

        # Resize to match input if needed
        if alpha_tensor.shape != (h, w):
            alpha_tensor = cv2.resize(alpha_tensor, (w, h), interpolation=cv2.INTER_LINEAR)

        # Convert to uint8 [0, 255]
        soft_alpha = (alpha_tensor.clip(0, 1) * 255).astype(np.uint8)

        duration_ms = (time.perf_counter() - start) * 1000
        logger.info("alpha_matte completed in %.1f ms", duration_ms)

        return PipelineResult(
            image=PipelineImage(bgr=image.bgr, alpha=soft_alpha),
            metadata=[
                StageMetadata(
                    stage_name="alpha_matte",
                    duration_ms=duration_ms,
                    input_shape=image.bgr.shape,
                    output_shape=image.bgr.shape,
                    extra={"trimap_unknown_pixels": int(np.sum(trimap == 128))},
                )
            ],
            masks={"trimap": trimap, "soft_alpha": soft_alpha},
        )

    except Exception as e:
        raise StageError("alpha_matte", str(e), cause=e) from e
