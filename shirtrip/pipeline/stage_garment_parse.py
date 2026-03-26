from __future__ import annotations

import logging
import time
from typing import Any

import cv2
import numpy as np
import torch

from shirtrip.config.settings import Settings
from shirtrip.models.model_registry import ModelRegistry
from shirtrip.pipeline.errors import GarmentNotFoundError, StageError
from shirtrip.pipeline.image_utils import bgr_to_rgb, crop_to_content
from shirtrip.pipeline.types import PipelineImage, PipelineResult, StageMetadata

logger = logging.getLogger(__name__)

SEGFORMER_MODEL_ID = "mattmdjaga/segformer_b2_clothes"
UPPER_CLOTHES_LABEL = 4
MIN_MASK_AREA_RATIO = 0.005  # Minimum mask area as fraction of image area


def _get_segformer(settings: Settings) -> tuple[Any, Any]:
    """Load SegFormer processor and model via the registry."""
    from transformers import AutoModelForSemanticSegmentation, SegformerImageProcessor

    registry = ModelRegistry.get()

    def _load() -> tuple[Any, Any]:
        processor = SegformerImageProcessor.from_pretrained(
            SEGFORMER_MODEL_ID, cache_dir=str(settings.model_cache_dir)
        )
        model = AutoModelForSemanticSegmentation.from_pretrained(
            SEGFORMER_MODEL_ID, cache_dir=str(settings.model_cache_dir)
        )
        model.to(settings.device)
        model.eval()
        return processor, model

    return registry.load("segformer", _load)


def _extract_garment_mask(logits: torch.Tensor, h: int, w: int) -> np.ndarray:
    """Extract upper-clothes mask from SegFormer logits."""
    upsampled = torch.nn.functional.interpolate(
        logits, size=(h, w), mode="bilinear", align_corners=False
    )
    seg_map = upsampled.argmax(dim=1).squeeze().cpu().numpy()
    mask = (seg_map == UPPER_CLOTHES_LABEL).astype(np.uint8) * 255

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


def garment_parse(image: PipelineImage, settings: Settings) -> PipelineResult:
    """Stage 1: Segment upper garment and extract the graphic region.

    Uses SegFormer to identify upper-body clothing, then crops and masks
    the result. All output pixels come from the original input image.
    """
    start = time.perf_counter()
    h, w = image.height, image.width

    try:
        processor, model = _get_segformer(settings)

        # Convert BGR to RGB for the model
        rgb = bgr_to_rgb(image.bgr)

        # Run SegFormer inference
        with torch.inference_mode():
            inputs = processor(images=rgb, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(settings.device)
            outputs = model(pixel_values=pixel_values)

        mask = _extract_garment_mask(outputs.logits, h, w)

        # Check if garment was found
        mask_area = np.sum(mask > 0)
        total_area = h * w
        if mask_area < total_area * MIN_MASK_AREA_RATIO:
            raise GarmentNotFoundError()

        # Crop to the garment region
        cropped_bgr, cropped_mask, bbox = crop_to_content(image.bgr, mask)

        # Create output with alpha from the mask
        output_image = PipelineImage(bgr=cropped_bgr, alpha=cropped_mask)

        duration_ms = (time.perf_counter() - start) * 1000
        logger.info("garment_parse completed in %.1f ms", duration_ms)

        return PipelineResult(
            image=output_image,
            metadata=[
                StageMetadata(
                    stage_name="garment_parse",
                    duration_ms=duration_ms,
                    input_shape=image.bgr.shape,
                    output_shape=cropped_bgr.shape,
                    extra={"bbox": (bbox.x, bbox.y, bbox.w, bbox.h), "mask_area": int(mask_area)},
                )
            ],
            masks={"garment": mask, "garment_cropped": cropped_mask},
        )

    except GarmentNotFoundError:
        raise
    except Exception as e:
        raise StageError("garment_parse", str(e), cause=e) from e
