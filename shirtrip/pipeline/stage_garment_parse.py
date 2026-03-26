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
GRAPHIC_COLOR_THRESHOLD = 30  # Min color distance from fabric to count as graphic
GRAPHIC_SATURATION_THRESHOLD = 20  # Min saturation to be considered "colored" (not shadow)


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


def _estimate_fabric_color(image_bgr: np.ndarray, garment_mask: np.ndarray) -> np.ndarray:
    """Estimate the base fabric color by sampling the edges of the garment mask.

    The graphic is typically in the center of the shirt, so edge pixels
    of the garment region are more likely to be plain fabric.
    """
    h, w = garment_mask.shape

    # Erode the mask significantly to get the interior
    erode_size = max(h, w) // 6
    if erode_size < 5:
        erode_size = 5
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_size, erode_size))
    interior = cv2.erode(garment_mask, kernel, iterations=1)

    # The fabric sampling region = garment mask minus the interior (border strip)
    border = cv2.bitwise_and(garment_mask, cv2.bitwise_not(interior))

    border_pixels = image_bgr[border > 0]

    if len(border_pixels) < 50:
        # Fallback: use the garment mask corners / top and bottom edges
        garment_pixels = image_bgr[garment_mask > 0]
        if len(garment_pixels) == 0:
            return np.array([255, 255, 255], dtype=np.uint8)
        return np.median(garment_pixels, axis=0).astype(np.uint8)

    # Use median to be robust against any graphic that bleeds to edges
    return np.median(border_pixels, axis=0).astype(np.uint8)


def _classify_shadow_vs_graphic(
    image_bgr: np.ndarray,
    color_distance: np.ndarray,
    fabric_lab_l: float,
) -> np.ndarray:
    """Distinguish wrinkle shadows from actual graphic ink.

    Shadows are characterized by:
    - Low saturation (they're just darker versions of the fabric color)
    - Luminance drop without significant chrominance change

    Graphics have:
    - Higher saturation OR significant chrominance shift
    """
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1].astype(np.float32)

    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_channel = lab[:, :, 0]
    a_channel = lab[:, :, 1]
    b_channel = lab[:, :, 2]

    # Chrominance distance from fabric (a,b channels only — ignoring luminance)
    fabric_a = 128.0  # Neutral a (will be overridden below)
    fabric_b = 128.0  # Neutral b

    # Use the median a,b of low-color-distance pixels as the fabric chrominance
    fabric_region = color_distance < 15
    if np.sum(fabric_region) > 100:
        fabric_a = np.median(a_channel[fabric_region])
        fabric_b = np.median(b_channel[fabric_region])

    chroma_dist = np.sqrt((a_channel - fabric_a) ** 2 + (b_channel - fabric_b) ** 2)

    # A pixel is likely GRAPHIC (not shadow) if:
    # 1. It has significant saturation, OR
    # 2. It has significant chrominance shift from fabric, OR
    # 3. It is much darker than fabric AND has some color (ink, not shadow)
    is_saturated = saturation > GRAPHIC_SATURATION_THRESHOLD
    has_chroma = chroma_dist > 12
    is_very_dark = l_channel < (fabric_lab_l * 0.4)  # Very dark = likely black ink

    graphic_confidence = np.zeros_like(color_distance, dtype=np.float32)
    graphic_confidence[is_saturated] += 0.4
    graphic_confidence[has_chroma] += 0.4
    graphic_confidence[is_very_dark] += 0.3

    return graphic_confidence


def _isolate_graphic(
    image_bgr: np.ndarray,
    garment_mask: np.ndarray,
    fabric_color: np.ndarray,
    threshold: int = GRAPHIC_COLOR_THRESHOLD,
) -> np.ndarray:
    """Isolate the graphic within the garment by finding pixels that differ
    significantly from the base fabric color, while filtering out wrinkle shadows.

    Uses LAB color space for perceptual distance and HSV for shadow detection.
    """
    h, w = image_bgr.shape[:2]

    # Convert to LAB for perceptual color distance
    image_lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    fabric_lab = cv2.cvtColor(
        fabric_color.reshape(1, 1, 3), cv2.COLOR_BGR2LAB
    ).astype(np.float32).squeeze()

    # Compute per-pixel color distance from fabric
    diff = image_lab - fabric_lab
    color_distance = np.sqrt(np.sum(diff ** 2, axis=2))

    # Classify shadow vs graphic
    graphic_confidence = _classify_shadow_vs_graphic(
        image_bgr, color_distance, fabric_lab[0]
    )

    # A pixel is graphic if:
    # - color distance exceeds threshold AND
    # - graphic confidence is above minimum (not just a shadow)
    graphic_mask = np.zeros((h, w), dtype=np.uint8)
    is_different = color_distance > threshold
    is_graphic = graphic_confidence > 0.3
    graphic_mask[is_different & is_graphic] = 255

    # Also include pixels with very high color distance regardless of confidence
    # (catches dark graphic elements that might look like shadows)
    very_high_distance = color_distance > (threshold * 2.5)
    graphic_mask[very_high_distance & (garment_mask > 0)] = 255

    # Only keep graphic pixels within the garment
    graphic_mask = cv2.bitwise_and(graphic_mask, garment_mask)

    # Morphological cleanup: close gaps within the graphic
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    graphic_mask = cv2.morphologyEx(graphic_mask, cv2.MORPH_CLOSE, kernel_close)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    graphic_mask = cv2.morphologyEx(graphic_mask, cv2.MORPH_OPEN, kernel_open)

    # Remove small connected components (noise, stray shadow patches)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        graphic_mask, connectivity=8
    )
    if num_labels > 1:
        garment_area = np.sum(garment_mask > 0)
        min_component_area = max(garment_area * 0.005, 100)

        # Find the largest component (main graphic body)
        areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)]
        max_area = max(areas) if areas else 0

        for label_idx in range(1, num_labels):
            area = stats[label_idx, cv2.CC_STAT_AREA]
            if area < min_component_area:
                # Small component: check if it's near the main graphic
                # (keep nearby small parts like dots, remove distant noise)
                graphic_mask[labels == label_idx] = 0

    # Fill holes within the graphic (e.g., light-colored areas inside the design)
    # Use convex hull of the main components to fill interior gaps
    contours, _ = cv2.findContours(graphic_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Create a filled version and blend with original
        filled = np.zeros_like(graphic_mask)
        cv2.drawContours(filled, contours, -1, 255, -1)

        # Only fill internal holes — don't expand the boundary
        # Use the original mask OR internal fill (closes interior gaps)
        graphic_mask = cv2.bitwise_or(graphic_mask, cv2.bitwise_and(filled, garment_mask))

    # Dilate to capture anti-aliased edges
    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    graphic_mask = cv2.dilate(graphic_mask, kernel_dilate, iterations=1)

    # Re-intersect with garment mask
    graphic_mask = cv2.bitwise_and(graphic_mask, garment_mask)

    return graphic_mask


def garment_parse(image: PipelineImage, settings: Settings) -> PipelineResult:
    """Stage 1: Segment upper garment and extract the graphic design.

    Two-phase approach:
    1. SegFormer segments the shirt region
    2. Color-distance analysis isolates the printed graphic within the shirt

    All output pixels come from the original input image.
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

        garment_mask = _extract_garment_mask(outputs.logits, h, w)

        # Check if garment was found
        garment_area = np.sum(garment_mask > 0)
        total_area = h * w
        if garment_area < total_area * MIN_MASK_AREA_RATIO:
            raise GarmentNotFoundError()

        # Phase 2: Isolate the graphic within the garment
        fabric_color = _estimate_fabric_color(image.bgr, garment_mask)
        logger.info(
            "Estimated fabric color (BGR): %s",
            fabric_color.tolist(),
        )

        graphic_mask = _isolate_graphic(image.bgr, garment_mask, fabric_color)

        graphic_area = np.sum(graphic_mask > 0)
        if graphic_area < total_area * MIN_MASK_AREA_RATIO:
            raise GarmentNotFoundError("No graphic design detected on the garment")

        logger.info(
            "Graphic isolation: garment=%d px, graphic=%d px (%.1f%% of garment)",
            garment_area,
            graphic_area,
            100 * graphic_area / garment_area,
        )

        # Crop to the graphic region
        cropped_bgr, cropped_mask, bbox = crop_to_content(image.bgr, graphic_mask)

        # Create output with alpha from the graphic mask
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
                    extra={
                        "bbox": (bbox.x, bbox.y, bbox.w, bbox.h),
                        "garment_area": int(garment_area),
                        "graphic_area": int(graphic_area),
                        "fabric_color_bgr": fabric_color.tolist(),
                    },
                )
            ],
            masks={
                "garment": garment_mask,
                "graphic": graphic_mask,
                "garment_cropped": cropped_mask,
            },
        )

    except GarmentNotFoundError:
        raise
    except Exception as e:
        raise StageError("garment_parse", str(e), cause=e) from e
