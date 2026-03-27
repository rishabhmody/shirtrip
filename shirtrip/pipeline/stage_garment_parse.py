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
from shirtrip.pipeline.image_utils import bgr_to_rgb
from shirtrip.pipeline.types import BBox, PipelineImage, PipelineResult, StageMetadata

logger = logging.getLogger(__name__)

SEGFORMER_MODEL_ID = "mattmdjaga/segformer_b2_clothes"
UPPER_CLOTHES_LABEL = 4
MIN_MASK_AREA_RATIO = 0.005


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

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


def _get_grounding_dino(settings: Settings) -> tuple[Any, Any]:
    """Load Grounding DINO processor and model via the registry."""
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

    registry = ModelRegistry.get()
    model_id = settings.dino_model_id

    def _load() -> tuple[Any, Any]:
        processor = AutoProcessor.from_pretrained(
            model_id, cache_dir=str(settings.model_cache_dir)
        )
        model = AutoModelForZeroShotObjectDetection.from_pretrained(
            model_id, cache_dir=str(settings.model_cache_dir)
        )
        model.to(settings.device)
        model.eval()
        return processor, model

    return registry.load("grounding_dino", _load)


def _get_sam(settings: Settings) -> tuple[Any, Any]:
    """Load SAM processor and model via the registry."""
    from transformers import SamModel, SamProcessor

    registry = ModelRegistry.get()
    model_id = settings.sam_model_id

    def _load() -> tuple[Any, Any]:
        processor = SamProcessor.from_pretrained(
            model_id, cache_dir=str(settings.model_cache_dir)
        )
        model = SamModel.from_pretrained(
            model_id, cache_dir=str(settings.model_cache_dir)
        )
        model.to(settings.device)
        model.eval()
        return processor, model

    return registry.load("sam", _load)


def _unload_model(key: str) -> None:
    """Unload a model from the registry and free GPU memory."""
    registry = ModelRegistry.get()
    registry.unload(key)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Stage helpers
# ---------------------------------------------------------------------------

def _extract_garment_mask(logits: torch.Tensor, h: int, w: int) -> np.ndarray:
    """Extract upper-clothes mask from SegFormer logits."""
    upsampled = torch.nn.functional.interpolate(
        logits, size=(h, w), mode="bilinear", align_corners=False
    )
    seg_map = upsampled.argmax(dim=1).squeeze().cpu().numpy()
    mask = (seg_map == UPPER_CLOTHES_LABEL).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    return mask


def _detect_graphic_boxes(
    image_rgb: np.ndarray,
    garment_mask: np.ndarray,
    settings: Settings,
) -> list[list[int]]:
    """Run Grounding DINO to detect graphic elements within the garment region.

    Returns a list of [x1, y1, x2, y2] boxes (pixel coords) that overlap
    significantly with the garment mask.
    """
    from PIL import Image as PILImage

    processor, model = _get_grounding_dino(settings)
    pil_image = PILImage.fromarray(image_rgb)
    h, w = image_rgb.shape[:2]

    with torch.inference_mode():
        inputs = processor(
            images=pil_image,
            text=settings.graphic_prompt,
            return_tensors="pt",
        ).to(settings.device)
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        threshold=settings.dino_box_threshold,
        text_threshold=settings.dino_text_threshold,
        target_sizes=[(h, w)],
    )[0]

    boxes = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    labels = results["labels"]

    logger.info(
        "DINO raw detections: %d boxes (threshold=%.2f, text_threshold=%.2f)",
        len(boxes), settings.dino_box_threshold, settings.dino_text_threshold,
    )

    filtered_boxes: list[list[int]] = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        if x2 <= x1 or y2 <= y1:
            logger.info("DINO detection %d: degenerate box [%d,%d,%d,%d], skipping", i, x1, y1, x2, y2)
            continue

        # Check overlap with garment mask
        box_region = garment_mask[y1:y2, x1:x2]
        overlap_ratio = np.sum(box_region > 0) / ((x2 - x1) * (y2 - y1))

        logger.info(
            "DINO detection %d: [%d,%d,%d,%d] score=%.3f label='%s' "
            "garment_overlap=%.1f%%",
            i, x1, y1, x2, y2, scores[i], labels[i], overlap_ratio * 100,
        )

        if overlap_ratio > 0.1:
            filtered_boxes.append([x1, y1, x2, y2])

    return filtered_boxes


def _segment_graphic_with_sam(
    image_rgb: np.ndarray,
    boxes: list[list[int]],
    settings: Settings,
) -> np.ndarray:
    """Run SAM with box prompts to produce a pixel-accurate graphic mask.

    Each DINO box is used as a box prompt for SAM.  The best-scoring mask
    from each prompt is kept, and all are unioned into a single mask.
    """
    from PIL import Image as PILImage

    h, w = image_rgb.shape[:2]
    processor, model = _get_sam(settings)
    pil_image = PILImage.fromarray(image_rgb)

    combined_mask = np.zeros((h, w), dtype=np.uint8)

    # Process each box individually for reliability (avoids batching edge cases)
    for box in boxes:
        input_boxes = [[[float(c) for c in box]]]  # [batch[image[box]]]

        with torch.inference_mode():
            inputs = processor(
                pil_image,
                input_boxes=input_boxes,
                return_tensors="pt",
            ).to(settings.device)
            outputs = model(**inputs)

        masks = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )

        # masks[0] shape: (num_boxes, num_masks_per_box, H, W)
        iou_scores = outputs.iou_scores.cpu().numpy()[0]  # (num_boxes, 3)
        best_idx = int(np.argmax(iou_scores[0]))
        mask = masks[0][0, best_idx].numpy().astype(np.uint8) * 255
        combined_mask = cv2.bitwise_or(combined_mask, mask)

        logger.info(
            "SAM mask for box %s: best_iou=%.3f, mask_pixels=%d",
            box, iou_scores[0][best_idx], np.sum(mask > 0),
        )

    return combined_mask


def _merge_boxes(boxes: list[list[int]]) -> list[int]:
    """Merge multiple boxes into a single bounding box that encloses all."""
    if not boxes:
        return [0, 0, 0, 0]
    x1 = min(b[0] for b in boxes)
    y1 = min(b[1] for b in boxes)
    x2 = max(b[2] for b in boxes)
    y2 = max(b[3] for b in boxes)
    return [x1, y1, x2, y2]


# ---------------------------------------------------------------------------
# Main stage
# ---------------------------------------------------------------------------

def garment_parse(image: PipelineImage, settings: Settings) -> PipelineResult:
    """Stage 1: Segment upper garment and extract the graphic design.

    Three-phase approach:
    1. SegFormer segments the shirt region (garment mask)
    2. Grounding DINO detects graphic elements via text-prompted object detection
    3. SAM refines DINO boxes into pixel-accurate masks

    All output pixels come from the original input image.
    """
    start = time.perf_counter()
    h, w = image.height, image.width

    try:
        # --- Phase 1: SegFormer garment segmentation ---
        processor, model = _get_segformer(settings)
        rgb = bgr_to_rgb(image.bgr)

        with torch.inference_mode():
            inputs = processor(images=rgb, return_tensors="pt")
            pixel_values = inputs["pixel_values"].to(settings.device)
            outputs = model(pixel_values=pixel_values)

        garment_mask = _extract_garment_mask(outputs.logits, h, w)

        garment_area = np.sum(garment_mask > 0)
        total_area = h * w
        if garment_area < total_area * MIN_MASK_AREA_RATIO:
            raise GarmentNotFoundError()

        logger.info("Garment mask: %d px (%.1f%% of image)", garment_area, 100 * garment_area / total_area)

        if settings.sequential_gpu_loading:
            _unload_model("segformer")

        # --- Phase 2: Grounding DINO graphic detection ---
        dino_boxes = _detect_graphic_boxes(rgb, garment_mask, settings)

        if not dino_boxes:
            raise GarmentNotFoundError("No graphic design detected on the garment")

        if settings.sequential_gpu_loading:
            _unload_model("grounding_dino")

        # --- Phase 3: SAM mask refinement ---
        graphic_mask = _segment_graphic_with_sam(rgb, dino_boxes, settings)

        if settings.sequential_gpu_loading:
            _unload_model("sam")

        # Intersect with garment mask to remove any bleed outside the shirt
        graphic_mask = cv2.bitwise_and(graphic_mask, garment_mask)

        graphic_area = np.sum(graphic_mask > 0)
        if graphic_area < total_area * MIN_MASK_AREA_RATIO:
            raise GarmentNotFoundError("SAM mask too small — no valid graphic detected")

        logger.info(
            "Graphic mask: %d px (%.1f%% of garment)",
            graphic_area, 100 * graphic_area / garment_area,
        )

        # --- Crop and white background ---
        # Use the merged DINO bounding box (padded) as the crop region.
        # SAM mask provides pixel-accurate graphic vs fabric distinction.
        merged = _merge_boxes(dino_boxes)
        gx1, gy1, gx2, gy2 = merged
        gw, gh = gx2 - gx1, gy2 - gy1

        pad_x = max(int(gw * 0.10), 10)
        pad_y = max(int(gh * 0.10), 10)
        x1 = max(gx1 - pad_x, 0)
        y1 = max(gy1 - pad_y, 0)
        x2 = min(gx2 + pad_x, w)
        y2 = min(gy2 + pad_y, h)

        cropped_bgr = image.bgr[y1:y2, x1:x2].copy()
        cropped_mask = graphic_mask[y1:y2, x1:x2]
        bbox = BBox(x1, y1, x2 - x1, y2 - y1)

        # Replace non-graphic pixels with white
        white_bgr = cropped_bgr.copy()
        white_bgr[cropped_mask == 0] = [255, 255, 255]

        # Solid rectangular alpha — the full bounding box is opaque
        solid_alpha = np.full((y2 - y1, x2 - x1), 255, dtype=np.uint8)
        output_image = PipelineImage(bgr=white_bgr, alpha=solid_alpha)

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
                        "dino_boxes": dino_boxes,
                        "num_dino_detections": len(dino_boxes),
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
