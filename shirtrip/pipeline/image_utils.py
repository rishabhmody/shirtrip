from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

from shirtrip.pipeline.types import BBox, PipelineImage


def rgb_to_bgr(img: np.ndarray) -> np.ndarray:
    """Convert RGB image to BGR."""
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def bgr_to_rgb(img: np.ndarray) -> np.ndarray:
    """Convert BGR image to RGB."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def pil_to_bgr(img: Image.Image) -> np.ndarray:
    """Convert PIL Image to BGR numpy array."""
    rgb = np.array(img.convert("RGB"))
    return rgb_to_bgr(rgb)


def bgr_to_pil(img: np.ndarray) -> Image.Image:
    """Convert BGR numpy array to PIL Image."""
    rgb = bgr_to_rgb(img)
    return Image.fromarray(rgb)


def apply_mask(image_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Apply a binary mask to a BGR image, returning BGRA.

    Masked pixels get alpha=255, unmasked get alpha=0.
    """
    alpha = (mask > 0).astype(np.uint8) * 255
    return np.dstack([image_bgr, alpha])


def crop_to_content(
    image_bgr: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray, BBox]:
    """Crop image and mask to the bounding box of non-zero mask pixels.

    Returns (cropped_bgr, cropped_mask, bbox).
    """
    coords = cv2.findNonZero(mask.astype(np.uint8))
    if coords is None:
        return image_bgr, mask, BBox(0, 0, image_bgr.shape[1], image_bgr.shape[0])
    x, y, w, h = cv2.boundingRect(coords)
    return image_bgr[y : y + h, x : x + w], mask[y : y + h, x : x + w], BBox(x, y, w, h)


def load_image(path: str) -> np.ndarray:
    """Load an image from disk as BGR, preserving alpha if present."""
    return cv2.imread(path, cv2.IMREAD_UNCHANGED)


def save_rgba_png(image_bgra: np.ndarray, path: str) -> None:
    """Save a BGRA image as a PNG file."""
    cv2.imwrite(path, image_bgra)


def resize_if_needed(image: np.ndarray, max_dim: int) -> np.ndarray:
    """Downscale image if its largest dimension exceeds max_dim."""
    h, w = image.shape[:2]
    if max(h, w) <= max_dim:
        return image
    scale = max_dim / max(h, w)
    new_w, new_h = int(w * scale), int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def to_pipeline_image(bgr: np.ndarray, alpha: np.ndarray | None = None) -> PipelineImage:
    """Create a PipelineImage from raw arrays."""
    return PipelineImage(bgr=bgr, alpha=alpha)
