from __future__ import annotations

import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def compute_ssim(extracted: np.ndarray, ground_truth: np.ndarray) -> float:
    """Compute structural similarity index between extracted and ground truth.

    Both inputs should be uint8 BGR images of the same dimensions.
    Returns a float in [0, 1] where 1 means identical.
    """
    # Convert to grayscale for SSIM if color
    if extracted.ndim == 3:
        import cv2

        extracted_gray = cv2.cvtColor(extracted, cv2.COLOR_BGR2GRAY)
        gt_gray = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2GRAY)
    else:
        extracted_gray = extracted
        gt_gray = ground_truth

    return float(structural_similarity(gt_gray, extracted_gray, data_range=255))


def compute_psnr(extracted: np.ndarray, ground_truth: np.ndarray) -> float:
    """Compute peak signal-to-noise ratio between extracted and ground truth.

    Returns float in dB. Higher is better. Returns inf for identical images.
    """
    return float(peak_signal_noise_ratio(ground_truth, extracted, data_range=255))


def compute_mask_iou(predicted: np.ndarray, ground_truth: np.ndarray) -> float:
    """Compute intersection over union between two binary masks.

    Returns float in [0, 1] where 1 means perfect overlap.
    """
    pred_bool = predicted > 0
    gt_bool = ground_truth > 0

    intersection = np.logical_and(pred_bool, gt_bool).sum()
    union = np.logical_or(pred_bool, gt_bool).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)


def pixel_origin_check(
    input_img: np.ndarray,
    output_img: np.ndarray,
    output_mask: np.ndarray,
) -> tuple[bool, float]:
    """Verify every output pixel originated from the input.

    Checks that all opaque output pixels exist somewhere in the input image.
    This enforces the core invariant: no pixel hallucination.

    Returns (all_valid, percentage_valid).
    """
    if output_mask is None:
        return True, 100.0

    # Get opaque output pixels
    opaque = output_mask > 0
    if not np.any(opaque):
        return True, 100.0

    # Flatten input pixel colors for lookup
    input_flat = input_img.reshape(-1, input_img.shape[2])
    input_set = set(map(tuple, input_flat))

    # Check each opaque output pixel
    output_opaque = output_img[opaque]
    total = len(output_opaque)
    valid = sum(1 for pixel in output_opaque if tuple(pixel) in input_set)

    percentage = (valid / total) * 100 if total > 0 else 100.0
    return valid == total, percentage
