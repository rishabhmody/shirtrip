from __future__ import annotations

import logging
import time
from typing import Callable

import torch

from shirtrip.config.settings import Settings
from shirtrip.pipeline.errors import PipelineError, StageError
from shirtrip.pipeline.stage_garment_parse import garment_parse
from shirtrip.pipeline.types import PipelineImage, PipelineResult, StageMetadata

logger = logging.getLogger(__name__)

StageFunction = Callable[[PipelineImage, Settings], PipelineResult]


def _wrap_depth_estimate(image: PipelineImage, settings: Settings) -> PipelineResult:
    from shirtrip.pipeline.stage_depth_estimate import depth_estimate

    return depth_estimate(image, settings)


def _wrap_dewarp(image: PipelineImage, settings: Settings, **kwargs) -> PipelineResult:
    from shirtrip.pipeline.stage_dewarp import dewarp

    return dewarp(image, settings, **kwargs)


def _wrap_illuminate(image: PipelineImage, settings: Settings, **kwargs) -> PipelineResult:
    from shirtrip.pipeline.stage_illuminate import illuminate

    return illuminate(image, settings, **kwargs)


def _wrap_alpha_matte(image: PipelineImage, settings: Settings) -> PipelineResult:
    from shirtrip.pipeline.stage_alpha_matte import alpha_matte

    return alpha_matte(image, settings)


STAGE_REGISTRY: dict[str, StageFunction] = {
    "garment_parse": garment_parse,
    "depth_estimate": _wrap_depth_estimate,
    "dewarp": _wrap_dewarp,
    "illuminate": _wrap_illuminate,
    "alpha_matte": _wrap_alpha_matte,
}

DEFAULT_PIPELINE: list[str] = [
    "garment_parse",
    "depth_estimate",
    "dewarp",
    "illuminate",
    "alpha_matte",
]


def run_pipeline(
    image: PipelineImage,
    settings: Settings,
    stages: list[str] | None = None,
) -> PipelineResult:
    """Execute pipeline stages sequentially.

    Each stage receives the output image from the previous stage.
    Metadata is accumulated across all stages.
    """
    stage_names = stages if stages is not None else DEFAULT_PIPELINE
    current_image = image
    all_metadata: list[StageMetadata] = []
    all_masks: dict[str, any] = {}

    logger.info("Starting pipeline with stages: %s", stage_names)
    pipeline_start = time.perf_counter()

    for stage_name in stage_names:
        if stage_name not in STAGE_REGISTRY:
            raise StageError(stage_name, f"Unknown stage: {stage_name}")

        stage_fn = STAGE_REGISTRY[stage_name]
        logger.info("Running stage: %s", stage_name)

        try:
            # Special handling: stages that need accumulated masks
            if stage_name in ("dewarp", "illuminate"):
                result = stage_fn(
                    current_image, settings,
                    depth_map=all_masks.get("depth"),
                )
            else:
                result = stage_fn(current_image, settings)
        except PipelineError:
            raise
        except Exception as e:
            raise StageError(stage_name, str(e), cause=e) from e

        current_image = result.image
        all_metadata.extend(result.metadata)
        all_masks.update(result.masks)

        # Free GPU memory between stages
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    total_ms = (time.perf_counter() - pipeline_start) * 1000
    logger.info("Pipeline completed in %.1f ms", total_ms)

    return PipelineResult(
        image=current_image,
        metadata=all_metadata,
        masks=all_masks,
    )
