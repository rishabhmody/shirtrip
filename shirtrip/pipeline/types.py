from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class BBox:
    x: int
    y: int
    w: int
    h: int


@dataclass(frozen=True)
class PipelineImage:
    """Wraps a BGR image with an optional alpha channel."""

    bgr: np.ndarray  # HWC uint8 BGR
    alpha: np.ndarray | None = None  # HW uint8, None if fully opaque

    @property
    def height(self) -> int:
        return self.bgr.shape[0]

    @property
    def width(self) -> int:
        return self.bgr.shape[1]

    @property
    def rgba(self) -> np.ndarray:
        """Return BGRA image (4-channel)."""
        if self.alpha is not None:
            return np.dstack([self.bgr, self.alpha])
        return np.dstack([self.bgr, np.full(self.bgr.shape[:2], 255, dtype=np.uint8)])


@dataclass
class StageMetadata:
    stage_name: str
    duration_ms: float
    input_shape: tuple[int, ...]
    output_shape: tuple[int, ...]
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class PipelineResult:
    image: PipelineImage
    metadata: list[StageMetadata] = field(default_factory=list)
    masks: dict[str, np.ndarray] = field(default_factory=dict)
