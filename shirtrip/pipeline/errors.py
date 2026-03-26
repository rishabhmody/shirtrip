from __future__ import annotations


class PipelineError(Exception):
    """Base exception for all pipeline errors."""


class StageError(PipelineError):
    """Error raised by a specific pipeline stage."""

    def __init__(self, stage_name: str, message: str, cause: Exception | None = None) -> None:
        self.stage_name = stage_name
        self.cause = cause
        super().__init__(f"[{stage_name}] {message}")


class ModelLoadError(PipelineError):
    """Error raised when a model fails to load."""

    def __init__(self, model_key: str, cause: Exception | None = None) -> None:
        self.model_key = model_key
        self.cause = cause
        super().__init__(f"Failed to load model '{model_key}': {cause}")


class InvalidInputError(PipelineError):
    """Error raised for invalid input images."""


class GarmentNotFoundError(StageError):
    """Error raised when no garment is detected in the image."""

    def __init__(self, message: str = "No upper garment detected in the image") -> None:
        super().__init__(stage_name="garment_parse", message=message)
