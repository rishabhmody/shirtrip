from __future__ import annotations

from pathlib import Path

import torch
from pydantic_settings import BaseSettings, SettingsConfigDict


def _default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SHIRTRIP_")

    device: str = _default_device()
    model_cache_dir: Path = Path("models_cache")
    max_upload_mb: int = 20
    log_level: str = "INFO"
    max_image_dimension: int = 2048
    allowed_extensions: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".webp"})
    use_sam2: bool = True

    # Grounding DINO + SAM settings
    dino_model_id: str = "IDEA-Research/grounding-dino-tiny"
    sam_model_id: str = "facebook/sam-vit-base"
    graphic_prompt: str = "graphic design . text . logo . printed artwork"
    dino_box_threshold: float = 0.3
    dino_text_threshold: float = 0.25
    sequential_gpu_loading: bool = True  # Unload models between stages to save VRAM

    upload_dir: Path = Path("uploads")
    output_dir: Path = Path("outputs")
    host: str = "0.0.0.0"
    port: int = 8000
