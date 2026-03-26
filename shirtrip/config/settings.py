from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="SHIRTRIP_")

    device: str = "cuda"
    model_cache_dir: Path = Path("models_cache")
    max_upload_mb: int = 20
    log_level: str = "INFO"
    max_image_dimension: int = 2048
    allowed_extensions: frozenset[str] = frozenset({".jpg", ".jpeg", ".png", ".webp"})
    use_sam2: bool = True
    upload_dir: Path = Path("uploads")
    output_dir: Path = Path("outputs")
    host: str = "0.0.0.0"
    port: int = 8000
