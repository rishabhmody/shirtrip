from __future__ import annotations

import logging
import threading
from typing import Any, Callable

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Thread-safe lazy-loading singleton registry for ML models."""

    _instance: ModelRegistry | None = None
    _instance_lock: threading.Lock = threading.Lock()

    def __init__(self) -> None:
        self._models: dict[str, Any] = {}
        self._lock = threading.Lock()

    @classmethod
    def get(cls) -> ModelRegistry:
        """Return the singleton instance."""
        if cls._instance is None:
            with cls._instance_lock:
                if cls._instance is None:
                    cls._instance = ModelRegistry()
        return cls._instance

    @classmethod
    def reset(cls) -> None:
        """Reset the singleton (for testing)."""
        with cls._instance_lock:
            if cls._instance is not None:
                cls._instance.unload_all()
            cls._instance = None

    def load(self, key: str, loader: Callable[[], Any]) -> Any:
        """Load a model by key, using the loader function on cache miss."""
        if key in self._models:
            return self._models[key]
        with self._lock:
            if key in self._models:
                return self._models[key]
            logger.info("Loading model: %s", key)
            model = loader()
            self._models[key] = model
            logger.info("Model loaded: %s", key)
            return model

    def unload(self, key: str) -> None:
        """Remove a model from the cache."""
        with self._lock:
            if key in self._models:
                del self._models[key]
                logger.info("Model unloaded: %s", key)

    def unload_all(self) -> None:
        """Remove all models from the cache."""
        with self._lock:
            self._models.clear()
            logger.info("All models unloaded")

    @property
    def loaded_models(self) -> list[str]:
        """Return list of currently loaded model keys."""
        return list(self._models.keys())
