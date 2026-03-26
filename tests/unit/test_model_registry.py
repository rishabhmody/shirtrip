from __future__ import annotations

import threading

from shirtrip.models.model_registry import ModelRegistry


class TestModelRegistry:
    def setup_method(self) -> None:
        ModelRegistry.reset()

    def teardown_method(self) -> None:
        ModelRegistry.reset()

    def test_singleton(self) -> None:
        r1 = ModelRegistry.get()
        r2 = ModelRegistry.get()
        assert r1 is r2

    def test_lazy_loading(self) -> None:
        registry = ModelRegistry.get()
        call_count = 0

        def loader() -> str:
            nonlocal call_count
            call_count += 1
            return "model_value"

        result = registry.load("test_model", loader)
        assert result == "model_value"
        assert call_count == 1

        # Second call should reuse cached value
        result2 = registry.load("test_model", loader)
        assert result2 == "model_value"
        assert call_count == 1  # loader NOT called again

    def test_unload(self) -> None:
        registry = ModelRegistry.get()
        registry.load("m1", lambda: "v1")
        assert "m1" in registry.loaded_models
        registry.unload("m1")
        assert "m1" not in registry.loaded_models

    def test_unload_nonexistent_is_noop(self) -> None:
        registry = ModelRegistry.get()
        registry.unload("nonexistent")  # should not raise

    def test_unload_all(self) -> None:
        registry = ModelRegistry.get()
        registry.load("m1", lambda: "v1")
        registry.load("m2", lambda: "v2")
        assert len(registry.loaded_models) == 2
        registry.unload_all()
        assert len(registry.loaded_models) == 0

    def test_thread_safety(self) -> None:
        registry = ModelRegistry.get()
        call_count = 0
        lock = threading.Lock()

        def loader() -> str:
            nonlocal call_count
            with lock:
                call_count += 1
            return "value"

        threads = [
            threading.Thread(target=lambda: registry.load("shared", loader))
            for _ in range(10)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert call_count == 1  # Only one thread should have called the loader
        assert registry.load("shared", loader) == "value"

    def test_loaded_models_list(self) -> None:
        registry = ModelRegistry.get()
        assert registry.loaded_models == []
        registry.load("a", lambda: 1)
        registry.load("b", lambda: 2)
        assert sorted(registry.loaded_models) == ["a", "b"]

    def test_reset_clears_singleton(self) -> None:
        r1 = ModelRegistry.get()
        r1.load("x", lambda: 42)
        ModelRegistry.reset()
        r2 = ModelRegistry.get()
        assert r1 is not r2
        assert r2.loaded_models == []
