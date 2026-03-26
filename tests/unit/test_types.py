from __future__ import annotations

import numpy as np

from shirtrip.pipeline.types import BBox, PipelineImage, PipelineResult, StageMetadata


class TestBBox:
    def test_creation(self) -> None:
        bbox = BBox(10, 20, 100, 200)
        assert bbox.x == 10
        assert bbox.y == 20
        assert bbox.w == 100
        assert bbox.h == 200


class TestPipelineImage:
    def test_dimensions(self) -> None:
        bgr = np.zeros((100, 200, 3), dtype=np.uint8)
        img = PipelineImage(bgr=bgr)
        assert img.height == 100
        assert img.width == 200

    def test_rgba_without_alpha(self) -> None:
        bgr = np.zeros((10, 10, 3), dtype=np.uint8)
        img = PipelineImage(bgr=bgr)
        rgba = img.rgba
        assert rgba.shape == (10, 10, 4)
        assert np.all(rgba[:, :, 3] == 255)

    def test_rgba_with_alpha(self) -> None:
        bgr = np.zeros((10, 10, 3), dtype=np.uint8)
        alpha = np.full((10, 10), 128, dtype=np.uint8)
        img = PipelineImage(bgr=bgr, alpha=alpha)
        rgba = img.rgba
        assert rgba.shape == (10, 10, 4)
        assert np.all(rgba[:, :, 3] == 128)

    def test_frozen(self) -> None:
        bgr = np.zeros((10, 10, 3), dtype=np.uint8)
        img = PipelineImage(bgr=bgr)
        try:
            img.bgr = np.ones((10, 10, 3), dtype=np.uint8)  # type: ignore[misc]
            assert False, "Should not allow attribute assignment"
        except AttributeError:
            pass


class TestStageMetadata:
    def test_creation(self) -> None:
        meta = StageMetadata(
            stage_name="test",
            duration_ms=42.0,
            input_shape=(100, 200, 3),
            output_shape=(100, 200, 4),
        )
        assert meta.stage_name == "test"
        assert meta.duration_ms == 42.0
        assert meta.extra == {}

    def test_extra_dict(self) -> None:
        meta = StageMetadata(
            stage_name="test",
            duration_ms=0.0,
            input_shape=(1,),
            output_shape=(1,),
            extra={"key": "value"},
        )
        assert meta.extra["key"] == "value"


class TestPipelineResult:
    def test_accumulates_metadata(self) -> None:
        bgr = np.zeros((10, 10, 3), dtype=np.uint8)
        result = PipelineResult(image=PipelineImage(bgr=bgr))
        meta = StageMetadata("s1", 10.0, (10, 10, 3), (10, 10, 4))
        result.metadata.append(meta)
        assert len(result.metadata) == 1

    def test_masks_dict(self) -> None:
        bgr = np.zeros((10, 10, 3), dtype=np.uint8)
        result = PipelineResult(image=PipelineImage(bgr=bgr))
        mask = np.ones((10, 10), dtype=np.uint8)
        result.masks["garment"] = mask
        assert "garment" in result.masks
