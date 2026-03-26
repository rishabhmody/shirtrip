# ShirtRip Project Rules

## Core Invariant
NO neural network output is ever used directly as pixel data. Networks produce masks, depth maps, displacement fields, or alpha mattes. These are applied to ORIGINAL input pixels via cv2.remap, cv2.bitwise_and, or alpha compositing.

## Code Standards
- Python 3.11+, type hints on all function signatures
- `from __future__ import annotations` in every file
- BGR np.ndarray internally, RGB only at API boundary
- `torch.inference_mode()` for all inference (not torch.no_grad)
- `logging` module only, no print statements
- Conventional commits: feat/fix/refactor/docs/test/chore

## Testing
- pytest, 80%+ branch coverage
- Mock model inference in unit tests (saved tensor fixtures)
- Integration tests with @pytest.mark.slow for real models
- TDD: write tests BEFORE implementation

## Architecture
- Each pipeline stage is a pure function: (PipelineImage, Settings) -> PipelineResult
- Models lazy-loaded via ModelRegistry singleton
- Pipeline stages have zero FastAPI imports
- Orchestrator composes stages but contains no ML logic
