# Implementation Plan - Project Reorganization & Refactoring

This plan aims to reorganize the MemeCap project into a modular, maintainable structure. We will move the core logic into a `src/` directory, grouping components by their role (common utilities, models/backends, and tasks).

## User Review Required

> [!IMPORTANT]
> - All existing logic will remain unchanged.
> - Scripts will be moved to new locations. If you have automated pipelines relying on exact paths, they will need to be updated.
> - We will introduce `__init__.py` files to support proper Python module discovery.
> - We will ensure all scripts can still be run from the root of the project.

## Proposed Structure

```
src/
├── common/             # Shared utilities across all tasks
│   ├── dataset.py      # MemeSample, data loading
│   ├── metrics.py      # Recall@K, MRR
│   └── utils.py        # Seeding, plotting, logging
├── models/
│   ├── custom/         # Custom architecture (Task 2.1c)
│   │   ├── architecture.py
│   │   ├── loss.py
│   │   └── data_utils.py
│   └── pretrained/     # Pretrained backends (Task 2.1b, d)
│       ├── openclip.py
│       ├── siglip.py
│       ├── lora.py
│       └── blip.py
└── tasks/
    └── retrieval/      # Task 2.1 specific runner scripts
        ├── evaluate_zero_shot.py
        ├── train_custom.py
        ├── evaluate_custom.py
        ├── train_lora.py
        └── inference_lora.py
```

## Proposed Changes

### [NEW] Directory Creation
1. Create `src/`, `src/common/`, `src/models/`, `src/models/custom/`, `src/models/pretrained/`, `src/tasks/`, `src/tasks/retrieval/`.
2. Add `__init__.py` to all new directories.

### [MOVE & REFACTOR] Common Modules
- **[MOVE]** `dataset.py` -> `src/common/dataset.py`
- **[MOVE]** `metrics.py` -> `src/common/metrics.py`
- **[MOVE]** `utils.py` -> `src/common/utils.py`

### [MOVE & REFACTOR] Model Backends
- **[MOVE]** `zero_shot/openclip_backend.py` -> `src/models/pretrained/openclip.py`
- **[MOVE]** `zero_shot/siglip_backend.py` -> `src/models/pretrained/siglip.py`
- **[MOVE]** `zero_shot/blip_reranker.py` -> `src/models/pretrained/blip.py`
- **[MOVE]** `finetune/lora_backend.py` -> `src/models/pretrained/lora.py`
- **[MOVE]** `custom/model.py` -> `src/models/custom/architecture.py`
- **[MOVE]** `custom/loss.py` -> `src/models/custom/loss.py`
- **[MOVE]** `custom/custom_dataset.py` -> `src/models/custom/data_utils.py`

### [MOVE & REFACTOR] Task Scripts
- **[MOVE]** `zero_shot/evaluation.py` -> `src/tasks/retrieval/evaluate_zero_shot.py`
- **[MOVE]** `custom/train.py` -> `src/tasks/retrieval/train_custom.py`
- **[MOVE]** `custom/evaluate.py` -> `src/tasks/retrieval/evaluate_custom.py`
- **[MOVE]** `finetune/train.py` -> `src/tasks/retrieval/train_lora.py`
- **[MOVE]** `finetune/inference.py` -> `src/tasks/retrieval/inference_lora.py`

### [CLEANUP]
- Remove empty directories: `zero_shot/`, `custom/`, `finetune/`.

## Open Questions

- Should we keep the root-level scripts (`dataset.py`, etc.) as stubs for backward compatibility, or strictly move them?
- Do you prefer running scripts via `python src/tasks/.../script.py` or as modules `python -m src.tasks...`? (I will configure them to work with both).

## Verification Plan

### Automated Tests
- Run each script with a small subset (`--limit 10`) to ensure imports and logic are intact.
- Verify `OpenCLIP`, `SigLIP`, `Custom Model`, and `LoRA` loading.

### Manual Verification
- Check the directory structure for clarity and logical grouping.
