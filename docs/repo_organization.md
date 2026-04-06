# Repository Organization

This document outlines the final modular and maintainable structure of the MemeCap project. The reorganization centralizes core logic, separates task-specific scripts, and standardizes output directories.

## Project Structure

```
CMPE597-group11/
├── src/
│   ├── common/             # Shared utilities across all tasks
│   │   ├── dataset.py      # MemeSample, data loading
│   │   ├── metrics.py      # Recall@K, MRR
│   │   └── utils.py        # Seeding, plotting, logging
│   ├── models/
│   │   ├── custom/         # Custom architecture (Task 2.1c)
│   │   │   ├── architecture.py
│   │   │   ├── loss.py
│   │   │   └── data_utils.py
│   │   └── pretrained/     # Pretrained backends (Task 2.1b, d)
│   │       ├── openclip.py
│   │       ├── siglip.py
│   │       ├── lora.py
│   │       └── blip.py
│   └── tasks/
│       └── retrieval/      # Task 2.1 (Cross-Modal Retrieval)
│           ├── train/      # Training scripts
│           │   ├── custom.py
│           │   └── lora.py
│           └── evaluate/   # Evaluation scripts
│               ├── zero_shot.py
│               ├── custom.py
│               └── lora.py
├── data/                   # Dataset files (JSON and images)
├── docs/                   # Project documentation and plans
│   └── repo_organization.md
├── outputs/                # Standardized training and evaluation outputs
│   ├── custom/             # Outputs from custom model (Task 2.1c)
│   │   ├── type1/
│   │   └── type2/
│   ├── finetune/           # Outputs from LoRA finetuning (Task 2.1d)
│   │   ├── type1/
│   │   └── type2/
│   └── zero_shot/          # Evaluation metrics for baseline models
├── venv/                   # Python virtual environment
└── README.md
```

## Core Components

### 1. Source Code (`src/`)
-   **`common/`**: Contains code used by multiple tasks. Everything here is task-agnostic to prevent code duplication.
-   **`models/`**: Distinguishes between custom-built models and wrappers for pretrained backends. This layer provides a uniform API (`encode_images`, `encode_texts`) for task runners.
-   **`tasks/`**: Grouped by project objective (e.g., `retrieval`, `classification`). Within each task, scripts are strictly divided into `train/` and `evaluate/` to keep local environments clean.

### 2. Standardized Outputs (`outputs/`)
All model outputs are moved into a hierarchical structure that mirrors the task input types:
-   `outputs/custom/` handles from-scratch models.
-   `outputs/finetune/` handles adapter and full-finetune sessions.
-   Each contains `type1/` and `type2/` subfolders for immediate comparability.

## Rationale for Changes
-   **Maintainability**: Centralizing data loaders and metrics ensures that updates fix all scripts at once.
-   **Scalability**: New tasks (Task 2.2, 2.3) can easily be added as new directories under `tasks/` while reusing existing components in `common/` and `models/`.
-   **Discoverability**: Subfolders like `train/` and `evaluate/` provide immediate clarity on a script's purpose.

## Running the Scripts
Scripts should be run from the repository root using absolute module imports.
Examples:
```bash
# Evaluate zero-shot retrieval:
python src/tasks/retrieval/evaluate/zero_shot.py --model_family openclip --input_type type1

# Train LoRA adapter:
python src/tasks/retrieval/train/lora.py --task type2
```
