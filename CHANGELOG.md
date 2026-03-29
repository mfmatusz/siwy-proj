# Changelog

All notable changes to this project will be documented in this file.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Sections: Added, Changed, Removed, Fixed.

## 29.03.2026

### Added

- Inseq integration (attention attribution)
- Dataset: 10 par promptów w 5 kategoriach
- GQA-aware head pooling (2 query heads per 1 KV head)
- Analiza 34 warstw z podziałem lokalne/globalne
- HTML report generator
- src/config/model.py
- src/models/attention_utils.py
- src/attribution/inseq_analysis.py
- src/visualization/report.py
- wandb enabled/disabled toggle
- Memory management (CPU offload, MPS cache clearing)
- Invoke tasks: run, run-inseq, report, lint, format, test, check
- docs/TODO.md

### Changed

- setuptools to uv + hatchling
- logging to loguru
- device cpu to mps, quantization none to bf16
- skrypty przeniesione do scripts/
- design-proposal.md to report.md
- README.md
- ruff (line-length 120, import sorting)
- hydra: get_original_cwd dla poprawnych sciezek
- NF4 fallback do BF16 na MPS
- type annotations

### Removed

- src/models/xai_utils.py
- martwy kod z inseq_analysis.py

### Fixed

- GQA: naiwne mean to grupowanie par query heads
- hydra cwd: sciezki trafialy do outputs/ zamiast roota
- wandb: brak fallbacka offline

## 25.03.2026

### Added

- Initial project structure for XAI LLM Attention Analysis.
- Model pipeline to extract attention matrices from Gemma-3-4B-IT.
- Heatmap visualization module.
- Integration with Hydra for configuration and Weights & Biases (WandB) for ML tracking.
- Updated `.gitignore` to exclude Python environments, IDE artifacts, WandB logs, and processed run outputs.
