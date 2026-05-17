# Changelog

All notable changes to this project will be documented in this file.
Format based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Sections: Added, Changed, Removed, Fixed.

## 17.05.2026

### Added

- `data/raw/prompts.json` — dataset rozbudowany z 10 do 25 par (5 per kategoria): style, tone, formality, framing, reformulation
- `data/raw/token_categories.json` — ręczne adnotacje słów-kluczy per para promptów (instrukcja / treść) do kategoryzacji tokenów
- `src/data/token_labels.py` — `load_token_categories()` i `categorize_tokens()`: strip prefixu ▁ SentencePiece, dopasowanie case-insensitive
- `src/metrics/__init__.py` — `mean_attention_by_category()`: średnia waga attention per kategorię tokenu (instrukcja / treść / funkcyjny)
- `src/visualization/visualize.py` — `plot_category_attention_barchart()`: grouped bar chart (base vs modified) per kategoria promptów, osobno dla tokenów instrukcji i treści
- `src/config/model.py` — `validate_model_config()`: walidacja stałych modelu przy starcie aplikacji (fail-fast)
- `tests/test_integration.py` — 4 testy integracyjne pipeline'u ekstrakcji attention (process_prompt_pair → aggregate → metrics), bez GPU i sieci
- `tests/test_token_labels.py` — 11 testów dla `load_token_categories` i `categorize_tokens`
- `tests/test_model_config.py` — 9 testów dla `validate_model_config`
- `.python-version` — pin Python 3.11 dla projektu (uv respektuje przy tworzeniu venv)
- `Makefile` — target `clean` (usuwa `.venv`); sekcja help aktualizacja

### Changed

- `conf/config.yaml` — stałe modelu (`num_layers: 34`, `gqa_group_size: 2`, `global_layer_indices: [5,11,17,23,29]`) przeniesione z kodu do sekcji `model:`
- `src/config/model.py` — stałe odczytywane z `config.yaml` (z fallbackiem do defaults); wywołanie sieci usunięte z poziomu modułu pozostaje zrealizowane
- `scripts/run_experiment.py` — `validate_model_config(cfg)` przy starcie; integracja token categorization, `mean_attention_by_category` i bar chart; logowanie wykresu zbiorczego do W&B
- `scripts/run_inseq.py` — `validate_model_config(cfg)` przy starcie
- `tests/test_metrics.py` — dodano `TestMeanAttentionByCategory` (6 testów); łącznie 24 testy
- `tests/test_visualize.py` — dodano `TestPlotCategoryAttentionBarchart` (4 testy); łącznie 8 testów
- `Makefile` — `uv run pytest` → `uv run python -m pytest` (cross-platform); `uv sync --extra dev` przed każdym uruchomieniem testów
- `pyproject.toml` — `[tool.uv] link-mode = "copy"` (kompatybilność z WSL / cross-filesystem)

Łącznie: 71 testów (było 37).

## 03.05.2026

### Added

- `src/metrics/__init__.py` — moduł metryk ilościowych: `attention_entropy`, `sparsity_ratio`, `pairwise_attention_diff`, `mean_attention_by_token_position` (pure functions, bez I/O, bez GPU)
- `src/visualization/visualize.py` — `plot_diff_heatmap()` z colormap RdBu_r, zakres ±max_abs
- Diff heatmapy (modified − base) per para promptów: `local_diff`, `global_diff`, `overall_diff`
- Logowanie metryk ilościowych do W&B: entropia, sparsity, normy L1/L2 diffy
- `tests/test_dataset.py`, `tests/test_attention_utils.py`, `tests/test_visualize.py`, `tests/test_metrics.py` — 37 testów pytest działających bez GPU i bez połączenia sieciowego
- `Makefile` — skróty do częstych operacji (install, lint, format, test, check, run, run-inseq, report); obsługuje `ARGS=` dla Hydra overrides
- `src/config/model.py` — funkcja `initialize_from_pretrained()` do opcjonalnego pobierania stałych modelu z HF API

### Changed

- `src/config/model.py` — usunięto wywołanie `AutoConfig.from_pretrained()` na poziomie modułu (było side effectem każdego importu); stałe `GLOBAL_LAYER_INDICES`, `NUM_LAYERS`, `GQA_GROUP_SIZE` są teraz hardkodowanymi defaults dla Gemma 3 4B IT
- `scripts/run_experiment.py` — `dotenv.load_dotenv()` przeniesiony przed pozostałe importy; dodano `torch.manual_seed(cfg.seed)`; dodano logowanie metryk do W&B
- `scripts/run_inseq.py` — `dotenv.load_dotenv()` przeniesiony przed pozostałe importy; dodano `torch.manual_seed(cfg.seed)`
- `tasks.py` — usunięto `PYTHONPATH=.` z tasków `run`, `run_inseq`, `report` (nie działa na Windows; `src` jest instalowane przez hatchling)
- `conf/config.yaml` — usunięto martwy klucz `paths.notebooks`
- `pyproject.toml` — dodano `per-file-ignores` dla `scripts/*.py` (E402) w konfiguracji ruff
- Komentarze i docstringi w nowych plikach standaryzowane do języka angielskiego

## 31.03.2026

### Added

- USAGE.md — szczegółowa instrukcja konfiguracji środowiska, uruchamiania eksperymentów i rozwiązywania problemów.
- Pokrycie tematyczne literatury w docs/report.md, zgodnie z z §1 pkt 6 regulaminu (wybór literatury powinien być uzasadniony)

### Changed

- Przegląd literatury i bibliografię w docs/report.md, zgodnie z §1 pkt 6 regulaminu — dodane kolumny: autorski komentarz, dostępność kodu/modeli, metryki ewaluacji w artykule, zasoby obliczeniowe autorów.
- README.md — dodany link do USAGE.md, rozbudowanie opisu projektu i sekcji
- docs/TODO.md.

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
