# Wpływ struktury promptu na odpowiedzi modelu językowego - analiza attention

Przedmiot: Wyjaśnialna Sztuczna Inteligencja (SIWY)

Zespół: Bartłomiej Dmitruk, Maciej Matuszewski, Magdalena Kalińska

Analiza wag attention w Gemma 3 4B na zbiorze par promptów (bazowy vs zmodyfikowany) w pięciu kategoriach: styl, ton/rola, formalność, framing, reformulacja. Pipeline ekstrakcji attention z podziałem na warstwy lokalne (sliding window) i globalne + atrybucja via Inseq.

## Wymagania

- Python >= 3.11
- GPU z min. ~8 GB VRAM
- uv (package manager)

## Instalacja

```bash
uv sync
```

## Uruchamianie

Konfiguracja: `conf/config.yaml`

```bash
uv run invoke run              # ekstrakcja attention (all 34 layers, local/global split)
uv run invoke run-inseq        # atrybucja Inseq
uv run invoke report           # generowanie raportu HTML
```

Nadpisywanie parametrów hydra:

```bash
uv run invoke run --config-overrides="model.name=google/gemma-3-4b-pt experiment_name=test_run"
```

## Struktura projektu

```
conf/               konfiguracja hydra
data/raw/           dataset promptów (niemutowalny)
data/processed/     wyniki eksperymentów (tensory, heatmapy)
docs/               design proposal, TODO
scripts/            entry pointy (run_experiment, run_inseq, generate_report)
src/
  config/           stałe modelu (warstwy, indeksy)
  data/             ładowanie datasetu
  models/           ekstrakcja attention, agregacja per-layer
  attribution/      analiza Inseq
  visualization/    heatmapy, raport HTML
tests/              testy pytest
```

## Narzędzia

uv, hydra, invoke, ruff, pytest, transformers, inseq, bertviz, wandb, loguru
