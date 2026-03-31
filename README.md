# Wpływ struktury promptu na odpowiedzi modelu językowego — analiza attention

Przedmiot: Wyjaśnialna Sztuczna Inteligencja (SIWY), Semestr: 26L

Zespół: Bartłomiej Dmitruk, Maciej Matuszewski, Magdalena Kalińska

---

Małe zmiany w treści promptu potrafią znacząco zmienić odpowiedź modelu językowego — ale co dokładnie się wtedy dzieje wewnątrz modelu? Projekt bada ten mechanizm od strony mechanizmu attention.

Używamy modelu Gemma 3 4B i zestawu ręcznie przygotowanych **par promptów** — każda para to ten sam sens, ale inaczej sformułowane zapytanie (np. „Wyjaśnij grawitację" vs. „Wyjaśnij grawitację krótko i prosto"). Dla każdej pary ekstrahujemy wagi attention ze wszystkich 34 warstw modelu i analizujemy, czy i jak zmienia się rozkład uwagi modelu po modyfikacji promptu.

Pary promptów pokrywają pięć kategorii zmian: **styl**, **ton/rola**, **formalność**, **framing** i **reformulacja**. Warstwy analizowane są osobno jako lokalne (sliding window, 1024 tokeny) i globalne (pełny kontekst), co wynika bezpośrednio z architektury Gemma 3. Analizę uzupełnia atrybucja gradientowa via Inseq (saliency, integrated gradients).

Projekt łączy budowę reprodukowalnego narzędzia analitycznego z kontrolowanym eksperymentem — wyniki mają charakter zarówno ilościowy (entropia attention, sparsity, średnie wagi per kategoria tokenu) jak i jakościowy (wizualizacje heatmap, diff między parami).

## Wymagania

- Python >= 3.11
- GPU z min. ~8 GB VRAM
- [`uv`](https://docs.astral.sh/uv/) — menadżer pakietów

> Model Gemma 3 jest bramkowany — wymaga akceptacji licencji i tokenu Hugging Face.
> Szczegóły konfiguracji środowiska → **[USAGE.md](USAGE.md)**

## Szybki start

```bash
git clone https://github.com/mfmatusz/siwy-proj.git
cd siwy-proj
uv sync
uv run invoke run
```

> Szczegółowa instrukcja konfiguracji, uruchamiania i rozwiązywania problemów → **[USAGE.md](USAGE.md)**

## Struktura projektu

```
conf/               konfiguracja Hydra (config.yaml)
data/raw/           dataset promptów (niemutowalny)
data/processed/     wyniki eksperymentów (tensory, heatmapy, raporty)
docs/               design proposal, analiza literatury, TODO
scripts/            entry pointy (run_experiment, run_inseq, generate_report)
src/
  config/           stałe modelu (warstwy, indeksy GQA)
  data/             ładowanie datasetu
  models/           ekstrakcja attention, agregacja per-layer
  attribution/      analiza Inseq
  visualization/    heatmapy, raport HTML
tests/              testy pytest
```

## Narzędzia deweloperskie

```bash
uv run invoke lint    # linter (ruff)
uv run invoke format  # formatowanie (ruff)
uv run invoke test    # testy (pytest)
uv run invoke check   # lint + testy
```

## Narzędzia i technologie

uv, Hydra, invoke, ruff, pytest, transformers, inseq, bertviz, wandb, loguru, bitsandbytes