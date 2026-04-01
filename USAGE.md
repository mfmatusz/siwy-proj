# Instrukcja użytkowania

---

## Spis treści

1. [Przed rozpoczęciem](#1-przed-rozpoczęciem)
2. [Instalacja](#2-instalacja)
3. [Konfiguracja](#3-konfiguracja)
4. [Uruchamianie eksperymentów](#4-uruchamianie-eksperymentów)
5. [Atrybucja Inseq](#5-atrybucja-inseq)
6. [Generowanie raportu](#6-generowanie-raportu)
7. [Narzędzia deweloperskie](#7-narzędzia-deweloperskie)
8. [Wyniki — gdzie szukać](#8-wyniki--gdzie-szukać)
9. [Rozwiązywanie problemów](#9-rozwiązywanie-problemów)

---

## 1. Przed rozpoczęciem

### Wymagania Sprzętowe
- GPU z co najmniej **8 GB VRAM** (wymagane do uruchomienia Gemma 3 4B)
- Minimum 16 GB RAM systemowego

> **Uwaga:** Atrybucja attention via Inseq działa szybko (~12s/prompt). Metody gradientowe (saliency, integrated gradients) wymagają GPU z CUDA — na MPS są zbyt wolne (~40 min/prompt) i są wyłączone domyślnie.

### Wymagania Programowe
- Python >= 3.11
- [`uv`](https://docs.astral.sh/uv/) — menadżer pakietów i środowisk wirtualnych
- Dostęp do modelu `google/gemma-3-4b-it` na Hugging Face (wymagana akceptacja warunków licencji)

### Instalacja `uv`
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Token Hugging Face
Model Gemma jest bramkowany — potrzebny jest token HF z uprawnieniami do pobrania modelu.

```bash
# Po zalogowaniu na huggingface.co i zaakceptowaniu licencji Gemma:
uv run huggingface-cli login
# lub ustaw zmienną środowiskową:
export HF_TOKEN=hf_twoj_token
```

---

## 2. Instalacja

```bash
git clone https://github.com/mfmatusz/siwy-proj.git
cd siwy-proj

# Zainstaluj wszystkie zależności (tworzy środowisko wirtualne automatycznie)
uv sync

# Dla środowiska deweloperskiego (z pytest, ruff):
uv sync --extra dev
```

---

## 3. Konfiguracja

Konfiguracja projektu znajduje się w `conf/config.yaml`. Używamy [Hydra](https://hydra.cc/) do zarządzania konfiguracją.

Aktualna konfiguracja domyślna:

```yaml
experiment_name: "attention_baseline"
seed: 42

paths:
  data_raw: "data/raw"
  data_processed: "data/processed"
  outputs: "outputs"

model:
  name: "google/gemma-3-4b-it"   # model instruction-tuned
  quantization: "bf16"            # bf16 / nf4
  device: "mps"                   # mps / cuda / cpu

wandb:
  project: "siwy-xai-llm"
  enabled: true
```

### Nadpisywanie parametrów w linii komend

Parametry konfiguracji można nadpisać bez edytowania pliku YAML:

```bash
uv run invoke run --config-overrides="model.name=google/gemma-3-4b-it experiment_name=test_run"
uv run invoke run --config-overrides="model.device=cuda"
uv run invoke run --config-overrides="model.quantization=nf4"
uv run invoke run --config-overrides="wandb.enabled=false"
```

---

## 4. Uruchamianie eksperymentów

### Pełny pipeline — ekstrakcja wag attention

```bash
uv run invoke run
```

Skrypt `scripts/run_experiment.py`:
1. Wczytuje dataset par promptów z `data/raw/prompts.json`
2. Ładuje model Gemma 3 4B IT z Hugging Face
3. Dla każdej pary (bazowy, zmodyfikowany) uruchamia forward pass z ekstrakcją attention ze wszystkich 34 warstw
4. Agreguje wagi po głowicach z uwzględnieniem GQA (group-aware pooling) i rozdziela warstwy na **lokalne** (sliding window) i **globalne**
5. Zapisuje tensory (`.pt`) i heatmapy (`.png`) do `data/processed/<experiment_name>/<prompt_id>/`
6. Loguje heatmapy do W&B

#### Kategorie promptów w datasecie

| Kategoria       | Opis                                                       |
|-----------------|------------------------------------------------------------|
| `style`         | Zmiana stylu językowego (np. formalny ↔ potoczny)          |
| `tone_role`     | Zmiana tonu lub przypisanej roli (np. ekspert, nauczyciel) |
| `formality`     | Zmiana poziomu formalności                                 |
| `framing`       | Zmiana ramowania pytania (np. aktywne ↔ pasywne)           |
| `reformulation` | Pełna parafrazacja przy zachowaniu sensu                   |

---

## 5. Atrybucja Inseq

```bash
uv run invoke run-inseq
```

Skrypt `scripts/run_inseq.py` uruchamia analizę atrybucji przy użyciu biblioteki [Inseq](https://github.com/inseq-team/inseq). Aktualnie używana metoda atrybucji: `attention`. Wyniki zapisywane są do:

```
data/processed/<experiment_name>_inseq/
└── <prompt_id>/
    └── attention/
        ├── base.json
        └── modified.json
```

Wizualizacje HTML atrybucji są równolegle logowane do W&B.

> Metoda attention działa szybko na MPS (~12s/prompt). Metody gradientowe (saliency, integrated gradients) wymagają CUDA i są wyłączone domyślnie.

---

## 6. Generowanie raportu

```bash
uv run invoke report
```

Skrypt `scripts/generate_report.py` łączy wyniki ekstrakcji attention i atrybucji Inseq w jeden samowystarczalny raport HTML (obrazy osadzone jako base64 — plik działa bez dostępu do serwera). Raport zawiera dwie sekcje, każda renderowana tylko jeśli odpowiedni katalog istnieje:

- **Attention Heatmaps** — siatka 3-kolumnowa z heatmapami per para promptów (local/global/overall × base/modified)
- **Inseq Attribution** — interaktywne wizualizacje atrybucji wczytane z plików `.json`

Raport zapisywany jest jako:

```
data/processed/<experiment_name>_report.html
```

i otwiera się automatycznie w domyślnej przeglądarce po wygenerowaniu.

> Sekcje attention i inseq są niezależne — raport wygeneruje się poprawnie nawet jeśli uruchomiono tylko jeden z poprzednich kroków.

### Typowy pełny przepływ pracy

```bash
# 1. Ekstrakcja attention
uv run invoke run --config-overrides="experiment_name=moj_eksperyment"
# wyniki: data/processed/moj_eksperyment/

# 2. Atrybucja Inseq (attention, ~2 min na 10 par)
uv run invoke run-inseq --config-overrides="experiment_name=moj_eksperyment"
# wyniki: data/processed/moj_eksperyment_inseq/

# 3. Wygenerowanie raportu — wymaga obu powyższych kroków
uv run invoke report --config-overrides="experiment_name=moj_eksperyment"
# raport: data/processed/moj_eksperyment_report.html (otwiera się automatycznie w przeglądarce)
```

---

## 7. Narzędzia deweloperskie

Wszystkie komendy deweloperskie obsługiwane są przez `invoke`:

```bash
# Sprawdzenie kodu (linter ruff)
uv run invoke lint

# Formatowanie kodu (ruff format)
uv run invoke format

# Uruchomienie testów
uv run invoke test

# Pełne sprawdzenie: lint + testy
uv run invoke check
```

### Bezpośrednie wywołanie pytest

```bash
uv run pytest tests/ -v
```

---

## 8. Wyniki — gdzie szukać

Wyniki każdego eksperymentu zapisywane są do `data/processed/<experiment_name>/`, w osobnym podkatalogu per para promptów:

```
data/processed/<experiment_name>/
└── <prompt_id>/
    ├── tensors/
    │   ├── per_layer_base.pt     # wagi attention per warstwa, prompt bazowy
    │   ├── per_layer_mod.pt      # wagi attention per warstwa, prompt zmodyfikowany
    │   ├── local_base.pt         # średnia po warstwach lokalnych, bazowy
    │   ├── global_base.pt        # średnia po warstwach globalnych, bazowy
    │   ├── overall_base.pt       # średnia po wszystkich warstwach, bazowy
    │   ├── local_mod.pt          # średnia po warstwach lokalnych, zmodyfikowany
    │   ├── global_mod.pt         # średnia po warstwach globalnych, zmodyfikowany
    │   └── overall_mod.pt        # średnia po wszystkich warstwach, zmodyfikowany
    └── heatmaps/
        ├── local_base.png
        ├── global_base.png
        ├── overall_base.png
        ├── local_mod.png
        ├── global_mod.png
        └── overall_mod.png
```

Dane surowe (pary promptów) znajdują się w `data/raw/prompts.json` i są **tylko do odczytu** — nie należy ich modyfikować.

Heatmapy są równolegle logowane do [Weights & Biases](https://wandb.ai/) jako obrazy. Logowanie metryk ilościowych (entropia, sparsity, średnie wagi per kategoria tokenów) jest jeszcze w trakcie implementacji — patrz sekcja 10.

Dostęp do dashboardu W&B wymaga konta i zalogowania:

```bash
uv run wandb login
```

---

## 9. Rozwiązywanie problemów

**`CUDA out of memory`**
Włącz kwantyzację NF4 (~2.6 GB wag zamiast ~8 GB w BF16):
```bash
uv run invoke run --config-overrides="model.quantization=nf4"
```

**Model nie ładuje się / błąd przy inicjalizacji**
Pipeline przejdzie automatycznie w tryb dry-run — przetworzy dataset bez uruchamiania modelu i zaloguje eksperyment do W&B. Pozwala to zweryfikować konfigurację bez dostępu do GPU.

**`401 Unauthorized` przy pobieraniu modelu**
Upewnij się, że zaakceptowałeś warunki licencji Gemma na stronie [huggingface.co/google/gemma-3-4b-it](https://huggingface.co/google/gemma-3-4b-it) i że token HF jest poprawnie ustawiony (`huggingface-cli login` lub `HF_TOKEN`).

**Powolna atrybucja gradientowa na MacOS (MPS)**
Dotyczy tylko metod gradientowych (saliency, integrated gradients), nie attention. Attention attribution działa szybko (~12s/prompt). Metody gradientowe wymagają CUDA.

**`ModuleNotFoundError`**
Upewnij się, że używasz `uv run` (nie `python` bezpośrednio), lub że środowisko wirtualne jest aktywne:
```bash
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows
```

---

## 10. Stan implementacji — co jeszcze nie gotowe

Poniższe elementy są zaplanowane, ale jeszcze niezaimplementowane. Dokumentacja opisuje ich docelowe działanie.

| Element                                     | Status      | Uwagi                                                                                                                                           |
|---------------------------------------------|-------------|-------------------------------------------------------------------------------------------------------------------------------------------------|
| `tests/` — testy jednostkowe i integracyjne | ⏳ W trakcie | Folder istnieje, testy do napisania (`pytest` zwróci "no tests found")                                                                          |
| `src/metrics/` — moduł metryk               | ⏳ W trakcie | Funkcje `attention_entropy`, `sparsity_ratio`, `pairwise_attention_diff`, `mean_attention_by_category` zaplanowane, jeszcze niezaimplementowane |
| Logowanie metryk do W&B                     | ⏳ W trakcie | W&B działa i loguje heatmapy jako obrazy; logowanie metryk ilościowych (entropia, sparsity) czeka na `src/metrics/`                             |
| Diff heatmapy i wykresy zbiorcze            | ⏳ W trakcie | Wizualizacja `modified - base` i bar charty per kategoria tokenów zaplanowane                                                                   |
| Ręczna kategoryzacja tokenów                | ⏳ W trakcie | Wymagana do `mean_attention_by_category`; kategorie: instrukcja / treść / funkcyjny                                                             |