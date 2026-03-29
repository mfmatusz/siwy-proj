# SIWY: Wpływ struktury promptu na odpowiedzi llm - analiza attention

Projekt badawczy realizowany w ramach przedmiotu **Wyjaśnialna Sztuczna Inteligencja (SIWY)**.
Celem projektu jest zbadanie za pomocą narzędzi XAI, w jaki sposób drobne zmiany w poleceniach (np. ton, kontekst, ograniczenia) wpływają na rozkład mechanizmu "uwagi" (Attention Matrix) w dużych modelach językowych (na przykładzie *Gemma-3-4B*).

**Autorzy (Zespół):**
- Bartłomiej Dmitruk
- Maciej Matuszewski
- Magdalena Kalińska

## Wymagania i instalacja

Projekt używa zarządzania zależnościami za pomocą `pyproject.toml` i wymaga Pythona w wersji np. 3.11.

```bash
# 1. Tworzenie i aktywacja wirtualnego środowiska
python -m venv .venv
source .venv/Scripts/activate  # (Windows: powershell)
# lub: source .venv/bin/activate (Linux/Mac)

# 2. Instalacja zależności projektu w trybie edytowalnym
pip install -e .
```

## Konfiguracja i Uruchamianie

Projekt jest integrowany z narzędziem **Weights & Biases (WandB)** do śledzenia eksperymentów oraz systemem **Hydra** do wstrzykiwania konfiguracji.
Cała główna konfiguracja hiperparametrów eksperymentu żyje w pliku `conf/config.yaml`.

```bash
# Wykonanie eksperymentu
python run_experiment.py
```