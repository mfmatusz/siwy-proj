# TODO

## Dane

- [ ] Rozbudować dataset do pełnego rozmiaru (więcej par per kategoria)
- [x] Zapewnić niemutowalność oryginalnych danych — dane surowe w `data/raw/` tylko do odczytu, wyniki w `data/processed/` (regulamin §2)

## Konfiguracja

- [ ] Przenieść stałe modelu (NUM_LAYERS, GLOBAL_LAYER_INDICES, GQA_GROUP_SIZE) do config.yaml z walidacją przy starcie

## Śledzenie eksperymentów

- [x] Dodać logowanie metryk (entropia, sparsity, normy diff L1/L2) do W&B

## Testy

- [x] Napisać testy `pytest` (regulamin §1 pkt 3, 9)
- [x] Testy jednostkowe: ekstrakcja attention, agregacja GQA, dataset, metryki, wizualizacja
- [ ] Testy integracyjne: pipeline end-to-end na małym modelu/mock

## Dokumentacja

- [x] Rozbudować `README.md` — struktura repo
- [x] Edytować `USAGE.md` — instrukcja użytkowania krok po kroku
- [ ] Przygotować system notatek do weekly standup (regulamin §1 pkt 7) — np. `docs/weekly/` lub GitHub Issues

## Analiza literatury (design-proposal)

- [ ] Rozbudować tabelę literatury w miarę rozwoju projektu

## Design-proposal

- [ ] Opisać planowaną funkcjonalność programu (co dostaje użytkownik: CLI? skrypt? jakie komendy?)

## Moduł metryk (src/metrics/)

- [x] Utworzyć moduł `src/metrics/` z funkcjami:
  - [x] `attention_entropy` — entropia rozkładu attention per token
  - [x] `mean_attention_by_token_position` — średnia waga attention per pozycja tokenu
  - [x] `pairwise_attention_diff` — różnica rozkładów attention między parami promptów (base vs modified)
  - [x] `sparsity_ratio` — procent near-zero wag attention
- [x] Zintegrować metryki z pipeline'em (`run_experiment.py`) i logowaniem do W&B
- [ ] Ręczna kategoryzacja tokenów jako instrukcja/treść/funkcyjny — potrzebne do `mean_attention_by_category`
- [ ] `mean_attention_by_category` (po ukończeniu kategoryzacji tokenów)

## Analiza i wizualizacje

- [x] Diff heatmapy (modified minus base) — wizualizacja co się zmienia między parami
- [ ] Wykresy zbiorcze: bar chart średniej attention na tokeny instrukcji vs treści per kategoria promptów
- [ ] Gradient attribution (saliency, integrated gradients) via Inseq — wymaga GPU z CUDA, na MPS zbyt wolne (>40 min/prompt)

## Deliverables

- [ ] Raport z obserwacji
- [x] Dokumentacja + instrukcja użytkowania
- [x] Testy pytest
- [ ] Filmik demo (3–5 min)
- [ ] Prezentacja finalna
