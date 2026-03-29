# TODO

## Setup projektu

- [ ] Dodać `USAGE.md` (instrukcja użytkowania — wymagana przez regulamin §1 pkt 3, 9)

## Dane

- [ ] Rozbudować dataset do pełnego rozmiaru (więcej par per kategoria)
- [ ] Zapewnić niemutowalność oryginalnych danych — dane surowe w `data/raw/` tylko do odczytu, wyniki w `data/processed/` (regulamin §2)

## Konfiguracja

- [ ] Przenieść stałe modelu (NUM_LAYERS, GLOBAL_LAYER_INDICES, GQA_GROUP_SIZE) do config.yaml z walidacją przy starcie

## Śledzenie eksperymentów

- [ ] Dodać logowanie metryk (entropia, sparsity, średnie wagi per kategoria) do W&B

## Testy

- [ ] Napisać testy `pytest` (regulamin §1 pkt 3, 9 — brak testów = niezaliczenie)
- [ ] Testy jednostkowe: ekstrakcja attention, agregacja GQA, kategoryzacja tokenów, obliczanie metryk
- [ ] Testy integracyjne: pipeline end-to-end na małym modelu/mock

## Dokumentacja

- [ ] Rozbudować `README.md` — opis projektu, setup, wymagania, struktura repo
- [ ] Napisać `USAGE.md` — instrukcja użytkowania krok po kroku
- [ ] Przygotować system notatek do weekly standup (regulamin §1 pkt 7) — np. `docs/weekly/` lub GitHub Issues

## Analiza literatury (design-proposal)

- [ ] Rozbudować tabelę literatury o brakujące kolumny wymagane przez regulamin §1 pkt 6: autorski komentarz, dostępność kodu/modeli, metryki ewaluacji w artykule, zasoby obliczeniowe autorów

## Design-proposal

- [ ] Opisać planowaną funkcjonalność programu (co dostaje użytkownik: CLI? skrypt? jakie komendy?)

## Moduł metryk (src/metrics/)

- [ ] Utworzyć moduł `src/metrics/` z funkcjami:
  - [ ] `attention_entropy` — entropia rozkładu attention per token
  - [ ] `mean_attention_by_category` — średnia waga attention per kategoria tokenu (instrukcja/treść/funkcyjny)
  - [ ] `pairwise_attention_diff` — różnica rozkładów attention między parami promptów (base vs modified)
  - [ ] `sparsity_ratio` — procent near-zero wag attention
- [ ] Zintegrować metryki z pipeline'em (`run_experiment.py`) i logowaniem do W&B
- [ ] Ręczna kategoryzacja tokenów jako instrukcja/treść/funkcyjny — potrzebne do `mean_attention_by_category`

## Analiza i wizualizacje

- [ ] Diff heatmapy (modified minus base) — wizualizacja co się zmienia między parami
- [ ] Wykresy zbiorcze: bar chart średniej attention na tokeny instrukcji vs treści per kategoria promptów
- [ ] Gradient attribution (saliency, integrated gradients) via Inseq — wymaga GPU z CUDA, na MPS zbyt wolne (~40 min/prompt)

## Deliverables

- [ ] Raport z obserwacji
- [ ] Dokumentacja + instrukcja użytkowania
- [ ] Testy pytest
- [ ] Filmik demo (3–5 min)
- [ ] Prezentacja finalna
