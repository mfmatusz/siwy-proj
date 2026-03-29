# TODO

## Setup projektu

- [x] Przejść z setuptools na `uv` jako package manager
- [x] Zastąpić `logging` na `loguru`
- [ ] Dodać `USAGE.md` (instrukcja użytkowania — wymagana przez regulamin §1 pkt 3, 9)
- [x] Dodać `invoke` (pyinvoke) do budowania, testowania, uruchamiania (regulamin §2)
- [x] Skonfigurować `ruff` w `pyproject.toml` (linter + formatter, styl PEP8 ze zwiększonym limitem linii)
- [x] Dodać `ruff check` i `ruff format` do tasks.py (invoke)
- [ ] Ustalić strukturę projektu przez cookiecutter/copier (regulamin §2 — rzetelna struktura projektu)

## Kod — rozbieżności z raportem

- [ ] `run_experiment.py` analizuje tylko ostatnią warstwę (`attrs[-1]`) — raport deklaruje analizę wszystkich 34 warstw z podziałem na lokalne vs globalne
- [x] Brak `attn_implementation="eager"` przy ładowaniu modelu — już było w kodzie
- [ ] Dodać analizę osobno per typ warstwy (lokalna sliding window vs globalna)
- [ ] Uwzględnić strukturę GQA przy agregacji głowic (grupy po 2 query na 1 KV)

## Dane

- [ ] Przygotować dataset promptów w 5 kategoriach (plik `data/raw/prompts.json`)
- [ ] Zapewnić niemutowalność oryginalnych danych — dane surowe w `data/raw/` tylko do odczytu, wyniki w `data/processed/` (regulamin §2)
- [x] Dodać `.gitkeep` w `data/processed/`

## Konfiguracja

- [ ] Odseparować konfigurację od kodu wykonawczego (regulamin §2) — config.yaml istnieje, ale upewnić się że wszystkie parametry są konfigurowalne (warstwy do analizy, metryki, ścieżki wyjściowe)

## Śledzenie eksperymentów

- [x] W&B jest w kodzie, ale raport mówi "opcjonalnie" — zmienione na wymagane w design-proposal
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
- [x] Dodać do harmonogramu wzmiankę o zasobach obliczeniowych

## Design-proposal — drobne poprawki

- [ ] Opisać planowaną funkcjonalność programu (co dostaje użytkownik: CLI? skrypt? jakie komendy?)
- [x] Zmienić "opcjonalnie W&B" na wymagane w tabeli narzędzi
- [x] Zmienić typer na hydra + invoke w tabeli narzędzi

## Deliverables (przed końcem projektu)

- [ ] Dataset promptów (5 kategorii, pary bazowy/zmodyfikowany)
- [ ] Skrypt do ekstrakcji i wizualizacji attention
- [ ] Raport z obserwacji
- [ ] Dokumentacja + instrukcja użytkowania
- [ ] Testy pytest
- [ ] Filmik demo (3–5 min)
- [ ] Prezentacja finalna
