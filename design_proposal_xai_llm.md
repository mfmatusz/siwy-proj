# Wpływ struktury promptu na odpowiedzi modelu językowego - analiza attention

Przedmiot: Wyjaśnialna Sztuczna Inteligencja (SIWY)
Zespół: Bartłomiej Dmitruk, Maciej Matuszewski, Magdalena Kalińska
Semestr: 26L

---

## Cel i problem badawczy

Małe zmiany w treści promptu potrafią znacząco zmienić charakter odpowiedzi modelu językowego. Celem projektu jest zbadanie, które tokeny w prompcie przyciągają największą uwagę modelu (mierzoną wagami attention) i czy pokrywa się to z intuicją człowieka co do „ważnych" części zapytania.

Projekt łączy budowę narzędzia do wizualizacji i analizy attention z kontrolowanym eksperymentem na zbiorze promptów, z którego wyciągamy obserwacje ilościowe i jakościowe.

---

## Pytania badawcze

RQ1: Czy tokeny pełniące funkcję instrukcji (np. „krótko", „jako ekspert", „napisz formalnie") mają wyższe wagi attention niż tokeny treści merytorycznej?

RQ2: Czy zmiana typu modyfikacji promptu (styl, ton, formalność, framing, reformulacja) prowadzi do różnych wzorców attention przy tej samej treści bazowej?

RQ3: Czy różnice w rozkładzie attention między parami promptów są widoczne jednakowo w warstwach lokalnych (sliding window) i globalnych?

---

## Przegląd literatury

| #   | Tytuł / Autorzy                                                                       | Rok  | Główna teza                                                                                                                              |
| --- | ------------------------------------------------------------------------------------- | ---- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| 1   | _Attention Is All You Need_ - Vaswani et al.                                          | 2017 | Architektura Transformer i mechanizm attention                                                                                           |
| 2   | _Attention is not Explanation_ - Jain & Wallace                                       | 2019 | Wagi attention nie są bezpośrednim wyjaśnieniem decyzji modelu                                                                           |
| 3   | _Attention is not not Explanation_ - Wiegreffe & Pinter                               | 2019 | Attention może być częściowym wyjaśnieniem przy odpowiedniej metodologii                                                                 |
| 4   | _What Does BERT Look At?_ - Clark et al.                                              | 2019 | Klasyfikacja głów attention wg funkcji (syntaktyczne, pozycyjne, separator-attending); metodologia analizy per-head                      |
| 5   | _A Survey of XAI for NLP_ - Danilevsky et al.                                         | 2020 | Przegląd metod wyjaśnialności dla modeli językowych                                                                                      |
| 6   | _Prompt Programming for LLMs_ - Reynolds & McDonell                                   | 2021 | Analiza efektów różnych typów promptów na zachowanie modelu                                                                              |
| 7   | _How Good is Your Tokenizer?_ - Rust et al.                                           | 2021 | Wpływ tokenizera na wydajność modelu w językach nieanglojęzycznych; tokenizacja determinuje rozkład attention na subtokenach             |
| 8   | _Eliciting Latent Predictions from Transformers with the Tuned Lens_ - Belrose et al. | 2023 | Analiza predykcji każdej warstwy transformera przez wyuczoną projekcję (tuned lens); uzupełnia analizę attention o perspektywę per-layer |
| 9   | _NNsight and NDIF_ - Fiotto-Kaufman et al.                                            | 2024 | Narzędzie do introspekcji wewnętrznych stanów modeli HuggingFace; dostęp do attention, aktywacji i gradientów w jednym API               |

---

## Metodologia

### Model

Gemma 3 4B (Google, licencja Gemma) - dense transformer, 34 warstwy, 8 głowic attention (GQA 2:1), kontekst 128k tokenów. Inferencja lokalna przez HuggingFace Transformers z `output_attentions=True`, co daje pełny dostęp do macierzy attention per warstwa i per głowica. W BF16 sam model zajmuje ~8 GB VRAM, z KV cache przy 32k kontekście ~12.7 GB. Z kwantyzacją 4-bit (`bitsandbytes` NF4) wagi zajmują ~2.6 GB, z KV cache przy 32k ~7.3 GB. Sliding window attention w większości warstw + globalne attention co kilka warstw — istotne przy interpretacji wzorców attention.

### Zbiór promptów

Ręcznie przygotowany zestaw promptów w parach (bazowy vs zmodyfikowany) w pięciu kategoriach:

| Kategoria               | Opis                                              | Przykład pary                                                                         |
| ----------------------- | ------------------------------------------------- | ------------------------------------------------------------------------------------- |
| Instrukcja stylistyczna | Ta sama treść +/- dyrektywa stylu                 | „Wyjaśnij grawitację" vs. „Wyjaśnij grawitację krótko i prosto"                       |
| Ton / rola              | Ta sama treść, zmiana tonu lub przypisanie roli   | „Co to jest inflacja?" vs. „Jako ekonomista, wyjaśnij czym jest inflacja"             |
| Język / formalność      | Ta sama treść, zmiana rejestru językowego         | „Jak działa internet?" vs. „Proszę o wyjaśnienie zasad funkcjonowania sieci Internet" |
| Kontekst / framing      | Ta sama treść, dodanie kontekstu lub ograniczenia | „Opisz fotosyntezę" vs. „Dla ucznia 5. klasy opisz fotosyntezę"                       |
| Negacja / reformulacja  | Ta sama intencja, inna struktura zdania           | „Co powoduje deszcz?" vs. „Dlaczego pada deszcz?"                                     |

### Analiza

Dla każdego promptu:

1. Forward pass przez model z `output_attentions=True`, ekstrakcja wag attention ze wszystkich 34 warstw.
2. Osobna analiza warstw lokalnych (sliding window, 1024 tokeny) i globalnych (pełny kontekst) — porównanie wzorców.
3. Agregacja wag po głowicach (mean pooling) z uwzględnieniem struktury GQA (grupy po 2 głowice query na 1 KV).
4. Wizualizacja heatmap attention na tokenach promptu (`bertviz` + wykresy per-warstwa).
5. Ręczna kategoryzacja tokenów jako: _instrukcja_, _treść_, _funkcyjny_ (spójnik, przyimek itp.).
6. Porównanie średnich wag attention między kategoriami tokenów, osobno per typ warstwy.

---

## Metryki ewaluacji

| Metryka                                     | Opis                                                                     |
| ------------------------------------------- | ------------------------------------------------------------------------ |
| Średnia waga attention per kategoria tokenu | Czy tokeny-instrukcje mają wyższe wagi niż tokeny treści?                |
| Entropia rozkładu attention                 | Czy model skupia uwagę czy rozkłada ją równomiernie?                     |
| Różnica wag między parami promptów          | O ile zmienia się rozkład attention po dodaniu instrukcji stylistycznej? |

Analiza jest jakościowa i ilościowa - obok liczb ważne są obserwacje i komentarz do wizualizacji.

---

## Narzędzia i technologie

| Element                 | Technologia                                     |
| ----------------------- | ----------------------------------------------- |
| Język                   | Python 3.11                                     |
| Model / inference       | `transformers` (Gemma 3 4B, `bitsandbytes` NF4) |
| Wizualizacja attention  | `bertviz`                                       |
| Obliczenia              | `numpy`, `pandas`                               |
| Wykresy                 | `matplotlib`, `seaborn`                         |
| Środowisko              | `venv` + `pyproject.toml`                       |
| Linting / formatowanie  | `ruff`                                          |
| Testy                   | `pytest`                                        |
| Uruchamianie            | Skrypty `typer`                                 |
| Dokumentacja            | `README.md` + `USAGE.md`                        |
| Wersjonowanie           | Git + Conventional Commits                      |
| Śledzenie eksperymentów | Pliki `.csv` z wynikami + opcjonalnie W&B       |

Wymagany GPU z min. ~8 GB VRAM (NF4 + KV cache przy 32k) lub ~13 GB (BF16 + KV cache przy 32k). Inferencja lokalna przez HuggingFace Transformers z `attn_implementation="eager"`.

---

## Harmonogram

| Tydzień | Daty          | Zadania                                                                                                     |
| ------- | ------------- | ----------------------------------------------------------------------------------------------------------- |
| 1       | 01.04 – 07.04 | Setup repozytorium, środowisko, pierwszy forward pass przez Gemma 3 4B (Transformers), ekstrakcja attention |
| 2       | 08.04 – 14.04 | Przygotowanie datasetu promptów (5 kategorii), skrypt do batch inference i zapisu wyników                   |
| 3       | 15.04 – 21.04 | Prototyp: działające wizualizacje attention dla kilku przykładów, wstępna analiza literaturowa              |
| 4       | 22.04 – 28.04 | Obliczenie metryk dla pełnego datasetu, wstępne obserwacje, testy jednostkowe                               |
| 5       | 29.04 – 05.05 | Analiza wyników, wykresy porównawcze, dokumentacja                                                          |
| 6       | 06.05 – 12.05 | Finalizacja, nagranie filmiku (3–5 min), przygotowanie prezentacji                                          |
| 7       | 13.05 – ...   | Prezentacja finalna                                                                                         |

---

## Ryzyka i ograniczenia

| Ryzyko                                                                                               | Mitygacja                                                                                       |
| ---------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| Sliding window attention w Gemma 3 zmienia rozkład wag w zależności od warstwy (lokalna vs globalna) | Osobna analiza warstw z globalnym attention vs sliding window; porównanie wzorców               |
| Wymagany GPU z min. ~8 GB VRAM (NF4 + KV cache)                                                      | Kwantyzacja NF4 przez bitsandbytes (~7.3 GB przy 32k kontekście); alternatywnie BF16 (~12.7 GB) |
| Ręczna kategoryzacja tokenów jest subiektywna                                                        | Dwie osoby kategoryzują niezależnie, liczymy zgodność                                           |
| Mały dataset ogranicza generalizowalność                                                             | Świadomie traktujemy to jako studium przypadku, nie twierdzimy o ogólności wniosków             |

---

## Deliverables

- [x] Design Proposal (ten dokument)
- [ ] Dataset promptów
- [ ] Skrypt do ekstrakcji i wizualizacji attention
- [ ] Raport z obserwacji
- [ ] Dokumentacja
- [ ] Testy
- [ ] Filmik demo (3–5 min)
- [ ] Prezentacja finalna

---

## Bibliografia

1. Vaswani, A., et al. (2017). _Attention Is All You Need_. NeurIPS 2017. https://arxiv.org/abs/1706.03762
2. Jain, S., & Wallace, B. C. (2019). _Attention is not Explanation_. NAACL 2019. https://arxiv.org/abs/1902.10186
3. Wiegreffe, S., & Pinter, Y. (2019). _Attention is not not Explanation_. EMNLP 2019. https://arxiv.org/abs/1908.04626
4. Clark, K., Khandelwal, U., Levy, O., & Manning, C. D. (2019). _What Does BERT Look At? An Analysis of BERT's Attention_. BlackboxNLP @ ACL 2019. https://arxiv.org/abs/1906.04341
5. Danilevsky, M., et al. (2020). _A Survey of the State of Explainable AI for Natural Language Processing_. AACL-IJCNLP 2020. https://arxiv.org/abs/2010.00711
6. Reynolds, L., & McDonell, K. (2021). _Prompt Programming for Large Language Models: Beyond the Few-Shot Paradigm_. CHI EA 2021. https://arxiv.org/abs/2102.07350
7. Rust, P., Pfeiffer, J., Vulic, I., Ruder, S., & Gurevych, I. (2021). _How Good is Your Tokenizer? On the Monolingual Performance of Multilingual Language Models_. ACL-IJCNLP 2021. https://arxiv.org/abs/2012.15613
8. Belrose, N., et al. (2023). _Eliciting Latent Predictions from Transformers with the Tuned Lens_. arXiv 2023. https://arxiv.org/abs/2303.08112
9. Fiotto-Kaufman, J., et al. (2024). _NNsight and NDIF: Democratizing Access to Foundation Model Internals_. ICLR 2025. https://arxiv.org/abs/2407.14561
