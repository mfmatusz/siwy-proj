# Wpływ struktury promptu na odpowiedzi modelu językowego - analiza attention

Przedmiot: Wyjaśnialna Sztuczna Inteligencja (SIWY), Semestr: 26L

Zespół: Bartłomiej Dmitruk, Maciej Matuszewski, Magdalena Kalińska

---

## Design Proposal

### Cel i problem badawczy

Małe zmiany w treści promptu potrafią znacząco zmienić charakter odpowiedzi modelu językowego. Celem projektu jest zbadanie, które tokeny w prompcie przyciągają największą uwagę modelu (mierzoną wagami attention) i czy pokrywa się to z intuicją człowieka co do „ważnych" części zapytania.

Projekt łączy budowę narzędzia do wizualizacji i analizy attention z kontrolowanym eksperymentem na zbiorze promptów, z którego wyciągamy obserwacje ilościowe i jakościowe.

---

### Pytania badawcze

RQ1: Czy tokeny pełniące funkcję instrukcji (np. „krótko", „jako ekspert", „napisz formalnie") mają wyższe wagi attention niż tokeny treści merytorycznej?

RQ2: Czy zmiana typu modyfikacji promptu (styl, ton, formalność, framing, reformulacja) prowadzi do różnych wzorców attention przy tej samej treści bazowej?

RQ3: Czy różnice w rozkładzie attention między parami promptów są widoczne jednakowo w warstwach lokalnych (sliding window) i globalnych?

---

### Przegląd literatury

Tabela spełnia wymagania §1 pkt 6 regulaminu.

| #   | Tytuł / Autorzy                                                                                                   | Rok  | Główna teza                                                                                                                                                                    | Link                                      | Kod / modele                                                                                                                                       | Metryki ewaluacji w artykule                                                                                                    | Zasoby obliczeniowe autorów                                                              | Autorski komentarz                                                                                                                                                                                                                                                             |
| --- | ----------------------------------------------------------------------------------------------------------------- | ---- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 1   | _Attention Is All You Need_ — Vaswani et al.                                                                      | 2017 | Architektura Transformer oparta wyłącznie na mechanizmie attention, bez rekurencji                                                                                             | [arXiv](https://arxiv.org/abs/1706.03762) | ✅ Kod: [tensor2tensor](https://github.com/tensorflow/tensor2tensor); modele niedostępne bezpośrednio, lecz BERT/GPT-2 są następnikami             | BLEU (WMT En-De: 28.4, En-Fr: 41.0)                                                                                             | 8× NVIDIA P100 GPU; model bazowy ~12h, model duży ~3.5 doby                              | Definiuje mechanizm attention używany w Gemma 3. Niezbędna jako punkt wyjścia. Sliding window attention w Gemma to bezpośrednie rozwinięcie tego mechanizmu.                                                                                                                   |
| 2   | _Attention is not Explanation_ — Jain & Wallace                                                                   | 2019 | Wagi attention nie są wiarygodnym wyjaśnieniem decyzji modelu — alternatywne rozkłady attention dają te same predykcje                                                         | [arXiv](https://arxiv.org/abs/1902.10186) | ✅ Kod: [github.com/successar/AttentionExplanation](https://github.com/successar/AttentionExplanation); brak pre-trenowanych modeli                | Korelacja Kendall τ między attention a gradientami; wskaźnik sukcesu ataków permutacyjnych na attention                         | Pojedyncze GPU; eksperymenty na klasycznych zadaniach NLU (NLI, klasyfikacja sentymentu) | Kluczowa dla kontekstowania wyników — nasze obserwacje attention muszą być interpretowane ostrożnie. Uzasadnia stosowanie uzupełniających metod atrybucji (Inseq).                                                                                                             |
| 3   | _Attention is not not Explanation_ — Wiegreffe & Pinter                                                           | 2019 | Attention może być częściowym wyjaśnieniem modelu przy właściwej metodologii i odpowiednich założeniach                                                                        | [arXiv](https://arxiv.org/abs/1908.04626) | ✅ Kod dostępny; brak pre-trenowanych modeli                                                                                                       | Korelacja Kendall τ; testy diagnostycznych klasyfikatorów; miara wierności (faithfulness)                                       | Pojedyncze GPU; te same datasety co Jain & Wallace                                       | Razem z 2) stanowią tło metodologiczne — projekt musi pozycjonować się względem obu prac przy interpretacji heatmap attention.                                                                                                                                                 |
| 4   | _What Does BERT Look At?_ — Clark et al.                                                                          | 2019 | Głowice attention BERT specjalizują się w konkretnych funkcjach: syntaktycznej, pozycyjnej, skupionej na tokenach specjalnych                                                  | [arXiv](https://arxiv.org/abs/1906.04341) | ✅ Kod: [github.com/clarkkev/attention-analysis](https://github.com/clarkkev/attention-analysis); modele BERT dostępne przez HuggingFace           | UAS (unlabeled attachment score) dla parsowania zależnościowego; accuracy dla koreferencji                                      | BERT-Large na TPU v3 (infrastruktura Google)                                             | Metodologia analizy per-head jest wzorcem — projekt stosuje analogiczną agregację po głowicach i warstwy. Pojęcie „głowic syntaktycznych" vs. „pozycyjnych" pomocne przy interpretacji wyników.                                                                                |
| 5   | _A Survey of XAI for NLP_ — Danilevsky et al.                                                                     | 2020 | Systematyczny przegląd metod wyjaśnialności dla modeli językowych: attention, gradienty, surrogate models                                                                      | [arXiv](https://arxiv.org/abs/2010.00711) | ❌ Artykuł przeglądowy; brak kodu ani modeli                                                                                                       | N/A (survey)                                                                                                                    | N/A (survey)                                                                             | Dobry punkt orientacyjny w krajobrazie XAI dla NLP. Uzasadnia wybór attention i gradientowej atrybucji (Inseq) jako uzupełniających metod. Nieco przestarzały — nie obejmuje najnowszych LLM.                                                                                  |
| 6   | _Prompt Programming for LLMs_ — Reynolds & McDonell                                                               | 2021 | Struktura promptu istotnie zmienia zachowanie modelu; meta-prompting i few-shot to odrębne techniki z różnymi efektami                                                         | [arXiv](https://arxiv.org/abs/2102.07350) | ❌ Brak kodu; GPT-3 dostępny tylko przez API (OpenAI)                                                                                              | Jakościowa analiza wyników; accuracy na benchmarkach few-shot (SuperGLUE)                                                       | GPT-3 API (model 175B, infrastruktura OpenAI)                                            | Prompt ma znaczenie. Projekt rozszerza tę obserwację o mechanistyczną analizę attention. Brak kodu i zamknięty model to ograniczenie.                                                                                                                                          |
| 7   | _How Good is Your Tokenizer?_ — Rust et al.                                                                       | 2021 | Jakość tokenizatora dla danego języka silnie wpływa na wydajność modelu; subtokenizacja zmienia rozkład attention na morfemach                                                 | [arXiv](https://arxiv.org/abs/2012.15613) | ✅ Kod dostępny; modele mBERT i XLM-R dostępne przez HuggingFace                                                                                   | F1 NER, accuracy POS, UAS — ewaluacja na 9 językach                                                                             | Standard GPU (fine-tuning)                                                               | Istotna przy interpretacji attention na subtokenach — Gemma 3 używa SentencePiece, więc tokeny polskie mogą być rozbite na wiele podjednostek. Wpływa na sposób kategoryzacji tokenów (instrukcja/treść/funkcyjny).                                                            |
| 8   | _Eliciting Latent Predictions from Transformers with the Tuned Lens_ — Belrose et al.                             | 2023 | Wyuczone projekcje (tuned lens) pozwalają śledzić ewolucję predykcji modelu warstwa po warstwie                                                                                | [arXiv](https://arxiv.org/abs/2303.08112) | ✅ Kod: [github.com/AlignmentResearch/tuned-lens](https://github.com/AlignmentResearch/tuned-lens); modele GPT-2, GPT-J dostępne przez HuggingFace | Perplexity; KL-dywergencja między predykcją pośrednią a finalną                                                                 | Klaster A100 (multi-GPU)                                                                 | Uzupełnia analizę attention o perspektywę per-layer: nie tylko „na co patrzy model", ale „co przewiduje model w każdej warstwie". Możliwa przyszła integracja z projektem jako metoda weryfikacji hipotez.                                                                     |
| 9   | _NNsight and NDIF_ — Fiotto-Kaufman et al.                                                                        | 2024 | Otwarta infrastruktura do introspekcji wewnętrznych stanów dużych modeli (attention, aktywacje, gradienty) przez zunifikowane API                                              | [arXiv](https://arxiv.org/abs/2407.14561) | ✅ Kod: [github.com/ndif-team/nnsight](https://github.com/ndif-team/nnsight); kompatybilny z dowolnym modelem HuggingFace                          | N/A (narzędzie); case studies na GPT-2 i Llama                                                                                  | Infrastruktura NDIF (Northeastern University); rozproszony klaster GPU                   | Alternatywne podejście do ekstrakcji attention względem naszego pipeline HuggingFace + `output_attentions=True`. Bardziej elastyczne API, ale wprowadza zewnętrzną zależność od infrastruktury NDIF.                                                                           |
| 10  | _A Multiscale Visualization of Attention in the Transformer Model (BERTViz)_ — Vig                                | 2019 | Interaktywna wizualizacja attention na wielu skalach (head view, model view, neuron view) dla modeli Transformer                                                               | [arXiv](https://arxiv.org/abs/1906.05714) | ✅ Kod: [github.com/jessevig/bertviz](https://github.com/jessevig/bertviz); kompatybilny z modelami HuggingFace                                    | Jakościowe case studies (BERT, GPT-2): bias, powiązania koreferentne                                                            | Pojedyncze GPU; BERT-base i GPT-2 (small)                                                | Projekt rozbudowuje podejście BERTViz o ilościowe metryki i analizę diff między parami promptów.                                                                                                                                                                               |
| 11  | _Inseq: An Interpretability Toolkit for Sequence Generation Models_ — Sarti et al.                                | 2023 | Pythonowa biblioteka do post-hoc atrybucji wag (attention, saliency, integrated gradients) dla modeli generatywnych                                                            | [arXiv](https://arxiv.org/abs/2302.13942) | ✅ Kod: [github.com/inseq-team/inseq](https://github.com/inseq-team/inseq); kompatybilny z modelami HuggingFace (w tym Gemma)                      | Jakościowe: lokalizacja wiedzy faktograficznej w GPT-2; wykrywanie bias w tłumaczeniu maszynowym                                | GPU (MarianNMT + GPT-2); skala akademicka                                                | Uzasadnia wybór saliency i integrated gradients jako metod komplementarnych do analizy attention. Praca pokazuje, że `attention` jako metoda atrybucji w Inseq ma dobrą efektywność na modelach decoder-only.                                                                  |
| 12  | _Gemma 3 Technical Report_ — Gemma Team, Google DeepMind                                                          | 2025 | Opis architektury rodziny modeli Gemma 3 (1B–27B): multimodalność, kontekst 128k tokenów, GQA, sliding window attention z wyższą proporcją warstw lokalnych do globalnych      | [arXiv](https://arxiv.org/abs/2503.19786) | ✅ Modele dostępne na HuggingFace ([google/gemma-3-4b-pt](https://huggingface.co/google/gemma-3-4b-pt)) pod licencją Gemma; brak kodu treningowego | MMLU, MATH, HumanEval, WMT; benchmarki wizyjne (ActivityNet-QA, RealWorldQA); model 4B trenowany na 4 bilionach tokenów         | Infrastruktura TPU Google; skala danych i zasobów nie ujawniona publicznie               | Opisuje dokładną architekturę modelu użytego w projekcie. Kluczowe szczegóły: stosunek warstw lokalnych do globalnych (istotny dla interpretacji heatmap), GQA (2:1), RMSNorm, QK-norm. Uzasadnia metodologiczny podział na analizę warstw lokalnych vs globalnych w pipeline. |
| 13  | _Large Language Models Sensitivity to The Order of Options in Multiple-Choice Questions_ — Pezeshkpour & Hruschka | 2023 | Modele językowe wykazują znaczną wrażliwość na kolejność opcji w pytaniach wielokrotnego wyboru — zmiana kolejności powoduje spadek accuracy o 13–75%                          | [arXiv](https://arxiv.org/abs/2308.11483) | ❌ Brak publicznego kodu; eksperymenty na modelach przez API (GPT-3, GPT-4, LLaMA)                                                                 | Accuracy na MMLU, TruthfulQA, ARC; performance gap przy przestawieniu opcji; poprawa o do 8 pp. po kalibracji                   | API inference (GPT-3/4, OpenAI); LLaMA inference na GPU                                  | Pokazuje, że forma promptu wpływa na zachowanie modelu nawet przy zachowaniu treści. Uzupełnia pracę 6) o ilościowe dowody wrażliwości. Nasz projekt bada mechanizm tego zjawiska od strony attention.                                                                         |
| 14  | _Successor Heads: Recurring, Interpretable Attention Heads In The Wild_ — Gould, Ong, Ogden, Conmy                | 2023 | Niektóre głowice attention konsekwentnie realizują konkretną, interpretowalną funkcję (inkrementację sekwencji) we wszystkich badanych architekturach (GPT-2, Pythia, Llama-2) | [arXiv](https://arxiv.org/abs/2312.09230) | ✅ Modele użyte w badaniach dostępne przez HuggingFace; kod eksperymentów dostępny; prezentacja na ICLR 2024                                       | Accuracy predykcji następnika w sekwencjach; loss na przykładach z następstwem; wyniki arytmetyki wektorowej na reprezentacjach | Klaster GPU (akademicki); modele od 31M do 12B parametrów                                | Wzorzec metodologiczny. Potwierdza, że analiza per-head ujawnia specjalizację głowic. Wspiera hipotezę, że tokeny instrukcyjne mogą być obsługiwane przez wyspecjalizowane głowice. Razem z pracą 4) tworzy zaplecze dla interpretacji wyników per-head.                       |

### Pokrycie tematyczne literatury

| Obszar                                                     | Pokryte przez                                        |
| ---------------------------------------------------------- | ---------------------------------------------------- |
| Fundamenty mechanizmu attention                            | #1 (Vaswani 2017)                                    |
| Wyjaśnialność attention — debata metodologiczna            | #2 (Jain 2019), #3 (Wiegreffe 2019)                  |
| Analiza per-head i per-layer                               | #4 (Clark 2019), #8 (Belrose 2023), #14 (Gould 2023) |
| Przegląd metod XAI dla NLP                                 | #5 (Danilevsky 2020)                                 |
| Wpływ struktury promptu na zachowanie modelu               | #6 (Reynolds 2021), #13 (Pezeshkpour 2023)           |
| Tokenizacja i jej wpływ na attention                       | #7 (Rust 2021)                                       |
| Narzędzia do introspekcji modeli                           | #9 (NNsight 2024)                                    |
| Wizualizacja attention — narzędzie w projekcie             | #10 (BERTViz / Vig 2019)                             |
| Atrybucja dla modeli generatywnych — narzędzie w projekcie | #11 (Inseq / Sarti 2023)                             |
| Architektura modelu użytego w projekcie                    | #12 (Gemma 3 Technical Report 2025)                  |

---

### Metodologia

#### Model

Gemma 3 4B (Google, licencja Gemma) - dense transformer, 34 warstwy, 8 głowic attention (GQA 2:1), kontekst 128k tokenów. Inferencja lokalna przez HuggingFace Transformers z `output_attentions=True`, co daje pełny dostęp do macierzy attention per warstwa i per głowica. W BF16 sam model zajmuje ~8 GB VRAM, z KV cache przy 32k kontekście ~12.7 GB. Z kwantyzacją 4-bit (`bitsandbytes` NF4) wagi zajmują ~2.6 GB, z KV cache przy 32k ~7.3 GB. Sliding window attention w większości warstw + globalne attention co kilka warstw — istotne przy interpretacji wzorców attention.

#### Zbiór promptów

Ręcznie przygotowany zestaw promptów w parach (bazowy vs zmodyfikowany) w pięciu kategoriach:

| Kategoria               | Opis                                              | Przykład pary                                                                         |
| ----------------------- | ------------------------------------------------- | ------------------------------------------------------------------------------------- |
| Instrukcja stylistyczna | Ta sama treść +/- dyrektywa stylu                 | „Wyjaśnij grawitację" vs. „Wyjaśnij grawitację krótko i prosto"                       |
| Ton / rola              | Ta sama treść, zmiana tonu lub przypisanie roli   | „Co to jest inflacja?" vs. „Jako ekonomista, wyjaśnij czym jest inflacja"             |
| Język / formalność      | Ta sama treść, zmiana rejestru językowego         | „Jak działa internet?" vs. „Proszę o wyjaśnienie zasad funkcjonowania sieci Internet" |
| Kontekst / framing      | Ta sama treść, dodanie kontekstu lub ograniczenia | „Opisz fotosyntezę" vs. „Dla ucznia 5. klasy opisz fotosyntezę"                       |
| Negacja / reformulacja  | Ta sama intencja, inna struktura zdania           | „Co powoduje deszcz?" vs. „Dlaczego pada deszcz?"                                     |

#### Analiza

Dla każdego promptu:

1. Forward pass przez model z `output_attentions=True`, ekstrakcja wag attention ze wszystkich 34 warstw.
2. Osobna analiza warstw lokalnych (sliding window, 1024 tokeny) i globalnych (pełny kontekst) — porównanie wzorców.
3. Agregacja wag po głowicach (mean pooling) z uwzględnieniem struktury GQA (grupy po 2 głowice query na 1 KV).
4. Wizualizacja heatmap attention na tokenach promptu (`bertviz` + wykresy per-warstwa).
5. Ręczna kategoryzacja tokenów jako: _instrukcja_, _treść_, _funkcyjny_ (spójnik, przyimek itp.).
6. Porównanie średnich wag attention między kategoriami tokenów, osobno per typ warstwy.

---

### Metryki ewaluacji

| Metryka                                     | Opis                                                                     |
| ------------------------------------------- | ------------------------------------------------------------------------ |
| Średnia waga attention per kategoria tokenu | Czy tokeny-instrukcje mają wyższe wagi niż tokeny treści?                |
| Entropia rozkładu attention                 | Czy model skupia uwagę czy rozkłada ją równomiernie?                     |
| Różnica wag między parami promptów          | O ile zmienia się rozkład attention po dodaniu instrukcji stylistycznej? |

Analiza jest jakościowa i ilościowa - obok liczb ważne są obserwacje i komentarz do wizualizacji.

---

### Narzędzia i technologie

| Element                 | Technologia                                         |
| ----------------------- | --------------------------------------------------- |
| Język                   | Python 3.11                                         |
| Model / inference       | `transformers` (Gemma 3 4B, `bitsandbytes` NF4)     |
| Atrybucja XAI           | `inseq` (attention, saliency, integrated gradients) |
| Wizualizacja attention  | `bertviz`                                           |
| Obliczenia              | `numpy`, `pandas`                                   |
| Wykresy                 | `matplotlib`, `seaborn`                             |
| Środowisko              | `venv` + `pyproject.toml`                           |
| Linting / formatowanie  | `ruff`                                              |
| Testy                   | `pytest`                                            |
| Uruchamianie            | `hydra` + `invoke` + `Makefile`                     |
| Dokumentacja            | `README.md` + `docs/manual.md`                      |
| Wersjonowanie           | Git + Conventional Commits                          |
| Śledzenie eksperymentów | W&B + pliki `.csv` z wynikami                       |

Wymagany GPU z min. ~8 GB VRAM (NF4 + KV cache przy 32k) lub ~13 GB (BF16 + KV cache przy 32k). Inferencja lokalna przez HuggingFace Transformers z `attn_implementation="eager"`.

---

### Harmonogram

| Tydzień | Daty          | Zadania                                                                                                                                                                                   |
| ------- | ------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1       | 01.04 – 07.04 | Setup repozytorium, środowisko, pierwszy forward pass przez Gemma 3 4B (Transformers), ekstrakcja attention. Brak trenowania modeli — wyłącznie inferencja na GPU z min. ~8 GB VRAM (NF4) |
| 2       | 08.04 – 14.04 | Przygotowanie datasetu promptów (5 kategorii), skrypt do batch inference i zapisu wyników                                                                                                 |
| 3       | 15.04 – 21.04 | Prototyp: działające wizualizacje attention dla kilku przykładów, wstępna analiza literaturowa                                                                                            |
| 4       | 22.04 – 28.04 | Obliczenie metryk dla pełnego datasetu, wstępne obserwacje, testy jednostkowe                                                                                                             |
| 5       | 29.04 – 05.05 | Analiza wyników, wykresy porównawcze, dokumentacja                                                                                                                                        |
| 6       | 06.05 – 12.05 | Finalizacja, nagranie filmiku (3–5 min), przygotowanie prezentacji                                                                                                                        |
| 7       | 13.05 – ...   | Prezentacja finalna                                                                                                                                                                       |

---

### Ryzyka i ograniczenia

| Ryzyko                                                                                               | Mitygacja                                                                                       |
| ---------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------- |
| Sliding window attention w Gemma 3 zmienia rozkład wag w zależności od warstwy (lokalna vs globalna) | Osobna analiza warstw z globalnym attention vs sliding window; porównanie wzorców               |
| Wymagany GPU z min. ~8 GB VRAM (NF4 + KV cache)                                                      | Kwantyzacja NF4 przez bitsandbytes (~7.3 GB przy 32k kontekście); alternatywnie BF16 (~12.7 GB) |
| Ręczna kategoryzacja tokenów jest subiektywna                                                        | Dwie osoby kategoryzują niezależnie, liczymy zgodność                                           |
| Mały dataset ogranicza generalizowalność                                                             | Świadomie traktujemy to jako studium przypadku, nie twierdzimy o ogólności wniosków             |

---

## Raport

Raport podsumowuje wyniki eksperymentu przeprowadzonego na modelu Gemma 3 4B. Analizie poddano 25 par promptów (5 kategorii × 5 par): _style_, _tone_, _formality_, _framing_, _reformulation_. Dla każdej pary zmierzono wagi attention ze wszystkich 34 warstw modelu i obliczono metryki ilościowe. Analizę uzupełniono o atrybucję attention metodą Inseq.

### Wyniki ilościowe

#### Zestawienie per para promptów

| ID | Kategoria | Entropia base | Entropia mod | ΔEntropii  | Sparsity base | Sparsity mod | Diff L1 | Diff L2 | ΔEntropii local | ΔEntropii global |
|---|---|---|---|------------|---|---|---|---|---|---|
| style_01 | style | 0.7668 | 1.0434 | **0.2766** | 0.4568 | 0.5422 | 0.000112 | 0.000460 | 0.2920 | 0.1713 |
| style_02 | style | 0.7577 | 1.2653 | **0.5076** | 0.4375 | 0.5750 | 0.000116 | 0.000268 | 0.5320 | 0.3359 |
| style_03 | style | 0.8860 | 1.3127 | **0.4267** | 0.4793 | 0.5974 | 0.000188 | 0.000655 | 0.4457 | 0.2873 |
| style_04 | style | 0.7929 | 1.2463 | **0.4534** | 0.5200 | 0.6343 | 0.000123 | 0.000448 | 0.4726 | 0.3133 |
| style_05 | style | 0.8019 | 1.3096 | **0.5077** | 0.4444 | 0.5847 | 0.000126 | 0.000481 | 0.5279 | 0.3609 |
| tone_01 | tone | 0.7481 | 0.9741 | **0.2260** | 0.4286 | 0.5089 | 0.015640 | 0.028448 | 0.2427 | 0.1203 |
| tone_02 | tone | 0.8135 | 1.2248 | **0.4114** | 0.4375 | 0.6219 | 0.013621 | 0.023404 | 0.4329 | 0.2711 |
| tone_03 | tone | 0.6399 | 1.0188 | **0.3789** | 0.4167 | 0.4852 | 0.010861 | 0.019539 | 0.3923 | 0.2731 |
| tone_04 | tone | 0.9572 | 1.2058 | **0.2486** | 0.4700 | 0.6175 | 0.015428 | 0.029710 | 0.2612 | 0.1610 |
| tone_05 | tone | 0.6677 | 1.1993 | **0.5316** | 0.4167 | 0.5457 | 0.018634 | 0.031696 | 0.5620 | 0.3160 |
| formality_01 | formality | 0.5947 | 0.9361 | **0.3414** | 0.4000 | 0.5444 | 0.018311 | 0.030842 | 0.3647 | 0.1729 |
| formality_02 | formality | 0.8268 | 1.1270 | **0.3002** | 0.4444 | 0.6644 | 0.012933 | 0.025026 | 0.3187 | 0.1697 |
| formality_03 | formality | 0.8672 | 1.0919 | **0.2248** | 0.4691 | 0.6080 | 0.015067 | 0.026687 | 0.2401 | 0.1266 |
| formality_04 | formality | 0.9472 | 1.1320 | **0.1849** | 0.4700 | 0.6177 | 0.017437 | 0.030373 | 0.1965 | 0.1056 |
| formality_05 | formality | 0.8376 | 1.0752 | **0.2376** | 0.4375 | 0.5813 | 0.014061 | 0.025736 | 0.2555 | 0.1167 |
| framing_01 | framing | 0.7393 | 1.0359 | **0.2966** | 0.4286 | 0.5689 | 0.011410 | 0.020126 | 0.3176 | 0.1578 |
| framing_02 | framing | 0.8009 | 1.4085 | **0.6076** | 0.5200 | 0.6922 | 0.000108 | 0.000429 | 0.6348 | 0.4086 |
| framing_03 | framing | 0.6675 | 1.1369 | **0.4695** | 0.4167 | 0.5926 | 0.018048 | 0.029743 | 0.5021 | 0.2409 |
| framing_04 | framing | 0.6038 | 1.4147 | **0.8109** | 0.4000 | 0.6502 | 0.022666 | 0.039779 | 0.8566 | 0.4847 |
| framing_05 | framing | 0.8761 | 1.4433 | **0.5672** | 0.4500 | 0.7344 | 0.012021 | 0.022546 | 0.5969 | 0.3609 |
| reformulation_01 | reformulation | 0.7356 | 0.7575 | **0.0219** | 0.4286 | 0.4286 | 0.006990 | 0.012610 | 0.0219 | 0.0224 |
| reformulation_02 | reformulation | 0.9764 | 1.0407 | **0.0643** | 0.5207 | 0.4792 | 0.016856 | 0.031087 | 0.0593 | 0.1018 |
| reformulation_03 | reformulation | 0.9376 | 1.0020 | **0.0644** | 0.4793 | 0.5148 | 0.011345 | 0.021253 | 0.0713 | 0.0248 |
| reformulation_04 | reformulation | 0.7433 | 0.9834 | **0.2401** | 0.4286 | 0.4545 | 0.005062 | 0.010408 | 0.2522 | 0.1602 |
| reformulation_05 | reformulation | 0.7382 | 0.8079 | **0.0697** | 0.4286 | 0.4444 | 0.016088 | 0.033460 | 0.0780 | 0.0101 |

#### Zestawienie per kategoria

| Kategoria     | Entropia base | Entropia mod | ΔEntropii  | Sparsity base | Sparsity mod | Diff L1 | Diff L2 | ΔEntropii local | ΔEntropii global | Stosunek local/global |
|---------------|---|---|------------|---|---|---|---|---|---|---|
| style         | 0.8011 | 1.2355 | **0.4344** | 0.4676 | 0.5867 | 0.0001 | 0.0005 | 0.4540 | 0.2937 | 1.5× |
| tone          | 0.7653 | 1.1246 | **0.3593** | 0.4339 | 0.5558 | 0.0148 | 0.0266 | 0.3782 | 0.2283 | 1.7× |
| formality     | 0.8147 | 1.0724 | **0.2578** | 0.4442 | 0.6032 | 0.0156 | 0.0277 | 0.2751 | 0.1383 | 2.0× |
| framing       | 0.7375 | 1.2879 | **0.5504** | 0.4431 | 0.6477 | 0.0129 | 0.0225 | 0.5816 | 0.3306 | 1.8× |
| reformulation | 0.8262 | 0.9183 | **0.0921** | 0.4572 | 0.4643 | 0.0113 | 0.0218 | 0.0965 | 0.0639 | 1.5× |
| **Razem**     | **0.7890** | **1.1277** | **0.3388** | **0.4492** | **0.5715** | **0.0109** | **0.0198** | **0.3571** | **0.2110** | **1.7×** |

---

### Odpowiedzi na pytania badawcze

#### RQ1: Czy tokeny instrukcji mają wyższe wagi attention niż tokeny treści?

**Tak — z zastrzeżeniami metodologicznymi.**

Analiza heatmap i wykresu zbiorczego (`category_attention_barchart.png`) pokazuje, że tokeny kategorii _instruction_ (np. „wyjaśnij", „opisz", „proszę") przyciągają uwagę modelu przede wszystkim w pierwszych warstwach lokalnych. W promptach bazowych — które są bardzo krótkie (2–6 tokenów) — token instrukcyjny stanowi znaczny ułamek sekwencji, co automatycznie zawyża jego średnią wagę attention.

W promptach zmodyfikowanych (szczególnie _style_, _framing_, _tone_) do sekwencji dochodzi wiele nowych tokenów (np. „krótko i prosto", „Jako ekonomista", „Dla ucznia 5. klasy"). Nowe tokeny modyfikatora też funkcjonują jako „instrukcja" dla modelu, co widać w diff heatmapach: po dodaniu modyfikatora wzrasta waga attention skupiona na tokenach stylistycznych, a rozkład staje się bardziej równomierny (wyższa entropia).

**Zastrzeżenie:** kategoryzacja tokenów opiera się na dopasowaniu słów kluczowych, co jest z natury niedoskonałe — szczególnie przy subtokenizacji SentencePiece (np. „Wyjaśnij" → `['Wy', 'ja', 'ś', 'nij']`). Tokeny funkcyjne (przyimki, spójniki) często mają wyższe wagi niż wynikałoby to z ich semantycznej roli, co jest znane z literatury (Clark et al. 2019 wskazuje na głowice skupione na tokenach pozycyjnych i gramatycznych).

#### RQ2: Czy różne typy modyfikacji promptu prowadzą do różnych wzorców attention?

**Tak — kategorie różnią się wyraźnie.**

|   | Kategoria | ΔEntropii | Interpretacja |
|---|---|-----------|---|
| 1 | framing | 0.5504    | Określenie odbiorcy radykalnie zmienia rozkład uwagi |
| 2 | style | 0.4344    | Dyrektywy stylistyczne istotnie rozpraszają attention |
| 3 | tone | 0.3593    | Przypisanie roli umiarkowanie zmienia wzorzec |
| 4 | formality | 0.2578    | Wzrost formalności — umiarkowany efekt |
| 5 | reformulation | 0.0921    | Parafraza niemal nie zmienia attention |

**Framing** wywołuje największą zmianę, bo dodawany kontekst odbiorcy jest semantycznie bogaty i znacznie rozszerza sekwencję tokenów (np. „Wyjaśnij jak działa komputer osobie po 70. roku życia, która nigdy nie korzystała z technologii" — 26 tokenów wobec 4 w wersji bazowej). Para `framing_04` osiągnęła najwyższy Δentropii w całym eksperymencie: **0.8109**.

**Style** ma bardzo niskie wartości diff L1 i diff L2 (rzędu 0.0001 — najniższe ze wszystkich kategorii), co wynika z metodologicznego artefaktu: gdy długości sekwencji base i modified znacznie się różnią, metryka diff operuje na obciętych macierzach i nie oddaje skali rzeczywistej zmiany. Wzrost entropii **0.4344** jest natomiast porównywalny z kategorią tone i wskazuje na rzeczywistą redystrybucję uwagi.

**Reformulation** jest wyjątkiem — parafraza tej samej treści przy zachowaniu podobnej długości i struktury gramatycznej nie zmienia istotnie rozkładu attention. Para `reformulation_01` **ΔE=0.0219, diff L1=0.0070** praktycznie nie różni się od promptu bazowego w przestrzeni attention.

#### RQ3: Czy różnice w rozkładzie attention między warstwami lokalnymi (sliding window) a globalnymi są widoczne?

**Tak — warstwy lokalne reagują wyraźnie silniej na modyfikacje promptów we wszystkich kategoriach.**

|   | Kategoria | ΔEntropii local | ΔEntropii global | Stosunek |
|---|---|---|---|---|
| 1 | formality | 0.2751 | 0.1383 | **2.0× silniejszy efekt w local** |
| 2 | framing | 0.5816 | 0.3306 | **1.8× silniejszy efekt w local** |
| 3 | tone | 0.3782 | 0.2283 | **1.7× silniejszy efekt w local** |
| 4 | style | 0.4540 | 0.2937 | **1.5× silniejszy efekt w local** |
| 5 | reformulation | 0.0965 | 0.0639 | **1.5× silniejszy efekt w local** |

Warstwy z globalnym attention (indeksy 5, 11, 17, 23, 29 w 34-warstwowym Gemma 3 4B) integrują informację z całego kontekstu — ich rozkłady są z natury bardziej równomierne. Warstwy lokalne (sliding window, okno 1024 tokenów) są bardziej czułe na zmiany w bezpośrednim sąsiedztwie tokenu, stąd silniejsza reakcja na dodanie modyfikatora do krótkiego promptu.

Efekt jest szczególnie wyraźny dla kategorii **formality**: stosunek local/global wynosi niemal 2:1. Może to wynikać ze struktury leksykalnej zmodyfikowanych promptów formalnych — zwroty grzecznościowe („Proszę o wyjaśnienie", „Uprzejmie proszę") tworzą lokalny wzorzec n-gramowy, który warstwy sliding window rozpoznają silniej niż warstwy z dostępem do pełnego kontekstu.

---

### Obserwacje z analizy Inseq

Atrybucja attention (Inseq, metoda `attention`) pozwala zobaczyć, które tokeny wejściowe mają najwyższe wagi przy generowaniu kolejnych tokenów.

**Obserwacja 1 — degeneracja generowania dla bardzo krótkich promptów.**

Dla pary `framing_04` prompt bazowy to 4 tokeny: `['Jak', '▁działa', '▁komputer', '?']`. Model wygenerował wyłącznie znaki nowej linii (`\n×20`), co jest typową degeneracją dla zbyt krótkiego, nieukierunkowanego promptu bez kontekstu systemowego. Prompt zmodyfikowany (26 tokenów) wygenerował sensowną, choć powtarzającą się odpowiedź. Ten kontrast ilustruje, że samo attention nie wystarczy do oceny „rozumienia" — długość i struktura promptu wpływają na to, czy model w ogóle inicjuje generowanie treści.

**Obserwacja 2 — subtokenizacja języka polskiego.**

SentencePiece tokenizuje polskie słowa na wiele podjednostek, np.:
- „Wyjaśnij" → `['Wy', 'ja', 'ś', 'nij']` (4 subtokeny)
- „komputer" → `['▁komputer']` (1 token)
- „korzystała" → `['▁korzyst', 'a', 'ła']` (3 subtokeny)

To ma bezpośrednie przełożenie na attention: tokeny-instrukcje w języku polskim są często rozbite na wiele subtokenów, co może rozmywać sygnał attention przypisany do „funkcji instrukcji" jako całości. Ręczna kategoryzacja tokenów jako `instruction`/`content`/`functional` działa poprawnie dla całych słów, ale nie uwzględnia tego, że jeden konstrukt semantyczny może być rozłożony na kilka pozycji.

**Obserwacja 3 — reformulacja zachowuje tokeny kluczowe.**

Dla pary `reformulation_01`:
- Base: `['Co', '▁powod', 'uje', '▁des', 'zcz', '?']`
- Modified: `['Dl', 'aczego', '▁pada', '▁des', 'zcz', '?']`

Token kluczowy treści `▁des` + `zcz` (= „deszcz") pozostaje na tej samej pozycji w obu wersjach, co tłumaczy minimalną zmianę wzorców attention **ΔE = 0.0219**. Model „wie", czego dotyczy zapytanie, niezależnie od zmiany czasownika pytającego.

**Obserwacja 4 — atrybucje Inseq dla stylu.**

Dla `style_01`:
- Base: `['Wy', 'ja', 'ś', 'nij', '▁g', 'raw', 'it', 'ację']` — 8 tokenów
- Modified: te same 8 + `['▁kr', 'ót', 'ko', '▁i', '▁pro', 'sto']` — łącznie 14 tokenów

`source_attributions` jest obecne w danych Inseq, ale wymaga dalszej analizy wizualnej heatmap, by stwierdzić, które z dodanych tokenów stylistycznych mają najwyższe wagi atrybucji przy generowaniu pierwszego tokenu odpowiedzi.

---

### Wnioski końcowe

1. **Każda modyfikacja promptu zwiększa entropię attention** — bez wyjątku we wszystkich 25 parach. Średni wzrost entropii wynosi **0.34**. To spójny wynik sugerujący, że dłuższe i bardziej opisowe prompty rozpraszają uwagę modelu na większą liczbę tokenów.

2. **Sparsity rośnie wraz z modyfikacją**, z jednym wyjątkiem — `reformulation_02` wykazuje nieznaczny spadek sparsity. Wzrost sparsity oznacza, że zmodyfikowane prompty prowadzą do bardziej skoncentrowanych rozkładów attention — pozornie sprzeczne z wzrostem entropii, ale wyjaśnialne przez dłuższe sekwencje: więcej tokenów z bardzo niską wagą (< 0.01) podnosi odsetek sparsity, a jednocześnie więcej tokenów z umiarkowaną wagą podnosi entropię Shannona.

3. **Diff L1/L2 jest nierzetelną miarą przy dużych różnicach długości sekwencji.** Kategorie _style_ i niektóre pary _framing_ mają diff **L1 ≈ 0.0001 — 100× niższy** niż inne kategorie z podobnymi zmianami entropii. Wynika to z faktu, że `pairwise_attention_diff` operuje na obciętej macierzy o rozmiarze `min(seq_len_base, seq_len_mod)`, co przy promptach bazowych liczących 4–8 tokenów i zmodyfikowanych 14–26 tokenów drastycznie redukuje obszar porównania. Metryka diff działa rzetelnie tylko przy zbliżonych długościach sekwencji.

4. **Warstwy lokalne (sliding window) są bardziej wrażliwe na modyfikacje** niż globalne we wszystkich kategoriach. Potwierdza to hipotezę, że zmiany w bezpośrednim sąsiedztwie tokenu (dodanie modyfikatora na końcu krótkiego promptu) są silniej uchwycone przez mechanizm lokalnego attention.

5. **Framing wykazuje największy i najbardziej zróżnicowany efekt.** Szczegółowy opis odbiorcy tworzy najsilniejszą redystrybucję uwagi — model musi „wziąć pod uwagę" wiele dodatkowych informacji kontekstowych. Największy efekt: `framing_04` — opis „osoby po 70. roku życia, która nigdy nie korzystała z technologii".

6. **Reformulacja semantycznie równoważna nie zmienia attention.** Wyniki reformulation wskazują, że Gemma 3 4B nie różnicuje istotnie wzorców attention między „Co powoduje deszcz?" a „Dlaczego pada deszcz?". Jest to zgodne z intuicją — te zdania mają identyczną treść merytoryczną i podobną strukturę składniową.

---

### Deliverables

- [x] Design Proposal (ten dokument)
- [x] Dataset promptów (25 par, 5 kategorii × 5 par: style, tone, formality, framing, reformulation)
- [x] Adnotacje kategorii tokenów (`data/raw/token_categories.json`: instrukcja / treść / funkcyjny)
- [x] Skrypt do ekstrakcji i wizualizacji attention (`scripts/run_experiment.py`)
- [x] Moduł metryk ilościowych (`src/metrics/`: entropia, sparsity, diff, mean per pozycja, mean per kategorię tokenu)
- [x] Diff heatmapy (modified − base) z diverging colormap
- [x] Wykresy zbiorcze bar chart (base vs modified per kategoria promptów i tokenu)
- [x] Logowanie metryk i wykresów zbiorczych do W&B per para promptów
- [x] Atrybucja Inseq attention (`scripts/run_inseq.py`)
- [x] Raport HTML (`scripts/generate_report.py`)
- [x] Testy pytest — 71 testów (jednostkowe + integracyjne), bez GPU/internetu
- [x] Dokumentacja (`README.md`, `docs/manual.md`)
- [x] Makefile do łatwiejszego korzystania z projektu
- [x] Raport z obserwacji i wniosków
- [ ] Filmik demo (3–5 min)
- [ ] Prezentacja finalna

---

### Bibliografia

1. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A. N., Kaiser, Ł., & Polosukhin, I. (2017). _Attention Is All You Need_. NeurIPS 2017. https://arxiv.org/abs/1706.03762
2. Jain, S., & Wallace, B. C. (2019). _Attention is not Explanation_. NAACL-HLT 2019. https://arxiv.org/abs/1902.10186
3. Wiegreffe, S., & Pinter, Y. (2019). _Attention is not not Explanation_. EMNLP 2019. https://arxiv.org/abs/1908.04626
4. Clark, K., Khandelwal, U., Levy, O., & Manning, C. D. (2019). _What Does BERT Look At? An Analysis of BERT's Attention_. BlackboxNLP @ ACL 2019. https://arxiv.org/abs/1906.04341
5. Danilevsky, M., Qian, K., Aharonov, R., Katsis, Y., Kawas, B., & Sen, P. (2020). _A Survey of the State of Explainable AI for Natural Language Processing_. AACL-IJCNLP 2020. https://arxiv.org/abs/2010.00711
6. Reynolds, L., & McDonell, K. (2021). _Prompt Programming for Large Language Models: Beyond the Few-Shot Paradigm_. CHI EA 2021. https://arxiv.org/abs/2102.07350
7. Rust, P., Pfeiffer, J., Vulić, I., Ruder, S., & Gurevych, I. (2021). _How Good is Your Tokenizer? On the Monolingual Performance of Multilingual Language Models_. ACL-IJCNLP 2021. https://arxiv.org/abs/2012.15613
8. Belrose, N., Furman, Z., Smith, L., Strauss, D., Gat, I., & Sontag, D. (2023). _Eliciting Latent Predictions from Transformers with the Tuned Lens_. arXiv preprint. https://arxiv.org/abs/2303.08112
9. Fiotto-Kaufman, J., Laber, A., Todd, E., Brinkmann, J., Juang, C., Pal, K., Rager, C., Mueller, A., Marks, S., Sharma, A., Bau, D., Lieberum, T., Conmy, A., & Nanda, N. (2024). _NNsight and NDIF: Democratizing Access to Foundation Model Internals_. ICLR 2025. https://arxiv.org/abs/2407.14561
10. Vig, J. (2019). _A Multiscale Visualization of Attention in the Transformer Model_. ACL 2019 (System Demonstrations). https://arxiv.org/abs/1906.05714
11. Sarti, G., Feldhus, N., Sickert, L., van der Wal, O., Nissim, M., & Bisazza, A. (2023). _Inseq: An Interpretability Toolkit for Sequence Generation Models_. ACL 2023 (System Demonstrations). https://arxiv.org/abs/2302.13942
12. Gemma Team, Google DeepMind. (2025). _Gemma 3 Technical Report_. arXiv preprint. https://arxiv.org/abs/2503.19786
13. Pezeshkpour, P., & Hruschka, E. (2023). _Large Language Models Sensitivity to The Order of Options in Multiple-Choice Questions_. arXiv preprint. https://arxiv.org/abs/2308.11483
14. Gould, J., Ong, E., Ogden, G., & Conmy, A. (2023). _Successor Heads: Recurring, Interpretable Attention Heads In The Wild_. ICLR 2024. https://arxiv.org/abs/2312.09230
