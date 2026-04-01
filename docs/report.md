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
| Uruchamianie            | `hydra` + `invoke`                                  |
| Dokumentacja            | `README.md` + `USAGE.md`                            |
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

### Deliverables

- [x] Design Proposal (ten dokument)
- [ ] Dataset promptów
- [ ] Skrypt do ekstrakcji i wizualizacji attention
- [ ] Raport z obserwacji
- [ ] Dokumentacja
- [ ] Testy
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
