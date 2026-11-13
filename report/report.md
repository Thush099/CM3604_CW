# Automated Fact-Checking of LLM-Generated Biographies via Lexical Overlap and Neural Textual Entailment

- **Module:** CM3604 - Deep Learning (Academic Year 2026/2027, Semester 1)
- **Module Coordinator:** Prasan Yapa
- **Group Members:**
  - Thushanth Mahendran (`[Insert Student ID]`)
  - Kavindu Sudasinghe (`[Insert Student ID]`)
  - Yasith Kirihena (`[Insert Student ID]`)

---

## 1. Introduction and Dataset Overview

Large language models (LLMs) frequently produce fluent but factually incorrect statements ("hallucinations"), especially when summarizing biographies of less prominent people (Min et al., 2023). Following the **FActScore** evaluation setup, this coursework investigates automatic fact-checking of atomic biographical statements extracted from ChatGPT generations against Wikipedia passages retrieved via BM25 (Robertson and Zaragoza, 2009).

Each record in `data/dev_labeled_ChatGPT.jsonl` and `data/passages_bm25_ChatGPT_humfacts.jsonl` pairs an atomic claim with between 2 and 5 retrieved Wikipedia chunks (mean `3.13` passages per claim). Across the **31 biographies** in the development set, there are **221 human-annotated atomic facts** labeled **112 Supported (`S`, 50.68%)**, **93 Not Supported (`NS`, 42.08%)**, and **16 Irrelevant (`IR`, 7.24%)**. In `factchecking_main.py`, `IR` labels are mapped to `NS` during evaluation, producing a nearly balanced binary split of **112 `S` (50.68%)** and **109 `NS` (49.32%)**. The majority-class baseline (`AlwaysEntailedFactChecker`) therefore achieves **50.68% accuracy** (`112/221`).

---

## 2. Part 1: Bag-of-Words Overlap Fact-Checking

### 2.1 Preprocessing Decisions and Justification

In discrete symbolic text models over vocabulary $V$, one-hot vectors in $\{0, 1\}^{|V|}$ are sparse, orthogonal, and equidistant ($\|e_i - e_j\|_2 = \sqrt{2}$ for $i \ne j$), while Bag-of-Words (BoW) aggregation discards word order (`Deep Learning Lectures (1)`, Slides 6–11; `Deep Learning Lectures (2)`, Slides 14–16, 53). Consequently, surface-level normalization that maps morphological variants into shared terms in $V$ is essential before computing set overlap or TF-IDF cosine similarity. The retrieved passages in `passages_bm25_ChatGPT_humfacts.jsonl` are untokenized Wikipedia excerpts containing structural tags (`<s>`, `</s>`) and empty quotation marks (`""`) left behind after markup stripping. In [`WordRecallThresholdFactChecker`](../factcheck.py), we tested five preprocessing stages:

1. **Markup Removal and Lowercasing:** `preprocess()` replaces `<s>`, `</s>`, and `""` with spaces and lowercases text. Otherwise, boundary tokens such as `<s>marianne` or `actor.</s>` fail to match clean claim words.
2. **Alphanumeric Tokenization (`[a-z0-9]+`):** Extracting alphanumeric words strips punctuation cleanly while retaining dates and years (such as `1931` and `1987`), which are often needed to verify biographical facts.
3. **Stop-Word Filtering:** We remove an offline set of **115 English stop words and common biographical filler words** (`is`, `was`, `he`, `she`, `later`, `known`, `one`, `two`, etc.). Because template phrasing appears in almost every passage, removing high-frequency function words keeps overlap scores focused on predicate content words.
4. **Porter Stemming (`PorterStemmer`):** Biographical claims often paraphrase inflected verbs or nouns in Wikipedia (for instance, *"invented"* vs. *"inventing"*, or *"acted"* vs. *"acting"*). Porter stemming (Porter, 1980) merges these morphological variants locally without requiring external downloads.
5. **Subject-Entity Base-Title Exclusion:** Because BM25 retrieves passages from the subject's own Wikipedia article, the subject's name appears in nearly every chunk. Without filtering it out, a false claim like *"Marianne McAndrew is a singer"* receives inflated recall simply by matching *"Marianne"* and *"McAndrew"*. Before scoring, `WordRecallThresholdFactChecker.predict()` strips parenthetical disambiguators from `passages[0]["title"]` (such as `"(inventor)"` in *"Gerhard Fischer (inventor)"*), stems the name tokens, and subtracts them from the fact's token set. If excluding the title would leave an empty set, the method falls back to keeping all non-stopword claim tokens.

### 2.2 Mathematical Formulation of Scoring Metrics

Passages are split along `<s>`/`</s>` block boundaries and sentence punctuation (`(?<=[.!?])\s+`) prior to scoring. Let $W(F)$ denote the preprocessed token set of the atomic claim (after base-title exclusion) and $\mathcal{U}(P) = \{u_1, \dots, u_m\}$ denote the sentence units extracted from passages $P$. Using `factcheck.py` and `experiments/ablation_study.py`, we evaluated four lexical scoring measures:

1. **Asymmetric Unigram Recall (`metric="recall"`, Selected):**
   $$\text{Score}_{\text{Recall}}(F, P) = \max_{u \in \mathcal{U}(P)} \frac{|W(F) \cap W(u)|}{|W(F)|}$$
   Atomic facts typically contain 3 to 8 content words, whereas Wikipedia sentences often contain 15 to 40 tokens. Normalizing by $|W(F)|$ checks whether the claim's content words appear together in one sentence without penalizing unrelated encyclopedic details in $u$.

2. **Symmetric Jaccard Similarity (`metric="jaccard"`):**
   $$\text{Score}_{\text{Jaccard}}(F, P) = \max_{u \in \mathcal{U}(P)} \frac{|W(F) \cap W(u)|}{|W(F) \cup W(u)|}$$

3. **Local TF-IDF Cosine Similarity:** Treating $\{F\} \cup \mathcal{U}(P)$ as a local collection of $N = m + 1$ texts with smoothed inverse document frequency $\text{idf}(t) = \ln\!\left(\frac{N+1}{\text{df}(t)+1}\right) + 1$ and term weights $v_x(t) = \text{tf}(t, x)\cdot\text{idf}(t)$:
   $$\text{Score}_{\text{TF-IDF}}(F, P) = \max_{u \in \mathcal{U}(P)} \frac{\sum_{t \in W(F) \cap W(u)} v_F(t)\,v_u(t)}{\|v_F\|_2 \, \|v_u\|_2}$$

4. **ROUGE-1 Recall (`rouge-score`):** Stemmed unigram recall (Lin, 2004) computed across concatenated passage text.

### 2.3 Validation Protocol and Part 1 Experimental Results

From the classical error decomposition (`Lecture 8`, Slides 2–13), a linear threshold over BoW features has bounded hypothesis-class capacity $\mathcal{F}$ (yielding higher *approximation error* on compositional claims), whereas optimizing $\tau$ directly across all 221 facts risks *estimation error* by overfitting to specific biography vocabularies. Because no separate test split is provided, `experiments/ablation_study.py` runs **5-fold grouped cross-validation partitioned by biography index** ($\text{bio\_idx} \bmod 5$), keeping all facts from the same biography in a single fold to control estimation error.

| ID | Scoring Metric & Granularity | Preprocessing Configuration | Best $\tau$ | Dev Accuracy | 5-Fold Grouped CV | Runtime (s) |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: |
| `B0` | Majority Baseline (`always_entail`) | None | N/A | 50.68% | 50.68% | 0.00 |
| `P1-1` | Jaccard Similarity (Sentence) | Lowercase + alphanumeric | 0.27 | 64.25% | 57.01% | 0.07 |
| `P1-2` | Jaccard Similarity (Sentence) | Stop-words + Porter stemmer | 0.21 | 66.97% | 61.99% | 1.26 |
| `P1-3` | TF-IDF Cosine (Sentence) | Lowercase + alphanumeric | 0.29 | 70.14% | 64.25% | 0.16 |
| `P1-4` | TF-IDF Cosine (Sentence) | Stop-words + Porter stemmer | 0.26 | 76.47% | 74.66% | 1.36 |
| `P1-5` | ROUGE-1 Recall (All Passages) | Porter stemmer | 0.81 | 81.00% | 75.11% | 1.41 |
| `P1-6` | Unigram Recall (Passage) | Lowercase + alphanumeric | 0.72 | 76.92% | 75.57% | 0.04 |
| `P1-7` | Unigram Recall (Passage) | Stop-words removed (no stem) | 0.61 | 79.64% | 79.64% | 0.04 |
| `P1-8` | Unigram Recall (Passage) | Stop-words + Porter stemmer | 0.61 | 81.90% | 81.90% | 1.24 |
| `P1-9` | Unigram Recall (Sentence) | Stop-words + base-title excl. (no stem) | 0.41 | 81.90% | 80.09% | 0.07 |
| **`P1-10`** | **Unigram Recall (Sentence) [Selected]** | **Stop-words + Porter + title excl.** | **0.50** | **84.16%** (`186/221`) | **82.35%** | **1.27** |

*Table 1: Part 1 lexical overlap, preprocessing, and 5-fold grouped cross-validation ablation results across 221 development claims.*

Running `WordRecallThresholdFactChecker(threshold=0.20, metric="jaccard")` directly in `factcheck.py` (including base-title exclusion) reaches `69.68%` (`154/221`), confirming that symmetric Jaccard consistently trails asymmetric recall even when using identical preprocessing.

#### Analysis of Part 1 Results
Symmetric similarity metrics underperform (`64.25%` to `69.68%` for Jaccard; `76.47%` for TF-IDF cosine) because $|W(F) \cup W(u)|$ and $\|v_u\|_2$ grow as sentences become longer, penalizing valid matches embedded inside detailed Wikipedia sentences.

Each preprocessing stage improves generalization on unigram recall: filtering the 115 stop words lifts 5-fold CV accuracy from `75.57%` (`P1-6`) to `79.64%` (`P1-7`), adding Porter stemming reaches `81.90%` (`P1-8`), and shifting to sentence-level matching with base-title exclusion (`P1-10`) raises 5-fold CV to **82.35%** and dev accuracy to **84.16%** (`Precision(S) = 82.35%`, `Recall(S) = 87.50%`, `F1(S) = 0.8485`; `Precision(NS) = 86.27%`, `Recall(NS) = 80.73%`, `F1(NS) = 0.8341`), exceeding the 75% requirement by **9.16 percentage points**. Sweeping $\tau$ across `0.35–0.70` confirms a wide, stable plateau around `0.50`: `81.90%` (`0.35`), `82.35%` (`0.40`), **84.16%** (`0.45` and `0.50`), `83.71%` (`0.55` and `0.60`), `82.81%` (`0.65`), and `75.11%` (`0.70`).

---

## 3. Part 2: Neural Textual Entailment Fact-Checking

### 3.1 Passage Cleaning and Contextual Windowing (`get_windows`)

For Part 2, `factchecking_main.py` loads `MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli` (He et al., 2021; Laurer et al., 2024), a Transformer architecture trained via self-supervised pre-training and supervised transfer learning (Devlin et al., 2019; `Deep Learning Lectures (2)`, Slides 28, 53; `Lecture 10`, Slides 10–27) on MNLI (Williams et al., 2018), FEVER (Thorne et al., 2018), and ANLI (Nie et al., 2020), whose output logits represent `[0: entailment, 1: neutral, 2: contradiction]`.

Unlike orderless BoW models, the Transformer tokenizer applies **sub-word tokenization** (`Lecture 7`, Slides 20, 50) to handle out-of-vocabulary names and morphological suffixes, and encodes each premise-hypothesis pair as `[CLS] premise [SEP] hypothesis [SEP]` combining token, segment ($E_A, E_B$), and disentangled relative positional embeddings (`Lecture 7`, Slides 43–50). Across encoder layers, **Multi-Head Self-Attention** (Vaswani et al., 2017; `Lecture 7`, Slides 38–42):
$$\text{SelfAttention}(Q_i, K, V) = \sum_j \text{softmax}_j\!\left(\frac{Q_i \cdot K^T}{\sqrt{d_k}}\right) V_j$$
computes bidirectional cross-attention between premise and hypothesis tokens before projecting the pooled `[CLS]` representation into 3-class logits $z = [z_{\text{ent}}, z_{\text{neu}}, z_{\text{con}}]$.

Evaluating isolated single sentences frequently breaks cross-sentence pronoun references and section lists. Conversely, passing an entire 200-word Wikipedia chunk dilutes attention across unrelated biographical events. To balance local precision with coreference context, [`EntailmentFactChecker.get_windows`](../factcheck.py) applies four steps:

1. **Template and Artifact Cleaning:** Within each `<s>`/`</s>` block, removes truncated or intact Wikipedia template fragments (`{{...}}`), empty quotes `""`, empty parentheses `()`, and duplicated leading title prefixes (`"{raw_title} {raw_title}"`).
2. **Abbreviation-Safe Sentence Splitting and Short-Header Merging:** Splits blocks on sentence endings while lookbehinds protect abbreviations (`Mr.`, `Ms.`, `Dr.`, `St.`, `Jr.`, `Sr.`, `No.`) and single-letter initials. Short fragments of at most 4 alphanumeric words (such as section headers like *"Bibliography: Standalone novels."*) are joined to the sentence immediately following them.
3. **Adjacent 1- and 2-Sentence Sliding Windows:** For each merged sentence `merged[j]` within a block, emits both `merged[j]` and, when `j > 0`, the adjacent pair `merged[j - 1] + " " + merged[j]` so local pronoun antecedents remain visible to DeBERTa.
4. **Entity-Title Grounding:** Extracts base title words (>2 characters, with parentheticals stripped); if none appear in window `u`, prepends `"{clean_title}: {u}"` and removes duplicate windows.

### 3.2 Converting Three-Class Outputs to Binary Decisions

For a premise window $w$ and hypothesis $F$, [`EntailmentModel.check_entailment`](../factcheck.py) applies softmax normalization over the three NLI logits $z = [z_{\text{ent}}, z_{\text{neu}}, z_{\text{con}}]$ (`Lecture 2`, Slide 16; `Lecture 7`, Slide 25):
$$P(y = c \mid w, F) = \frac{\exp(z_c)}{\sum_{k \in \{\text{ent}, \text{neu}, \text{con}\}} \exp(z_k)}$$
and enforces a **contradiction guard** before returning a calibrated confidence score (`Lecture 9`, Slide 20):
$$s(w, F) = \begin{cases} 0.0 & \text{if } P(\text{con} \mid w, F) > P(\text{ent} \mid w, F), \\ P(\text{ent} \mid w, F) & \text{otherwise.} \end{cases}$$
A claim is predicted `"S"` if any retained candidate window satisfies $s(w, F) \ge \tau_{\text{ent}}$ (`ent_threshold=0.38`), which is mathematically equivalent to checking $\max_{w} s(w, F) \ge \tau_{\text{ent}}$, and `"NS"` otherwise. Rejecting cases where contradiction outweighs entailment prevents high-contradiction examples from passing a threshold below `0.50`, while setting $\tau_{\text{ent}} = 0.38$ captures legitimate biographical paraphrases where minor phrasing omissions divide probability mass between `entailment` and `neutral`.

### 3.3 Word-Overlap Pruning, Short-Circuiting, and Memory Management

Across all 221 examples, `get_windows()` produces `8,765` candidate windows. Running NLI inference on every window is too slow on CPU and increases false-positive exposure from unrelated sentences. [`EntailmentFactChecker.predict`](../factcheck.py) reduces compute and memory usage cleanly:
- **Overlap Filtering and Top-$K$ Ranking:** Computes title-excluded stemmed unigram recall against each raw window sentence, discards candidates below `prune_threshold = 0.20`, sorts survivors by `(overlap, -len(raw_sentence))` descending (preferring shorter single sentences on ties), and retains at most `max_candidates = 5`.
- **Early Exit Equivalence:** Candidates are scored in rank order and `predict()` returns `"S"` immediately once the running maximum satisfies $\max_w s(w, F) \ge 0.38$. Because $\max_w s(w, F) \ge \tau_{\text{ent}} \iff \exists w : s(w, F) \ge \tau_{\text{ent}}$, early termination yields identical binary predictions while executing only `438` NLI calls across the 221 examples (`95.00%` compute saved vs. scoring all `8,765` windows, finishing in `248s` / `4 min 08 s` on CPU — a `20x` speedup over unpruned evaluation).
- **OOM Prevention:** `EntailmentModel.__init__` calls `self.model.eval()` to freeze **Dropout regularization** masks ($p$) and enable deterministic test-time scaling (`Deep Learning Lectures (1)`, Slides 35–38). Every forward pass in `check_entailment()` runs under `with torch.no_grad():`—disabling **computation graph** construction and intermediate activation caching required for backpropagation (`Lecture 1`, Slide 44; `Lecture 2`, Slide 35)—with `truncation=True, max_length=512`, extracts Python `float` scalars via `.item()`, and calls `del inputs, outputs, logits, probs; gc.collect()` immediately afterwards.

### 3.4 Part 2 Experimental Ablation Study

To isolate the contribution of premise representation, title grounding, candidate pruning, and the continuous scoring rule, nine configurations (`P2-1` to `P2-9`) were evaluated over the 221 development claims (692 passages, 5,764 single sentences, and 8,765 contextual 1- and 2-sentence windows; 1,511 unique claim-premise pairs cached and scored in 235.7s on Apple M4 `mps`) alongside a 5-fold group cross-validation protocol grouped by passage-set context. Table 2 summarises the ablation results, and Table 3 reports threshold sensitivity around the chosen configuration (`P2-9`).

| Config | Premise Representation | Scoring & Decision Rule | Pruning Rule | NLI Calls (Saved) | Dev Acc. | 5-Fold CV Acc. |
| :--- | :--- | :--- | :--- | :--- | :---: | :---: |
| `P2-1` | Full passages (`<s>` cleaned) | Discrete `argmax == 0` | `ov >= 0.34, topK=5` | 193 / 692 (72.11%) | 82.81% | — |
| `P2-2` | Full passages (`<s>` cleaned) | Guard + $P(\text{ent}) \ge 0.38$ | `ov >= 0.34, topK=5` | 193 / 692 (72.11%) | 83.26% | 83.26% |
| `P2-3` | Isolated 1-sentence | Discrete `argmax == 0` | `ov >= 0.20, topK=5` | 406 / 5,764 (92.96%) | 82.81% | — |
| `P2-4` | Isolated 1-sentence | Guard + $P(\text{ent}) \ge 0.38$ | `ov >= 0.20, topK=5` | 401 / 5,764 (93.04%) | 83.26% | 81.45% |
| `P2-5` | Title-grounded 1-sentence | Guard + $P(\text{ent}) \ge 0.38$ | `ov >= 0.20, topK=5` | 379 / 5,764 (93.42%) | 83.71% | 81.00% |
| `P2-6` | Contextual 1&2-sent windows | Discrete `argmax == 0` | `ov >= 0.20, topK=5` | 449 / 8,765 (94.88%) | 86.43% | — |
| `P2-7` | Contextual 1&2-sent windows | $P(\text{ent}) \ge 0.38$ (no guard) | `ov >= 0.20, topK=5` | 438 / 8,765 (95.00%) | 88.24% | 87.78% |
| `P2-8` | Contextual 1&2-sent windows | Guard + $P(\text{ent}) \ge 0.38$ | `ov >= 0.34, topK=5` | 256 / 8,765 (97.08%) | 87.78% | 85.97% |
| **`P2-9`** | **Contextual 1&2-sent windows [Selected]** | **Guard + $P(\text{ent}) \ge 0.38$** | **`ov >= 0.20, topK=5`** | **438 / 8,765 (95.00%)** | **88.24%** | **86.88%** |

*Table 2: Part 2 premise representation, scoring rule, and candidate-pruning ablation results across 221 development claims.*

| Threshold ($\tau_{\text{ent}}$) | 0.25 | 0.30 | 0.35 | **0.38** | 0.40 | 0.45 | 0.50 | 0.55 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Dev Accuracy (`P2-9`) | 87.78% (194/221) | 87.78% (194/221) | 87.78% (194/221) | **88.24% (195/221)** | 87.33% (193/221) | 86.43% (191/221) | 86.43% (191/221) | 86.43% (191/221) |
| NLI Calls Executed | 435 | 435 | 436 | **438** | 441 | 449 | 453 | 454 |

*Table 3: Sensitivity of selected system `P2-9` to the entailment threshold $\tau_{\text{ent}}$.*

Four main findings emerge from these ablations:

1. **Contextual 1- and 2-sentence sliding windows give the largest single gain.** Full passages (`P2-2`) top out at 83.26% because long retrieved blocks mix multiple events and routinely exceed or dilute the 512-token NLI budget. Isolated sentences (`P2-4`, 83.26%) and title-grounded single sentences (`P2-5`, 83.71%) avoid truncation but break multi-sentence reasoning across adjacent lines and coreference (`"He"`, `"The system"`). Adding short-header merging and adjacent 2-sentence windows (`P2-9`) lifts dev accuracy by **+4.53 percentage points** over `P2-5` (88.24% vs. 83.71%) and **+5.88 points** in 5-fold grouped CV (86.88% vs. 81.00%).
2. **Continuous threshold calibration outperforms discrete NLI `argmax`.** Switching from hard `argmax == 0` (`P2-6`, 86.43%) to `P(ent) >= 0.38` with the contradiction guard (`P2-9`, 88.24%) adds **+1.81 percentage points** (195/221 vs. 191/221). Across Table 3, accuracy stays above 87.3% throughout $\tau_{\text{ent}} \in [0.25, 0.40]$ before dropping back to 86.43% once $\tau_{\text{ent}} \ge 0.45$, where valid paraphrases begin to get rejected.
3. **Conservative overlap pruning (`ov >= 0.20, topK=5`) balances recall and efficiency.** Tightening the pruning floor to `ov >= 0.34` (`P2-8`) cuts NLI calls further to 256 (97.08% saved) but drops both dev (87.78%) and 5-fold CV (85.97%) accuracy because valid paraphrased windows near `0.20 <= ov < 0.34` get skipped before DeBERTa sees them. Retaining `ov >= 0.20` with `topK = 5` keeps 88.24% dev accuracy (86.88% CV) while still pruning 95.00% of the 8,765 windows (438 calls with early exit).
4. **Contradiction guard trade-off.** Removing the `contra_prob > entail_prob` guard (`P2-7`) reaches the same 88.24% peak dev accuracy and slightly edges `P2-9` on 5-fold CV (87.78% vs. 86.88%), because in low-threshold folds an unguarded `P(ent)` above 0.38 can still recover a partial match when `P(con)` sits slightly higher on unrelated clauses. We retain the guard in `P2-9` so explicit factual contradictions (such as `IDX 8` / `FN_01` on the 1881 Bell sentence, or `FN_05` at `IDX 141` for Jessica Barboza's conflicting 1987 birth date) reliably return `0.0` rather than passing noisy entailment mass.

---

## 4. Part 3: Error Analysis of the Entailment Model

### 4.1 Sampling Protocol and Overall Error Breakdown

On the full 221-example benchmark, [`EntailmentFactChecker`](../factcheck.py) reaches **88.24% accuracy** (`195/221`; `Precision(S) = 93.00%`, `Recall(S) = 83.04%`, `F1(S) = 0.8774`; `Precision(NS) = 84.30%`, `Recall(NS) = 93.58%`, `F1(NS) = 0.8870`). It makes **26 total errors**: **7 False Positives** (predicted `S`, binary ground truth `NS`) and **19 False Negatives** (predicted `NS`, ground truth `S`). As specified in the brief for classes with fewer than 10 mistakes, we examined **all 7 False Positives (`FP_01`–`FP_07`)** and **10 False Negatives (`FN_01`–`FN_10`, the first 10 False Negatives in dataset order across 7 distinct biographies)**, totaling **17 manually analyzed errors** (verified in [`experiments/error_analysis_sheet.csv`](../experiments/error_analysis_sheet.csv) and listed in Appendix A).

### 4.2 Definitions of Fine-Grained Error Categories

1. **Lexical-Semantic & Geographic World-Knowledge Gap:** Claims requiring non-surface semantic equivalence, such as near-synonyms (*pastor* vs. *minister*), abbreviations (*TV* vs. *television*), divergent demonym stems (*Argentinian* vs. *Argentine*), taxonomic hypernyms (*domestic titles* vs. specific competitions), or geographic containment (*London* $\subset$ *United Kingdom*). When surface stem overlap falls below `0.20`, the lexical pruner filters out supporting windows before DeBERTa runs.
2. **Scope / Quantifier & Degree Mismatch:** Instances where the passage and claim refer to the same event or role but differ in quantifier scope (*3 years* vs. *"many years"* or *one lab* vs. *"various organizations"*), hedged phrasing (*"contributed to"* vs. *"best known for inventing"*), or implied temporal order (*"discovered in"* vs. *"began career in"*). This divergence triggers either over-generalized `S` predictions or overly conservative `neutral` outputs.
3. **Associative Window Conflation & Unstated Commonsense Inference:** False positives in which BM25 boundary clipping, title grounding, or adjacent 2-sentence windows cause DeBERTa to misattribute a relationship, or where strong pre-trained cultural/institutional priors fill in facts absent from the passage. The model relies on statistical associations rather than explicit textual support.
4. **Dataset Label-Mapping & Annotation Noise:** Errors caused by dataset or evaluation quirks rather than faulty model inference, including geographically true background claims labeled `IR` that `factchecking_main.py` converts to `NS`, or human annotations that contradict explicit text in Wikipedia.

### 4.3 Aggregate Error Statistics and Key Trends

| Fine-Grained Error Category | False Positives (`Pred=S, Gold=NS`) | False Negatives (`Pred=NS, Gold=S`) | Total Examined (out of 17) | Share (%) |
| :--- | :---: | :---: | :---: | :---: |
| 1. Lexical-Semantic & Geographic World-Knowledge Gap | 0 | 5 | 5 | 29.4% |
| 2. Scope / Quantifier & Degree Mismatch | 2 | 4 | 6 | 35.3% |
| 3. Associative Window Conflation & Unstated Commonsense Inference | 3 | 0 | 3 | 17.6% |
| 4. Dataset Label-Mapping & Annotation Noise | 2 | 1 | 3 | 17.6% |
| **Total Examined** | **7** *(all available)* | **10** | **17** | **100.0%** |

*Table 4: Distribution of examined false positives and false negatives across fine-grained error categories.*

**Key Trends:** False negatives outnumber false positives nearly 3-to-1 (`19` vs. `7`), consistent with the model's `93.00%` precision on `S`. Notably, **100% of Category 1 errors (`5/5`) are False Negatives**, pinpointing the main limitation of placing a strict stem-overlap filter ahead of DeBERTa: if non-title content stems fail to match `>= 0.20`, neural inference is skipped entirely (e.g., `0` NLI calls for `FN_04`, `FN_07`, and `FN_10`). On the false-positive side, Category 3 accounts for **42.9% of all False Positives (`3/7`)**, showing how unstated priors and entity-grounded clipped chunks induce spurious `S` predictions. Furthermore, **17.6% (`3/17`)** of examined errors arise from ground-truth/mapping artifacts (`FP_02`, `FP_07` Rob Furlong *"Canadian former military sniper"*, and `FN_05` Jessica Barboza *July 18, 1987* vs. *14 August 1987*).

### 4.4 Detailed Analysis of Three Representative Examples

#### Example 1 (`FN_04` — Index 104, Subject: *Hibo Wardere*)
- **(a) Text of Example & Passage Excerpt:**
  - *Fact:* `"Hibo Wardere is based in the United Kingdom."`
  - *Passage Excerpt:* `"Born in Somalia, she moved to London, England when just a teenager in 1989... She currently resides in Walthamstow, London, where she worked as a mediator... Her testimonials and campaigning work have made her one of Britain's most prominent campaigners..."`
- **(b) Ground Truth Label:** `S` (Supported)
- **(c) Model Predicted Label:** `NS` (Not Supported)
- **(d) Fine-Grained Error Category:** Lexical-Semantic & Geographic World-Knowledge Gap
- **(e) Explanation:** After removing stop words and subject name tokens (`"hibo"`, `"warder"`), the remaining claim stems `{"base", "unit", "kingdom"}` have zero overlap (`0.00 < 0.20`) with passage stems such as `{"resid", "walthamstow", "london", "england", "britain"}`. Consequently, the pruner discards all windows before NLI runs (`0` DeBERTa calls), whereas evaluating the unpruned passage directly yields $P(\text{entailment}) = 0.999$ as DeBERTa resolves `London` $\subset$ `United Kingdom`.

#### Example 2 (`FP_03` — Index 67, Subject: *Mike Trivisonno*)
- **(a) Text of Example & Passage Excerpt:**
  - *Fact:* `"Trivisonno worked at WKNR."`
  - *Triggered Candidate & Passage Excerpt:* `"Mike Trivisonno: in the ratings, even though it was usually up against Cleveland Indians games on WKNR."` *(followed by: `"It did so well, the decision was made to put Trivisonno in afternoon drive-time..."`)*
- **(b) Ground Truth Label:** `NS` (Not Supported)
- **(c) Model Predicted Label:** `S` (Supported; $P(\text{entailment}) = 0.958$)
- **(d) Fine-Grained Error Category:** Associative Window Conflation & Unstated Commonsense Inference
- **(e) Explanation:** Because BM25 chunking clipped earlier text naming **WTAM** as Trivisonno's employer, the chunk begins mid-sentence with the clause referencing competing Cleveland Indians broadcasts on **WKNR**. When `get_windows()` prepends `"Mike Trivisonno: "` ahead of this truncated clause, DeBERTa associates Trivisonno directly with WKNR ($P(\text{entailment}) = 0.958$), and pairing the following afternoon drive-time sentence pushes confidence to `0.993`.

#### Example 3 (`FN_01` — Index 8, Subject: *Gerhard Fischer (inventor)*)
- **(a) Text of Example & Passage Excerpt:**
  - *Fact:* `"Gerhard Fischer is best known for inventing a metal detector."`
  - *Passage Excerpt:* `"Gerhard Julius Richard Fischer ... was a U.S. entrepreneur of German descent. He contributed to the development and popularity of the hand held metal detector... In 1937, he patented his metal detector."`
- **(b) Ground Truth Label:** `S` (Supported)
- **(c) Model Predicted Label:** `NS` (Not Supported; $\max P(\text{ent}) = 0.018, P(\text{neu}) = 0.981$)
- **(d) Fine-Grained Error Category:** Scope / Quantifier & Degree Mismatch
- **(e) Explanation:** Although human FActScore annotators treated the biography's opening summary as pragmatic support for Fischer's reputation, Wikipedia carefully hedges his role as *"contributed to the development and popularity"* (`0.006` entailment) and notes earlier detectors from 1881 and 1925 (`0.000` after the contradiction guard). Trained on strict MNLI/ANLI logic, DeBERTa refuses to infer the stronger superlative *"best known for inventing"* from *"patented"* (`0.018` entailment, `0.981` neutral) or *"contributed to"*.

### 4.5 Critical Reflection and Implications for Future Improvements

Our error analysis highlights three concrete directions for strengthening biographical fact verification:

1. **Semantic Fallback in Candidate Pruning:** Requiring hard Porter-stem overlap (`ov >= 0.20`) saves substantial CPU time and filters noisy sentences, yet causes every Category 1 False Negative (*London* vs. *United Kingdom*, *TV* vs. *television*, *Argentine* vs. *Argentinian*). Adding a **Siamese dense bi-encoder** (`all-MiniLM-L6-v2`; `Lecture 9`, Slides 21–34) that maps sentences into continuous embedding space $\mathbb{R}^d$ and scores **cosine similarity** $\cos(u, v) = \frac{u \cdot v}{\|u\|_2 \|v\|_2}$ (`Deep Learning Lectures (1)`, Slide 20)—or a WordNet synonym/lemma fallback whenever stem overlap is low—would retrieve paraphrases and geographic links without full cross-attention cost.
2. **Boundary-Aware Context & Coreference Resolution:** Clipped BM25 chunks combined with title prepending and 2-sentence windows resolve pronouns cleanly in most cases, yet induce false employer links in `FP_03` (*WKNR* vs. *WTAM*). Dropping incomplete leading fragments that start mid-clause (`lowercase` start) and running in-place coreference substitution (*"He"* $\to$ *"Ardeshir Zahedi"*) would preserve referent clarity without cross-entity leakage.
3. **Pragmatic Calibration and Relevance Filtering:** Strict logical NLI models conflict with pragmatic FActScore annotations in both directions, rejecting hedged summaries (`FN_01`) while endorsing off-subject background truths (`FP_02`, *"Cleveland is in Ohio"*). Fine-tuning an explicit calibration head on biographical annotations alongside a simple subject-relevance check would better separate true support from `NS` and `IR`.

---

## References

- Devlin, J., Chang, M.-W., Lee, K. and Toutanova, K. (2019) 'BERT: Pre-training of deep bidirectional transformers for language understanding', *Proceedings of NAACL-HLT*, pp. 4171–4186.
- He, P., Gao, J. and Chen, W. (2021) 'DeBERTaV3: Improving DeBERTa using ELECTRA-style pre-training with gradient-disentangled embedding sharing', *arXiv preprint arXiv:2111.09543*.
- Laurer, M., van Atteveldt, W., Casas, A. and Welbers, K. (2024) 'Less annotating, more classifying: Addressing the data scarcity issue of supervised machine learning with deep transfer learning and active learning', *Political Analysis*, 32(1), pp. 84–100.
- Lin, C.-Y. (2004) 'ROUGE: A package for automatic evaluation of summaries', *Text Summarization Branches Out*, ACL, pp. 74–81.
- Min, S., Krishna, K., Lyu, X., Lewis, M., Yih, W., Koh, P.W., Iyyer, M., Zettlemoyer, L. and Hajishirzi, H. (2023) 'FActScore: Fine-grained atomic evaluation of factual precision in long form text generation', *Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing (EMNLP)*, pp. 12076–12100.
- Nie, Y., Williams, A., Dinan, E., Bansal, M., Weston, J. and Kiela, D. (2020) 'Adversarial NLI: A new benchmark for natural language understanding', *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL)*, pp. 4885–4901.
- Porter, M.F. (1980) 'An algorithm for suffix stripping', *Program*, 14(3), pp. 130–137.
- Robertson, S. and Zaragoza, H. (2009) 'The probabilistic relevance framework: BM25 and beyond', *Foundations and Trends in Information Retrieval*, 3(4), pp. 333–389.
- Thorne, J., Vlachos, A., Christodoulopoulos, C. and Mittal, A. (2018) 'FEVER: a large-scale dataset for fact extraction and VERification', *Proceedings of NAACL-HLT*, pp. 809–819.
- Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, Ł. and Polosukhin, I. (2017) 'Attention is all you need', *Advances in Neural Information Processing Systems (NeurIPS)*, 30, pp. 5998–6008.
- Williams, A., Nangia, N. and Bowman, S.R. (2018) 'A broad-coverage challenge corpus for sentence understanding through inference', *Proceedings of NAACL-HLT*, pp. 1112–1122.

---

## Acknowledgement of Generative AI Use

*(Complete/adjust prior to CampusMoodle submission in accordance with RGU Category 3 - Editing guidelines on Pages 12-13 of the Assessment Brief)*

- `[ ]` I/We have not used GenAI to help complete this assessment.
- `[ ]` I/We acknowledge use of `[insert name of GenAI tool(s)]` from `[url of GenAI tool(s)]` to help complete this assessment in the following ways:
  - `[ ]` To help proofread the assessment.
  - `[ ]` To help improve wording, phrasing, and formatting/presentation of original student work (Category 3: Editing).
- Prompts entered on `[insert date]`: `[insert prompt summary and description of how any generated suggestions were reviewed and adapted by the group]`.

---

## Appendix A: Full Set of Examined Errors (7 False Positives and 10 False Negatives)

| ID | Idx | Subject | Gold (Raw) → Pred | Atomic Fact | Fine-Grained Category | Brief Justification |
| :--- | :---: | :--- | :---: | :--- | :--- | :--- |
| FP_01 | 0 | Marianne McAndrew | NS (NS) → S | Marianne McAndrew is an American. | Associative Window Conflation & Unstated Commonsense Inference | Show title 'Love, American Style' passes stem filtering and DeBERTa infers nationality from U.S. TV cues (0.733). |
| FP_02 | 5 | Mike Trivisonno | NS (IR) → S | Cleveland is in Ohio. | Dataset Label-Mapping & Annotation Noise | Supported by East Cleveland, Ohio birth line (0.994); FActScore labels it IR and driver maps IR to NS. |
| FP_03 | 67 | Mike Trivisonno | NS (NS) → S | Trivisonno worked at WKNR. | Associative Window Conflation & Unstated Commonsense Inference | BM25 clips WTAM context; prepending 'Mike Trivisonno: ' to rival WKNR clause induces 0.958 score. |
| FP_04 | 105 | Hibo Wardere | NS (NS) → S | She has worked with a number of organizations. | Scope / Quantifier & Degree Mismatch | Mediator/educator role plus sessions to doctors and Police (0.656) over-generalized to multiple organizations. |
| FP_05 | 122 | Travis Oliphant | NS (NS) → S | He has worked as a researcher at various organizations. | Scope / Quantifier & Degree Mismatch | Directing and researching at one lab (BYU Biomedical Imaging Lab, 0.825) over-generalized to various organizations. |
| FP_06 | 124 | Travis Oliphant | NS (NS) → S | Mayo Clinic is an organization. | Associative Window Conflation & Unstated Commonsense Inference | Pre-trained knowledge marks Mayo Clinic an organization (0.842) despite passage only listing a Ph.D. |
| FP_07 | 133 | Rob Furlong | NS (NS) → S | He served as a sniper. | Dataset Label-Mapping & Annotation Noise | Lead explicitly says 'Canadian former military sniper' and triggers sniper kills (0.886); noisy NS label. |
| FN_01 | 8 | Gerhard Fischer | S (S) → NS | Gerhard Fischer is best known for inventing a metal detector. | Scope / Quantifier & Degree Mismatch | Passage hedges credit ('contributed to development', 0.006; patented 1937, 0.018), insufficient for 'best known'. |
| FN_02 | 9 | Gerhard Fischer | S (S) → NS | Gerhard Fischer invented the first handheld, battery-operated metal detector. | Scope / Quantifier & Degree Mismatch | Passage omits 'battery-operated' and notes invention is disputed, yielding near-zero entailment. |
| FN_03 | 99 | Malcolm Hedding | S (S) → NS | He served for many years as a pastor. | Scope / Quantifier & Degree Mismatch | '1986 and 1989, Malcolm a pastor' spans only 3 years (0.000), failing 'many years'; longer service uses 'minister'. |
| FN_04 | 104 | Hibo Wardere | S (S) → NS | Hibo Wardere is based in the United Kingdom. | Lexical-Semantic & Geographic World-Knowledge Gap | Stems `{"base", "unit", "kingdom"}` have zero overlap with London/England/Britain, causing complete pruning (0 calls). |
| FN_05 | 141 | Jessica Barboza | S (S) → NS | She was born on July 18, 1987. | Dataset Label-Mapping & Annotation Noise | Passage states 14 August 1987; contradiction guard zeros entailment (0.000), exposing noisy gold S label. |
| FN_06 | 144 | Daniel Carvalho | S (S) → NS | With CSKA, he won domestic titles. | Lexical-Semantic & Geographic World-Knowledge Gap | 'Domestic' and 'titles' fail to overlap Russian league/cup names, so top-5 only checks UEFA Cup chunks. |
| FN_07 | 149 | Florencia Bertotti | S (S) → NS | Florencia Bertotti is an Argentinian. | Lexical-Semantic & Geographic World-Knowledge Gap | Porter stemmer maps 'Argentinian' to `argentinian` vs. 'Argentine' to `argentin`, causing zero overlap (0 calls). |
| FN_08 | 160 | Jonathan Haagensen | S (S) → NS | Jonathan Haagensen was born in Rio de Janeiro. | Lexical-Semantic & Geographic World-Knowledge Gap | Linking lifelong 'Vidigal community' residence to Rio de Janeiro birthplace requires external geographic knowledge. |
| FN_09 | 162 | Jonathan Haagensen | S (S) → NS | Haagensen began his acting career in theater. | Scope / Quantifier & Degree Mismatch | 'Discovered in... theatrical company' differs in stem (`theatric` vs. `theater`) and lacks explicit career-start phrasing. |
| FN_10 | 164 | Jonathan Haagensen | S (S) → NS | Haagensen later transitioned to television. | Lexical-Semantic & Geographic World-Knowledge Gap | After stop-word/title filtering, passage uses 'TV' ('TV Record', 'TV series') instead of 'television' (0 calls). |
