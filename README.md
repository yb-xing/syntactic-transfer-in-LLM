# syntactic-echo

> *Does French syntax echo in how language models process English?*

[![Code License: MIT](https://img.shields.io/badge/Code%20License-MIT-green.svg)](LICENSE)
[![Data License: CC BY-NC 4.0](https://img.shields.io/badge/Data%20License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

A computational psycholinguistics project that uses **surprisal analysis across multilingual and monolingual LLMs** to model cross-lingual syntactic transfer — a phenomenon documented in human bilingual processing and grounded in a published behavioral study.

---

## TL;DR

- **Problem:** When English-French bilinguals read ungrammatical English sentences that match French word order, they process them *easier* (but not neccessarily faster) than monolingual English speakers. This is syntactic transfer.
- **Question:** Do multilingual LLMs show the same pattern — lower surprisal on French-order English sentences than monolingual models?
- **Method:** Compare token-level surprisal across XLM-R vs. RoBERTa and mGPT vs. GPT-2 on matched sentence stimuli. Correlate per-item surprisal differences with human reaction time data from a maze task experiment.
- **Why it matters:** If the correlation holds, it suggests multilingual training induces the same cross-lingual syntactic biases observed in human bilingual cognition — with implications for how we evaluate and interpret multilingual model behavior.

---

## Background

Consider this English sentence:

> *John watches **often** television.*

It's ungrammatical in English — adverbs don't go between the verb and the object. But in French, Verb + Adverb + Object is perfectly natural:

> *Jean regarde **souvent** la télé.*

In a published maze task study (*Bilingualism: Language and Cognition*, in production), L1 English speakers showed measurable RT facilitation when reading V+Adv English sentences compared to L1 English monolinguals. The interpretation: their French grammar partially licenses the ungrammatical English structure, reducing processing difficulty.

This project asks whether that same signal is detectable in LLMs — using surprisal as a proxy for processing difficulty, and multilingual vs. monolingual training as a proxy for bilingual vs. monolingual cognition.

---

## Experimental Design

### Stimuli

Four sentence conditions, item-aligned with the behavioral study where possible:

| Condition | Example | Purpose |
|---|---|---|
| **V+Adv (critical)** | *John watches often television* | Main experimental condition |
| **Adv+V (grammatical)** | *John often watches television* | Grammatical baseline |
| **Unrelated violation** | *John television watches often* | Specificity control |
| **French source** | *Jean regarde souvent la télé* | Model sanity check |

**Critical measurement position:** the adverb token (*often*), matching the RT measurement point in the maze task.

### Model Comparison

#### Masked LMs (PLL via Salazar et al., 2020)

| Monolingual English | Multilingual | Notes |
|---|---|---|
| RoBERTa-base | XLM-R-base | **Primary pair** — identical architecture and training objective; multilingualism is the only variable |
| BERT-base-uncased | mBERT (bert-base-multilingual-cased) | Classic pair; adds BERT vs. RoBERTa architecture contrast |
| DistilBERT-base-uncased | DistilmBERT | Tests whether the effect survives distillation |
| RoBERTa-base | XLM-R-large | Scale-up check within the same architecture family |

#### Causal LMs (token-level surprisal from logits)

| Monolingual English | Multilingual / Bilingual | Notes |
|---|---|---|
| GPT-2 | mGPT | **Primary pair** |
| GPT-2 | CroissantLLM-Base | 50/50 EN/FR training — closest LLM analog to the human bilingual participants |
| OPT-125M | BLOOM-560M | Larger-scale; note different positional encoding (ALiBi vs. learned) |
| Pythia-160M | BLOOM-560M | Well-documented English-only baseline vs. BLOOM |

The **XLM-R vs. RoBERTa** contrast remains the primary comparison. All multilingual models listed include substantial French training data.

### Surprisal Metrics

- **Causal LMs:** standard token-level surprisal (negative log probability given left context)
- **Masked LMs:** pseudo-log-likelihood per token (Salazar et al., 2020) — mask each token, record the model's predicted probability for it given all other tokens

**Key derived metric:** per-item surprisal delta

```
ΔS(item) = surprisal_monolingual(item) − surprisal_multilingual(item)
```

A positive delta means the multilingual model is *less surprised* by the V+Adv order — the predicted direction if cross-lingual transfer is present.

### Region-Level Surprisal Aggregation

Behavioral RT is recorded per region, not per word. Surprisal is aggregated to match:

> **Summed surprisal over the region** = Σ surprisal(token) for all tokens in the region span

Summing is theoretically motivated: under surprisal theory (Levy, 2008), processing difficulty is additive across words, so a region's total cognitive cost equals the sum of its token-level surprisals.

The RT data covers four regions:

| Region | Content (V+Adv condition) | Content (Adv+V condition) | Role |
|---|---|---|---|
| Region 1 | *John* | *John* | Subject NP baseline |
| **Region 2** | *watches often* | *often watches* | **Primary — contains the syntactic manipulation** |
| **Region 3** | *television* | *television* | **Spillover check — RT often lags one region** |
| Region 4 | *at home* | *at home* | Post-critical baseline |

Both Region 2 and Region 3 are correlated with ΔS. Region 2 is the primary measure; Region 3 tests for downstream spillover effects that are common in reading-time studies.

### Tokenization Alignment Strategy

> *(Implementation note — may be removed before publication)*

LLM tokenizers (BPE, WordPiece, SentencePiece) do not split on word boundaries, so region text cannot be matched to token indices by word count alone. The strategy used here:

1. **Tokenize the full sentence** with `return_offsets_mapping=True` to get character-level `(start, end)` spans for each token.
2. **Locate the region** by finding the character offset of `region_text` as a substring of `sentence`.
3. **Select tokens** whose span falls within `[region_char_start, region_char_end)`.
4. **Sum their surprisal / PLL values** to produce the region-level score.

Edge cases to handle:
- **GPT-2 / mGPT:** prepend a space to non-initial tokens (Ġ prefix); strip whitespace from `region_text` before the substring search.
- **BERT / mBERT (WordPiece):** words may be split into `##` subword pieces; include all subword pieces that fall within the region span.
- **XLM-R / CroissantLLM (SentencePiece):** uses `▁` as a word-boundary marker; alignment via offset mapping is reliable but verify against the raw token strings.
- **BLOOM (ALiBi + byte-level BPE):** tokenizes similarly to GPT-2; offset mapping is straightforward.

### Correlation with Human Behavioral Data

Per-item ΔS is correlated with per-item mean residualized RT from the maze task (RT residualized on word length, frequency, and serial position). Pearson *r* with permutation test; Spearman *ρ* as a robustness check.

Correlations are run separately for:
- **Region 2 ΔS × Region 2 RT** — primary analysis
- **Region 3 ΔS × Region 3 RT** — spillover analysis

**Prediction:** items where multilingual models show the largest surprisal reduction are items where L1 French speakers showed the most RT facilitation — at Region 2, and potentially propagating to Region 3.

---

## Repo Structure

```
syntactic-echo/
├── data/
│   ├── stimuli.csv            # All sentence conditions with item IDs
│   └── behavioral_rt.csv      # Per-item maze task RT (available on request)
├── src/
│   ├── surprisal_causal.py    # Token surprisal for GPT-2 / mGPT
│   ├── surprisal_masked.py    # PLL extraction for RoBERTa / XLM-R
│   └── correlation.py         # Delta computation + RT correlation analysis
├── notebooks/
│   └── analysis.ipynb         # End-to-end walkthrough with figures
├── results/
│   └── figures/               # Condition bar charts, ΔS vs. RT scatter plots
├── requirements.txt
└── README.md
```

---

## Results

> 🚧 In progress — figures and correlation outputs will be added here.

**Predicted pattern:**

| Finding | Direction | Status |
|---|---|---|
| XLM-R < RoBERTa surprisal at *often* (V+Adv) | Multilingual advantage | Pending |
| No multilingual advantage in unrelated violation | Effect is syntax-specific | Pending |
| Positive ΔS × RT facilitation correlation | Model tracks human behavior | Pending |

---

## Future Directions

This project establishes a correlational baseline. Next steps toward a stronger causal claim:

1. **Fine-tuning experiment** — train a monolingual English model on French text and measure whether surprisal at V+Adv positions decreases post-training, isolating the causal role of French syntactic exposure
2. **Representational probing** — probe hidden states to test whether multilingual models encode adverb placement as a language-specific feature, and whether that shifts after French exposure
3. **Scaling analysis** — does the surprisal delta grow or shrink with model size?

---

## Requirements

```
transformers>=4.30.0
torch>=2.0.0
scipy
numpy
pandas
matplotlib
seaborn
jupyter
```

```bash
pip install -r requirements.txt
```

---

## Citation

```bibtex
@article{xing2025syntactic,
  author    = {Xing, Yubin},
  title     = {[Title — update when available]},
  journal   = {Bilingualism: Language and Cognition},
  year      = {2025},
  publisher = {Cambridge University Press},
  note      = {In production}
}
```

---

## License

This repository uses a dual license to distinguish between code and research materials.

**Code** — all files in `src/` and `notebooks/` are released under the **MIT License**. See [`LICENSE`](LICENSE) for details.

**Stimuli & behavioral data** — all files in `data/` are derived from or associated with the published behavioral study below and are shared under **CC BY-NC 4.0**. You are free to use, adapt, and redistribute these materials for non-commercial purposes with attribution. Commercial use is not permitted.

> Xing, Y. (2025). *[Title — update when available]*. *Bilingualism: Language and Cognition*. Cambridge University Press. In production.

[![CC BY-NC 4.0](https://licensebuttons.net/l/by-nc/4.0/88x31.png)](https://creativecommons.org/licenses/by-nc/4.0/)

---

## Author

**Yubin Xing**  
PhD Candidate, Psycholinguistics — University of Ottawa  
Computational Linguist / NLP Researcher  
[GitHub](https://github.com/) · [LinkedIn](https://linkedin.com/)
