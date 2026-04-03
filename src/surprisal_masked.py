"""
surprisal_masked.py
Pseudo-log-likelihood (PLL) extraction for masked language models.

Two PLL variants are computed for every region:

  Standard PLL — Salazar et al. (2020):
      For each token position i, replace token_i with [MASK], run a forward
      pass, and record log P(token_i | all other tokens).
      Stored as 'surprisal_region*' columns (negative PLL, higher = more
      surprising), consistent with causal surprisal naming.

  PLL-word-l2r — Kauf & Ivanova (2023):
      Adjusted metric that prevents the model from using within-word future
      subword information.  For each token t at position p within word w,
      mask token t AND all subsequent tokens t' ≥ t in the same word before
      running the forward pass:

          PLL-word-l2r(S) = Σ_w Σ_{t=1}^{|w|}
                                log P_MLM(s_wt | S \\ {s_wt' : t' ≥ t})

      For single-token words the two metrics are identical.
      Stored as 'surprisal_region*_PLL_word_l2r' columns.

LLM input format:
    "{context_sentence} {full_sentence}"
    The context sentence is included in the bidirectional attention window.
    Only tokens in the target regions (Region 2, Region 3) are scored.

Supported model pairs (monolingual → multilingual):
    roberta      → xlmr          (primary pair; identical architecture, best-controlled)
    bert         → mbert         (classic pair; adds BERT vs. RoBERTa contrast)
    distilbert   → distilmbert   (tests whether the effect survives distillation)
    roberta      → xlmr-large    (scale-up check within the RoBERTa family)
"""

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForMaskedLM


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MODELS = {
    # Monolingual English (masked)
    "roberta":    "roberta-base",
    "bert":       "bert-base-uncased",
    "distilbert": "distilbert-base-uncased",
    # Multilingual (masked) — all include substantial French
    "xlmr":        "xlm-roberta-base",
    "mbert":       "bert-base-multilingual-cased",
    "distilmbert": "distilbert-base-multilingual-cased",
    "xlmr-large":  "xlm-roberta-large",
}

# Each tuple: (monolingual_key, multilingual_key)
MODEL_PAIRS = [
    ("roberta",    "xlmr"),
    ("bert",       "mbert"),
    ("distilbert", "distilmbert"),
    ("roberta",    "xlmr-large"),
]


# ---------------------------------------------------------------------------
# Device helper
# ---------------------------------------------------------------------------

def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_key: str) -> tuple:
    """Load a masked LM tokenizer and model from the MODELS registry.

    Models are loaded in fp32 and moved to the best available device
    (CUDA → MPS → CPU).

    Args:
        model_key: One of the keys in MODELS
                   (e.g. 'roberta', 'xlmr', 'bert', 'mbert').

    Returns:
        (tokenizer, model) — fast tokenizer; model in eval mode.
    """
    if model_key not in MODELS:
        raise ValueError(f"Unknown model key '{model_key}'. Choose from: {list(MODELS)}")

    model_id = MODELS[model_key]
    device   = _get_device()
    print(f"  Loading {model_key} ({model_id}) on {device} …")

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model     = AutoModelForMaskedLM.from_pretrained(model_id, torch_dtype=torch.float32)
    model     = model.to(device)
    model.eval()

    return tokenizer, model


# ---------------------------------------------------------------------------
# Core PLL computation
# ---------------------------------------------------------------------------

def get_token_pll(sentence: str, tokenizer, model) -> list[dict]:
    """Compute per-token pseudo-log-likelihood (PLL) for a single sentence.

    For each non-special token i:
      1. Replace token_i with the model's MASK token.
      2. Run a forward pass to get logits.
      3. Record log P(token_i | all other tokens) = log_softmax(logits[i])[token_id_i].

    The returned value is the NEGATIVE PLL (i.e. -log P), so it is directly
    comparable in sign convention to causal surprisal (higher = more surprising).

    Special tokens ([CLS], [SEP], <s>, </s>) are identified via offset_mapping:
    fast tokenizers assign them offset (0, 0), which is unambiguous because real
    content tokens always have char_end > 0.

    WordPiece subword pieces (## prefix in BERT/mBERT) are handled correctly
    because offset_mapping covers each piece individually — all pieces within a
    region's character span are included automatically.

    Args:
        sentence:  Input string (context + target sentence).
        tokenizer: HuggingFace fast tokenizer.
        model:     Masked LM in eval mode.

    Returns:
        List of dicts with keys:
            'token'       — string representation of the subword token
            'token_index' — integer position in the tokenized sequence
            'pll'         — float, -log P(token | context), in nats
            'char_start'  — character start offset in `sentence`
            'char_end'    — character end offset in `sentence`
    """
    encoding = tokenizer(
        sentence,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    input_ids      = encoding["input_ids"].to(model.device)        # (1, seq_len)
    offset_mapping = encoding["offset_mapping"][0].tolist()        # list of (start, end)
    tokens         = tokenizer.convert_ids_to_tokens(input_ids[0])

    results = []

    for i in range(len(tokens)):
        char_start, char_end = offset_mapping[i]

        # Skip special tokens: offset (0, 0) is unambiguous for special tokens
        # because real content tokens always have char_end > char_start > 0,
        # or char_start == 0 with char_end > 0 for the very first content token.
        if char_start == 0 and char_end == 0:
            continue

        original_id = input_ids[0, i].item()

        # Build masked input: clone and replace position i with MASK token id
        masked_ids       = input_ids.clone()
        masked_ids[0, i] = tokenizer.mask_token_id

        with torch.no_grad():
            logits = model(masked_ids).logits                      # (1, seq_len, vocab)

        log_probs = torch.nn.functional.log_softmax(logits[0, i], dim=-1)  # (vocab,)
        neg_pll   = -log_probs[original_id].item()

        results.append({
            "token":       tokens[i],
            "token_index": i,
            "pll":         neg_pll,
            "char_start":  char_start,
            "char_end":    char_end,
        })

    return results


# ---------------------------------------------------------------------------
# Region-level aggregation
# ---------------------------------------------------------------------------

def get_region_pll(
    sentence:    str,
    region_text: str,
    tokenizer,
    model,
    search_start: int = 0,
) -> float:
    """Return the summed negative-PLL for all tokens within a region span.

    Tokenization alignment (README §Tokenization Alignment Strategy):
      1. Tokenize the full sentence with return_offsets_mapping=True.
      2. Locate region_text as a substring of sentence (from search_start
         to avoid matching the context sentence).
      3. Select tokens whose character span overlaps [region_start, region_end).
      4. For each selected token, run a masked forward pass and record -log P.
      5. Sum the per-token values.

    WordPiece subword pieces for words in the region are all included because
    their character spans fall within the region boundaries.

    Args:
        sentence:     Full input string (context + target sentence).
        region_text:  Text chunk for the region (e.g. "watches often").
        tokenizer:    HuggingFace fast tokenizer.
        model:        Masked LM in eval mode.
        search_start: Character offset to begin searching for region_text,
                      preventing false matches in the context sentence.

    Returns:
        Summed -log P (float) over all tokens in the region, or NaN if
        region_text is not found in sentence.
    """
    region_text_stripped = region_text.strip()
    region_start = sentence.find(region_text_stripped, search_start)

    if region_start == -1:
        return float("nan")

    region_end = region_start + len(region_text_stripped)

    token_plls = get_token_pll(sentence, tokenizer, model)

    total   = 0.0
    matched = 0
    for tok in token_plls:
        # Token overlaps with region if its span intersects [region_start, region_end)
        if tok["char_start"] < region_end and tok["char_end"] > region_start:
            total   += tok["pll"]
            matched += 1

    return total if matched > 0 else float("nan")


# ---------------------------------------------------------------------------
# Word-grouping helper (for PLL-word-l2r)
# ---------------------------------------------------------------------------

def _group_tokens_into_words(token_records: list[dict]) -> list[list[dict]]:
    """Group token records into words based on character-span contiguity.

    Two consecutive tokens belong to the same word when there is no character
    gap between them (char_start of token i+1 == char_end of token i), i.e.
    they are subword pieces of the same surface form (e.g. BERT "##ing" pieces).

    Args:
        token_records: Non-special token dicts produced by get_token_pll or
                       the equivalent encoding loop.  Each must have
                       'char_start' and 'char_end'.

    Returns:
        List of word groups; each group is an ordered list of token dicts.
    """
    if not token_records:
        return []

    words: list[list[dict]] = []
    current_word = [token_records[0]]

    for tok in token_records[1:]:
        if tok["char_start"] == current_word[-1]["char_end"]:
            # No gap → continuation of the same surface word
            current_word.append(tok)
        else:
            words.append(current_word)
            current_word = [tok]

    words.append(current_word)
    return words


# ---------------------------------------------------------------------------
# PLL-word-l2r computation (Kauf & Ivanova 2023)
# ---------------------------------------------------------------------------

def get_token_pll_word_l2r(sentence: str, tokenizer, model) -> list[dict]:
    """Compute per-token PLL-word-l2r for a single sentence.

    For each token t at position p within word w, the masked input replaces
    token t AND all subsequent within-word tokens (positions p, p+1, …, |w|-1)
    with the MASK token before running the forward pass.  Only one forward
    pass is needed per token.

    For single-token words this is identical to standard PLL.  For multi-token
    words, earlier subword pieces are scored without access to the future pieces
    that would not be available in a left-to-right reading.

    Reference: Kauf & Ivanova (2023), "A Better Way to Do Masked Language
    Model Scoring", ACL 2023.

    Args:
        sentence:  Input string (context + target sentence).
        tokenizer: HuggingFace fast tokenizer.
        model:     Masked LM in eval mode.

    Returns:
        List of dicts with the same keys as get_token_pll:
            'token', 'token_index', 'pll', 'char_start', 'char_end'
        but 'pll' is computed with the l2r within-word masking strategy.
    """
    encoding = tokenizer(
        sentence,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    input_ids      = encoding["input_ids"].to(model.device)     # (1, seq_len)
    offset_mapping = encoding["offset_mapping"][0].tolist()     # list[(start, end)]
    tokens         = tokenizer.convert_ids_to_tokens(input_ids[0])

    # Collect non-special token records (same filter as get_token_pll)
    non_special: list[dict] = []
    for i, (char_start, char_end) in enumerate(offset_mapping):
        if char_start == 0 and char_end == 0:
            continue
        non_special.append({
            "token":       tokens[i],
            "token_index": i,
            "char_start":  char_start,
            "char_end":    char_end,
            "original_id": input_ids[0, i].item(),
        })

    word_groups = _group_tokens_into_words(non_special)
    results: list[dict] = []

    for word_tokens in word_groups:
        for pos_in_word, tok in enumerate(word_tokens):
            # Mask: current token + all following tokens within the same word
            mask_indices = [t["token_index"] for t in word_tokens[pos_in_word:]]

            masked_ids = input_ids.clone()
            for idx in mask_indices:
                masked_ids[0, idx] = tokenizer.mask_token_id

            with torch.no_grad():
                logits = model(masked_ids).logits                # (1, seq_len, vocab)

            i         = tok["token_index"]
            log_probs = torch.nn.functional.log_softmax(logits[0, i], dim=-1)
            neg_pll   = -log_probs[tok["original_id"]].item()

            results.append({
                "token":       tok["token"],
                "token_index": i,
                "pll":         neg_pll,
                "char_start":  tok["char_start"],
                "char_end":    tok["char_end"],
            })

    return results


# ---------------------------------------------------------------------------
# Region-level aggregation — PLL-word-l2r
# ---------------------------------------------------------------------------

def get_region_pll_word_l2r(
    sentence:    str,
    region_text: str,
    tokenizer,
    model,
    search_start: int = 0,
) -> float:
    """Return summed PLL-word-l2r for all tokens within a region span.

    Identical alignment logic to get_region_pll, but uses
    get_token_pll_word_l2r for the per-token scores.

    Args:
        sentence:     Full input string (context + target sentence).
        region_text:  Text chunk for the region (e.g. "watches often").
        tokenizer:    HuggingFace fast tokenizer.
        model:        Masked LM in eval mode.
        search_start: Character offset to begin searching for region_text.

    Returns:
        Summed PLL-word-l2r (float) over all tokens in the region, or NaN if
        region_text is not found in sentence.
    """
    region_text_stripped = region_text.strip()
    region_start = sentence.find(region_text_stripped, search_start)

    if region_start == -1:
        return float("nan")

    region_end = region_start + len(region_text_stripped)

    token_plls = get_token_pll_word_l2r(sentence, tokenizer, model)

    total   = 0.0
    matched = 0
    for tok in token_plls:
        if tok["char_start"] < region_end and tok["char_end"] > region_start:
            total   += tok["pll"]
            matched += 1

    return total if matched > 0 else float("nan")


def get_critical_pll(
    sentence:       str,
    critical_token: str,
    tokenizer,
    model,
    search_start:   int = 0,
) -> float:
    """Return the negative-PLL of the critical token (adverb) within the sentence.

    Delegates to get_region_pll with critical_token as the region text —
    this correctly sums over all subword pieces of the adverb.

    Args:
        sentence:       Full input string (context + target).
        critical_token: The adverb of interest (e.g. "often").
        tokenizer:      HuggingFace fast tokenizer.
        model:          Masked LM in eval mode.
        search_start:   Character offset to begin the substring search.

    Returns:
        Summed -log P (float) for the critical token, or NaN if not found.
    """
    return get_region_pll(sentence, critical_token, tokenizer, model, search_start)


# ---------------------------------------------------------------------------
# Full extraction pipeline
# ---------------------------------------------------------------------------

def run_pll_extraction(stimuli_path: str, model_key: str) -> pd.DataFrame:
    """Run PLL extraction over all stimuli for one masked LM.

    For each item × condition row in stimuli.csv, the LLM input is:
        "{context_sentence} {full_sentence}"

    Negative PLL is measured at three granularities (matching surprisal_causal.py):
        - Region 2 (V+Adv or Adv+V chunk) — primary analysis
        - Region 3 (object NP)             — spillover analysis
        - Critical token (the adverb)      — single-word control

    Note: PLL requires one forward pass per token per sentence, so this is
    significantly slower than causal surprisal extraction.  PLL-word-l2r
    requires an additional forward pass per within-word token (but is only
    slower than standard PLL for multi-token words).  A progress bar is shown
    at the sentence level; per-token passes are silent.

    Expected stimuli.csv columns (produced by data_preprocessing.py):
        item_no, condition, advTYPE, context_sentence,
        region1_text, region2_text, region3_text, region4_text,
        full_sentence, critical_token

    Args:
        stimuli_path: Path to stimuli.csv.
        model_key:    One of the keys in MODELS.

    Returns:
        DataFrame with columns:
            item_no, condition, advTYPE, model,
            surprisal_region2,              surprisal_region2_PLL_word_l2r,
            surprisal_region3,              surprisal_region3_PLL_word_l2r,
            surprisal_critical_token,       surprisal_critical_token_PLL_word_l2r

        The 'surprisal_region*' columns (standard PLL) use the same names as
        surprisal_causal.py so that correlation.py can treat all models
        uniformly.  The '_PLL_word_l2r' columns are additional outputs for
        the Kauf & Ivanova (2023) adjusted metric.
    """
    stimuli = pd.read_csv(stimuli_path)
    tokenizer, model = load_model(model_key)

    records = []
    for _, row in tqdm(stimuli.iterrows(), total=len(stimuli),
                       desc=f"[{model_key}] PLL"):

        context      = str(row["context_sentence"]).strip()
        full_sent    = str(row["full_sentence"]).strip()
        region2_text = str(row["region2_text"]).strip()
        region3_text = str(row["region3_text"]).strip()
        crit_token   = str(row["critical_token"]).strip()

        # Build the combined input; note where the target sentence begins
        combined    = f"{context} {full_sent}"
        sent_offset = len(context) + 1    # char position where full_sent starts

        # Standard PLL (Salazar et al. 2020)
        pll_r2   = get_region_pll(combined, region2_text, tokenizer, model, sent_offset)
        pll_r3   = get_region_pll(combined, region3_text, tokenizer, model, sent_offset)
        pll_crit = get_critical_pll(combined, crit_token, tokenizer, model, sent_offset)

        # PLL-word-l2r (Kauf & Ivanova 2023)
        pll_l2r_r2   = get_region_pll_word_l2r(combined, region2_text, tokenizer, model, sent_offset)
        pll_l2r_r3   = get_region_pll_word_l2r(combined, region3_text, tokenizer, model, sent_offset)
        pll_l2r_crit = get_region_pll_word_l2r(combined, crit_token,   tokenizer, model, sent_offset)

        records.append({
            "item_no":                              row["item_no"],
            "condition":                            row["condition"],
            "advTYPE":                              row["advTYPE"],
            "model":                                model_key,
            # Standard PLL (compatible with correlation.py)
            "surprisal_region2":                    pll_r2,
            "surprisal_region3":                    pll_r3,
            "surprisal_critical_token":             pll_crit,
            # PLL-word-l2r (Kauf & Ivanova 2023)
            "surprisal_region2_PLL_word_l2r":       pll_l2r_r2,
            "surprisal_region3_PLL_word_l2r":       pll_l2r_r3,
            "surprisal_critical_token_PLL_word_l2r": pll_l2r_crit,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract pseudo-log-likelihood from masked LMs over maze stimuli."
    )
    parser.add_argument("--stimuli", required=True,
                        help="Path to stimuli.csv (output of data_preprocessing.py)")
    parser.add_argument("--model",   required=True, choices=list(MODELS),
                        help="Model key to run (see MODELS dict)")
    parser.add_argument("--output",  required=True,
                        help="Path to write PLL output CSV")
    args = parser.parse_args()

    results = run_pll_extraction(args.stimuli, args.model)
    results.to_csv(args.output, index=False)
    print(f"Saved → {args.output}  ({len(results)} rows)")
