"""
surprisal_masked.py
Pseudo-log-likelihood (PLL) extraction for masked language models.

Method: Salazar et al. (2020)
    For each token position i, replace token_i with [MASK], run a forward pass,
    and record log P(token_i | all other tokens).
    PLL is used as a surprisal proxy: higher PLL = lower surprise.
    To keep sign convention consistent with causal surprisal, we store
    NEGATIVE PLL, i.e. -log P(token_i | context), so higher = more surprising.

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
    significantly slower than causal surprisal extraction. A progress bar
    is shown at the sentence level; per-token passes are silent.

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
            surprisal_region2, surprisal_region3, surprisal_critical_token
        (column names use 'surprisal_' prefix to stay consistent with
        the causal output so correlation.py can treat them uniformly)
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

        pll_r2   = get_region_pll(combined, region2_text, tokenizer, model, sent_offset)
        pll_r3   = get_region_pll(combined, region3_text, tokenizer, model, sent_offset)
        pll_crit = get_critical_pll(combined, crit_token, tokenizer, model, sent_offset)

        records.append({
            "item_no":                   row["item_no"],
            "condition":                 row["condition"],
            "advTYPE":                   row["advTYPE"],
            "model":                     model_key,
            "surprisal_region2":         pll_r2,
            "surprisal_region3":         pll_r3,
            "surprisal_critical_token":  pll_crit,
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
