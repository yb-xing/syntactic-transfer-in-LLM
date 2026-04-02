"""
surprisal_causal.py
Token-level surprisal extraction for causal language models.

Surprisal = -log P(token_i | token_1 ... token_{i-1}), derived from the model's
output logits at each position.

LLM input format:
    "{context_sentence} {full_sentence}"
    The context sentence primes the model; surprisal is measured only on tokens
    belonging to the target regions (Region 2 and Region 3) within full_sentence.

Supported model pairs (monolingual → multilingual):
    gpt2        → mgpt          (original pair)
    gpt2        → croissantllm  (50/50 EN/FR; closest LLM analog to human bilinguals)
    opt-125m    → bloom-560m    (larger-scale; ALiBi vs. learned positional encoding)
    pythia-160m → bloom-560m    (well-documented English-only baseline vs. BLOOM)
"""

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MODELS = {
    # Monolingual English (causal)
    "gpt2":        "openai-community/gpt2",
    "opt-125m":    "facebook/opt-125m",
    "pythia-160m": "EleutherAI/pythia-160m",
    # Multilingual / bilingual (causal) — all include substantial French
    "mgpt":         "ai-forever/mGPT",
    "bloom-560m":   "bigscience/bloom-560m",
    "croissantllm": "croissantllm/CroissantLLM-Base",
}

# Each tuple: (monolingual_key, multilingual_key)
MODEL_PAIRS = [
    ("gpt2",        "mgpt"),
    ("gpt2",        "croissantllm"),
    ("opt-125m",    "bloom-560m"),
    ("pythia-160m", "bloom-560m"),
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
    """Load a causal LM tokenizer and model from the MODELS registry.

    Models are loaded in fp32 and moved to the best available device
    (CUDA → MPS → CPU). For BLOOM and CroissantLLM on CPU, inference will
    be slow; a GPU is recommended.

    Args:
        model_key: One of the keys in MODELS
                   (e.g. 'gpt2', 'mgpt', 'bloom-560m', 'croissantllm').

    Returns:
        (tokenizer, model) — tokenizer is the fast tokenizer; model is in eval mode.
    """
    if model_key not in MODELS:
        raise ValueError(f"Unknown model key '{model_key}'. Choose from: {list(MODELS)}")

    model_id = MODELS[model_key]
    device   = _get_device()
    print(f"  Loading {model_key} ({model_id}) on {device} …")

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    model     = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float32)
    model     = model.to(device)
    model.eval()

    # Some causal tokenizers (GPT-2, Pythia) have no pad token; set to eos.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model


# ---------------------------------------------------------------------------
# Core surprisal computation
# ---------------------------------------------------------------------------

def get_token_surprisals(sentence: str, tokenizer, model) -> list[dict]:
    """Compute per-token surprisal for a single sentence.

    Surprisal of token i = -log P(token_i | token_0 ... token_{i-1})
                         = -log_softmax(logits[i-1])[token_id_i]

    The first token has no left context and is skipped (no surprisal assigned).

    Special tokens (BOS, EOS) added by the tokenizer are also skipped via
    offset_mapping: special tokens receive offset (0, 0) from fast tokenizers.

    Args:
        sentence:  Input string (e.g. "{context_sentence} {full_sentence}").
        tokenizer: HuggingFace fast tokenizer.
        model:     Causal LM in eval mode.

    Returns:
        List of dicts with keys:
            'token'      — string representation of the token
            'token_index'— integer position in the tokenized sequence
            'surprisal'  — float, -log P(token | left context), in nats
            'char_start' — character start offset in `sentence`
            'char_end'   — character end offset in `sentence`
    """
    encoding = tokenizer(
        sentence,
        return_tensors="pt",
        return_offsets_mapping=True,
    )
    input_ids      = encoding["input_ids"].to(model.device)        # (1, seq_len)
    offset_mapping = encoding["offset_mapping"][0].tolist()        # list of (start, end)

    with torch.no_grad():
        logits = model(input_ids).logits                           # (1, seq_len, vocab)

    # log P(token_i | context) = log_softmax(logits[i-1])[token_id_i]
    log_probs = torch.nn.functional.log_softmax(logits[0], dim=-1) # (seq_len, vocab)

    tokens  = tokenizer.convert_ids_to_tokens(input_ids[0])
    results = []

    for i in range(1, len(tokens)):                                # skip position 0 (no context)
        char_start, char_end = offset_mapping[i]

        # Skip special tokens: fast tokenizers give special tokens offset (0, 0).
        # Note: (0, 0) is unambiguous because real tokens always have char_end > 0.
        if char_start == 0 and char_end == 0:
            continue

        token_id  = input_ids[0, i].item()
        surprisal = -log_probs[i - 1, token_id].item()

        results.append({
            "token":       tokens[i],
            "token_index": i,
            "surprisal":   surprisal,
            "char_start":  char_start,
            "char_end":    char_end,
        })

    return results


# ---------------------------------------------------------------------------
# Region-level aggregation
# ---------------------------------------------------------------------------

def get_region_surprisal(
    sentence:    str,
    region_text: str,
    tokenizer,
    model,
    search_start: int = 0,
) -> float:
    """Return the summed surprisal for all tokens within a region span.

    Tokenization alignment (README §Tokenization Alignment Strategy):
      1. Tokenize the full sentence with return_offsets_mapping=True.
      2. Locate region_text as a substring of sentence (starting from search_start
         to avoid false matches in the context sentence).
      3. Select tokens whose character span overlaps [region_start, region_end).
      4. Sum their surprisal values.

    Summing is theoretically justified: under surprisal theory (Levy, 2008),
    processing difficulty is additive, so a region's cost = Σ token surprisals.

    Args:
        sentence:     Full input string (context + target sentence).
        region_text:  The text chunk for the region (e.g. "watches often").
        tokenizer:    HuggingFace fast tokenizer.
        model:        Causal LM in eval mode.
        search_start: Character offset in `sentence` from which to begin searching
                      for region_text (use to avoid context-sentence false matches).

    Returns:
        Summed surprisal (float) over all tokens in the region, or NaN if
        region_text is not found in sentence.
    """
    region_text_stripped = region_text.strip()
    region_start = sentence.find(region_text_stripped, search_start)

    if region_start == -1:
        return float("nan")

    region_end = region_start + len(region_text_stripped)

    token_surprisals = get_token_surprisals(sentence, tokenizer, model)

    total = 0.0
    matched = 0
    for tok in token_surprisals:
        # Overlap: token span intersects [region_start, region_end)
        if tok["char_start"] < region_end and tok["char_end"] > region_start:
            total   += tok["surprisal"]
            matched += 1

    return total if matched > 0 else float("nan")


def get_critical_surprisal(
    sentence:       str,
    critical_token: str,
    tokenizer,
    model,
    search_start:   int = 0,
) -> float:
    """Return the surprisal of a critical token (the adverb) within the sentence.

    Delegates to get_region_surprisal with critical_token as the region text —
    this correctly sums surprisal over all subword pieces of the adverb.

    Args:
        sentence:       Full input string (context + target).
        critical_token: The adverb of interest (e.g. "often").
        tokenizer:      HuggingFace fast tokenizer.
        model:          Causal LM in eval mode.
        search_start:   Character offset to begin the substring search from.

    Returns:
        Surprisal (float) for the critical token span, or NaN if not found.
    """
    return get_region_surprisal(sentence, critical_token, tokenizer, model, search_start)


# ---------------------------------------------------------------------------
# Full extraction pipeline
# ---------------------------------------------------------------------------

def run_surprisal_extraction(stimuli_path: str, model_key: str) -> pd.DataFrame:
    """Run surprisal extraction over all stimuli for one causal LM.

    For each item × condition row in stimuli.csv, the LLM input is:
        "{context_sentence} {full_sentence}"

    Surprisal is measured at three granularities:
        - Region 2 (V+Adv or Adv+V chunk) — primary analysis
        - Region 3 (object NP)             — spillover analysis
        - Critical token (the adverb)      — single-word control

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
    """
    stimuli = pd.read_csv(stimuli_path)
    tokenizer, model = load_model(model_key)

    records = []
    for _, row in tqdm(stimuli.iterrows(), total=len(stimuli),
                       desc=f"[{model_key}] surprisal"):

        context      = str(row["context_sentence"]).strip()
        full_sent    = str(row["full_sentence"]).strip()
        region2_text = str(row["region2_text"]).strip()
        region3_text = str(row["region3_text"]).strip()
        crit_token   = str(row["critical_token"]).strip()

        # Build the combined input; note where the target sentence begins
        combined     = f"{context} {full_sent}"
        sent_offset  = len(context) + 1   # character position where full_sent starts

        surp_r2   = get_region_surprisal(combined, region2_text,  tokenizer, model, sent_offset)
        surp_r3   = get_region_surprisal(combined, region3_text,  tokenizer, model, sent_offset)
        surp_crit = get_critical_surprisal(combined, crit_token,  tokenizer, model, sent_offset)

        records.append({
            "item_no":                    row["item_no"],
            "condition":                  row["condition"],
            "advTYPE":                    row["advTYPE"],
            "model":                      model_key,
            "surprisal_region2":          surp_r2,
            "surprisal_region3":          surp_r3,
            "surprisal_critical_token":   surp_crit,
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Extract token-level surprisal from causal LMs over maze stimuli."
    )
    parser.add_argument("--stimuli", required=True,
                        help="Path to stimuli.csv (output of data_preprocessing.py)")
    parser.add_argument("--model",   required=True, choices=list(MODELS),
                        help="Model key to run (see MODELS dict)")
    parser.add_argument("--output",  required=True,
                        help="Path to write surprisal output CSV")
    args = parser.parse_args()

    results = run_surprisal_extraction(args.stimuli, args.model)
    results.to_csv(args.output, index=False)
    print(f"Saved → {args.output}  ({len(results)} rows)")
