"""
utils.py
Embedding extraction and dimensionality-reduction visualisation for the
syntactic-transfer experiment.

Two conditions per item:
    EN_Word_order — grammatical English (Adv+V): "My neighbor usually drinks …"
    FR_Word_order — French-order English  (V+Adv): "My neighbor drinks usually …"

The core claim is that multilingual models represent the two conditions more
similarly (smaller inter-condition distance) than their monolingual counterparts.

Typical workflow
----------------
    from src.utils import extract_embeddings, reduce_dimensions, plot_pair

    df = pd.read_csv("data/stimuli_generated.csv")

    # masked LM pair
    emb_roberta = extract_embeddings(df, "roberta", model_type="masked")
    emb_xlmr    = extract_embeddings(df, "xlmr",    model_type="masked")

    # causal LM pair
    emb_gpt2 = extract_embeddings(df, "gpt2", model_type="causal")
    emb_mgpt = extract_embeddings(df, "mgpt", model_type="causal")

    fig = plot_pair(emb_roberta, emb_xlmr, method="pca",  label="RoBERTa vs XLM-R")
    fig = plot_pair(emb_gpt2,    emb_mgpt,  method="tsne", label="GPT-2 vs mGPT")
"""

from __future__ import annotations

import warnings
from typing import Literal

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForCausalLM,
)

# ---------------------------------------------------------------------------
# Model registries (mirrors surprisal_masked.py / surprisal_causal.py)
# ---------------------------------------------------------------------------

MASKED_MODELS: dict[str, str] = {
    "roberta":     "roberta-base",
    "bert":        "bert-base-uncased",
    "distilbert":  "distilbert-base-uncased",
    "xlmr":        "xlm-roberta-base",
    "mbert":       "bert-base-multilingual-cased",
    "distilmbert": "distilbert-base-multilingual-cased",
    "xlmr-large":  "xlm-roberta-large",
}

CAUSAL_MODELS: dict[str, str] = {
    "gpt2":        "openai-community/gpt2",
    "opt-125m":    "facebook/opt-125m",
    "pythia-160m": "EleutherAI/pythia-160m",
    "mgpt":        "ai-forever/mGPT",
    "bloom-560m":  "bigscience/bloom-560m",
    "croissantllm":"croissantllm/CroissantLLMBase",
}

MULTILINGUAL_KEYS = {"xlmr", "mbert", "distilmbert", "xlmr-large", "mgpt", "bloom-560m", "croissantllm"}

CONDITION_COLORS = {
    "EN_Word_order": "#2196F3",  # blue  — grammatical
    "FR_Word_order": "#F44336",  # red   — French-order
}
CONDITION_LABELS = {
    "EN_Word_order": "EN order (Adv+V)",
    "FR_Word_order": "FR order (V+Adv)",
}

# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

def _get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _free_device_memory() -> None:
    import gc
    gc.collect()
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()


def _load_model(model_key: str, model_type: Literal["masked", "causal"]):
    registry = MASKED_MODELS if model_type == "masked" else CAUSAL_MODELS
    if model_key not in registry:
        raise ValueError(f"Unknown model key '{model_key}'. Available: {list(registry)}")

    _free_device_memory()  # release any previously held tensors before loading

    model_id = registry[model_key]
    device   = _get_device()
    print(f"  Loading {model_key} ({model_id}) on {device} …")

    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    cls       = AutoModelForMaskedLM if model_type == "masked" else AutoModelForCausalLM
    # Load weights on CPU first, then move — avoids a double-allocation spike on MPS
    model = cls.from_pretrained(model_id, torch_dtype=torch.float32,
                                output_hidden_states=True)
    model = model.to(device).eval()

    if model_type == "causal" and tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model


def _get_critical_token_index(sentence: str, critical_token: str, tokenizer) -> int | None:
    """Return the token index of the first occurrence of critical_token in the sentence."""
    encoding       = tokenizer(sentence, return_offsets_mapping=True)
    offset_mapping = encoding["offset_mapping"]
    tokens         = tokenizer.convert_ids_to_tokens(encoding["input_ids"])

    # Locate critical_token by character span search
    lower = sentence.lower()
    char_pos = lower.find(critical_token.lower())
    if char_pos == -1:
        return None

    for idx, (cs, ce) in enumerate(offset_mapping):
        if cs == 0 and ce == 0:
            continue  # special token
        if cs <= char_pos < ce:
            return idx

    return None


@torch.no_grad()
def _forward_hidden_states(sentence: str, tokenizer, model, layer: int):
    """Single forward pass; returns (hidden_state_tensor, offset_mapping)."""
    inputs = tokenizer(sentence, return_tensors="pt", return_offsets_mapping=True)
    offset_mapping = inputs.pop("offset_mapping")[0].tolist()
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    outputs = model(**inputs)
    # hidden_states: tuple of (n_layers+1) tensors, shape (1, seq_len, hidden_dim)
    hs = outputs.hidden_states[layer][0].float().cpu()  # (seq_len, hidden_dim)
    return hs, offset_mapping


def _token_indices_for_span(offset_mapping: list, char_start: int, char_end: int) -> list[int]:
    """Return token indices whose character span overlaps [char_start, char_end)."""
    return [
        i for i, (cs, ce) in enumerate(offset_mapping)
        if not (cs == 0 and ce == 0)   # skip special tokens
        and ce > char_start and cs < char_end
    ]


def _vec_for_region2(sentence: str, critical_token: str,
                     tokenizer, model, layer: int) -> np.ndarray | None:
    """Hidden state at the critical (adverb) token — single token, no averaging."""
    hs, offsets = _forward_hidden_states(sentence, tokenizer, model, layer)
    lower    = sentence.lower()
    char_pos = lower.find(critical_token.lower())
    if char_pos == -1:
        return None
    idxs = _token_indices_for_span(offsets, char_pos, char_pos + len(critical_token))
    if not idxs:
        return None
    return hs[idxs[0]].numpy()  # first subword token of the adverb


def _vec_for_region3(sentence: str, region3_text: str,
                     tokenizer, model, layer: int) -> np.ndarray | None:
    """Mean hidden state over all tokens in the Region 3 span."""
    hs, offsets = _forward_hidden_states(sentence, tokenizer, model, layer)
    lower    = sentence.lower()
    char_pos = lower.find(region3_text.lower().strip())
    if char_pos == -1:
        return None
    char_end = char_pos + len(region3_text.strip())
    idxs = _token_indices_for_span(offsets, char_pos, char_end)
    if not idxs:
        return None
    return hs[idxs].mean(dim=0).numpy()  # mean-pool over region tokens


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_embeddings(
    stimuli_df: pd.DataFrame,
    model_key: str,
    model_type: Literal["masked", "causal"],
    regions: list[Literal["region2", "region3"]] | None = None,
    layer: int = -1,
    context_col: str = "context_sentence",
    sentence_col: str = "full_sentence",
    critical_col: str = "critical_token",
    region3_col: str = "region3_text",
) -> dict[str, pd.DataFrame]:
    """
    Extract hidden-state embeddings for one or more regions for every
    item × condition row in stimuli_df.

    Region 2 — single hidden state at the critical (adverb) token.
    Region 3 — mean hidden state over all tokens in the region3_text span.

    Parameters
    ----------
    stimuli_df  : DataFrame with columns: item_no, condition,
                  context_sentence, full_sentence, critical_token, region3_text.
    model_key   : Key from MASKED_MODELS or CAUSAL_MODELS.
    model_type  : "masked" or "causal".
    regions     : Which regions to extract. Defaults to ["region2", "region3"].
    layer       : Transformer layer index (-1 = last layer).
    *_col       : Column name overrides.

    Returns
    -------
    Dict mapping region name → DataFrame with columns:
        item_no, condition, model, is_multilingual, embedding (ndarray).
    Rows where the target span cannot be located are dropped with a warning.
    """
    if regions is None:
        regions = ["region2", "region3"]

    tokenizer, model = _load_model(model_key, model_type)
    is_multi = model_key in MULTILINGUAL_KEYS

    records: dict[str, list] = {r: [] for r in regions}

    for _, row in stimuli_df.iterrows():
        full_input = f"{row[context_col]} {row[sentence_col]}"
        meta = {
            "item_no":         row["item_no"],
            "condition":       row["condition"],
            "model":           model_key,
            "is_multilingual": is_multi,
        }

        if "region2" in regions:
            vec = _vec_for_region2(full_input, row[critical_col], tokenizer, model, layer)
            if vec is None:
                warnings.warn(
                    f"item {row['item_no']} / {row['condition']}: critical token "
                    f"'{row[critical_col]}' not found. Region 2 row skipped."
                )
            else:
                records["region2"].append({**meta, "embedding": vec})

        if "region3" in regions:
            vec = _vec_for_region3(full_input, row[region3_col], tokenizer, model, layer)
            if vec is None:
                warnings.warn(
                    f"item {row['item_no']} / {row['condition']}: region3 text "
                    f"'{row[region3_col]}' not found. Region 3 row skipped."
                )
            else:
                records["region3"].append({**meta, "embedding": vec})

    del model
    _free_device_memory()

    return {r: pd.DataFrame(records[r]) for r in regions}


def reduce_dimensions(
    emb_df: pd.DataFrame,
    method: Literal["pca", "tsne", "umap"] = "pca",
    n_components: int = 2,
    random_state: int = 42,
    **method_kwargs,
) -> pd.DataFrame:
    """
    Apply dimensionality reduction to the embedding column and return a copy
    of emb_df with added columns dim_0, dim_1 (and dim_2 if n_components==3).

    Parameters
    ----------
    emb_df       : Output of extract_embeddings().
    method       : "pca", "tsne", or "umap".
    n_components : Target number of dimensions (usually 2).
    random_state : Seed for stochastic methods.
    **method_kwargs : Extra kwargs forwarded to the reducer constructor.

    Returns
    -------
    emb_df copy with dim_0, dim_1 (and optionally dim_2) columns added.
    """
    X = np.stack(emb_df["embedding"].values)
    X = StandardScaler().fit_transform(X)

    if method == "pca":
        reducer = PCA(n_components=n_components, random_state=random_state, **method_kwargs)
        coords  = reducer.fit_transform(X)

    elif method == "tsne":
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=n_components, random_state=random_state, **method_kwargs)
        coords  = reducer.fit_transform(X)

    elif method == "umap":
        try:
            import umap
        except ImportError:
            raise ImportError("Install umap-learn: pip install umap-learn")
        reducer = umap.UMAP(n_components=n_components, random_state=random_state, **method_kwargs)
        coords  = reducer.fit_transform(X)

    else:
        raise ValueError(f"Unknown method '{method}'. Choose from: pca, tsne, umap")

    out = emb_df.copy()
    for i in range(n_components):
        out[f"dim_{i}"] = coords[:, i]
    return out


def mean_cosine_distance_between_conditions(emb_df: pd.DataFrame) -> float:
    """
    Compute the mean cosine distance between paired EN / FR embeddings for the
    same item.  Lower values mean the two conditions are more similar in the
    model's representation space — the expected direction for multilingual models.

    Returns a scalar in [0, 2] (0 = identical, 2 = opposite).
    """
    en = emb_df[emb_df["condition"] == "EN_Word_order"].set_index("item_no")["embedding"]
    fr = emb_df[emb_df["condition"] == "FR_Word_order"].set_index("item_no")["embedding"]
    shared = en.index.intersection(fr.index)

    dists = []
    for item in shared:
        a = en[item].astype(float)
        b = fr[item].astype(float)
        cos_sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10)
        dists.append(1.0 - cos_sim)

    return float(np.mean(dists)) if dists else float("nan")


def condition_separability(emb_df: pd.DataFrame) -> float:
    """
    Compute d' (d-prime): the ratio of between-class centroid distance to
    pooled within-class spread, computed in the full high-dimensional space.

        d' = ‖μ_EN − μ_FR‖₂ / √(σ²_EN + σ²_FR)

    where σ²_EN and σ²_FR are the mean per-dimension variances of each cloud.

    Interpretation
    --------------
    Lower d' → the two condition clouds overlap more → the model treats the
    two word orders as more geometrically similar.  This is the direction
    predicted for multilingual models and maps directly onto the linear
    separability of EN vs FR conditions in embedding space.

    Unlike cosine distance, d' is insensitive to item identity: it asks
    whether the two *clouds* overlap, not whether each item's EN/FR pair is
    close.  It is therefore more directly comparable to a linear classifier
    separating the two conditions.

    Returns a non-negative scalar (lower = more mixed / less separable).
    """
    X_en = np.stack(emb_df[emb_df["condition"] == "EN_Word_order"]["embedding"].values).astype(float)
    X_fr = np.stack(emb_df[emb_df["condition"] == "FR_Word_order"]["embedding"].values).astype(float)

    if len(X_en) == 0 or len(X_fr) == 0:
        return float("nan")

    mu_en, mu_fr = X_en.mean(axis=0), X_fr.mean(axis=0)
    between = np.linalg.norm(mu_en - mu_fr)
    within  = np.sqrt(X_en.var(axis=0).mean() + X_fr.var(axis=0).mean() + 1e-10)
    return float(between / within)


def plot_pair(
    mono_emb: pd.DataFrame,
    multi_emb: pd.DataFrame,
    method: Literal["pca", "tsne", "umap"] = "pca",
    label: str = "",
    figsize: tuple[float, float] = (12, 5),
    alpha: float = 0.65,
    s: float = 45,
    random_state: int = 42,
    **reducer_kwargs,
) -> plt.Figure:
    """
    Plot dimensionality-reduced embeddings for a monolingual / multilingual pair
    side by side.  Each subplot shows EN_Word_order vs FR_Word_order points
    coloured by condition; the inter-condition cosine distance is annotated.

    Parameters
    ----------
    mono_emb  : Output of extract_embeddings() for the monolingual model.
    multi_emb : Output of extract_embeddings() for the multilingual model.
    method    : Dimensionality reduction method ("pca", "tsne", "umap").
    label     : Figure title suffix, e.g. "RoBERTa vs XLM-R".
    figsize   : (width, height) in inches.
    alpha     : Point transparency.
    s         : Marker size.
    random_state : Seed for stochastic methods.
    **reducer_kwargs : Forwarded to reduce_dimensions().

    Returns
    -------
    matplotlib Figure (call fig.savefig(...) to save).
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    method_label = method.upper()

    for ax, emb_df in zip(axes, [mono_emb, multi_emb]):
        model_name  = emb_df["model"].iloc[0]
        is_multi    = emb_df["is_multilingual"].iloc[0]
        model_label = f"{'[multi] ' if is_multi else '[mono]  '}{model_name}"

        reduced = reduce_dimensions(emb_df, method=method,
                                    random_state=random_state, **reducer_kwargs)
        dist    = mean_cosine_distance_between_conditions(emb_df)

        for cond, grp in reduced.groupby("condition"):
            ax.scatter(
                grp["dim_0"], grp["dim_1"],
                color=CONDITION_COLORS[cond],
                label=CONDITION_LABELS[cond],
                alpha=alpha, s=s, edgecolors="none",
            )

        # Connect paired items with thin grey lines
        en_r = reduced[reduced["condition"] == "EN_Word_order"].set_index("item_no")
        fr_r = reduced[reduced["condition"] == "FR_Word_order"].set_index("item_no")
        shared = en_r.index.intersection(fr_r.index)
        for item in shared:
            ax.plot(
                [en_r.loc[item, "dim_0"], fr_r.loc[item, "dim_0"]],
                [en_r.loc[item, "dim_1"], fr_r.loc[item, "dim_1"]],
                color="grey", alpha=0.15, linewidth=0.6, zorder=0,
            )

        ax.set_title(model_label, fontsize=11, fontweight="bold")
        ax.set_xlabel(f"{method_label} dim 1", fontsize=9)
        ax.set_ylabel(f"{method_label} dim 2", fontsize=9)
        ax.annotate(
            f"mean cosine dist = {dist:.4f}",
            xy=(0.03, 0.96), xycoords="axes fraction",
            fontsize=8, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7),
        )
        ax.tick_params(labelsize=8)

    # Shared legend
    legend_elements = [
        mpatches.Patch(color=CONDITION_COLORS["EN_Word_order"], label=CONDITION_LABELS["EN_Word_order"]),
        mpatches.Patch(color=CONDITION_COLORS["FR_Word_order"], label=CONDITION_LABELS["FR_Word_order"]),
        Line2D([0], [0], color="grey", alpha=0.5, linewidth=0.8, label="paired items"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               fontsize=9, framealpha=0.8, bbox_to_anchor=(0.5, -0.05))

    title = f"Embedding Space ({method_label}) — {label}" if label else f"Embedding Space ({method_label})"
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig


def plot_all_pairs(
    embs: dict[str, dict[str, pd.DataFrame]],
    model_pairs: list[tuple[str, str]],
    regions: list[str],
    method: Literal["pca", "tsne", "umap"] = "pca",
    save_dir: str | None = None,
    **reducer_kwargs,
) -> list[plt.Figure]:
    """
    Plot one figure per (monolingual, multilingual) pair, with both regions
    shown side by side (4 panels: mono-R2 | multi-R2 | mono-R3 | multi-R3).

    Parameters
    ----------
    embs        : Nested dict {region: {model_key: emb_df}}, as returned by
                  extract_all() in the notebook.
    model_pairs : List of (mono_key, multi_key) tuples.
    regions     : Region names to display, e.g. ["region2", "region3"].
    method      : Dimensionality reduction method.
    save_dir    : If provided, figures are saved as
                  {save_dir}/{mono}_vs_{multi}_{method}.png.
    **reducer_kwargs : Forwarded to reduce_dimensions().

    Returns
    -------
    List of matplotlib Figures (one per pair).
    """
    import os
    figs = []
    for mono_key, multi_key in model_pairs:
        fig = _plot_pair_with_regions(
            mono_key=mono_key,
            multi_key=multi_key,
            embs=embs,
            regions=regions,
            method=method,
            **reducer_kwargs,
        )
        figs.append(fig)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, f"{mono_key}_vs_{multi_key}_{method}.png")
            fig.savefig(path, dpi=150, bbox_inches="tight")
            print(f"  Saved → {path}")
    return figs


# ---------------------------------------------------------------------------
# Grouped plots — masked LMs and causal LMs
# ---------------------------------------------------------------------------

# Default model pairs matching the experiment design
MASKED_PAIRS: list[tuple[str, str]] = [
    ("roberta",    "xlmr"),
    ("bert",       "mbert"),
    ("distilbert", "distilmbert"),
    ("roberta",    "xlmr-large"),
]

CAUSAL_PAIRS: list[tuple[str, str]] = [
    ("gpt2",        "mgpt"),
    ("gpt2",        "croissantllm"),
    ("opt-125m",    "bloom-560m"),
    ("pythia-160m", "bloom-560m"),
]

REGION_LABELS = {
    "region2": "Region 2 (critical)",
    "region3": "Region 3 (spillover)",
}


def _draw_scatter(ax, emb_df, method, method_label, **reducer_kwargs):
    """Reduce dimensions and draw scatter + paired connector lines on ax. Returns d'."""
    reduced = reduce_dimensions(emb_df, method=method, **reducer_kwargs)
    d_prime = condition_separability(emb_df)

    for cond, grp in reduced.groupby("condition"):
        ax.scatter(grp["dim_0"], grp["dim_1"],
                   color=CONDITION_COLORS[cond], label=CONDITION_LABELS[cond],
                   alpha=0.65, s=40, edgecolors="none")

    en_r = reduced[reduced["condition"] == "EN_Word_order"].set_index("item_no")
    fr_r = reduced[reduced["condition"] == "FR_Word_order"].set_index("item_no")
    for item in en_r.index.intersection(fr_r.index):
        ax.plot([en_r.loc[item, "dim_0"], fr_r.loc[item, "dim_0"]],
                [en_r.loc[item, "dim_1"], fr_r.loc[item, "dim_1"]],
                color="grey", alpha=0.15, linewidth=0.6, zorder=0)

    ax.set_xlabel(f"{method_label} dim 1", fontsize=8)
    ax.set_ylabel(f"{method_label} dim 2", fontsize=8)
    ax.annotate(f"d′ = {d_prime:.4f}",
                xy=(0.03, 0.96), xycoords="axes fraction", fontsize=8, va="top",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.7))
    ax.tick_params(labelsize=7)
    return d_prime


def _plot_pair_with_regions(
    mono_key: str,
    multi_key: str,
    embs: dict[str, dict[str, pd.DataFrame]],
    regions: list[str],
    method: Literal["pca", "tsne", "umap"],
    **reducer_kwargs,
) -> plt.Figure:
    """
    Single model pair → one figure with 2 × len(regions) panels.
    Columns: one pair of [mono | multi] per region, grouped left-to-right.
    """
    n_regions   = len(regions)
    method_label = method.upper()
    fig, axes   = plt.subplots(1, n_regions * 2, figsize=(6.5 * n_regions * 2 / 2, 5))
    if n_regions * 2 == 1:
        axes = [axes]

    col = 0
    for region in regions:
        for key, tag in [(mono_key, "[mono]"), (multi_key, "[multi]")]:
            ax      = axes[col]
            emb_df  = embs[region][key]
            is_multi = key in MULTILINGUAL_KEYS
            ax.set_title(f"{tag} {key}\n{REGION_LABELS.get(region, region)}",
                         fontsize=9, fontweight="bold")
            _draw_scatter(ax, emb_df, method, method_label, **reducer_kwargs)
            col += 1

    legend_elements = [
        mpatches.Patch(color=CONDITION_COLORS["EN_Word_order"], label=CONDITION_LABELS["EN_Word_order"]),
        mpatches.Patch(color=CONDITION_COLORS["FR_Word_order"], label=CONDITION_LABELS["FR_Word_order"]),
        Line2D([0], [0], color="grey", alpha=0.5, linewidth=0.8, label="paired items"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               fontsize=9, framealpha=0.8, bbox_to_anchor=(0.5, -0.08))
    fig.suptitle(f"{mono_key} vs {multi_key} — {method_label}",
                 fontsize=12, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def plot_group(
    embs: dict[str, dict[str, pd.DataFrame]],
    model_pairs: list[tuple[str, str]],
    regions: list[str],
    group_title: str,
    method: Literal["pca", "tsne", "umap"] = "pca",
    save_dir: str | None = None,
    save_name: str | None = None,
    **reducer_kwargs,
) -> plt.Figure:
    """
    All pairs in a single figure: rows = model pairs,
    columns = [mono-R2 | multi-R2 | mono-R3 | multi-R3] (or however many regions).

    Parameters
    ----------
    embs        : Nested dict {region: {model_key: emb_df}} from extract_all().
    model_pairs : List of (mono_key, multi_key) tuples.
    regions     : Ordered list of region names to show, e.g. ["region2", "region3"].
    group_title : Figure suptitle prefix, e.g. "Masked LMs".
    method      : Dimensionality reduction method.
    save_dir    : Directory for saving; skipped if None.
    save_name   : Filename; defaults to "{group_title}_{method}.png".
    **reducer_kwargs : Forwarded to reduce_dimensions().

    Returns
    -------
    matplotlib Figure.
    """
    import os

    n_pairs   = len(model_pairs)
    n_cols    = len(regions) * 2          # [mono | multi] per region
    method_label = method.upper()

    fig, axes = plt.subplots(n_pairs, n_cols,
                             figsize=(5.5 * n_cols, 4.5 * n_pairs),
                             squeeze=False)

    for row_idx, (mono_key, multi_key) in enumerate(model_pairs):
        col = 0
        for region in regions:
            for key, tag in [(mono_key, "[mono]"), (multi_key, "[multi]")]:
                ax     = axes[row_idx, col]
                emb_df = embs[region][key]
                title  = f"{tag} {key}"
                if row_idx == 0:
                    title = f"{REGION_LABELS.get(region, region)}\n{title}"
                ax.set_title(title, fontsize=9, fontweight="bold")
                _draw_scatter(ax, emb_df, method, method_label, **reducer_kwargs)
                col += 1

    legend_elements = [
        mpatches.Patch(color=CONDITION_COLORS["EN_Word_order"], label=CONDITION_LABELS["EN_Word_order"]),
        mpatches.Patch(color=CONDITION_COLORS["FR_Word_order"], label=CONDITION_LABELS["FR_Word_order"]),
        Line2D([0], [0], color="grey", alpha=0.5, linewidth=0.8, label="paired items"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               fontsize=9, framealpha=0.8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"{group_title} — Embedding Space ({method_label})",
                 fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fname = save_name or f"{group_title.replace(' ', '_')}_{method}.png"
        path  = os.path.join(save_dir, fname)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved → {path}")

    return fig


def separability_summary(
    embs: dict[str, dict[str, pd.DataFrame]],
    model_pairs: list[tuple[str, str]],
    regions: list[str] | None = None,
) -> pd.DataFrame:
    """
    Tabular summary of condition separability (d') from pre-extracted embeddings.

        d' = ‖μ_EN − μ_FR‖₂ / √(σ²_EN + σ²_FR)

    Lower d' = the two condition clouds overlap more = the model treats the two
    word orders as more similar.  This is the predicted direction for multilingual
    models and maps directly onto the t-test hypothesis.

    Parameters
    ----------
    embs        : Nested dict {region: {model_key: emb_df}} from extract_all().
    model_pairs : List of (mono_key, multi_key) tuples.
    regions     : Which regions to include. Defaults to all keys in embs.

    Returns
    -------
    DataFrame with columns: pair, region, model, is_multilingual, d_prime, direction.
    direction: '(ref)' for monolingual baseline, '✓' if d' < monolingual (predicted),
               '✗' if reversed.
    """
    if regions is None:
        regions = list(embs.keys())

    rows = []
    for mono_key, multi_key in model_pairs:
        pair_label = f"{mono_key} → {multi_key}"
        for region in regions:
            for key in (mono_key, multi_key):
                d_prime = condition_separability(embs[region][key])
                rows.append({
                    "pair":            pair_label,
                    "region":          region,
                    "model":           key,
                    "is_multilingual": key in MULTILINGUAL_KEYS,
                    "d_prime":         round(d_prime, 4),
                })

    df = pd.DataFrame(rows)
    mono_ref = (
        df[~df["is_multilingual"]]
        .set_index(["pair", "region"])["d_prime"]
    )
    df["direction"] = df.apply(
        lambda r: "(ref)" if not r["is_multilingual"] else
                  "✓" if r["d_prime"] < mono_ref.get((r["pair"], r["region"]), float("inf"))
                  else "✗",
        axis=1,
    )
    return df
