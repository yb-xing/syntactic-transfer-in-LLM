"""
correlation.py
Surprisal analysis for model-pair comparisons (H1: paired t-test).

Core metric per model pair:

    S_mono(FR−EN)  = surprisal_mono(FR)  − surprisal_mono(EN)   per item
    S_multi(FR−EN) = surprisal_multi(FR) − surprisal_multi(EN)  per item

H1 test: paired t-test — mean(S_mono(FR−EN)) > mean(S_multi(FR−EN))
A significantly positive mean difference means the monolingual model is
more penalised by French word order than the multilingual model,
consistent with cross-lingual syntactic transfer.
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# All model pairs: (monolingual_key, multilingual_key, lm_type)
MODEL_PAIRS = [
    # Masked LM pairs
    ("roberta",    "xlmr",        "masked"),
    ("bert",       "mbert",       "masked"),
    ("distilbert", "distilmbert", "masked"),
    ("roberta",    "xlmr-large",  "masked"),
    # Causal LM pairs
    ("gpt2",        "mgpt",         "causal"),
    ("gpt2",        "croissantllm", "causal"),
    ("opt-125m",    "bloom-560m",   "causal"),
    ("pythia-160m", "bloom-560m",   "causal"),
]

# Conditions in the data (matching data_preprocessing.py output)
COND_FR = "FR_Word_order"
COND_EN = "EN_Word_order"

# Surprisal column names (from surprisal_causal.py / surprisal_masked.py output).
# Two variants for masked LMs; causal models always use the standard names.
#   "PLL"          → standard Salazar et al. (2020) PLL  [default]
#   "PLL_word_l2r" → Kauf & Ivanova (2023) adjusted metric
SURP_REGION_COLS = {
    "PLL": {
        "region2": "surprisal_region2",
        "region3": "surprisal_region3",
    },
    "PLL_word_l2r": {
        "region2": "surprisal_region2_PLL_word_l2r",
        "region3": "surprisal_region3_PLL_word_l2r",
    },
}

# Regions to analyse
ANALYSIS_REGIONS = ["region2", "region3"]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_surprisal(paths) -> pd.DataFrame:
    """Load surprisal CSV(s) into a single DataFrame.

    Accepts either a single path (str) or a list of paths — useful when each
    model was run separately and saved to individual files.

    Expected columns (from surprisal_causal.py / surprisal_masked.py):
        item_no, condition, advTYPE, model,
        surprisal_region2, surprisal_region3, surprisal_critical_token

    Args:
        paths: str or list[str] — path(s) to surprisal CSV file(s).

    Returns:
        Combined DataFrame with all models and conditions.
    """
    if isinstance(paths, str):
        paths = [paths]
    frames = [pd.read_csv(p) for p in paths]
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Item-no pairing helper
# ---------------------------------------------------------------------------

def _add_base_item_no(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'base_item_no' column that strips the condition prefix from item_no.

    SVadvP_01 (FR) and SadvPV_01 (EN) represent the same semantic item.
    This extracts the numeric suffix so both map to base_item_no = '01',
    enabling correct pairing when pivoting by condition.

    For integer-only item IDs (generated stimuli), base_item_no = str(item_no).
    """
    extracted = df["item_no"].astype(str).str.extract(r"_(\d+)$")[0]
    df = df.copy()
    df["base_item_no"] = extracted.where(extracted.notna(), df["item_no"].astype(str))
    return df


# ---------------------------------------------------------------------------
# Pivot helper
# ---------------------------------------------------------------------------

def _pcol(pivot: pd.DataFrame, key: tuple) -> pd.Series:
    """Safely fetch a MultiIndex column from a pivot, returning NaN Series if absent."""
    if key in pivot.columns:
        return pivot[key]
    return pd.Series(np.nan, index=pivot.index, dtype=float)


# ---------------------------------------------------------------------------
# Model-vs-model paired t-test
# ---------------------------------------------------------------------------

def compute_model_delta(
    surprisal_df: pd.DataFrame,
    mono_model:   str,
    multi_model:  str,
    region:       str = "region2",
    pll_variant:  str = "PLL",
) -> pd.DataFrame:
    """Per-item S_mono(FR−EN) and S_multi(FR−EN) for a paired t-test.

    Each row is one item. The two columns capture how much each model is
    penalised by French word order relative to the English baseline.

    Expected direction: S_mono(FR−EN) > S_multi(FR−EN) on average —
    the monolingual model is more penalised by French word order than the
    multilingual model, reflecting transfer-driven surprisal reduction.

    Args:
        surprisal_df: Combined surprisal DataFrame from load_surprisal.
        mono_model:   Key for the monolingual model.
        multi_model:  Key for the multilingual model.
        region:       'region2' or 'region3'.
        pll_variant:  'PLL' or 'PLL_word_l2r' (causal models fall back to 'PLL').

    Returns:
        DataFrame with columns: base_item_no, mono_delta, multi_delta.
        Rows with NaN in either column are dropped.
    """
    if pll_variant not in SURP_REGION_COLS:
        raise ValueError(f"Unknown pll_variant '{pll_variant}'. "
                         f"Choose from: {list(SURP_REGION_COLS)}")
    surp_col = SURP_REGION_COLS[pll_variant][region]

    df = surprisal_df[
        surprisal_df["model"].isin([mono_model, multi_model])
    ].copy()

    if surp_col not in df.columns or df[surp_col].isna().all():
        fallback = SURP_REGION_COLS["PLL"][region]
        if fallback in df.columns:
            surp_col = fallback

    df = _add_base_item_no(df)
    pivot = df.pivot_table(
        index="base_item_no",
        columns=["model", "condition"],
        values=surp_col,
    )

    mono_delta  = (
        _pcol(pivot, (mono_model,  COND_FR)) - _pcol(pivot, (mono_model,  COND_EN))
    ).rename("mono_delta")
    multi_delta = (
        _pcol(pivot, (multi_model, COND_FR)) - _pcol(pivot, (multi_model, COND_EN))
    ).rename("multi_delta")

    combined = pd.concat([mono_delta, multi_delta], axis=1).dropna()
    return combined.reset_index()


def run_model_ttest(model_delta_df: pd.DataFrame) -> dict:
    """Paired t-test: is S_mono(FR−EN) significantly greater than S_multi(FR−EN)?

    Tests H1: mean(mono_delta) > mean(multi_delta), i.e. the monolingual model
    is more penalised by French word order than the multilingual model.

    Args:
        model_delta_df: Output of compute_model_delta.

    Returns:
        Dict with keys: t_stat, p_two_tailed, p_one_tailed, mean_mono,
        mean_multi, mean_diff, n_items.
    """
    mono  = model_delta_df["mono_delta"].values
    multi = model_delta_df["multi_delta"].values

    t_stat, p_two = stats.ttest_rel(mono, multi)
    p_one = p_two / 2 if t_stat > 0 else 1.0 - p_two / 2

    return {
        "t_stat":        float(t_stat),
        "p_two_tailed":  float(p_two),
        "p_one_tailed":  float(p_one),
        "mean_mono":     float(np.mean(mono)),
        "mean_multi":    float(np.mean(multi)),
        "mean_diff":     float(np.mean(mono - multi)),
        "n_items":       int(len(mono)),
    }


# ---------------------------------------------------------------------------
# Violin plot
# ---------------------------------------------------------------------------

def plot_model_delta(
    model_delta_df: pd.DataFrame,
    ttest_result:   dict,
    title:          str,
    output_path:    str,
) -> None:
    """Paired violin plot: S_mono(FR−EN) vs S_multi(FR−EN) across items.

    Args:
        model_delta_df: Output of compute_model_delta.
        ttest_result:   Output of run_model_ttest.
        title:          Plot title string.
        output_path:    File path to save the figure (PNG).
    """
    mono_vals  = model_delta_df["mono_delta"].values
    multi_vals = model_delta_df["multi_delta"].values

    fig, ax = plt.subplots(figsize=(5.5, 4.5))

    parts = ax.violinplot(
        [mono_vals, multi_vals],
        positions=[0, 1],
        showmedians=True,
        showextrema=True,
    )
    colors = ["#2196F3", "#9C27B0"]
    for pc, color in zip(parts["bodies"], colors):
        pc.set_facecolor(color)
        pc.set_alpha(0.55)
    for key in ("cmedians", "cmins", "cmaxes", "cbars"):
        parts[key].set_color("black")
        parts[key].set_linewidth(1.2)

    for m, ml in zip(mono_vals, multi_vals):
        ax.plot([0, 1], [m, ml], color="gray", alpha=0.25, linewidth=0.7)
    ax.scatter([0] * len(mono_vals),  mono_vals,  color="#2196F3",
               s=20, alpha=0.6, zorder=3)
    ax.scatter([1] * len(multi_vals), multi_vals, color="#9C27B0",
               s=20, alpha=0.6, zorder=3)

    ax.axhline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.4)

    mono_label  = model_delta_df.get("mono_model",  ["mono"])[0]  if "mono_model"  in model_delta_df.columns else "mono"
    multi_label = model_delta_df.get("multi_model", ["multi"])[0] if "multi_model" in model_delta_df.columns else "multi"
    ax.set_xticks([0, 1])
    ax.set_xticklabels([mono_label, multi_label], fontsize=10)

    t, p1, p2, n = (ttest_result["t_stat"], ttest_result["p_one_tailed"],
                    ttest_result["p_two_tailed"], ttest_result["n_items"])
    annot = (f"paired t = {t:.3f}\np(one-tail) = {p1:.3f},  p(two-tail) = {p2:.3f}"
             f"\nN = {n} items")
    ax.text(0.05, 0.97, annot, transform=ax.transAxes, va="top", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

    ax.set_ylabel("Surprisal(FR) − Surprisal(EN)", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight="bold")
    fig.tight_layout()

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Plot saved → {output_path}")


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_analysis(
    surprisal_paths,
    output_dir:  str = "results",
    pll_variant: str = "PLL",
) -> list[dict]:
    """End-to-end t-test analysis for all model pairs and both regions.

    For each (model pair × region) combination:
      1. Compute per-item S_mono(FR−EN) and S_multi(FR−EN).
      2. Run a paired t-test (H1: mono > multi).
      3. Save a paired violin plot.

    Results are saved to {output_dir}/ttest_results.json.

    Args:
        surprisal_paths: str or list[str] — path(s) to surprisal CSV file(s).
        output_dir:      Directory to save plots and results JSON.
        pll_variant:     Which surprisal column to use for masked LMs.
                         'PLL'          → Salazar et al. (2020) [default]
                         'PLL_word_l2r' → Kauf & Ivanova (2023)
                         Causal models are unaffected (always use 'PLL').

    Returns:
        List of result dicts, one per (model pair × region).
    """
    os.makedirs(output_dir, exist_ok=True)
    figures_dir = os.path.join(output_dir, "figures")
    os.makedirs(figures_dir, exist_ok=True)

    print("Loading surprisal data …")
    surprisal_df = load_surprisal(surprisal_paths)

    available_models = set(surprisal_df["model"].unique())
    print(f"  Models: {sorted(available_models)}")
    print(f"  PLL variant: {pll_variant}")

    all_results = []

    for mono_model, multi_model, lm_type in MODEL_PAIRS:
        if mono_model not in available_models or multi_model not in available_models:
            print(f"  Skipping {mono_model} → {multi_model} (data not available)")
            continue

        for region in ANALYSIS_REGIONS:
            print(f"\n--- {mono_model} → {multi_model}  |  {region} ---")

            model_delta = compute_model_delta(
                surprisal_df, mono_model, multi_model, region,
                pll_variant=pll_variant,
            )

            if len(model_delta) < 5:
                print("  Too few items — skipping.")
                continue

            tt = run_model_ttest(model_delta)
            print(f"  t = {tt['t_stat']:.3f}, "
                  f"p(two) = {tt['p_two_tailed']:.3f}, "
                  f"p(one) = {tt['p_one_tailed']:.3f}, "
                  f"mean diff = {tt['mean_diff']:.3f}, "
                  f"N = {tt['n_items']}")

            region_label = "Region 2 (V+Adv)" if region == "region2" else "Region 3 (Spillover)"
            variant_tag  = "" if pll_variant == "PLL" else "  [PLL-word-l2r]"
            title = (f"{mono_model} vs {multi_model}  |  {region_label}{variant_tag}\n"
                     f"S(FR−EN) per model  (expected: mono > multi)")
            plot_path = os.path.join(
                figures_dir,
                f"model_delta_{mono_model}_vs_{multi_model}_{region}.png",
            )
            plot_model_delta(model_delta, tt, title, plot_path)

            all_results.append({
                "mono_model":  mono_model,
                "multi_model": multi_model,
                "lm_type":     lm_type,
                "region":      region,
                "pll_variant": pll_variant,
                "ttest":       tt,
            })

    # Save summary JSON
    json_path = os.path.join(output_dir, "ttest_results.json")
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved → {json_path}")

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Paired t-test analysis of monolingual vs. multilingual surprisal."
    )
    parser.add_argument(
        "--surprisal", required=True, nargs="+",
        help="Path(s) to surprisal CSV file(s) — one per model, or a single combined file.",
    )
    parser.add_argument(
        "--output_dir", default="results",
        help="Directory to write plots and ttest_results.json.",
    )
    parser.add_argument(
        "--pll_variant", default="PLL", choices=list(SURP_REGION_COLS),
        help="Which surprisal column to use for masked LMs: "
             "'PLL' (Salazar et al. 2020, default) or "
             "'PLL_word_l2r' (Kauf & Ivanova 2023). "
             "Causal models are unaffected.",
    )
    args = parser.parse_args()

    run_analysis(
        surprisal_paths=args.surprisal,
        output_dir=args.output_dir,
        pll_variant=args.pll_variant,
    )
