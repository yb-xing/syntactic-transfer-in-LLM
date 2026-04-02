"""
correlation.py
Surprisal–RT interaction computation and correlation analysis.

Core logic (2×2 interaction on both sides):

    LLM interaction per item:
        ΔS_interaction(item) = [S_mono(FR) − S_mono(EN)] − [S_multi(FR) − S_multi(EN)]

    Human RT interaction per item:
        ΔRT_interaction(item) = [RT_mono(FR) − RT_mono(EN)] − [RT_ebilingual(FR) − RT_ebilingual(EN)]

    Main correlation:
        ΔS_interaction  ~  ΔRT_interaction     across N items

Rationale for the interaction formulation:
  - Subtracting the EN baseline from each model normalises out absolute surprisal
    scale differences between models (different vocab sizes, training corpora).
  - Subtracting the EN baseline from each human group removes item-level difficulty
    unrelated to word order.
  - Both sides then measure the *differential sensitivity to French word order*
    between the monolingual and multilingual entity — making the analogy structurally
    parallel and interpretable.

Predictions:
  - Primary:     ΔS_interaction × ΔRT_interaction at Region 2 — positive correlation
  - Spillover:   Same at Region 3
  - Specificity: Effect present for FR_Word_order; absent (or weaker) for unrelated
                 violations — tests that the result is syntax-specific
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
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

# RT column names (from behavioral_rt_cleaned.csv) mapped to readable labels
RT_REGION_COLS = {
    "region2": "R2_rt",
    "region3": "R3_rt",
}

# Surprisal column names (from surprisal_causal.py / surprisal_masked.py output)
SURP_REGION_COLS = {
    "region2": "surprisal_region2",
    "region3": "surprisal_region3",
}

# Regions to run the primary correlation over
CORRELATION_REGIONS = ["region2", "region3"]


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
    df = pd.concat(frames, ignore_index=True)
    return df


def load_behavioral_rt(rt_path: str) -> pd.DataFrame:
    """Load the cleaned trial-level RT data from data_preprocessing.py.

    Expected columns (behavioral_rt_cleaned.csv):
        participant, group, condition, item_no, advTYPE,
        R1_rt, R2_rt, R3_rt, R4_rt
        (RT cells are NaN for outlier trials; NaN is excluded from means automatically)

    Note: The RT values here are raw (not residualized on word length / frequency /
    serial position). For the main analysis, residualize before passing to this
    pipeline — e.g. using a linear mixed model via statsmodels or pingouin and
    saving residuals back as R2_rt, R3_rt. For exploratory use, raw means are fine.

    Args:
        rt_path: Path to behavioral_rt_cleaned.csv.

    Returns:
        DataFrame with trial-level RT data.
    """
    return pd.read_csv(rt_path)


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
# RT interaction computation
# ---------------------------------------------------------------------------

def compute_rt_interaction(rt_df: pd.DataFrame, region: str = "region2") -> pd.DataFrame:
    """Compute per-item RT interaction: [RT_mono(FR)−RT_mono(EN)] − [RT_ebilingual(FR)−RT_ebilingual(EN)].

    Steps:
      1. Average RT across participants within each (group × condition × item_no) cell.
         NaN outlier cells are excluded from the mean automatically.
      2. For each group, compute the condition difference: FR − EN per item.
      3. Subtract the bilingual condition difference from the monolingual one.

    A positive ΔRT_interaction means monolinguals find FR word order relatively
    harder than bilinguals do — the predicted human facilitation signal.

    Args:
        rt_df:  Trial-level DataFrame from load_behavioral_rt.
        region: One of 'region2' (primary) or 'region3' (spillover).

    Returns:
        DataFrame with columns: item_no, rt_interaction.
        Items with missing data in any cell are NaN.
    """
    rt_col = RT_REGION_COLS[region]

    # Add base_item_no for cross-condition pairing (SVadvP_01 + SadvPV_01 → "01")
    rt_df = _add_base_item_no(rt_df)

    # Per-item mean RT, grouped by group × condition × base_item_no
    means = (
        rt_df.groupby(["group", "condition", "base_item_no"])[rt_col]
        .mean()
        .reset_index()
        .rename(columns={rt_col: "mean_rt"})
    )

    # Pivot to wide: index=base_item_no, columns=(group, condition)
    pivot = means.pivot_table(
        index="base_item_no",
        columns=["group", "condition"],
        values="mean_rt",
    )

    # Condition difference per group (FR − EN): how much harder is FR for each group?
    mono_diff = (
        pivot.get(("Monolinguals", COND_FR), np.nan)
        - pivot.get(("Monolinguals", COND_EN), np.nan)
    )
    ebil_diff = (
        pivot.get(("Ebilinguals", COND_FR), np.nan)
        - pivot.get(("Ebilinguals", COND_EN), np.nan)
    )

    # Interaction: how much MORE does FR cost monolinguals vs. bilinguals?
    interaction = (mono_diff - ebil_diff).rename("rt_interaction")

    return interaction.reset_index()


# ---------------------------------------------------------------------------
# Surprisal interaction computation
# ---------------------------------------------------------------------------

def compute_surprisal_interaction(
    surprisal_df: pd.DataFrame,
    mono_model:   str,
    multi_model:  str,
    region:       str = "region2",
) -> pd.DataFrame:
    """Compute per-item surprisal interaction: [S_mono(FR)−S_mono(EN)] − [S_multi(FR)−S_multi(EN)].

    Steps:
      1. Filter to the two models of interest.
      2. For each model, compute the condition difference: S(FR) − S(EN) per item.
         This normalises out absolute surprisal scale differences between models.
      3. Subtract the multilingual model's condition difference from the monolingual's.

    A positive ΔS_interaction means the monolingual model is more penalised by FR
    word order (relative to its EN baseline) than the multilingual model is —
    the predicted transfer signal.

    Args:
        surprisal_df: Combined surprisal DataFrame from load_surprisal.
        mono_model:   Key for the monolingual model (e.g. 'roberta', 'gpt2').
        multi_model:  Key for the multilingual model (e.g. 'xlmr', 'mgpt').
        region:       One of 'region2' or 'region3'.

    Returns:
        DataFrame with columns: item_no, surprisal_interaction.
        Items with missing surprisal in any cell are NaN.
    """
    surp_col = SURP_REGION_COLS[region]

    df = surprisal_df[
        surprisal_df["model"].isin([mono_model, multi_model])
    ].copy()

    # Add base_item_no for cross-condition pairing (SVadvP_01 + SadvPV_01 → "01")
    df = _add_base_item_no(df)

    # Pivot: index=base_item_no, columns=(model, condition)
    pivot = df.pivot_table(
        index="base_item_no",
        columns=["model", "condition"],
        values=surp_col,
    )

    # Condition difference per model (FR − EN)
    mono_diff = (
        pivot.get((mono_model,  COND_FR), np.nan)
        - pivot.get((mono_model,  COND_EN), np.nan)
    )
    multi_diff = (
        pivot.get((multi_model, COND_FR), np.nan)
        - pivot.get((multi_model, COND_EN), np.nan)
    )

    # Interaction: how much MORE does FR cost the mono model vs. the multi model?
    interaction = (mono_diff - multi_diff).rename("surprisal_interaction")

    return interaction.reset_index()


# ---------------------------------------------------------------------------
# Merge interactions
# ---------------------------------------------------------------------------

def merge_interactions(
    surp_interaction: pd.DataFrame,
    rt_interaction:   pd.DataFrame,
) -> pd.DataFrame:
    """Join surprisal and RT interaction DataFrames on item_no.

    Args:
        surp_interaction: Output of compute_surprisal_interaction.
        rt_interaction:   Output of compute_rt_interaction.

    Returns:
        DataFrame with columns: item_no, surprisal_interaction, rt_interaction.
        Only items present in both DataFrames are kept; rows with NaN in either
        column are dropped before returning.
    """
    merged = surp_interaction.merge(rt_interaction, on="base_item_no", how="inner")
    merged = merged.dropna(subset=["surprisal_interaction", "rt_interaction"])
    return merged.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Correlation
# ---------------------------------------------------------------------------

def run_correlation(merged_df: pd.DataFrame) -> dict:
    """Compute Pearson r and Spearman rho between ΔS_interaction and ΔRT_interaction.

    Pearson r is the primary statistic (predicted direction: positive).
    Spearman rho is a robustness check against non-normality and outliers.

    Args:
        merged_df: Output of merge_interactions, with columns
                   'surprisal_interaction' and 'rt_interaction'.

    Returns:
        Dict with keys:
            pearson_r, pearson_p, spearman_rho, spearman_p, n_items
    """
    x = merged_df["surprisal_interaction"].values
    y = merged_df["rt_interaction"].values

    pearson_r,  pearson_p  = stats.pearsonr(x, y)
    spearman_r, spearman_p = stats.spearmanr(x, y)

    return {
        "pearson_r":   float(pearson_r),
        "pearson_p":   float(pearson_p),
        "spearman_rho": float(spearman_r),
        "spearman_p":  float(spearman_p),
        "n_items":     int(len(x)),
    }


def permutation_test(
    merged_df:      pd.DataFrame,
    n_permutations: int = 10_000,
    random_seed:    int = 42,
) -> float:
    """Two-tailed permutation test for the Pearson correlation.

    Builds a null distribution by repeatedly shuffling the RT interaction labels
    and recomputing r, then reports the proportion of null |r| >= observed |r|.

    Args:
        merged_df:      Output of merge_interactions.
        n_permutations: Number of shuffle iterations (default 10,000).
        random_seed:    Seed for reproducibility.

    Returns:
        Two-tailed permutation p-value (float).
    """
    rng = np.random.default_rng(random_seed)
    x   = merged_df["surprisal_interaction"].values
    y   = merged_df["rt_interaction"].values

    observed_r, _ = stats.pearsonr(x, y)

    null_rs = np.array([
        stats.pearsonr(x, rng.permutation(y))[0]
        for _ in range(n_permutations)
    ])

    p_perm = float(np.mean(np.abs(null_rs) >= np.abs(observed_r)))
    return p_perm


# ---------------------------------------------------------------------------
# Scatter plot
# ---------------------------------------------------------------------------

def plot_scatter(
    merged_df:   pd.DataFrame,
    corr_result: dict,
    title:       str,
    output_path: str,
) -> None:
    """Scatter plot of ΔS_interaction vs. ΔRT_interaction with a regression line.

    Args:
        merged_df:   Output of merge_interactions.
        corr_result: Output of run_correlation (used for annotation).
        title:       Plot title string.
        output_path: File path to save the figure (PNG).
    """
    x = merged_df["surprisal_interaction"].values
    y = merged_df["rt_interaction"].values

    fig, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(x, y, color="#2196F3", alpha=0.75, edgecolors="white",
               linewidths=0.5, s=60, zorder=3)

    # Regression line
    slope, intercept, *_ = stats.linregress(x, y)
    x_line = np.linspace(x.min(), x.max(), 200)
    ax.plot(x_line, slope * x_line + intercept, color="#F44336",
            linewidth=1.8, zorder=2)

    # Horizontal and vertical reference lines at zero
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
    ax.axvline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)

    # Annotation
    r      = corr_result["pearson_r"]
    p      = corr_result["pearson_p"]
    p_perm = corr_result.get("p_perm", None)
    n      = corr_result["n_items"]

    annot = f"r = {r:.3f},  p = {p:.3f}"
    if p_perm is not None:
        annot += f"\np_perm = {p_perm:.3f}"
    annot += f"\nN = {n} items"
    ax.text(0.05, 0.95, annot, transform=ax.transAxes,
            verticalalignment="top", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.set_xlabel("ΔS interaction\n[S_mono(FR−EN) − S_multi(FR−EN)]", fontsize=10)
    ax.set_ylabel("ΔRT interaction\n[RT_mono(FR−EN) − RT_ebilingual(FR−EN)]", fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")

    plt.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved → {output_path}")


def plot_all_pairs(
    all_results: list[dict],
    region:      str,
    output_dir:  str,
) -> None:
    """Grid scatter plot — one panel per model pair, for a given region.

    Args:
        all_results: List of result dicts from run_analysis (one per model pair).
                     Each dict must contain 'merged_df', 'mono_model', 'multi_model',
                     'corr_result', 'region'.
        region:      Which region's results to plot ('region2' or 'region3').
        output_dir:  Directory to save the figure.
    """
    subset = [r for r in all_results if r["region"] == region]
    n      = len(subset)
    ncols  = 4
    nrows  = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4.5 * nrows))
    axes = np.array(axes).flatten()

    for ax, result in zip(axes, subset):
        merged = result["merged_df"]
        corr   = result["corr_result"]
        x = merged["surprisal_interaction"].values
        y = merged["rt_interaction"].values

        ax.scatter(x, y, color="#2196F3", alpha=0.7, s=45, edgecolors="white",
                   linewidths=0.4, zorder=3)
        slope, intercept, *_ = stats.linregress(x, y)
        x_line = np.linspace(x.min(), x.max(), 200)
        ax.plot(x_line, slope * x_line + intercept, color="#F44336",
                linewidth=1.5, zorder=2)
        ax.axhline(0, color="gray", linewidth=0.6, linestyle="--", alpha=0.4)
        ax.axvline(0, color="gray", linewidth=0.6, linestyle="--", alpha=0.4)

        r = corr["pearson_r"]
        p = corr["pearson_p"]
        ax.set_title(
            f"{result['mono_model']} → {result['multi_model']}\n"
            f"r = {r:.3f},  p = {p:.3f}",
            fontsize=9, fontweight="bold",
        )
        ax.set_xlabel("ΔS interaction", fontsize=8)
        ax.set_ylabel("ΔRT interaction", fontsize=8)
        ax.tick_params(labelsize=7)

    # Hide unused axes
    for ax in axes[len(subset):]:
        ax.set_visible(False)

    region_label = "Region 2 (V+Adv)" if region == "region2" else "Region 3 (Spillover)"
    fig.suptitle(
        f"ΔS × ΔRT Interaction Correlations — {region_label}",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    out_path = os.path.join(output_dir, f"correlation_grid_{region}.png")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Grid plot saved → {out_path}")


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_analysis(
    surprisal_paths,
    rt_path:     str,
    output_dir:  str = "results",
    n_perm:      int = 10_000,
) -> list[dict]:
    """End-to-end correlation analysis for all model pairs and both regions.

    For each (model pair × region) combination:
      1. Compute ΔS_interaction per item from surprisal data.
      2. Compute ΔRT_interaction per item from RT data.
      3. Merge on item_no.
      4. Run Pearson r, Spearman rho, and permutation test.
      5. Save a scatter plot.

    Results are also saved to {output_dir}/correlation_results.json.

    Args:
        surprisal_paths: str or list[str] — path(s) to surprisal CSV file(s).
                         Each file is the output of surprisal_causal.py or
                         surprisal_masked.py for one model.
        rt_path:         Path to behavioral_rt_cleaned.csv
                         (output of data_preprocessing.py).
        output_dir:      Directory to save plots and results JSON.
        n_perm:          Number of permutation iterations.

    Returns:
        List of result dicts, one per (model pair × region).
        Each dict contains:
            mono_model, multi_model, lm_type, region,
            corr_result (pearson_r, pearson_p, spearman_rho, spearman_p, n_items),
            p_perm, merged_df
    """
    os.makedirs(output_dir, exist_ok=True)
    figures_dir = os.path.join(output_dir, "figures")

    print("Loading data …")
    surprisal_df = load_surprisal(surprisal_paths)
    rt_df        = load_behavioral_rt(rt_path)

    available_models = set(surprisal_df["model"].unique())
    print(f"  Models in surprisal data: {sorted(available_models)}")
    print(f"  RT data shape: {rt_df.shape}")

    all_results = []

    for mono_model, multi_model, lm_type in MODEL_PAIRS:
        # Skip pairs where either model is missing from surprisal data
        if mono_model not in available_models or multi_model not in available_models:
            print(f"  Skipping {mono_model} → {multi_model} (data not available)")
            continue

        for region in CORRELATION_REGIONS:
            print(f"\n--- {mono_model} → {multi_model}  |  {region} ---")

            # Compute interactions
            surp_int = compute_surprisal_interaction(
                surprisal_df, mono_model, multi_model, region
            )
            rt_int = compute_rt_interaction(rt_df, region)

            # Merge
            merged = merge_interactions(surp_int, rt_int)
            print(f"  Merged items: {len(merged)}")

            if len(merged) < 5:
                print("  Too few items to correlate — skipping.")
                continue

            # Correlation
            corr = run_correlation(merged)
            p_perm = permutation_test(merged, n_permutations=n_perm)
            corr["p_perm"] = p_perm

            print(f"  Pearson r = {corr['pearson_r']:.3f}, "
                  f"p = {corr['pearson_p']:.3f}, "
                  f"p_perm = {p_perm:.3f}, "
                  f"N = {corr['n_items']}")
            print(f"  Spearman ρ = {corr['spearman_rho']:.3f}, "
                  f"p = {corr['spearman_p']:.3f}")

            # Scatter plot (individual)
            title = (f"{mono_model} → {multi_model}  |  {region.replace('region', 'Region ')}\n"
                     f"r = {corr['pearson_r']:.3f},  p = {corr['pearson_p']:.3f}")
            plot_path = os.path.join(
                figures_dir,
                f"scatter_{mono_model}_vs_{multi_model}_{region}.png",
            )
            plot_scatter(merged, corr, title, plot_path)

            all_results.append({
                "mono_model":  mono_model,
                "multi_model": multi_model,
                "lm_type":     lm_type,
                "region":      region,
                "corr_result": corr,
                "p_perm":      p_perm,
                "merged_df":   merged,
            })

    # Grid plots (one per region)
    for region in CORRELATION_REGIONS:
        if any(r["region"] == region for r in all_results):
            plot_all_pairs(all_results, region, figures_dir)

    # Save summary JSON (exclude merged_df — not JSON-serialisable)
    summary = [
        {k: v for k, v in r.items() if k != "merged_df"}
        for r in all_results
    ]
    json_path = os.path.join(output_dir, "correlation_results.json")
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved → {json_path}")

    return all_results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Compute surprisal–RT interaction correlations."
    )
    parser.add_argument(
        "--surprisal", required=True, nargs="+",
        help="Path(s) to surprisal CSV file(s) — one per model, or a single combined file.",
    )
    parser.add_argument(
        "--rt", required=True,
        help="Path to behavioral_rt_cleaned.csv (output of data_preprocessing.py).",
    )
    parser.add_argument(
        "--output_dir", default="results",
        help="Directory to write plots and correlation_results.json.",
    )
    parser.add_argument(
        "--n_perm", type=int, default=10_000,
        help="Number of permutation iterations (default: 10,000).",
    )
    args = parser.parse_args()

    run_analysis(
        surprisal_paths=args.surprisal,
        rt_path=args.rt,
        output_dir=args.output_dir,
        n_perm=args.n_perm,
    )
