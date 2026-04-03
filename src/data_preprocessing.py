"""
data_preprocessing.py
Load, clean, and visualise maze task RT data for the LLM syntactic transfer project.

Pipeline:
  1.  Load maze_116_20231004.csv and lbq_116.csv, join on Participant Private ID
  2.  Parse FR_AoI from LBQ; classify groups (Monolinguals / Ebilinguals / Lbilinguals)
  3.  Remove excluded late bilinguals (FR_AoI >= 15) then drop remaining Lbilinguals
  4.  Extract stimuli from raw maze rows:
        context sentence (from Screen Name == "context" rows)
        region texts (re1–re4 target words, pivoted to wide)
        full target sentence and critical token (the adverb)
  5.  Filter to one trial row per participant × item
        (complete, item=="trial", Screen Name=="maze", region=="re2")
  6.  Retain only accurate trials (Accuracy == 4)
  7.  Apply region-specific RT cutoffs — outliers become NaN (row is kept)
        Rows with Accuracy != 4 are dropped entirely (genuinely invalid)
        Outlier RT cells within accurate trials are NaN (excluded from means only)
  8.  Attach stimuli columns to the wide RT DataFrame
  9.  Print descriptive statistics before and after cleaning
 10.  Plot mean RT per region, faceted by group, lines by condition
 11.  Save two CSVs:
        behavioral_rt_cleaned.csv  — wide trial-level RT + stimuli (for correlation.py)
        stimuli.csv                — unique stimuli per item × condition (for LLM surprisal)

Output columns for behavioral_rt_cleaned.csv:
    participant, group, condition, item_no, advTYPE,
    context_sentence, region1_text, region2_text, region3_text, region4_text,
    full_sentence, critical_token,
    R1_rt, R2_rt, R3_rt, R4_rt           — raw RT (NaN where outlier)
    R2_rt_resid, R3_rt_resid             — residualized on region text length (ablation)
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Region-specific RT upper bounds (ms), matching the original R script
RT_UPPER = {
    "R1_Reaction": 2000,
    "R2_Reaction": 2500,
    "R3_Reaction": 2500,
    "R4_Reaction": 3000,
}
RT_LOWER_GLOBAL = 300   # global lower bound (ms), applied to all regions

# AoI thresholds (years)
AOI_EARLY_MAX  = 6      # FR_AoI <= 6  → Ebilinguals
AOI_REMOVE_MIN = 15     # FR_AoI >= 15 → excluded entirely (matches R script)

# Region columns in wide format (before rename)
REGION_RT_COLS = ["R1_Reaction", "R2_Reaction", "R3_Reaction", "R4_Reaction"]

# Rename RT columns to cleaner names in output
REGION_RT_RENAME = {
    "R1_Reaction": "R1_rt",
    "R2_Reaction": "R2_rt",
    "R3_Reaction": "R3_rt",
    "R4_Reaction": "R4_rt",
}

# Region code → stimuli column name
REGION_TEXT_COLS = {
    "re1": "region1_text",
    "re2": "region2_text",
    "re3": "region3_text",
    "re4": "region4_text",
}

# Human-readable labels for the plot x-axis
REGION_PLOT_LABELS = {
    "R1_rt": "Region 1\n(Subject NP)",
    "R2_rt": "Region 2\n(V+Adv / Adv+V)",
    "R3_rt": "Region 3\n(Object NP)",
    "R4_rt": "Region 4\n(PP)",
}

CONDITION_PALETTE = {
    "EN_Word_order": "#2196F3",   # blue  — grammatical (Adv+V)
    "FR_Word_order": "#F44336",   # red   — ungrammatical (V+Adv, French-like)
}


# ---------------------------------------------------------------------------
# Step 1: Load and join
# ---------------------------------------------------------------------------

def load_and_join(maze_path: str, lbq_path: str) -> pd.DataFrame:
    """Load maze and LBQ CSVs and join on Participant Private ID.

    Only FR_AoI is taken from the LBQ — no other LBQ predictors are needed.

    Args:
        maze_path: Path to maze_116_20231004.csv.
        lbq_path:  Path to lbq_116.csv.

    Returns:
        Merged DataFrame with standardised column names.
    """
    maze = pd.read_csv(maze_path, low_memory=False)
    lbq  = pd.read_csv(lbq_path,  low_memory=False)

    maze.columns = maze.columns.str.strip()
    lbq.columns  = lbq.columns.str.strip()

    maze = maze.rename(columns={
        "Participant Private ID": "participant",
        "Participant Status":     "completeness",
        "branch-erwt":           "population",
        "Screen Name":           "screen_name",
    })

    # Fix known capitalisation typo in population values
    maze["population"] = maze["population"].replace("BIlinguals", "Bilinguals")

    # Extract only participant ID + FR_AoI from LBQ
    lbq = (
        lbq
        .rename(columns={
            "Participant Private ID": "participant",
            "otherlanguage_a_AoI":   "FR_AoI_raw",
        })
        [["participant", "FR_AoI_raw"]]
        .drop_duplicates(subset="participant")
    )

    return maze.merge(lbq, on="participant", how="left")


# ---------------------------------------------------------------------------
# Step 2: Parse FR_AoI and classify groups
# ---------------------------------------------------------------------------

def _parse_aoi(raw) -> float:
    """Convert a Gorilla LBQ AoI string to a numeric year value.

    Special cases:
        '0 (birth)' → 0.0
        '20+'       → 20.0
        NaN         → NaN  (Monolinguals have no AoI entry)
    """
    if pd.isna(raw):
        return np.nan
    s = str(raw).strip()
    if s == "0 (birth)":
        return 0.0
    if s == "20+":
        return 20.0
    try:
        return float(s)
    except ValueError:
        return np.nan


def classify_groups(df: pd.DataFrame) -> pd.DataFrame:
    """Add numeric FR_AoI and group columns; drop excluded and late bilinguals.

    Matching the original R script logic:
      - Remove bilinguals with FR_AoI >= 15 first (n=2 in original data)
      - Assign groups: Monolinguals | Ebilinguals (AoI <= 6) | Lbilinguals (AoI > 6)
      - Drop Lbilinguals — not used in this project

    Args:
        df: Merged DataFrame from load_and_join.

    Returns:
        DataFrame filtered to Monolinguals and Ebilinguals only.
    """
    df = df.copy()
    df["FR_AoI"] = df["FR_AoI_raw"].apply(_parse_aoi)

    # Remove extreme late bilinguals (FR_AoI >= 15)
    df = df[~((df["population"] == "Bilinguals") & (df["FR_AoI"] >= AOI_REMOVE_MIN))]

    def _assign_group(row):
        if row["population"] == "Monolinguals":
            return "Monolinguals"
        aoi = row["FR_AoI"]
        if pd.isna(aoi):
            return "Lbilinguals"
        return "Ebilinguals" if aoi <= AOI_EARLY_MAX else "Lbilinguals"

    df["group"] = df.apply(_assign_group, axis=1)
    df = df[df["group"].isin(["Monolinguals", "Ebilinguals"])].copy()

    print(f"  Groups retained (all rows): "
          f"Monolinguals={(df['group']=='Monolinguals').sum()}, "
          f"Ebilinguals={(df['group']=='Ebilinguals').sum()}")
    return df


# ---------------------------------------------------------------------------
# Step 3: Extract stimuli
# ---------------------------------------------------------------------------

def extract_stimuli(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Extract unique stimuli from raw maze data.

    Sources within the maze CSV:
      - Context sentences: rows where screen_name == "context" and item == "trial",
        text stored in the 'context' column.
      - Region texts: rows where screen_name == "maze" and item == "trial",
        the 'target' column holds the correct word/phrase for each region (re1–re4).

    Critical token (adverb) derivation:
      - EN condition (Adv+V order): adverb is the FIRST word of region 2
          e.g. "often watches" → critical_token = "often"
      - FR condition (V+Adv order): adverb is the LAST word of region 2
          e.g. "watches often" → critical_token = "often"

    Region 2 text for LLM input:
      - EN: "often watches"   (Adv+V — grammatical English baseline)
      - FR: "watches often"   (V+Adv — ungrammatical English, French-like)

    Args:
        df_raw: Full merged DataFrame from load_and_join (before any row filtering).

    Returns:
        DataFrame with one row per item_no × condition, columns:
            item_no, condition, advTYPE,
            context_sentence,
            region1_text, region2_text, region3_text, region4_text,
            full_sentence, critical_token
    """
    # --- Region texts ---
    maze_rows = df_raw[
        (df_raw["item"] == "trial") &
        (df_raw["screen_name"] == "maze")
    ].copy()

    # One unique text per item_no × condition × region (identical across participants)
    region_unique = (
        maze_rows[["item_no", "condition", "advTYPE", "region", "target"]]
        .drop_duplicates(subset=["item_no", "condition", "region"])
    )

    region_wide = (
        region_unique
        .pivot(index=["item_no", "condition", "advTYPE"], columns="region", values="target")
        .rename(columns=REGION_TEXT_COLS)
        .reset_index()
    )
    # Ensure all four region columns exist even if a region had no data
    for col in REGION_TEXT_COLS.values():
        if col not in region_wide.columns:
            region_wide[col] = np.nan

    # --- Context sentences ---
    context_rows = df_raw[
        (df_raw["item"] == "trial") &
        (df_raw["screen_name"] == "context")
    ][["item_no", "context"]].drop_duplicates(subset="item_no")
    context_rows = context_rows.rename(columns={"context": "context_sentence"})

    # --- Join ---
    stimuli = region_wide.merge(context_rows, on="item_no", how="left")

    # --- Full target sentence ---
    stimuli["full_sentence"] = (
        stimuli["region1_text"].fillna("").str.strip() + " " +
        stimuli["region2_text"].fillna("").str.strip() + " " +
        stimuli["region3_text"].fillna("").str.strip() + " " +
        stimuli["region4_text"].fillna("").str.strip()
    ).str.strip()

    # --- Critical token (adverb) ---
    def _get_critical_token(row) -> str:
        r2 = str(row.get("region2_text", "")).strip()
        words = r2.split()
        if not words:
            return ""
        # EN (Adv+V): adverb is first word; FR (V+Adv): adverb is last word
        return words[0] if row["condition"] == "EN" else words[-1]

    stimuli["critical_token"] = stimuli.apply(_get_critical_token, axis=1)

    # Rename condition values to project convention
    stimuli["condition"] = stimuli["condition"].map(
        {"EN": "EN_Word_order", "FR": "FR_Word_order"}
    )

    stimuli = stimuli.sort_values(["item_no", "condition"]).reset_index(drop=True)
    print(f"  Stimuli extracted: {len(stimuli)} rows "
          f"({stimuli['item_no'].nunique()} items × 2 conditions)")
    return stimuli


# ---------------------------------------------------------------------------
# Step 4: Filter to one trial row per participant × item
# ---------------------------------------------------------------------------

def filter_trials(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one row per participant × trial; normalise condition labels.

    Gorilla emits one row per region (re1–re4) per trial, but R1–R4 RT values
    are pre-computed and identical across all region rows of the same trial.
    Filtering for region == 're2' deduplicates without losing any RT data.

    Args:
        df: DataFrame from classify_groups.

    Returns:
        Wide trial-level DataFrame (one row per participant × item).
    """
    df = df[
        (df["completeness"] == "complete") &
        (df["item"] == "trial") &
        (df["screen_name"] == "maze") &
        (df["region"] == "re2")
    ].copy()

    df["condition"] = df["condition"].map({"EN": "EN_Word_order", "FR": "FR_Word_order"})

    print(f"  After trial filter: {len(df)} trials, "
          f"{df['participant'].nunique()} participants")
    return df


# ---------------------------------------------------------------------------
# Step 5: Accuracy filter
# ---------------------------------------------------------------------------

def filter_accuracy(df: pd.DataFrame) -> pd.DataFrame:
    """Drop trials where not all maze choices were correct (Accuracy != 4).

    These are genuinely invalid trials — dropped entirely, not NaN-coded.

    Args:
        df: Trial-level DataFrame from filter_trials.

    Returns:
        Filtered DataFrame.
    """
    before = len(df)
    df = df[df["Accuracy"] == 4].copy()
    print(f"  After accuracy filter: {len(df)} trials (dropped {before - len(df)})")
    return df


# ---------------------------------------------------------------------------
# Step 6: Apply RT cutoffs — outliers → NaN (row kept)
# ---------------------------------------------------------------------------

def apply_rt_cutoffs(df: pd.DataFrame) -> pd.DataFrame:
    """Replace outlier RT values with NaN; keep the trial row.

    Rationale: an outlier in one region does not invalidate the other regions.
    Replacing with NaN means the cell is excluded from per-item means while the
    trial still contributes data for non-outlier regions.

    Rules (matching original R script):
        All regions : rt < 300 ms             → NaN  (too fast; likely misfire)
        Region 1    : rt > 2000 ms            → NaN
        Region 2    : rt > 2500 ms            → NaN
        Region 3    : rt > 2500 ms            → NaN
        Region 4    : rt > 3000 ms            → NaN

    Args:
        df: Wide trial-level DataFrame from filter_accuracy, with columns
            R1_Reaction … R4_Reaction as numeric.

    Returns:
        DataFrame with same shape; outlier cells replaced by NaN.
    """
    df = df.copy()
    total_cells = 0
    nan_cells   = 0

    for col, upper in RT_UPPER.items():
        df[col] = pd.to_numeric(df[col], errors="coerce")
        mask = (df[col] < RT_LOWER_GLOBAL) | (df[col] > upper)
        total_cells += len(df)
        nan_cells   += mask.sum()
        df.loc[mask, col] = np.nan

    pct = 100 * nan_cells / total_cells if total_cells else 0
    print(f"  RT outliers replaced with NaN: {nan_cells} cells "
          f"({pct:.1f}% of {total_cells} RT values)")
    return df


# ---------------------------------------------------------------------------
# Step 7: Attach stimuli and rename RT columns
# ---------------------------------------------------------------------------

def attach_stimuli(rt_df: pd.DataFrame, stimuli_df: pd.DataFrame) -> pd.DataFrame:
    """Merge stimuli columns onto the wide trial-level RT DataFrame.

    Args:
        rt_df:      Wide trial-level DataFrame from apply_rt_cutoffs.
        stimuli_df: Stimuli DataFrame from extract_stimuli.

    Returns:
        Merged DataFrame with stimuli columns and renamed RT columns.
    """
    cols_needed = [
        "participant", "group", "condition", "item_no", "advTYPE",
    ] + REGION_RT_COLS

    df = rt_df[cols_needed].copy()
    df = df.rename(columns=REGION_RT_RENAME)
    df = df.merge(stimuli_df, on=["item_no", "condition", "advTYPE"], how="left")

    # Reorder columns for readability
    col_order = [
        "participant", "group", "condition", "item_no", "advTYPE",
        "context_sentence",
        "region1_text", "region2_text", "region3_text", "region4_text",
        "full_sentence", "critical_token",
        "R1_rt", "R2_rt", "R3_rt", "R4_rt",
    ]
    col_order = [c for c in col_order if c in df.columns]
    return df[col_order]


# ---------------------------------------------------------------------------
# Step 7b: Residualize RT on region text character length
# ---------------------------------------------------------------------------

def add_residualized_rt(df: pd.DataFrame) -> pd.DataFrame:
    """Add residualized RT columns (R2_rt_resid, R3_rt_resid) as ablation covariates.

    For each of Region 2 and Region 3, fits a simple OLS regression:

        RT ~ nchar(region_text)

    and stores the residuals.  Residualization is done across all groups
    (Monolinguals + Ebilinguals) so the covariate effect is removed uniformly
    without interacting with the group membership that is the variable of interest.

    NaN RT cells (outliers from apply_rt_cutoffs) remain NaN in the residualized
    columns.  The raw R2_rt / R3_rt columns are unchanged.

    Args:
        df: Wide trial-level DataFrame from attach_stimuli, containing
            R2_rt, R3_rt, region2_text, region3_text columns.

    Returns:
        DataFrame with two additional columns: R2_rt_resid, R3_rt_resid.
    """
    df = df.copy()

    for rt_col, text_col, resid_col in [
        ("R2_rt", "region2_text", "R2_rt_resid"),
        ("R3_rt", "region3_text", "R3_rt_resid"),
    ]:
        nchar = df[text_col].fillna("").str.len().astype(float)

        # Fit OLS only on non-NaN RT rows
        valid = df[rt_col].notna()
        x = nchar[valid].values
        y = df.loc[valid, rt_col].values

        slope, intercept = np.polyfit(x, y, 1)
        fitted    = intercept + slope * nchar
        residuals = df[rt_col] - fitted
        residuals[~valid] = np.nan

        df[resid_col] = residuals

        r_sq = float(np.corrcoef(x, y)[0, 1] ** 2)
        print(f"  {resid_col}: slope={slope:.2f} ms/char, "
              f"R²={r_sq:.3f}, N={valid.sum()}")

    return df


# ---------------------------------------------------------------------------
# Step 8: Descriptive statistics
# ---------------------------------------------------------------------------

def print_descriptives(df: pd.DataFrame) -> None:
    """Print mean, SD, and N per region and per group × condition × region.

    NaN values (outliers) are automatically excluded from all aggregations.

    Args:
        df: Wide trial-level DataFrame with R1_rt … R4_rt columns.
    """
    rt_cols = list(REGION_RT_RENAME.values())

    print("\n--- Mean RT by region (cleaned, outliers excluded) ---")
    summary = df[rt_cols].agg(["mean", "std", "count"]).round(1)
    print(summary.to_string())

    print("\n--- Mean RT by group × condition × region ---")
    long = df.melt(
        id_vars=["group", "condition"],
        value_vars=rt_cols,
        var_name="region", value_name="rt",
    )
    print(
        long.groupby(["group", "condition", "region"])["rt"]
        .agg(mean="mean", sd="std", n="count")
        .round(1)
        .to_string()
    )


# ---------------------------------------------------------------------------
# Step 9: Plot
# ---------------------------------------------------------------------------

def plot_rt_line(df: pd.DataFrame, output_dir: str = "results/figures") -> None:
    """Plot mean RT per region as a line graph, faceted by group.

    X-axis : Region 1 → Region 4
    Y-axis : Mean RT (ms)
    Lines  : EN_Word_order (grammatical) vs FR_Word_order (ungrammatical)
    Facets : Monolinguals | Ebilinguals

    Args:
        df:         Wide trial-level DataFrame with R1_rt … R4_rt columns.
        output_dir: Directory to save the figure.
    """
    os.makedirs(output_dir, exist_ok=True)

    rt_cols     = list(REGION_RT_RENAME.values())
    label_order = [REGION_PLOT_LABELS[c] for c in rt_cols]
    group_order = ["Monolinguals", "Ebilinguals"]

    long = df.melt(
        id_vars=["group", "condition"],
        value_vars=rt_cols,
        var_name="region", value_name="rt",
    )
    means = (
        long.groupby(["group", "condition", "region"])["rt"]
        .mean()
        .reset_index()
        .rename(columns={"rt": "mean_rt"})
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    fig.suptitle(
        "Mean Maze RT by Region, Group, and Word-Order Condition",
        fontsize=14, fontweight="bold", y=1.01,
    )

    for ax, group in zip(axes, group_order):
        subset = means[means["group"] == group]
        for condition, grp in subset.groupby("condition"):
            grp_sorted = (
                grp.set_index("region")
                   .reindex(rt_cols)
                   .reset_index()
            )
            ax.plot(
                label_order,
                grp_sorted["mean_rt"].values,
                marker="o",
                label=condition.replace("_", " "),
                color=CONDITION_PALETTE[condition],
                linewidth=2,
                markersize=8,
            )
        ax.set_title(group, fontsize=13, pad=8)
        ax.set_xlabel("Region", fontsize=11)
        ax.set_ylabel("Mean RT (ms)" if group == "Monolinguals" else "", fontsize=11)
        ax.legend(title="Condition", fontsize=9, loc="upper left")
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.tick_params(axis="x", labelsize=9)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "rt_by_region_group_condition.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"\n  Plot saved → {out_path}")


# ---------------------------------------------------------------------------
# Step 10: Save outputs
# ---------------------------------------------------------------------------

def save_outputs(
    rt_df:      pd.DataFrame,
    stimuli_df: pd.DataFrame,
    rt_path:    str,
    stimuli_path: str,
) -> None:
    """Save both output CSVs.

    behavioral_rt_cleaned.csv — wide trial-level data with stimuli columns.
        One row per participant × item. RT cells are NaN where the value was
        a region-specific outlier. Used by correlation.py after residualization.

    stimuli.csv — unique stimuli per item × condition.
        Used directly by surprisal_causal.py and surprisal_masked.py as LLM input.
        Columns: item_no, condition, advTYPE, context_sentence,
                 region1_text … region4_text, full_sentence, critical_token.

    Args:
        rt_df:        Wide trial-level DataFrame from attach_stimuli.
        stimuli_df:   Stimuli DataFrame from extract_stimuli.
        rt_path:      Output path for behavioral_rt_cleaned.csv.
        stimuli_path: Output path for stimuli.csv.
    """
    for path in [rt_path, stimuli_path]:
        out_dir = os.path.dirname(path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    rt_df.to_csv(rt_path, index=False)
    print(f"\n  behavioral_rt_cleaned.csv → {rt_path}")
    print(f"    Shape: {rt_df.shape}  "
          f"| Participants: {rt_df['participant'].nunique()}  "
          f"| Items: {rt_df['item_no'].nunique()}")

    stimuli_df.to_csv(stimuli_path, index=False)
    print(f"\n  stimuli.csv → {stimuli_path}")
    print(f"    Shape: {stimuli_df.shape}  "
          f"| Items: {stimuli_df['item_no'].nunique()}  "
          f"| Conditions: {stimuli_df['condition'].unique().tolist()}")


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def run_preprocessing(
    maze_path:    str = "data/maze_116_20231004.csv",
    lbq_path:     str = "data/lbq_116.csv",
    rt_output:    str = "data/behavioral_rt_cleaned.csv",
    stimuli_output: str = "data/stimuli.csv",
    figures_dir:  str = "results/figures",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """End-to-end preprocessing pipeline.

    Args:
        maze_path:       Path to raw maze CSV.
        lbq_path:        Path to LBQ CSV.
        rt_output:       Destination path for behavioral_rt_cleaned.csv.
        stimuli_output:  Destination path for stimuli.csv.
        figures_dir:     Directory to save plots.

    Returns:
        (rt_df, stimuli_df) — cleaned RT DataFrame and stimuli DataFrame.
    """
    print("=" * 60)
    print("Step 1 — Load and join maze + LBQ")
    print("=" * 60)
    df_raw = load_and_join(maze_path, lbq_path)
    print(f"  Loaded {len(df_raw)} rows, "
          f"{df_raw['participant'].nunique()} unique participants")

    print("\n" + "=" * 60)
    print("Step 2 — Classify groups (Monolinguals / Ebilinguals)")
    print("=" * 60)
    df = classify_groups(df_raw)

    print("\n" + "=" * 60)
    print("Step 3 — Extract stimuli (context + region texts)")
    print("=" * 60)
    stimuli_df = extract_stimuli(df)

    print("\n" + "=" * 60)
    print("Step 4 — Filter to one trial row per participant × item")
    print("=" * 60)
    df = filter_trials(df)

    print("\n" + "=" * 60)
    print("Step 5 — Accuracy filter (drop Accuracy != 4)")
    print("=" * 60)
    df = filter_accuracy(df)

    print("\n" + "=" * 60)
    print("Step 6 — RT cutoffs (outliers → NaN, row kept)")
    print("=" * 60)
    df = apply_rt_cutoffs(df)

    print("\n" + "=" * 60)
    print("Step 7 — Attach stimuli columns")
    print("=" * 60)
    rt_df = attach_stimuli(df, stimuli_df)
    print(f"  Final DataFrame shape: {rt_df.shape}")

    print("\n" + "=" * 60)
    print("Step 7b — Residualize RT on region text length (ablation)")
    print("=" * 60)
    rt_df = add_residualized_rt(rt_df)

    print("\n" + "=" * 60)
    print("Step 8 — Descriptive statistics")
    print("=" * 60)
    print_descriptives(rt_df)

    print("\n" + "=" * 60)
    print("Step 9 — Plot")
    print("=" * 60)
    plot_rt_line(rt_df, figures_dir)

    print("\n" + "=" * 60)
    print("Step 10 — Save outputs")
    print("=" * 60)
    save_outputs(rt_df, stimuli_df, rt_output, stimuli_output)

    return rt_df, stimuli_df


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess maze RT data.")
    parser.add_argument("--maze",     default="data/maze_116_20231004.csv")
    parser.add_argument("--lbq",      default="data/lbq_116.csv")
    parser.add_argument("--rt_out",   default="data/behavioral_rt_cleaned.csv")
    parser.add_argument("--stim_out", default="data/stimuli.csv")
    parser.add_argument("--figures",  default="results/figures")
    args = parser.parse_args()

    run_preprocessing(
        maze_path=args.maze,
        lbq_path=args.lbq,
        rt_output=args.rt_out,
        stimuli_output=args.stim_out,
        figures_dir=args.figures,
    )
