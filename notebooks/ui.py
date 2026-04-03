"""
ui.py
ipywidgets UI for the surprisal–RT correlation analysis pipeline.

Layout
------
  Section 0  Introduction — project description + registered model pairs table
  Section 1  Data Input   — surprisal CSV paths + RT data path + Load button
  Section 2  Options      — model pair checkboxes, region checkboxes, PLL variant
  Section 3  Run          — save-to-disk toggle + output directory + Run button
  Section 4  Output       — inline results table and scatter plots (or saved to disk)
"""

import io
import os
import sys
import importlib

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")          # non-interactive backend; figures rendered to PNG buffers
import matplotlib.pyplot as plt
import ipywidgets as widgets
from IPython.display import display, clear_output
from IPython.display import HTML as _IPyHTML   # only for display() calls, NOT inside widget containers
from IPython.display import Image as _IPyImage # for rendering Agg figures inline
from scipy import stats as scipy_stats

# Resolve project root (one level above notebooks/) and make it the working
# directory so that relative paths like "results/..." and "data/..." work
# regardless of where Jupyter was launched from.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(_ROOT)

_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

import correlation as _corr


# ---------------------------------------------------------------------------
# Styling helpers
# ---------------------------------------------------------------------------

_SECTION_STYLE = (
    "margin: 18px 0 8px 0; padding: 9px 14px;"
    "background: #f0f4ff; border-left: 4px solid #4285f4;"
    "border-radius: 3px; font-size: 14px; font-weight: bold;"
)

def _section(title: str) -> widgets.HTML:
    return widgets.HTML(f'<div style="{_SECTION_STYLE}">{title}</div>')


def _model_pair_table() -> widgets.HTML:
    """HTML table of all registered model pairs."""
    rows = "".join(
        f"<tr>"
        f"<td style='padding:4px 14px'>{mono}</td>"
        f"<td style='padding:4px 14px'>{multi}</td>"
        f"<td style='padding:4px 14px; color:#555'>{lm_type}</td>"
        f"</tr>"
        for mono, multi, lm_type in _corr.MODEL_PAIRS
    )
    return widgets.HTML(f"""
    <table style="border-collapse:collapse; font-size:13px; margin-top:6px">
      <thead>
        <tr style="background:#dce8ff">
          <th style="padding:6px 14px; text-align:left">Monolingual</th>
          <th style="padding:6px 14px; text-align:left">Multilingual</th>
          <th style="padding:6px 14px; text-align:left">LM type</th>
        </tr>
      </thead>
      <tbody>{rows}</tbody>
    </table>
    """)


# ---------------------------------------------------------------------------
# Main UI class
# ---------------------------------------------------------------------------

class CorrelationUI:
    """Interactive widget UI for the correlation analysis pipeline."""

    def __init__(self):
        self._surprisal_df = None
        self._rt_df        = None
        self._pair_checks  = {}       # (mono, multi) → Checkbox widget
        self._build()

    # ── Construction ────────────────────────────────────────────────────────

    def _build(self):
        importlib.reload(_corr)       # pick up any edits to correlation.py

        # ── Introduction ─────────────────────────────────────────────────────
        intro = widgets.VBox([
            widgets.HTML("""
            <div style="font-family:sans-serif; max-width:840px">
              <h2 style="margin-bottom:4px">
                Syntactic Transfer in LLMs — Correlation Analysis
              </h2>
              <p style="color:#444; margin:4px 0 10px 0">
                Computes a 2×2 surprisal–RT interaction per model pair and
                correlates it with human maze-task reading times:
              </p>
              <div style="font-family:monospace; background:#f1f3f4;
                          padding:8px 14px; border-radius:4px; font-size:13px">
                ΔS &nbsp;= [S<sub>mono</sub>(FR) − S<sub>mono</sub>(EN)] −
                           [S<sub>multi</sub>(FR) − S<sub>multi</sub>(EN)]<br>
                ΔRT = [RT<sub>mono</sub>(FR) − RT<sub>mono</sub>(EN)] −
                      [RT<sub>ebilingual</sub>(FR) − RT<sub>ebilingual</sub>(EN)]<br>
                <br>
                Correlation: &nbsp; ΔS &nbsp;~&nbsp; ΔRT &nbsp; across items
                &nbsp;(Pearson r + permutation test)
              </div>
              <p style="color:#444; margin:10px 0 4px 0">
                <b>Registered model pairs</b>
                (pairs with loaded data will be selectable below):
              </p>
            </div>
            """),
            _model_pair_table(),
        ])

        # ── Section 1 — Data Input ────────────────────────────────────────────
        _default_surp = "\n".join([
            "results/surprisal_bert.csv",
            "results/surprisal_bloom-560m.csv",
            "results/surprisal_croissantllm.csv",
            "results/surprisal_distilbert.csv",
            "results/surprisal_distilmbert.csv",
            "results/surprisal_gpt2.csv",
            "results/surprisal_mbert.csv",
            "results/surprisal_mgpt.csv",
            "results/surprisal_opt-125m.csv",
            "results/surprisal_pythia-160m.csv",
            "results/surprisal_roberta.csv",
            "results/surprisal_xlmr-large.csv",
            "results/surprisal_xlmr.csv",
        ])
        self._surp_w = widgets.Textarea(
            value=_default_surp,
            layout=widgets.Layout(width="540px", height="200px"),
        )
        self._rt_w = widgets.Text(
            value="data/behavioral_rt_cleaned.csv",
            layout=widgets.Layout(width="540px"),
        )
        self._load_btn = widgets.Button(
            description="Load Data",
            button_style="info",
            icon="folder-open",
            layout=widgets.Layout(width="130px"),
        )
        self._load_out = widgets.Output()
        self._load_btn.on_click(self._on_load)

        data_section = widgets.VBox([
            _section("1 · Data Input"),
            widgets.HTML("<b>Surprisal CSV paths</b> — one file path per line:"),
            self._surp_w,
            widgets.HTML("<b>RT data path:</b>"),
            self._rt_w,
            self._load_btn,
            self._load_out,
        ])

        # ── Section 2 — Options (hidden until data loaded) ────────────────────
        self._pair_box = widgets.VBox([])        # filled on load

        self._region_checks = {
            "region2": widgets.Checkbox(
                value=True,
                description="Region 2  —  V+Adv chunk  (primary)",
                style={"description_width": "initial"},
            ),
            "region3": widgets.Checkbox(
                value=True,
                description="Region 3  —  Object NP  (spillover)",
                style={"description_width": "initial"},
            ),
        }

        self._pll_w = widgets.RadioButtons(
            options=[
                ("Standard PLL  (Salazar et al. 2020)", "PLL"),
                ("PLL-word-l2r  (Kauf & Ivanova 2023)", "PLL_word_l2r"),
            ],
            value="PLL",
            layout=widgets.Layout(margin="0"),
        )

        self._resid_cb = widgets.Checkbox(
            value=False,
            description="Ablation: residualize RT on region text length  "
                        "(controls for nchar of region text before correlating)",
            style={"description_width": "initial"},
        )

        self._options_box = widgets.VBox([
            _section("2 · Model Pairs & Options"),
            widgets.HTML("<b>Model pairs</b> — only pairs present in the loaded data are enabled:"),
            self._pair_box,
            widgets.HTML("<br><b>Regions:</b>"),
            self._region_checks["region2"],
            self._region_checks["region3"],
            widgets.HTML("<br><b>RT residualization (ablation):</b>"),
            self._resid_cb,
            widgets.HTML("<br><b>PLL variant</b> (applies to masked LMs only):"),
            self._pll_w,
        ], layout=widgets.Layout(display="none"))

        # ── Section 3 — Output & Run ──────────────────────────────────────────
        self._save_cb = widgets.Checkbox(
            value=False,
            description="Save results and plots to disk",
            style={"description_width": "initial"},
        )
        self._outdir_w = widgets.Text(
            value="results",
            placeholder="Output directory",
            layout=widgets.Layout(width="320px", display="none"),
        )
        self._save_cb.observe(
            lambda change: setattr(
                self._outdir_w.layout, "display",
                "" if change["new"] else "none"
            ),
            names="value",
        )
        self._run_btn = widgets.Button(
            description="Run Analysis",
            button_style="success",
            icon="play",
            layout=widgets.Layout(width="155px", margin="10px 0 0 0"),
        )
        self._run_btn.on_click(self._on_run)

        run_section = widgets.VBox([
            _section("3 · Output & Run"),
            self._save_cb,
            widgets.HBox([
                widgets.HTML('<span style="margin:auto 8px auto 0">Output directory:</span>'),
                self._outdir_w,
            ]),
            self._run_btn,
        ], layout=widgets.Layout(display="none"))

        # ── Section 4 — Results output ────────────────────────────────────────
        self._results_out = widgets.Output()

        # ── Store refs needed across callbacks ────────────────────────────────
        self._options_box_ref = self._options_box
        self._run_section_ref = run_section

        self._root = widgets.VBox([
            intro,
            data_section,
            self._options_box,
            run_section,
            self._results_out,
        ], layout=widgets.Layout(max_width="880px"))

    # ── Callbacks ────────────────────────────────────────────────────────────

    def _on_load(self, _):
        with self._load_out:
            clear_output()
            importlib.reload(_corr)   # pick up any edits to correlation.py

            paths_raw = self._surp_w.value.strip()
            rt_path   = self._rt_w.value.strip()

            if not paths_raw:
                print("⚠  Enter at least one surprisal CSV path.")
                return
            if not rt_path:
                print("⚠  Enter the RT data path.")
                return

            paths   = [p.strip() for p in paths_raw.splitlines() if p.strip()]
            missing = [p for p in paths + [rt_path] if not os.path.exists(p)]
            if missing:
                for m in missing:
                    print(f"✗  File not found: {m}")
                return

            try:
                self._surprisal_df = _corr.load_surprisal(paths)
                self._rt_df        = _corr.load_behavioral_rt(rt_path)
            except Exception as exc:
                print(f"✗  Error loading data: {exc}")
                return

            available = set(self._surprisal_df["model"].unique())
            print(f"✔  Surprisal loaded  —  {len(self._surprisal_df):,} rows  "
                  f"|  models: {sorted(available)}")
            print(f"✔  RT data loaded    —  {len(self._rt_df):,} trials  "
                  f"|  participants: {self._rt_df['participant'].nunique()}")

            # Populate model pair checkboxes, disabled if data absent
            self._pair_checks = {}
            checks = []
            for mono, multi, lm_type in _corr.MODEL_PAIRS:
                has = mono in available and multi in available
                cb = widgets.Checkbox(
                    value=has,
                    description=f"{mono}  →  {multi}  ({lm_type})",
                    disabled=not has,
                    style={"description_width": "initial"},
                )
                self._pair_checks[(mono, multi)] = cb
                checks.append(cb)
            self._pair_box.children = tuple(checks)

            # Reveal options + run sections
            self._options_box.layout.display = ""
            self._run_section_ref.layout.display = ""

    def _on_run(self, _):
        with self._results_out:
            clear_output()

            importlib.reload(_corr)   # pick up any edits to correlation.py

            if self._surprisal_df is None or self._rt_df is None:
                print("⚠  Load data first.")
                return

            selected = [
                (mono, multi, lm_type)
                for (mono, multi, lm_type) in _corr.MODEL_PAIRS
                if (mono, multi) in self._pair_checks
                and self._pair_checks[(mono, multi)].value
            ]
            if not selected:
                print("⚠  Select at least one model pair.")
                return

            use_resid = self._resid_cb.value
            regions = [
                (r + "_resid" if use_resid else r)
                for r, cb in self._region_checks.items() if cb.value
            ]
            if not regions:
                print("⚠  Select at least one region.")
                return

            save    = self._save_cb.value
            out_dir = self._outdir_w.value.strip()
            pll_var = self._pll_w.value

            # Build a combo subfolder name, e.g. "PLL_raw" or "PLL_word_l2r_resid"
            rt_label   = "resid" if use_resid else "raw"
            combo_dir  = os.path.join(out_dir, f"{pll_var}_{rt_label}")
            figures_dir = os.path.join(combo_dir, "figures")

            if save:
                os.makedirs(figures_dir, exist_ok=True)

            rows = []

            for mono, multi, _ in selected:
                for region in regions:
                    label = f"{mono} → {multi}  |  {region}"
                    print(f"  Running {label} …", end="  ")

                    try:
                        surp_int = _corr.compute_surprisal_interaction(
                            self._surprisal_df, mono, multi, region,
                            pll_variant=pll_var,
                        )
                        rt_int = _corr.compute_rt_interaction(self._rt_df, region)
                        merged = _corr.merge_interactions(surp_int, rt_int)
                    except Exception as exc:
                        print(f"✗  {exc}")
                        continue

                    if len(merged) < 5:
                        print(f"too few items ({len(merged)}) — skipped")
                        continue

                    corr_res         = _corr.run_correlation(merged)
                    p_perm           = _corr.permutation_test(merged)
                    corr_res["p_perm"] = p_perm

                    r, p, rho = (corr_res["pearson_r"],
                                 corr_res["pearson_p"],
                                 corr_res["spearman_rho"])
                    print(f"r = {r:.3f}, p = {p:.3f}, "
                          f"p_perm = {p_perm:.3f}, N = {corr_res['n_items']}")

                    rows.append({
                        "Mono model":    mono,
                        "Multi model":   multi,
                        "Region":        region,
                        "Pearson r":     round(r,   3),
                        "p (parametric)": round(p,  3),
                        "p (permutation)": round(p_perm, 3),
                        "Spearman ρ":    round(rho, 3),
                        "N items":       corr_res["n_items"],
                    })

                    fig = self._scatter(merged, corr_res, mono, multi, region, pll_var)

                    if save:
                        fig_path = os.path.join(
                            figures_dir,
                            f"scatter_{mono}_vs_{multi}_{region}.png",
                        )
                        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
                        print(f"    → Figure saved: {fig_path}")
                    else:
                        buf = io.BytesIO()
                        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
                        buf.seek(0)
                        display(_IPyImage(buf.getvalue()))
                    plt.close(fig)

            if not rows:
                return

            # Summary table
            summary_df = pd.DataFrame(rows)
            print()
            display(_section("Results Summary"))
            display(_IPyHTML(
                summary_df.style
                    .format({"Pearson r": "{:.3f}", "p (parametric)": "{:.3f}",
                             "p (permutation)": "{:.3f}", "Spearman ρ": "{:.3f}"})
                    .set_table_styles([
                        {"selector": "th", "props": "background:#dce8ff; padding:6px 12px; text-align:left"},
                        {"selector": "td", "props": "padding:5px 12px; border-bottom:1px solid #eee"},
                    ])
                    .to_html()
            ))

            if save:
                csv_path = os.path.join(combo_dir, "correlation_results.csv")
                summary_df.to_csv(csv_path, index=False)
                print(f"\n✔  Summary saved → {csv_path}")

    # ── Scatter plot helper ──────────────────────────────────────────────────

    def _scatter(self, merged, corr_res, mono, multi, region, pll_var):
        x = merged["surprisal_interaction"].values
        y = merged["rt_interaction"].values

        fig, ax = plt.subplots(figsize=(5.5, 4.5))
        ax.scatter(x, y, color="#2196F3", alpha=0.75, edgecolors="white",
                   linewidths=0.5, s=60, zorder=3)

        slope, intercept, *_ = scipy_stats.linregress(x, y)
        x_line = np.linspace(x.min(), x.max(), 200)
        ax.plot(x_line, slope * x_line + intercept,
                color="#F44336", linewidth=1.8, zorder=2)

        ax.axhline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)
        ax.axvline(0, color="gray", linewidth=0.8, linestyle="--", alpha=0.5)

        r      = corr_res["pearson_r"]
        p      = corr_res["pearson_p"]
        p_perm = corr_res.get("p_perm")
        n      = corr_res["n_items"]

        annot = f"r = {r:.3f},  p = {p:.3f}"
        if p_perm is not None:
            annot += f"\np_perm = {p_perm:.3f}"
        annot += f"\nN = {n}"
        ax.text(0.05, 0.95, annot, transform=ax.transAxes,
                va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

        _base_region = region.replace("_resid", "")
        region_label = "Region 2 (V+Adv)" if _base_region == "region2" else "Region 3 (Spillover)"
        variant_tag  = "" if pll_var == "PLL" else "  [PLL-word-l2r]"
        ax.set_title(f"{mono} → {multi}  |  {region_label}{variant_tag}",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("ΔS interaction\n[S_mono(FR−EN) − S_multi(FR−EN)]", fontsize=9)
        ax.set_ylabel("ΔRT interaction\n[RT_mono(FR−EN) − RT_ebilingual(FR−EN)]", fontsize=9)
        fig.tight_layout()
        return fig

    # ── Public entry point ───────────────────────────────────────────────────

    def show(self):
        """Render the UI in the current notebook cell."""
        display(self._root)


# ---------------------------------------------------------------------------
# Convenience entry point (called from the notebook)
# ---------------------------------------------------------------------------

def run_analysis() -> CorrelationUI:
    """Instantiate and display the correlation UI. Returns the UI object."""
    ui = CorrelationUI()
    ui.show()
    return ui
