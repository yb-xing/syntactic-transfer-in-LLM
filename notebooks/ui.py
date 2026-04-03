"""
ui.py
ipywidgets UI for the surprisal t-test analysis pipeline.

Layout
------
  Section 0  Introduction — project description + registered model pairs table
  Section 1  Data Input   — surprisal CSV paths + Load button
  Section 2  Options      — model pair checkboxes, region checkboxes, PLL variant
  Section 3  Run          — save-to-disk toggle + output directory + Run button
  Section 4  Output       — inline results table and violin plots (or saved to disk)
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
from IPython.display import HTML as _IPyHTML
from IPython.display import Image as _IPyImage

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
    """Interactive widget UI for the surprisal t-test analysis pipeline."""

    def __init__(self):
        self._surprisal_df = None
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
                Syntactic Transfer in LLMs — Surprisal Analysis
              </h2>
              <p style="color:#444; margin:4px 0 10px 0">
                Computes per-item surprisal deltas and runs paired t-tests
                comparing monolingual vs. multilingual models (H1):
              </p>
              <div style="font-family:monospace; background:#f1f3f4;
                          padding:8px 14px; border-radius:4px; font-size:13px">
                S_mono(FR−EN)  = surprisal_mono(FR)  − surprisal_mono(EN)<br>
                S_multi(FR−EN) = surprisal_multi(FR) − surprisal_multi(EN)<br>
                <br>
                H1: &nbsp; mean(S_mono(FR−EN)) &gt; mean(S_multi(FR−EN)) &nbsp;
                (paired t-test)
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

        self._options_box = widgets.VBox([
            _section("2 · Model Pairs & Options"),
            widgets.HTML("<b>Model pairs</b> — only pairs present in the loaded data are enabled:"),
            self._pair_box,
            widgets.HTML("<br><b>Regions:</b>"),
            self._region_checks["region2"],
            self._region_checks["region3"],
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
            importlib.reload(_corr)

            paths_raw = self._surp_w.value.strip()
            if not paths_raw:
                print("⚠  Enter at least one surprisal CSV path.")
                return

            paths   = [p.strip() for p in paths_raw.splitlines() if p.strip()]
            missing = [p for p in paths if not os.path.exists(p)]
            if missing:
                for m in missing:
                    print(f"✗  File not found: {m}")
                return

            try:
                self._surprisal_df = _corr.load_surprisal(paths)
            except Exception as exc:
                print(f"✗  Error loading data: {exc}")
                return

            available = set(self._surprisal_df["model"].unique())
            print(f"✔  Surprisal loaded  —  {len(self._surprisal_df):,} rows  "
                  f"|  models: {sorted(available)}")

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

            if self._surprisal_df is None:
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

            regions = [r for r, cb in self._region_checks.items() if cb.value]
            if not regions:
                print("⚠  Select at least one region.")
                return

            save    = self._save_cb.value
            out_dir = self._outdir_w.value.strip()
            pll_var = self._pll_w.value

            # Organise output by PLL variant: results/PLL/ or results/PLL_word_l2r/
            combo_dir   = os.path.join(out_dir, pll_var)
            figures_dir = os.path.join(combo_dir, "figures")
            if save:
                os.makedirs(figures_dir, exist_ok=True)

            rows = []

            for mono, multi, _ in selected:
                for region in regions:
                    label = f"{mono} → {multi}  |  {region}"
                    print(f"  Running {label} …", end="  ")

                    try:
                        model_delta = _corr.compute_model_delta(
                            self._surprisal_df, mono, multi, region,
                            pll_variant=pll_var,
                        )
                    except Exception as exc:
                        print(f"✗  {exc}")
                        continue

                    if len(model_delta) < 5:
                        print(f"too few items ({len(model_delta)}) — skipped")
                        continue

                    tt = _corr.run_model_ttest(model_delta)
                    print(f"t = {tt['t_stat']:.3f}, "
                          f"p(two) = {tt['p_two_tailed']:.3f}, "
                          f"p(one) = {tt['p_one_tailed']:.3f}, "
                          f"mean diff = {tt['mean_diff']:.3f}, "
                          f"N = {tt['n_items']}")

                    rows.append({
                        "Mono model":     mono,
                        "Multi model":    multi,
                        "Region":         region,
                        "t":              round(tt["t_stat"],       3),
                        "p (two-tail)":   round(tt["p_two_tailed"], 3),
                        "p (one-tail)":   round(tt["p_one_tailed"], 3),
                        "Mean diff":      round(tt["mean_diff"],    3),
                        "Mean mono Δ":    round(tt["mean_mono"],    3),
                        "Mean multi Δ":   round(tt["mean_multi"],   3),
                        "N items":        tt["n_items"],
                    })

                    fig = self._violin_model_delta(
                        model_delta, tt, mono, multi, region, pll_var)
                    if save:
                        fig_path = os.path.join(
                            figures_dir,
                            f"model_delta_{mono}_vs_{multi}_{region}.png",
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
                    .format({
                        "t":            "{:.3f}",
                        "p (two-tail)": "{:.3f}",
                        "p (one-tail)": "{:.3f}",
                        "Mean diff":    "{:.3f}",
                        "Mean mono Δ":  "{:.3f}",
                        "Mean multi Δ": "{:.3f}",
                    })
                    .set_table_styles([
                        {"selector": "th", "props": "background:#dce8ff; padding:6px 12px; text-align:left"},
                        {"selector": "td", "props": "padding:5px 12px; border-bottom:1px solid #eee"},
                    ])
                    .to_html()
            ))

            if save:
                csv_path = os.path.join(combo_dir, "ttest_results.csv")
                summary_df.to_csv(csv_path, index=False)
                print(f"\n✔  Summary saved → {csv_path}")

    # ── Violin plot helper ───────────────────────────────────────────────────

    def _violin_model_delta(self, model_delta, tt, mono, multi, region, pll_var):
        """Paired violin: S_mono(FR−EN) vs S_multi(FR−EN) distributions across items."""
        mono_vals  = model_delta["mono_delta"].values
        multi_vals = model_delta["multi_delta"].values

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
        ax.set_xticks([0, 1])
        ax.set_xticklabels([mono, multi], fontsize=10)

        t, p1, p2, n = (tt["t_stat"], tt["p_one_tailed"],
                        tt["p_two_tailed"], tt["n_items"])
        annot = (f"paired t = {t:.3f}\np(one-tail) = {p1:.3f},"
                 f"  p(two-tail) = {p2:.3f}\nN = {n} items")
        ax.text(0.05, 0.97, annot, transform=ax.transAxes,
                va="top", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85))

        region_label = "Region 2 (V+Adv)" if region == "region2" else "Region 3 (Spillover)"
        variant_tag  = "" if pll_var == "PLL" else "  [PLL-word-l2r]"
        ax.set_title(f"{mono} vs {multi}  |  {region_label}{variant_tag}\n"
                     f"S(FR−EN) per model  (expected: mono > multi)",
                     fontsize=10, fontweight="bold")
        ax.set_ylabel("Surprisal(FR) − Surprisal(EN)", fontsize=9)
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
    """Instantiate and display the analysis UI. Returns the UI object."""
    ui = CorrelationUI()
    ui.show()
    return ui
