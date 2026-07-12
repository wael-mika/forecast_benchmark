"""Publication-quality figures for the w14 discharge forecasting benchmark.

Produces three figures:
  fig1_nse_rmse_by_stage.pdf/.png  — avg NSE and RMSE by model × stage
  fig2_ablation_average.pdf/.png   — avg NSE gain: Stage1→2 and Stage2→3
  fig3_ablation_by_horizon.pdf/.png — per-horizon NSE gain for both transitions

All figures include the newly trained XGBoost Hydro (w14) experiment.

Usage
-----
    python3 scripts/plot_journal_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
SEQ_DIR = ARTIFACT_DIR / "advanced_seq"
ADV_DIR = ARTIFACT_DIR / "advanced"
OUT_DIR = ARTIFACT_DIR / "journal_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Typography & style ────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "font.size": 8,
    "axes.titlesize": 9,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7,
    "legend.framealpha": 0.9,
    "legend.edgecolor": "0.7",
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
    "xtick.major.size": 3,
    "ytick.major.size": 3,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "grid.linewidth": 0.4,
    "grid.color": "0.88",
    "pdf.fonttype": 42,   # embeds fonts in PDF
    "ps.fonttype": 42,
})

# ── Colour palette ────────────────────────────────────────────────────────────
STAGE_COLORS = {
    "Context":  "#4E79A7",   # muted blue
    "Weather":  "#F28E2B",   # amber
    "Hydro":    "#59A14F",   # green
}
STAGE_LABELS = {
    "Context": "Stage 1: Discharge context",
    "Weather": "Stage 2: +ERA5 atmospheric",
    "Hydro":   "Stage 3: +ERA5-Land soil",
}
GAIN_POS = "#2ca02c"
GAIN_NEG = "#d62728"

MODELS_ORDERED = ["TFT", "N-HiTS", "ANN", "LSTM", "PatchTST", "xLSTM", "Mamba", "XGBoost"]
MODEL_KEY = {
    "TFT": "tft", "N-HiTS": "nhits", "ANN": "ann", "LSTM": "lstm",
    "PatchTST": "patchtst", "xLSTM": "xlstm", "Mamba": "mamba", "XGBoost": "xgboost",
}

# ── Data loading helpers ──────────────────────────────────────────────────────

def _micro_avg(path: Path) -> dict[str, float]:
    """Return avg-over-horizons micro NSE, RMSE, MAE from a metrics_summary.csv."""
    df = pd.read_csv(path)
    test = df[(df["split"] == "test") & (df["aggregation"] == "micro")]
    return {
        "nse":  float(test["nse"].mean()),
        "rmse": float(test["rmse"].mean()),
        "mae":  float(test["mae"].mean()),
    }


def _micro_per_horizon(path: Path) -> pd.DataFrame:
    """Return test micro NSE/RMSE per horizon (h1/h2/h3)."""
    df = pd.read_csv(path)
    return (
        df[(df["split"] == "test") & (df["aggregation"] == "micro")]
        [["horizon", "nse", "rmse", "mae"]]
        .set_index("horizon")
    )


def load_summary_data() -> pd.DataFrame:
    """Build a (model, stage) → avg metrics DataFrame."""
    rows = []
    for label, key in MODEL_KEY.items():
        # Context
        p = SEQ_DIR / f"{key}_context_w14_h3" / "metrics_summary.csv"
        if not p.exists():
            p = ADV_DIR / f"{key}_context_w14_h3" / "metrics_summary.csv"
        if p.exists():
            m = _micro_avg(p)
            rows.append({"model": label, "stage": "Context", **m})

        # Weather
        p = SEQ_DIR / f"{key}_weather_w14_h3" / "metrics_summary.csv"
        if not p.exists():
            p = ADV_DIR / f"{key}_weather_w14_h3" / "metrics_summary.csv"
        if p.exists():
            m = _micro_avg(p)
            rows.append({"model": label, "stage": "Weather", **m})

        # Hydro
        p = SEQ_DIR / f"{key}_hydro_weather_w14_h3" / "metrics_summary.csv"
        if p.exists():
            m = _micro_avg(p)
            rows.append({"model": label, "stage": "Hydro", **m})

    return pd.DataFrame(rows)


def load_ablation_average() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load pre-computed ablation CSVs and inject XGBoost rows."""
    weather = pd.read_csv(
        SEQ_DIR / "ablation" / "weather_ablation" / "weather_effect_average.csv"
    )
    hydro = pd.read_csv(
        SEQ_DIR / "ablation" / "hydro_ablation" / "hydro_effect_average.csv"
    )

    # Compute XGBoost weather gain from metrics files
    xgb_ctx = _micro_avg(ADV_DIR / "xgboost_context_w14_h3" / "metrics_summary.csv")
    xgb_wth = _micro_avg(ADV_DIR / "xgboost_weather_w14_h3" / "metrics_summary.csv")
    xgb_hyd = _micro_avg(SEQ_DIR / "xgboost_hydro_weather_w14_h3" / "metrics_summary.csv")

    xgb_w_row = pd.DataFrame([{
        "model_name": "xgboost",
        "rmse_gain": xgb_ctx["rmse"] - xgb_wth["rmse"],
        "mae_gain":  xgb_ctx["mae"]  - xgb_wth["mae"],
        "r2_gain":   xgb_wth["nse"]  - xgb_ctx["nse"],
        "nse_gain":  xgb_wth["nse"]  - xgb_ctx["nse"],
    }])
    xgb_h_row = pd.DataFrame([{
        "model_name": "xgboost",
        "rmse_gain": xgb_wth["rmse"] - xgb_hyd["rmse"],
        "mae_gain":  xgb_wth["mae"]  - xgb_hyd["mae"],
        "r2_gain":   xgb_hyd["nse"]  - xgb_wth["nse"],
        "nse_gain":  xgb_hyd["nse"]  - xgb_wth["nse"],
    }])

    weather = pd.concat([weather, xgb_w_row], ignore_index=True)
    hydro   = pd.concat([hydro,   xgb_h_row], ignore_index=True)
    return weather, hydro


def load_ablation_by_horizon() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load per-horizon ablation CSVs and inject XGBoost rows."""
    weather = pd.read_csv(
        SEQ_DIR / "ablation" / "weather_ablation" / "weather_effect_by_horizon.csv"
    )
    hydro = pd.read_csv(
        SEQ_DIR / "ablation" / "hydro_ablation" / "hydro_effect_by_horizon.csv"
    )

    # Per-horizon XGBoost
    ctx_h  = _micro_per_horizon(ADV_DIR / "xgboost_context_w14_h3" / "metrics_summary.csv")
    wth_h  = _micro_per_horizon(ADV_DIR / "xgboost_weather_w14_h3" / "metrics_summary.csv")
    hyd_h  = _micro_per_horizon(SEQ_DIR / "xgboost_hydro_weather_w14_h3" / "metrics_summary.csv")

    for h in [1, 2, 3]:
        weather = pd.concat([weather, pd.DataFrame([{
            "model_name": "xgboost", "horizon": h,
            "nse_gain": wth_h.loc[h, "nse"] - ctx_h.loc[h, "nse"],
        }])], ignore_index=True)
        hydro = pd.concat([hydro, pd.DataFrame([{
            "model_name": "xgboost", "horizon": h,
            "nse_gain": hyd_h.loc[h, "nse"] - wth_h.loc[h, "nse"],
        }])], ignore_index=True)

    return weather, hydro


# ── Figure 1: avg NSE and RMSE by model × stage ───────────────────────────────

def plot_fig1(summary: pd.DataFrame) -> None:
    models = MODELS_ORDERED
    stages = ["Context", "Weather", "Hydro"]
    n_m = len(models)
    n_s = len(stages)
    width = 0.22
    x = np.arange(n_m)
    offsets = np.array([-1, 0, 1]) * width

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.9))
    fig.subplots_adjust(wspace=0.38)

    for ax, metric, ylabel, panel in zip(
        axes,
        ["nse", "rmse"],
        ["Average NSE", r"Average RMSE (m$^3$/s)"],
        ["(a)", "(b)"],
    ):
        for i, (stage, offset) in enumerate(zip(stages, offsets)):
            vals = []
            for m in models:
                row = summary[(summary["model"] == m) & (summary["stage"] == stage)]
                vals.append(float(row[metric].values[0]) if len(row) else np.nan)

            bars = ax.bar(
                x + offset, vals, width,
                color=STAGE_COLORS[stage],
                label=STAGE_LABELS[stage],
                edgecolor="white", linewidth=0.4,
                zorder=3,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(models, rotation=35, ha="right", fontsize=7)
        ax.set_ylabel(ylabel)
        ax.yaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
        ax.text(-0.08, 1.04, panel, transform=ax.transAxes,
                fontsize=9, fontweight="bold", va="top")

        if metric == "nse":
            ax.set_ylim(0.88, 0.975)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
        else:
            ax.set_ylim(80, 165)
            ax.yaxis.set_major_locator(mticker.MultipleLocator(20))

    axes[0].legend(loc="lower right", ncol=1, frameon=True)

    fig.savefig(OUT_DIR / "fig1_nse_rmse_by_stage.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig1_nse_rmse_by_stage.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig1")


# ── Figure 2: avg NSE ablation bars ──────────────────────────────────────────

DISPLAY_NAME = {
    "ann": "ANN", "nhits": "N-HiTS", "tft": "TFT", "lstm": "LSTM",
    "patchtst": "PatchTST", "xlstm": "xLSTM", "mamba": "Mamba",
    "xgboost": "XGBoost", "hybrid": "Hybrid", "flownet": "FlowNet",
}

def _ablation_panel(ax, df: pd.DataFrame, title: str, show_xgboost: bool = True) -> None:
    key_col = "model_name"
    keep = [k for k in MODEL_KEY.values() if show_xgboost or k != "xgboost"]
    df = df[df[key_col].isin(keep)].copy()
    df["nse_gain_k"] = df["nse_gain"] * 1e3          # convert to 10⁻³ NSE
    df = df.sort_values("nse_gain_k", ascending=True)

    labels = [DISPLAY_NAME.get(r, r) for r in df[key_col]]
    vals   = df["nse_gain_k"].values
    colors = [GAIN_POS if v >= 0 else GAIN_NEG for v in vals]

    y = np.arange(len(vals))
    bars = ax.barh(y, vals, color=colors, edgecolor="white", linewidth=0.4,
                   height=0.6, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.axvline(0, color="0.3", linewidth=0.8, zorder=4)
    ax.xaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlabel(r"$\Delta$NSE ($\times$10$^{-3}$)")
    ax.set_title(title, pad=4)

    # value labels inside/outside bars
    for bar, v in zip(bars, vals):
        ha = "right" if v >= 0 else "left"
        offset = -0.15 if v >= 0 else 0.15
        ax.text(v + offset, bar.get_y() + bar.get_height() / 2,
                f"{v:+.1f}", va="center", ha=ha, fontsize=6.5)


def plot_fig2(weather_avg: pd.DataFrame, hydro_avg: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    fig.subplots_adjust(wspace=0.5)

    _ablation_panel(axes[0], weather_avg,
                    "Stage 1 → Stage 2\n(+ERA5 atmospheric)",
                    show_xgboost=True)
    _ablation_panel(axes[1], hydro_avg,
                    "Stage 2 → Stage 3\n(+ERA5-Land soil)",
                    show_xgboost=True)

    axes[0].text(-0.12, 1.04, "(a)", transform=axes[0].transAxes,
                 fontsize=9, fontweight="bold", va="top")
    axes[1].text(-0.12, 1.04, "(b)", transform=axes[1].transAxes,
                 fontsize=9, fontweight="bold", va="top")

    fig.savefig(OUT_DIR / "fig2_ablation_average.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig2_ablation_average.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig2")


# ── Figure 3: per-horizon ablation lines ─────────────────────────────────────

LINE_STYLES = {
    "TFT":      ("solid",   "o",  "#4E79A7"),
    "N-HiTS":   ("solid",   "s",  "#F28E2B"),
    "ANN":      ("solid",   "^",  "#59A14F"),
    "LSTM":     ("solid",   "D",  "#E15759"),
    "PatchTST": ("dashed",  "o",  "#76B7B2"),
    "xLSTM":    ("dashed",  "s",  "#EDC948"),
    "Mamba":    ("dashed",  "^",  "#B07AA1"),
    "XGBoost":  ("dotted",  "P",  "#FF9DA7"),
}

def _horizon_panel(ax, df: pd.DataFrame, title: str) -> None:
    horizons = [1, 2, 3]
    for display, key in MODEL_KEY.items():
        sub = df[df["model_name"] == key].copy()
        if sub.empty:
            continue
        sub = sub.sort_values("horizon")
        vals = sub["nse_gain"].values * 1e3
        ls, mk, col = LINE_STYLES.get(display, ("solid", "o", "gray"))
        ax.plot(horizons, vals, linestyle=ls, marker=mk, color=col,
                linewidth=1.2, markersize=4, label=display, zorder=3)

    ax.axhline(0, color="0.4", linewidth=0.7, linestyle="--", zorder=2)
    ax.set_xticks(horizons)
    ax.set_xticklabels(["h+1", "h+2", "h+3"])
    ax.set_xlabel("Forecast horizon")
    ax.set_ylabel(r"$\Delta$NSE ($\times$10$^{-3}$)")
    ax.set_title(title, pad=4)
    ax.yaxis.grid(True, zorder=0)
    ax.set_axisbelow(True)


def plot_fig3(weather_hz: pd.DataFrame, hydro_hz: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    fig.subplots_adjust(wspace=0.38)

    _horizon_panel(axes[0], weather_hz,
                   "Stage 1 → Stage 2\n(+ERA5 atmospheric)")
    _horizon_panel(axes[1], hydro_hz,
                   "Stage 2 → Stage 3\n(+ERA5-Land soil)")

    axes[0].text(-0.10, 1.04, "(a)", transform=axes[0].transAxes,
                 fontsize=9, fontweight="bold", va="top")
    axes[1].text(-0.10, 1.04, "(b)", transform=axes[1].transAxes,
                 fontsize=9, fontweight="bold", va="top")

    # single shared legend below both panels
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, -0.12), frameon=True,
               fontsize=7, handlelength=2.0)

    fig.savefig(OUT_DIR / "fig3_ablation_by_horizon.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / "fig3_ablation_by_horizon.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved fig3")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    print("Loading data…")
    summary       = load_summary_data()
    weather_avg, hydro_avg = load_ablation_average()
    weather_hz,  hydro_hz  = load_ablation_by_horizon()

    print("Plotting fig1…")
    plot_fig1(summary)

    print("Plotting fig2…")
    plot_fig2(weather_avg, hydro_avg)

    print("Plotting fig3…")
    plot_fig3(weather_hz, hydro_hz)

    print(f"\nAll figures saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
