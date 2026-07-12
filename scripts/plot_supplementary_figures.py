"""Supplementary publication figures for the w14 discharge forecasting benchmark.

Produces nine figures:
  fig_s1_spatial_nse_map.pdf/png        — per-station NSE map (best vs worst model)
  fig_s2_station_nse_boxplots.pdf/png   — per-station NSE distribution by model
  fig_s3_timeseries.pdf/png             — actual vs predicted time-series (3 stations)
  fig_s4_flow_regime_error.pdf/png      — RMSE by low/medium/high discharge regime
  fig_s5_rmsse_heatmap.pdf/png          — RMSSE heatmap model × horizon
  fig_s6_feature_importance.pdf/png     — XGBoost feature importance across stages
  fig_s7_bias_by_stage.pdf/png          — signed bias per model and stage
  fig_s8_nse_gain_map.pdf/png           — per-station NSE gain (weather − context)
  fig_s9_loss_curves.pdf/png            — training/validation loss curves

Usage
-----
    python3 scripts/plot_supplementary_figures.py
"""

from __future__ import annotations

import json
import pathlib
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import matplotlib.cm as cm
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = PROJECT_ROOT / "artifacts"
SEQ_DIR      = ARTIFACT_DIR / "advanced_seq"
ADV_DIR      = ARTIFACT_DIR / "advanced"
OUT_DIR      = ARTIFACT_DIR / "journal_figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "serif",
    "font.serif":  ["Times New Roman", "DejaVu Serif"],
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
    "pdf.fonttype": 42,
    "ps.fonttype":  42,
})

MODELS_8 = ["TFT", "N-HiTS", "ANN", "LSTM", "PatchTST", "xLSTM", "Mamba", "XGBoost"]
MODEL_KEY = {
    "TFT": "tft", "N-HiTS": "nhits", "ANN": "ann", "LSTM": "lstm",
    "PatchTST": "patchtst", "xLSTM": "xlstm", "Mamba": "mamba", "XGBoost": "xgboost",
}
STAGE_COLORS = {"Context": "#4E79A7", "Weather": "#F28E2B", "Hydro": "#59A14F"}
MODEL_COLORS = {
    "TFT": "#4E79A7", "N-HiTS": "#F28E2B", "ANN": "#59A14F", "LSTM": "#E15759",
    "PatchTST": "#76B7B2", "xLSTM": "#EDC948", "Mamba": "#B07AA1", "XGBoost": "#FF9DA7",
}

def _metrics_path(key: str, stage: str) -> Path | None:
    regime_map = {"Context": "context", "Weather": "weather", "Hydro": "hydro_weather"}
    reg = regime_map[stage]
    for base in [SEQ_DIR, ADV_DIR]:
        p = base / f"{key}_{reg}_w14_h3" / "metrics_summary.csv"
        if p.exists():
            return p
    return None

def _per_station_path(key: str, stage: str) -> Path | None:
    regime_map = {"Context": "context", "Weather": "weather", "Hydro": "hydro_weather"}
    reg = regime_map[stage]
    for base in [SEQ_DIR, ADV_DIR]:
        p = base / f"{key}_{reg}_w14_h3" / "metrics_by_station.csv"
        if p.exists():
            return p
    return None

def _predictions_path(key: str, stage: str) -> Path | None:
    regime_map = {"Context": "context", "Weather": "weather", "Hydro": "hydro_weather"}
    reg = regime_map[stage]
    for base in [SEQ_DIR, ADV_DIR]:
        p = base / f"{key}_{reg}_w14_h3" / "predictions.parquet"
        if p.exists():
            return p
    return None

def _save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT_DIR / f"{name}.pdf", dpi=300, bbox_inches="tight")
    fig.savefig(OUT_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {name}")


# ─────────────────────────────────────────────────────────────────────────────
# FIG S1 — Spatial NSE map
# ─────────────────────────────────────────────────────────────────────────────
def plot_s1_spatial_map() -> None:
    meta = (pd.read_csv(PROJECT_ROOT / "data/processed/reanalysis_station_metadata.csv")
            .drop_duplicates("unique_id")
            .set_index("unique_id"))

    def _station_nse(key: str, stage: str, horizon: int = 1) -> pd.Series:
        p = _per_station_path(key, stage)
        if p is None:
            return pd.Series(dtype=float)
        df = pd.read_csv(p)
        return (df[(df.split == "test") & (df.horizon == horizon)]
                .set_index("unique_id")["nse"])

    tft_nse   = _station_nse("tft",   "Weather")
    mamba_nse = _station_nse("mamba", "Hydro")

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    fig.subplots_adjust(wspace=0.35)

    cmap = plt.cm.RdYlGn
    vmin, vmax = 0.3, 1.0

    for ax, nse, title, panel in zip(
        axes,
        [tft_nse, mamba_nse],
        ["TFT — Stage 2 (Weather)", "Mamba — Stage 3 (Hydro)"],
        ["(a)", "(b)"],
    ):
        common = meta.index.intersection(nse.index)
        lons = meta.loc[common, "requested_longitude"]
        lats = meta.loc[common, "requested_latitude"]
        vals = nse.loc[common]

        sc = ax.scatter(lons, lats, c=vals, cmap=cmap, vmin=vmin, vmax=vmax,
                        s=110, edgecolors="0.3", linewidths=0.5, zorder=3)

        # annotate station ids briefly
        for uid in common:
            ax.annotate(str(uid)[-4:],
                        (meta.loc[uid, "requested_longitude"],
                         meta.loc[uid, "requested_latitude"]),
                        fontsize=4.5, ha="center", va="bottom",
                        xytext=(0, 5), textcoords="offset points", color="0.3")

        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        ax.set_title(title, pad=4)
        ax.text(-0.10, 1.04, panel, transform=ax.transAxes,
                fontsize=9, fontweight="bold", va="top")
        ax.yaxis.grid(True, zorder=0)
        ax.xaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin, vmax))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.025, pad=0.04)
    cb.set_label("NSE (h+1, test set)")

    _save(fig, "fig_s1_spatial_nse_map")


# ─────────────────────────────────────────────────────────────────────────────
# FIG S2 — Station-level NSE boxplots
# ─────────────────────────────────────────────────────────────────────────────
def plot_s2_station_boxplots() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0), sharey=True)
    fig.subplots_adjust(wspace=0.12)

    for ax, (stage, panel) in zip(axes, [("Context","(a)"),("Weather","(b)"),("Hydro","(c)")]):
        data, labels = [], []
        for display in MODELS_8:
            key = MODEL_KEY[display]
            p = _per_station_path(key, stage)
            if p is None:
                continue
            df = pd.read_csv(p)
            vals = df[(df.split == "test") & (df.horizon == 1)]["nse"].dropna().values
            if len(vals):
                data.append(vals)
                labels.append(display)

        bp = ax.boxplot(data, patch_artist=True, vert=True,
                        medianprops=dict(color="black", linewidth=1.2),
                        flierprops=dict(marker="o", markersize=2.5,
                                        markerfacecolor="0.5", markeredgecolor="0.5"),
                        whiskerprops=dict(linewidth=0.8),
                        capprops=dict(linewidth=0.8),
                        boxprops=dict(linewidth=0.8))

        colors = [MODEL_COLORS.get(lbl, "#aaaaaa") for lbl in labels]
        for patch, col in zip(bp["boxes"], colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.75)

        ax.axhline(0, color="0.4", linewidth=0.6, linestyle="--")
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=6.5)
        ax.set_title(f"Stage {['1','2','3'][['Context','Weather','Hydro'].index(stage)]}: {stage}", pad=4)
        ax.text(-0.14, 1.04, panel, transform=ax.transAxes,
                fontsize=9, fontweight="bold", va="top")
        ax.yaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Per-station NSE (h+1, test set)")
    _save(fig, "fig_s2_station_nse_boxplots")


# ─────────────────────────────────────────────────────────────────────────────
# FIG S3 — Actual vs predicted time-series
# ─────────────────────────────────────────────────────────────────────────────
def plot_s3_timeseries() -> None:
    # large river, medium, small/difficult
    STATION_LABELS = {
        "6142200": "Stn 6142200 (large, mean ≈ 1913 m³/s)",
        "6142150": "Stn 6142150 (medium, mean ≈ 77 m³/s)",
        "6144490": "Stn 6144490 (small, lowest NSE)",
    }
    HORIZON_STYLES = {1: ("solid", 1.4), 2: ("dashed", 1.0), 3: ("dotted", 1.0)}
    HORIZON_COLORS = {1: "#d62728", 2: "#ff7f0e", 3: "#9467bd"}

    key  = "tft"
    p    = _predictions_path(key, "Weather")
    preds = pd.read_parquet(p)
    preds = preds[preds.split == "test"].copy()
    preds["target_ds"] = pd.to_datetime(preds["target_ds"])

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 6.5))
    fig.subplots_adjust(hspace=0.45)

    for ax, (sid, slabel), panel in zip(axes, STATION_LABELS.items(), ["(a)","(b)","(c)"]):
        sub = preds[preds.unique_id == sid].copy()
        # show 3 years of test data to keep readable
        t_min = sub["target_ds"].min()
        t_max = t_min + pd.DateOffset(years=3)
        sub = sub[sub["target_ds"] <= t_max]

        # observed (use h=1 to avoid duplicates)
        obs = sub[sub.horizon == 1].sort_values("target_ds")
        ax.fill_between(obs["target_ds"], obs["y_true"],
                        alpha=0.18, color="0.3", zorder=1)
        ax.plot(obs["target_ds"], obs["y_true"],
                color="0.2", linewidth=0.6, label="Observed", zorder=2)

        for h in [1, 2, 3]:
            sub_h = sub[sub.horizon == h].sort_values("target_ds")
            ls, lw = HORIZON_STYLES[h]
            ax.plot(sub_h["target_ds"], sub_h["y_pred"],
                    color=HORIZON_COLORS[h], linewidth=lw, linestyle=ls,
                    label=f"h+{h}", zorder=3, alpha=0.9)

        ax.set_ylabel(r"Discharge (m$^3$/s)")
        ax.set_title(slabel, pad=3, fontsize=7.5)
        ax.text(-0.07, 1.06, panel, transform=ax.transAxes,
                fontsize=9, fontweight="bold", va="top")
        ax.yaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
        if ax == axes[0]:
            ax.legend(loc="upper right", ncol=4, fontsize=6.5)

    axes[-1].set_xlabel("Date")
    fig.suptitle("TFT — Stage 2 (Weather): test-set predictions", y=1.01, fontsize=9)
    _save(fig, "fig_s3_timeseries")


# ─────────────────────────────────────────────────────────────────────────────
# FIG S4 — Error by flow regime
# ─────────────────────────────────────────────────────────────────────────────
def plot_s4_flow_regime() -> None:
    # Load all model Weather predictions at h+1
    regime_rows = []
    for display in MODELS_8:
        key = MODEL_KEY[display]
        p = _predictions_path(key, "Weather")
        if p is None:
            continue
        df = pd.read_parquet(p)
        df = df[(df.split == "test") & (df.horizon == 1)].copy()
        q33 = df["y_true"].quantile(0.33)
        q67 = df["y_true"].quantile(0.67)
        df["regime"] = pd.cut(df["y_true"], bins=[-np.inf, q33, q67, np.inf],
                              labels=["Low", "Medium", "High"])
        for regime, grp in df.groupby("regime"):
            rmse = np.sqrt(np.mean((grp["y_pred"] - grp["y_true"])**2))
            nse_num = np.sum((grp["y_true"] - grp["y_pred"])**2)
            nse_den = np.sum((grp["y_true"] - grp["y_true"].mean())**2)
            nse  = 1 - nse_num / nse_den if nse_den > 0 else np.nan
            regime_rows.append({"model": display, "regime": regime,
                                 "rmse": rmse, "nse": nse})

    df_regime = pd.DataFrame(regime_rows)
    regimes = ["Low", "Medium", "High"]
    x = np.arange(len(MODELS_8))
    width = 0.25
    offsets = np.array([-1, 0, 1]) * width
    regime_colors = {"Low": "#74add1", "Medium": "#fdae61", "High": "#d73027"}

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    fig.subplots_adjust(wspace=0.38)

    for ax, metric, ylabel, panel in zip(
        axes, ["rmse", "nse"],
        [r"RMSE (m$^3$/s)", "NSE"],
        ["(a)", "(b)"]
    ):
        for regime, offset in zip(regimes, offsets):
            vals = []
            for m in MODELS_8:
                row = df_regime[(df_regime.model == m) & (df_regime.regime == regime)]
                vals.append(float(row[metric].values[0]) if len(row) else np.nan)
            ax.bar(x + offset, vals, width, label=regime,
                   color=regime_colors[regime],
                   edgecolor="white", linewidth=0.4, zorder=3)

        ax.set_xticks(x)
        ax.set_xticklabels(MODELS_8, rotation=35, ha="right", fontsize=7)
        ax.set_ylabel(ylabel)
        ax.yaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
        ax.text(-0.10, 1.04, panel, transform=ax.transAxes,
                fontsize=9, fontweight="bold", va="top")
        if metric == "nse":
            ax.axhline(0, color="0.3", linewidth=0.6, linestyle="--")

    axes[0].legend(title="Flow regime", fontsize=6.5, title_fontsize=6.5)
    fig.suptitle("Stage 2 (Weather), h+1: performance by flow regime", y=1.02, fontsize=9)
    _save(fig, "fig_s4_flow_regime_error")


# ─────────────────────────────────────────────────────────────────────────────
# FIG S5 — RMSSE heatmap
# ─────────────────────────────────────────────────────────────────────────────
def plot_s5_rmsse_heatmap() -> None:
    horizons = [1, 2, 3]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.4), sharey=True)
    fig.subplots_adjust(wspace=0.08)

    for ax, stage, panel in zip(axes,
                                 ["Context", "Weather", "Hydro"],
                                 ["(a)", "(b)", "(c)"]):
        matrix = np.full((len(MODELS_8), len(horizons)), np.nan)
        for i, display in enumerate(MODELS_8):
            key = MODEL_KEY[display]
            p = _metrics_path(key, stage)
            if p is None:
                continue
            df = pd.read_csv(p)
            for j, h in enumerate(horizons):
                row = df[(df.split == "test") & (df.horizon == h)
                         & (df.aggregation == "micro")]
                if len(row):
                    matrix[i, j] = float(row["rmsse"].values[0])

        # diverging colormap centred on 1.0
        vmax = max(np.nanmax(np.abs(matrix - 1.0)) + 1.0, 1.5)
        vmin = 2.0 - vmax
        cmap = plt.cm.RdYlGn_r

        im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")

        # annotate cells
        for i in range(len(MODELS_8)):
            for j in range(len(horizons)):
                v = matrix[i, j]
                if not np.isnan(v):
                    text_col = "white" if abs(v - 1.0) > 0.35 else "black"
                    ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                            fontsize=6.5, color=text_col, fontweight="bold")

        ax.set_xticks(range(len(horizons)))
        ax.set_xticklabels(["h+1", "h+2", "h+3"])
        ax.set_yticks(range(len(MODELS_8)))
        ax.set_yticklabels(MODELS_8 if ax == axes[0] else [])
        title_map = {"Context": "Stage 1\n(Context)", "Weather": "Stage 2\n(Weather)",
                     "Hydro": "Stage 3\n(Hydro)"}
        ax.set_title(title_map[stage], pad=4)
        ax.text(-0.08 if ax == axes[0] else -0.04, 1.04, panel,
                transform=ax.transAxes, fontsize=9, fontweight="bold", va="top")

    # shared colorbar
    sm = plt.cm.ScalarMappable(cmap=plt.cm.RdYlGn_r,
                                norm=mcolors.Normalize(vmin=0.6, vmax=2.0))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.025, pad=0.04,
                      shrink=0.85)
    cb.set_label("RMSSE")
    cb.ax.axhline(1.0, color="black", linewidth=1.0)
    cb.ax.text(1.05, 1.0, "persistence", va="center", ha="left",
               fontsize=5.5, transform=cb.ax.transAxes)

    _save(fig, "fig_s5_rmsse_heatmap")


# ─────────────────────────────────────────────────────────────────────────────
# FIG S6 — XGBoost feature importance
# ─────────────────────────────────────────────────────────────────────────────
def _feature_group(name: str) -> str:
    if name in ("current_y",):
        return "Current discharge"
    if name.startswith("lag_") and name[4:].isdigit():
        return "Discharge lags"
    if name in ("lag_mean", "lag_std", "lag_min", "lag_max"):
        return "Rolling stats"
    if name.startswith("delta_"):
        return "Lag deltas"
    if name.startswith("era5_precipitation") or name.startswith("era5_rain"):
        return "ERA5 Precipitation"
    if name.startswith("era5_temperature") or name.startswith("era5_snowfall"):
        return "ERA5 Temp/Snow"
    if name.startswith("era5l_soil_moisture") or name.startswith("era5l_soil_temperature"):
        return "ERA5-Land Soil"
    if name.startswith("era5_"):
        return "ERA5 Other"
    if name.startswith("flow_context"):
        return "Upstream flow"
    if name == "station_id_feature":
        return "Station ID"
    return "Other"


def plot_s6_feature_importance() -> None:
    stage_paths = {
        "Context": {
            h: ADV_DIR / "xgboost_context_w14_h3" / f"h{h}" / "feature_importance.csv"
            for h in [1, 2, 3]
        },
        "Weather": {
            h: ADV_DIR / "xgboost_weather_w14_h3" / f"h{h}" / "feature_importance.csv"
            for h in [1, 2, 3]
        },
        "Hydro": {
            h: SEQ_DIR / "xgboost_hydro_weather_w14_h3" / f"h{h}" / "feature_importance.csv"
            for h in [1, 2, 3]
        },
    }

    GROUP_COLORS = {
        "Current discharge": "#1f77b4",
        "Discharge lags":    "#aec7e8",
        "Rolling stats":     "#6baed6",
        "Lag deltas":        "#c6dbef",
        "ERA5 Precipitation":"#e6550d",
        "ERA5 Temp/Snow":    "#fd8d3c",
        "ERA5-Land Soil":    "#fdae6b",
        "ERA5 Other":        "#fdd0a2",
        "Upstream flow":     "#31a354",
        "Station ID":        "#969696",
        "Other":             "#cccccc",
    }

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.4))
    fig.subplots_adjust(wspace=0.05)

    for ax, (stage, panels, panel) in zip(axes, [
        ("Context", stage_paths["Context"], "(a)"),
        ("Weather", stage_paths["Weather"], "(b)"),
        ("Hydro",   stage_paths["Hydro"],   "(c)"),
    ]):
        # average gain across h1/h2/h3
        dfs = []
        for h, p in panels.items():
            if p.exists():
                df = pd.read_csv(p)
                df["group"] = df["feature"].apply(_feature_group)
                dfs.append(df.groupby("group")["gain"].sum())
        if not dfs:
            continue
        total = pd.concat(dfs, axis=1).fillna(0).mean(axis=1)
        total = total / total.sum() * 100   # % share
        total = total.sort_values(ascending=True)

        colors = [GROUP_COLORS.get(g, "#cccccc") for g in total.index]
        bars = ax.barh(range(len(total)), total.values,
                       color=colors, edgecolor="white", linewidth=0.4, height=0.65)

        ax.set_yticks(range(len(total)))
        ax.set_yticklabels(total.index if ax == axes[0] else [],
                           fontsize=6.5)
        ax.set_xlabel("Feature group importance (%)")
        title_map = {"Context": "Stage 1\n(Context)", "Weather": "Stage 2\n(Weather)",
                     "Hydro": "Stage 3\n(Hydro)"}
        ax.set_title(title_map[stage], pad=4)
        ax.text(-0.06, 1.04, panel, transform=ax.transAxes,
                fontsize=9, fontweight="bold", va="top")
        ax.xaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)

        # value labels
        for bar, v in zip(bars, total.values):
            if v > 1.5:
                ax.text(v + 0.3, bar.get_y() + bar.get_height() / 2,
                        f"{v:.1f}%", va="center", fontsize=5.5)

    _save(fig, "fig_s6_feature_importance")


# ─────────────────────────────────────────────────────────────────────────────
# FIG S7 — Bias per model and stage
# ─────────────────────────────────────────────────────────────────────────────
def plot_s7_bias() -> None:
    stages = ["Context", "Weather", "Hydro"]
    x = np.arange(len(MODELS_8))
    width = 0.25
    offsets = np.array([-1, 0, 1]) * width

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0), sharey=False)
    fig.subplots_adjust(wspace=0.4)

    for ax, h, panel in zip(axes, [1, 2, 3], ["(a)", "(b)", "(c)"]):
        for stage, offset in zip(stages, offsets):
            vals = []
            for display in MODELS_8:
                key = MODEL_KEY[display]
                p = _metrics_path(key, stage)
                if p is None:
                    vals.append(np.nan)
                    continue
                df = pd.read_csv(p)
                row = df[(df.split == "test") & (df.horizon == h)
                         & (df.aggregation == "micro")]
                vals.append(float(row["bias"].values[0]) if len(row) else np.nan)

            colors = [STAGE_COLORS[stage] if not np.isnan(v) else "white"
                      for v in vals]
            ax.bar(x + offset, vals, width, color=STAGE_COLORS[stage],
                   label=f"Stage {'123'[stages.index(stage)]}: {stage}",
                   edgecolor="white", linewidth=0.4, zorder=3,
                   alpha=0.85)

        ax.axhline(0, color="0.3", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS_8, rotation=38, ha="right", fontsize=6.5)
        ax.set_ylabel(r"Mean bias (m$^3$/s)")
        ax.set_title(f"h+{h}", pad=4)
        ax.text(-0.14, 1.04, panel, transform=ax.transAxes,
                fontsize=9, fontweight="bold", va="top")
        ax.yaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)

    axes[0].legend(fontsize=6, ncol=1, loc="upper left")
    _save(fig, "fig_s7_bias_by_stage")


# ─────────────────────────────────────────────────────────────────────────────
# FIG S8 — Per-station NSE gain bubble map (Weather − Context)
# ─────────────────────────────────────────────────────────────────────────────
def plot_s8_nse_gain_map() -> None:
    meta = (pd.read_csv(PROJECT_ROOT / "data/processed/reanalysis_station_metadata.csv")
            .drop_duplicates("unique_id")
            .set_index("unique_id"))

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8))
    fig.subplots_adjust(wspace=0.35)

    panels = [
        ("TFT",  "tft",  "Weather",  "Context", "Stage 1 → 2: TFT"),
        ("N-HiTS","nhits","Weather", "Context", "Stage 1 → 2: N-HiTS"),
        ("Mamba", "mamba","Weather", "Context", "Stage 1 → 2: Mamba"),
    ]

    cmap = plt.cm.RdYlGn
    vmax = 0.04
    vmin = -vmax

    for ax, (display, key, stage_b, stage_a, title), panel in zip(
        axes, panels, ["(a)","(b)","(c)"]
    ):
        def _nse(stage):
            p = _per_station_path(key, stage)
            if p is None:
                return pd.Series(dtype=float)
            df = pd.read_csv(p)
            return df[(df.split == "test") & (df.horizon == 1)].set_index("unique_id")["nse"]

        nse_a = _nse(stage_a)
        nse_b = _nse(stage_b)
        gain  = nse_b - nse_a

        common = meta.index.intersection(gain.index)
        lons = meta.loc[common, "requested_longitude"]
        lats = meta.loc[common, "requested_latitude"]
        vals = gain.loc[common]

        sc = ax.scatter(lons, lats, c=vals, cmap=cmap,
                        vmin=vmin, vmax=vmax,
                        s=np.abs(vals) * 3000 + 30,
                        edgecolors="0.3", linewidths=0.4, zorder=3)

        ax.set_xlabel("Lon (°E)")
        ax.set_ylabel("Lat (°N)" if ax == axes[0] else "")
        ax.set_title(title, pad=3, fontsize=7.5)
        ax.text(-0.12, 1.04, panel, transform=ax.transAxes,
                fontsize=9, fontweight="bold", va="top")
        ax.yaxis.grid(True, zorder=0)
        ax.xaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=mcolors.Normalize(vmin, vmax))
    sm.set_array([])
    cb = fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.022, pad=0.03)
    cb.set_label(r"$\Delta$NSE (Weather $-$ Context)")

    _save(fig, "fig_s8_nse_gain_map")


# ─────────────────────────────────────────────────────────────────────────────
# FIG S9 — Training / validation loss curves
# ─────────────────────────────────────────────────────────────────────────────
def plot_s9_loss_curves() -> None:
    SHOW = [("TFT",     "tft",     "Weather"),
            ("ANN",     "ann",     "Weather"),
            ("LSTM",    "lstm",    "Weather"),
            ("N-HiTS",  "nhits",   "Weather"),
            ("Mamba",   "mamba",   "Weather"),
            ("PatchTST","patchtst","Weather")]

    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.2))
    fig.subplots_adjust(hspace=0.5, wspace=0.38)

    for ax, (display, key, stage), panel in zip(
        axes.flat, SHOW,
        ["(a)","(b)","(c)","(d)","(e)","(f)"]
    ):
        regime_map = {"Context": "context", "Weather": "weather", "Hydro": "hydro_weather"}
        reg = regime_map[stage]
        p = SEQ_DIR / f"{key}_{reg}_w14_h3" / "loss_history.csv"
        if not p.exists():
            ax.set_visible(False)
            continue
        df = pd.read_csv(p)
        train = df[df.split == "train"]
        val   = df[df.split == "validation"]

        ax.plot(train.epoch, train.loss, color="#4E79A7",
                linewidth=1.2, label="Train")
        ax.plot(val.epoch, val.loss, color="#E15759",
                linewidth=1.2, linestyle="--", label="Validation")

        # mark best epoch (min val loss)
        best_idx = val.loss.idxmin()
        best_ep  = val.loc[best_idx, "epoch"]
        ax.axvline(best_ep, color="0.4", linewidth=0.7, linestyle=":")
        ax.set_title(f"{display} (best: ep {int(best_ep)})", pad=3, fontsize=7.5)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.text(-0.16, 1.06, panel, transform=ax.transAxes,
                fontsize=9, fontweight="bold", va="top")
        ax.yaxis.grid(True, zorder=0)
        ax.set_axisbelow(True)
        if ax == axes.flat[0]:
            ax.legend(fontsize=6.5, loc="upper right")

    _save(fig, "fig_s9_loss_curves")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    tasks = [
        ("S1 — Spatial NSE map",           plot_s1_spatial_map),
        ("S2 — Station NSE boxplots",       plot_s2_station_boxplots),
        ("S3 — Time-series predictions",    plot_s3_timeseries),
        ("S4 — Flow-regime error",          plot_s4_flow_regime),
        ("S5 — RMSSE heatmap",             plot_s5_rmsse_heatmap),
        ("S6 — Feature importance",         plot_s6_feature_importance),
        ("S7 — Bias by stage",              plot_s7_bias),
        ("S8 — NSE-gain map",               plot_s8_nse_gain_map),
        ("S9 — Loss curves",                plot_s9_loss_curves),
    ]
    for name, fn in tasks:
        print(f"Plotting {name}…")
        try:
            fn()
        except Exception as exc:
            print(f"  ERROR: {exc}")

    print(f"\nAll figures saved to: {OUT_DIR}")


if __name__ == "__main__":
    main()
