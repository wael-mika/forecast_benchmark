"""Comprehensive cross-model benchmark comparison: old (w14) vs new (w30) runs.

This script loads metrics from both artifact generations, merges them into a
single master frame, and produces 11 CSV tables, LaTeX versions of the key
tables, and 7 plots for scientific publication.

Generations
-----------
    w14 : artifacts/advanced_seq/*_w14_h3/  (neural)
          artifacts/advanced/*_w14_h3/      (XGBoost)
          14-day lookback window

    w30 : runs/w30_v2/*/
          30-day lookback window

Outputs (all under artifacts/comparison_all/)
---------------------------------------------
    table_01_full_test_micro.csv
    table_02_pivot_by_horizon.csv  +  _latex.tex
    table_03_avg_horizons.csv      +  _latex.tex
    table_04_ranking_{regime}.csv  +  _latex.tex   (context / weather / hydro)
    table_05_window_ablation.csv
    table_06_regime_ablation.csv
    table_07_best_per_metric.csv
    table_08_per_horizon_ranking.csv
    table_09_training_metadata.csv
    table_10_macro_vs_micro.csv
    table_11_all_metrics_full.csv
    plots/  (7 PNG files)
    manifest.json

Usage
-----
    .venv/Scripts/python scripts/compare_all_results.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.io import ensure_parent_dir, save_csv, save_json
from src.utils.logging import get_logger

# ---------------------------------------------------------------------------
# Registry of all known artifact directories
# Keys: (model_name, regime)   Values: relative path from PROJECT_ROOT
# ---------------------------------------------------------------------------

_REGIMES = ("context", "weather", "hydro")
_REGIME_DIR_SUFFIX = {"context": "context", "weather": "weather", "hydro": "hydro_weather"}

# Old generation (w14) -------------------------------------------------------
_OLD_NEURAL_MODELS = [
    "ann", "lstm", "nhits", "patchtst", "tft", "xlstm", "mamba", "hybrid", "flownet",
]
_OLD_XGBOOST_MODELS = ["xgboost"]

OLD_REGISTRY: dict[tuple[str, str], Path] = {}
for _model in _OLD_NEURAL_MODELS:
    for _regime in _REGIMES:
        _suffix = _REGIME_DIR_SUFFIX[_regime]
        OLD_REGISTRY[(_model, _regime)] = (
            PROJECT_ROOT / "artifacts" / "advanced_seq" / f"{_model}_{_suffix}_w14_h3"
        )
for _model in _OLD_XGBOOST_MODELS:
    for _regime in ("context", "weather"):  # only context + weather exist for old xgb
        _suffix = _REGIME_DIR_SUFFIX[_regime]
        OLD_REGISTRY[(_model, _regime)] = (
            PROJECT_ROOT / "artifacts" / "advanced" / f"{_model}_{_suffix}_w14_h3"
        )

# New generation (w30) -------------------------------------------------------
_NEW_NEURAL_MODELS = ["ann", "lstm", "nhits"]
_NEW_XGBOOST_MODELS = ["xgboost"]

NEW_REGISTRY: dict[tuple[str, str], Path] = {}
for _model in _NEW_NEURAL_MODELS:
    for _regime in _REGIMES:
        _suffix = _REGIME_DIR_SUFFIX[_regime]
        NEW_REGISTRY[(_model, _regime)] = (
            PROJECT_ROOT / "runs" / "w30_v2" / f"{_model}_{_suffix}"
        )
for _model in _NEW_XGBOOST_MODELS:
    for _regime in _REGIMES:
        _suffix = _REGIME_DIR_SUFFIX[_regime]
        NEW_REGISTRY[(_model, _regime)] = (
            PROJECT_ROOT / "runs" / "w30_v2" / f"{_model}_{_suffix}"
        )

# ---------------------------------------------------------------------------
# Metric columns
# ---------------------------------------------------------------------------
METRIC_COLS = ["bias", "mae", "mse", "rmse", "r2", "nse", "mape", "smape", "wape", "mase", "rmsse"]
KEY_METRICS = ["rmse", "nse", "r2", "mae", "mase", "smape"]

# Metrics where lower is better (for ranking/bolding)
LOWER_IS_BETTER = {"bias", "mae", "mse", "rmse", "mape", "smape", "wape", "mase", "rmsse"}
# Metrics where higher is better
HIGHER_IS_BETTER = {"r2", "nse"}

OUTPUT_DIR = PROJECT_ROOT / "artifacts" / "comparison_all"
PLOTS_DIR = OUTPUT_DIR / "plots"

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_training_summary(artifact_dir: Path) -> dict[str, Any]:
    path = artifact_dir / "training_summary.json"
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_all_runs(logger: Any) -> pd.DataFrame:
    """Load metrics_summary.csv from every known artifact directory.

    Returns a combined DataFrame tagged with generation, model, and regime.
    """
    frames: list[pd.DataFrame] = []

    registries = [
        ("w14", OLD_REGISTRY),
        ("w30", NEW_REGISTRY),
    ]

    for generation, registry in registries:
        for (model_name, regime), artifact_dir in registry.items():
            metrics_path = artifact_dir / "metrics_summary.csv"
            if not metrics_path.exists():
                logger.debug("Missing metrics for %s/%s [%s]: %s", generation, model_name, regime, metrics_path)
                continue

            df = pd.read_csv(metrics_path)
            df["generation"] = generation
            df["model"] = model_name
            df["regime"] = regime
            df["artifact_dir"] = str(artifact_dir)

            # Pull window_size and loss_name from training_summary if available
            summary = _load_training_summary(artifact_dir)
            df["window_size"] = summary.get("window_size", None)
            df["loss_name"] = summary.get("loss_name", None)
            df["best_epoch"] = summary.get("best_epoch", None)
            df["best_val_nse"] = summary.get("best_validation_macro_nse", None)

            frames.append(df)
            logger.info("Loaded %s [%s/%s] (%d rows)", generation, model_name, regime, len(df))

    if not frames:
        raise FileNotFoundError("No completed artifact directories were found.")

    combined = pd.concat(frames, ignore_index=True)
    logger.info("Total rows loaded: %d", len(combined))
    return combined


# ---------------------------------------------------------------------------
# Helper: test-set micro slice
# ---------------------------------------------------------------------------

def _test_micro(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[(df["split"] == "test") & (df["aggregation"] == "micro")].copy()


def _test_macro(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[(df["split"] == "test") & (df["aggregation"] == "macro")].copy()


# ---------------------------------------------------------------------------
# LaTeX helpers
# ---------------------------------------------------------------------------

def _bold_extremes(df: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
    """Return a string-typed DataFrame where the best value per column is bolded."""
    result = df.copy().astype(object)
    for col in metric_cols:
        if col not in df.columns:
            continue
        try:
            numeric = pd.to_numeric(df[col], errors="coerce")
        except Exception:
            continue
        if numeric.isna().all():
            continue
        if col in LOWER_IS_BETTER:
            best_idx = numeric.idxmin()
        else:
            best_idx = numeric.idxmax()
        result[col] = numeric.round(4).astype(str)
        result.at[best_idx, col] = r"\textbf{" + str(round(numeric[best_idx], 4)) + "}"
    return result


def _save_latex(df: pd.DataFrame, path: Path, caption: str, label: str, metric_cols: list[str]) -> None:
    """Write a LaTeX table to disk with best values bolded."""
    bolded = _bold_extremes(df, metric_cols)
    tex = bolded.to_latex(
        index=False,
        escape=False,
        caption=caption,
        label=label,
    )
    ensure_parent_dir(path)
    path.write_text(tex, encoding="utf-8")


# ---------------------------------------------------------------------------
# Table generators
# ---------------------------------------------------------------------------

def make_table_01(df: pd.DataFrame, output_dir: Path) -> Path:
    """T1: Full master table — test / micro / all horizons."""
    t = _test_micro(df).sort_values(["generation", "model", "regime", "horizon"], kind="stable")
    cols = ["generation", "model", "regime", "horizon"] + METRIC_COLS
    available = [c for c in cols if c in t.columns]
    out = output_dir / "table_01_full_test_micro.csv"
    save_csv(t[available], out)
    return out


def make_table_02(df: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    """T2: Wide pivot — one row per (model, regime, generation), h1/h2/h3 columns."""
    t = _test_micro(df)
    pivot_metrics = ["rmse", "nse", "r2", "mae"]
    rows = []
    for (gen, model, regime), grp in t.groupby(["generation", "model", "regime"], dropna=False):
        row: dict[str, Any] = {"generation": gen, "model": model, "regime": regime}
        for h in [1, 2, 3]:
            hgrp = grp.loc[grp["horizon"] == h]
            for m in pivot_metrics:
                key = f"h{h}_{m}"
                row[key] = float(hgrp[m].iloc[0]) if len(hgrp) > 0 and m in hgrp.columns else float("nan")
        rows.append(row)

    pivot_df = pd.DataFrame(rows).sort_values(["generation", "regime", "model"], kind="stable")
    out = output_dir / "table_02_pivot_by_horizon.csv"
    save_csv(pivot_df, out)

    metric_cols = [f"h{h}_{m}" for h in [1, 2, 3] for m in pivot_metrics]
    tex_out = output_dir / "table_02_pivot_by_horizon_latex.tex"
    _save_latex(
        pivot_df.round(4),
        tex_out,
        caption="Per-horizon test metrics (micro aggregation) for all models and data regimes.",
        label="tab:pivot_by_horizon",
        metric_cols=metric_cols,
    )
    return out, tex_out


def make_table_03(df: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    """T3: Average across h1+h2+h3 per (model, regime, generation)."""
    t = _test_micro(df)
    agg_df = (
        t.groupby(["generation", "model", "regime"], dropna=False)[KEY_METRICS]
        .mean()
        .reset_index()
        .rename(columns={m: f"avg_{m}" for m in KEY_METRICS})
    )
    agg_df = agg_df.sort_values(["generation", "regime", "avg_nse"], ascending=[True, True, False], kind="stable")
    out = output_dir / "table_03_avg_horizons.csv"
    save_csv(agg_df, out)

    tex_out = output_dir / "table_03_avg_horizons_latex.tex"
    _save_latex(
        agg_df.round(4),
        tex_out,
        caption="Average test metrics across forecast horizons h1--h3 (micro aggregation).",
        label="tab:avg_horizons",
        metric_cols=[f"avg_{m}" for m in KEY_METRICS],
    )
    return out, tex_out


def make_table_04(df: pd.DataFrame, output_dir: Path) -> list[Path]:
    """T4: Per-regime ranking tables sorted by avg NSE."""
    t = _test_micro(df)
    agg_df = (
        t.groupby(["generation", "model", "regime"], dropna=False)[KEY_METRICS]
        .mean()
        .reset_index()
    )
    paths: list[Path] = []
    for regime in ("context", "weather", "hydro"):
        rdf = agg_df.loc[agg_df["regime"] == regime].sort_values("nse", ascending=False, kind="stable").copy()
        rdf.insert(0, "rank", range(1, len(rdf) + 1))
        out = output_dir / f"table_04_ranking_{regime}.csv"
        save_csv(rdf, out)
        paths.append(out)

        tex_out = output_dir / f"table_04_ranking_{regime}_latex.tex"
        _save_latex(
            rdf.round(4),
            tex_out,
            caption=f"Model ranking by average NSE on the {regime} regime (test set, micro).",
            label=f"tab:ranking_{regime}",
            metric_cols=KEY_METRICS,
        )
        paths.append(tex_out)
    return paths


def make_table_05(df: pd.DataFrame, output_dir: Path) -> Path:
    """T5: Window ablation — w14 vs w30 per-horizon RMSE/NSE gains."""
    t = _test_micro(df)
    # Only include models with BOTH generations
    models_w14 = set(t.loc[t["generation"] == "w14", "model"].unique())
    models_w30 = set(t.loc[t["generation"] == "w30", "model"].unique())
    common_models = models_w14 & models_w30

    rows = []
    for model in sorted(common_models):
        for regime in _REGIMES:
            w14 = t.loc[(t["generation"] == "w14") & (t["model"] == model) & (t["regime"] == regime)]
            w30 = t.loc[(t["generation"] == "w30") & (t["model"] == model) & (t["regime"] == regime)]
            if w14.empty or w30.empty:
                continue
            row: dict[str, Any] = {"model": model, "regime": regime}
            for h in [1, 2, 3]:
                w14h = w14.loc[w14["horizon"] == h]
                w30h = w30.loc[w30["horizon"] == h]
                for m in ["rmse", "nse", "r2", "mae"]:
                    if len(w14h) > 0 and len(w30h) > 0 and m in w14h.columns:
                        v14 = float(w14h[m].iloc[0])
                        v30 = float(w30h[m].iloc[0])
                        if m in LOWER_IS_BETTER:
                            row[f"h{h}_{m}_gain"] = round(v14 - v30, 6)  # positive = w30 better
                        else:
                            row[f"h{h}_{m}_gain"] = round(v30 - v14, 6)  # positive = w30 better
            # Average gains across horizons
            for m in ["rmse", "nse", "r2", "mae"]:
                horizon_gains = [row.get(f"h{h}_{m}_gain") for h in [1, 2, 3] if f"h{h}_{m}_gain" in row]
                row[f"avg_{m}_gain"] = round(float(np.mean(horizon_gains)), 6) if horizon_gains else float("nan")
            rows.append(row)

    ablation_df = pd.DataFrame(rows).sort_values(["model", "regime"], kind="stable")
    out = output_dir / "table_05_window_ablation.csv"
    save_csv(ablation_df, out)
    return out


def make_table_06(df: pd.DataFrame, output_dir: Path) -> Path:
    """T6: Feature regime ablation — context → weather → hydro gains per (model, generation)."""
    t = _test_micro(df)
    # Average over horizons first
    agg = (
        t.groupby(["generation", "model", "regime"], dropna=False)[KEY_METRICS]
        .mean()
        .reset_index()
    )

    rows = []
    for (gen, model), grp in agg.groupby(["generation", "model"], dropna=False):
        ctx = grp.loc[grp["regime"] == "context"]
        wth = grp.loc[grp["regime"] == "weather"]
        hyd = grp.loc[grp["regime"] == "hydro"]
        if ctx.empty:
            continue
        row: dict[str, Any] = {"generation": gen, "model": model}
        for m in KEY_METRICS:
            if m not in ctx.columns:
                continue
            v_ctx = float(ctx[m].iloc[0])
            row[f"ctx_{m}"] = round(v_ctx, 6)
            if not wth.empty:
                v_wth = float(wth[m].iloc[0])
                row[f"wth_{m}"] = round(v_wth, 6)
                if m in LOWER_IS_BETTER:
                    row[f"ctx_to_wth_{m}_gain"] = round(v_ctx - v_wth, 6)
                else:
                    row[f"ctx_to_wth_{m}_gain"] = round(v_wth - v_ctx, 6)
            if not hyd.empty:
                v_hyd = float(hyd[m].iloc[0])
                row[f"hyd_{m}"] = round(v_hyd, 6)
                if not wth.empty and f"wth_{m}" in row:
                    if m in LOWER_IS_BETTER:
                        row[f"wth_to_hyd_{m}_gain"] = round(row[f"wth_{m}"] - v_hyd, 6)
                    else:
                        row[f"wth_to_hyd_{m}_gain"] = round(v_hyd - row[f"wth_{m}"], 6)
                if m in LOWER_IS_BETTER:
                    row[f"ctx_to_hyd_{m}_gain"] = round(v_ctx - v_hyd, 6)
                else:
                    row[f"ctx_to_hyd_{m}_gain"] = round(v_hyd - v_ctx, 6)
        rows.append(row)

    ablation_df = pd.DataFrame(rows).sort_values(["generation", "model"], kind="stable")
    out = output_dir / "table_06_regime_ablation.csv"
    save_csv(ablation_df, out)
    return out


def make_table_07(df: pd.DataFrame, output_dir: Path) -> Path:
    """T7: Best model per metric × regime × generation."""
    t = _test_micro(df)
    agg = (
        t.groupby(["generation", "model", "regime"], dropna=False)[METRIC_COLS]
        .mean()
        .reset_index()
    )

    rows = []
    for (gen, regime) in agg.groupby(["generation", "regime"], dropna=False).groups:
        grp = agg.loc[(agg["generation"] == gen) & (agg["regime"] == regime)]
        if grp.empty:
            continue
        for m in METRIC_COLS:
            if m not in grp.columns:
                continue
            numeric = pd.to_numeric(grp[m], errors="coerce")
            if numeric.isna().all():
                continue
            if m in LOWER_IS_BETTER:
                sorted_grp = grp.loc[numeric.notna()].iloc[numeric.dropna().argsort().values]
            else:
                sorted_grp = grp.loc[numeric.notna()].iloc[numeric.dropna().argsort().values[::-1]]
            best = sorted_grp.iloc[0]
            second = sorted_grp.iloc[1] if len(sorted_grp) > 1 else None
            row: dict[str, Any] = {
                "generation": gen,
                "regime": regime,
                "metric": m,
                "best_model": best["model"],
                "best_value": round(float(best[m]), 6),
                "second_model": second["model"] if second is not None else None,
                "second_value": round(float(second[m]), 6) if second is not None else None,
            }
            rows.append(row)

    best_df = pd.DataFrame(rows).sort_values(["generation", "regime", "metric"], kind="stable")
    out = output_dir / "table_07_best_per_metric.csv"
    save_csv(best_df, out)
    return out


def make_table_08(df: pd.DataFrame, output_dir: Path) -> Path:
    """T8: Per-horizon ranking within each (regime, generation)."""
    t = _test_micro(df)
    rows = []
    for (gen, regime, horizon), grp in t.groupby(["generation", "regime", "horizon"], dropna=False):
        if grp.empty:
            continue
        for m in ["rmse", "nse", "r2", "mae"]:
            if m not in grp.columns:
                continue
            numeric = pd.to_numeric(grp[m], errors="coerce")
            if numeric.isna().all():
                continue
            if m in LOWER_IS_BETTER:
                ranked = grp.loc[numeric.notna()].assign(_v=numeric.dropna()).sort_values("_v", ascending=True)
            else:
                ranked = grp.loc[numeric.notna()].assign(_v=numeric.dropna()).sort_values("_v", ascending=False)
            for rank_pos, (_, r_row) in enumerate(ranked.iterrows(), start=1):
                rows.append({
                    "generation": gen,
                    "regime": regime,
                    "horizon": horizon,
                    "metric": m,
                    "rank": rank_pos,
                    "model": r_row["model"],
                    "value": round(float(r_row[m]), 6),
                })

    rank_df = pd.DataFrame(rows).sort_values(
        ["generation", "regime", "horizon", "metric", "rank"], kind="stable"
    )
    out = output_dir / "table_08_per_horizon_ranking.csv"
    save_csv(rank_df, out)
    return out


def make_table_09(df: pd.DataFrame, output_dir: Path) -> Path:
    """T9: Training metadata per run."""
    meta_cols = ["generation", "model", "regime", "window_size", "loss_name", "best_epoch", "best_val_nse"]
    available = [c for c in meta_cols if c in df.columns]
    meta_df = (
        df[available]
        .drop_duplicates(subset=["generation", "model", "regime"])
        .sort_values(["generation", "model", "regime"], kind="stable")
    )
    out = output_dir / "table_09_training_metadata.csv"
    save_csv(meta_df, out)
    return out


def make_table_10(df: pd.DataFrame, output_dir: Path) -> Path:
    """T10: Macro vs micro RMSE/NSE side-by-side on test split."""
    micro = _test_micro(df).groupby(["generation", "model", "regime"], dropna=False)[["rmse", "nse", "r2", "mae"]].mean().reset_index()
    macro = _test_macro(df).groupby(["generation", "model", "regime"], dropna=False)[["rmse", "nse", "r2", "mae"]].mean().reset_index()

    merged = micro.merge(macro, on=["generation", "model", "regime"], suffixes=("_micro", "_macro"))
    for m in ["rmse", "nse", "r2", "mae"]:
        mic_col = f"{m}_micro"
        mac_col = f"{m}_macro"
        if mic_col in merged.columns and mac_col in merged.columns:
            merged[f"{m}_gap"] = (merged[mic_col] - merged[mac_col]).round(6)

    merged = merged.sort_values(["generation", "model", "regime"], kind="stable")
    out = output_dir / "table_10_macro_vs_micro.csv"
    save_csv(merged, out)
    return out


def make_table_11(df: pd.DataFrame, output_dir: Path) -> Path:
    """T11: Full 11-metric supplementary table (test, micro)."""
    t = _test_micro(df).sort_values(["generation", "model", "regime", "horizon"], kind="stable")
    cols = ["generation", "model", "regime", "horizon"] + METRIC_COLS
    available = [c for c in cols if c in t.columns]
    out = output_dir / "table_11_all_metrics_full.csv"
    save_csv(t[available], out)
    return out


# ---------------------------------------------------------------------------
# Plot generators
# ---------------------------------------------------------------------------

_REGIME_COLORS = {
    "context": "#1f77b4",
    "weather": "#ff7f0e",
    "hydro": "#2ca02c",
}
_GEN_HATCHES = {"w14": "", "w30": "///"}
_GEN_MARKERS = {"w14": "o", "w30": "s"}


def _style_axes(ax: plt.Axes) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.25)


def plot_heatmaps(df: pd.DataFrame, plots_dir: Path) -> list[Path]:
    """Heatmap: model × horizon for RMSE and NSE, one figure per (metric, regime)."""
    t = _test_micro(df)
    out_paths: list[Path] = []
    for metric in ["rmse", "nse"]:
        for regime in ("context", "weather", "hydro"):
            rdf = t.loc[t["regime"] == regime]
            if rdf.empty:
                continue
            # One panel per generation
            generations = sorted(rdf["generation"].unique())
            n_gen = len(generations)
            fig, axes = plt.subplots(1, n_gen, figsize=(6 * n_gen, 4.5), constrained_layout=True)
            if n_gen == 1:
                axes = [axes]
            for ax, gen in zip(axes, generations):
                gdf = rdf.loc[rdf["generation"] == gen]
                pivot = gdf.pivot_table(index="model", columns="horizon", values=metric, aggfunc="mean")
                if pivot.empty:
                    continue
                im = ax.imshow(
                    pivot.values,
                    aspect="auto",
                    cmap="RdYlGn_r" if metric in LOWER_IS_BETTER else "RdYlGn",
                )
                ax.set_xticks(range(len(pivot.columns)))
                ax.set_xticklabels([f"h{c}" for c in pivot.columns])
                ax.set_yticks(range(len(pivot.index)))
                ax.set_yticklabels(pivot.index)
                ax.set_title(f"{gen.upper()} — {metric.upper()}")
                ax.set_xlabel("Horizon")
                ax.set_ylabel("Model")
                for i in range(pivot.shape[0]):
                    for j in range(pivot.shape[1]):
                        val = pivot.values[i, j]
                        if not np.isnan(val):
                            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=7)
                fig.colorbar(im, ax=ax, shrink=0.8)
            fig.suptitle(f"{metric.upper()} heatmap — {regime} regime (test, micro)")
            out = plots_dir / f"heatmap_{metric}_{regime}.png"
            ensure_parent_dir(out)
            fig.savefig(out, dpi=200, bbox_inches="tight")
            plt.close(fig)
            out_paths.append(out)
    return out_paths


def plot_bar_avg_nse_by_regime(df: pd.DataFrame, plots_dir: Path) -> Path:
    """Grouped bar chart: avg NSE per model, grouped by regime, colored by generation."""
    t = _test_micro(df)
    agg = (
        t.groupby(["generation", "model", "regime"], dropna=False)["nse"]
        .mean()
        .reset_index()
    )

    models = sorted(agg["model"].unique())
    regimes = [r for r in ("context", "weather", "hydro") if r in agg["regime"].unique()]
    generations = sorted(agg["generation"].unique())

    n_models = len(models)
    n_groups = len(regimes)
    n_gen = len(generations)
    bar_width = 0.8 / (n_groups * n_gen)

    fig, ax = plt.subplots(figsize=(max(10, n_models * 1.2), 5.5), constrained_layout=True)
    x = np.arange(n_models)

    for g_idx, gen in enumerate(generations):
        for r_idx, regime in enumerate(regimes):
            offset = (r_idx * n_gen + g_idx - (n_groups * n_gen) / 2 + 0.5) * bar_width
            vals = []
            for model in models:
                sub = agg.loc[(agg["generation"] == gen) & (agg["model"] == model) & (agg["regime"] == regime), "nse"]
                vals.append(float(sub.iloc[0]) if len(sub) > 0 else float("nan"))
            color = _REGIME_COLORS.get(regime, "#888888")
            alpha = 0.9 if gen == "w30" else 0.55
            ax.bar(
                x + offset,
                vals,
                width=bar_width,
                color=color,
                alpha=alpha,
                hatch=_GEN_HATCHES[gen],
                label=f"{regime}/{gen}",
            )

    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=30, ha="right")
    ax.set_ylabel("Average NSE (h1–h3, test, micro)")
    ax.set_title("Average NSE by model, regime and generation")
    ax.axhline(0, color="#333333", linewidth=0.8, linestyle="--")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    _style_axes(ax)

    out = plots_dir / "bar_avg_nse_by_regime.png"
    ensure_parent_dir(out)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_window_ablation(ablation_df: pd.DataFrame, plots_dir: Path) -> Path:
    """Bar chart: RMSE gain from w14→w30 per (model, regime)."""
    if ablation_df.empty or "avg_rmse_gain" not in ablation_df.columns:
        out = plots_dir / "bar_window_ablation.png"
        return out

    fig, ax = plt.subplots(figsize=(max(8, len(ablation_df) * 0.5), 5), constrained_layout=True)
    labels = ablation_df.apply(lambda r: f"{r['model']}/{r['regime']}", axis=1)
    vals = ablation_df["avg_rmse_gain"].values
    colors = np.where(np.array(vals) >= 0, "#2ca02c", "#d62728")
    ax.bar(range(len(vals)), vals, color=colors)
    ax.axhline(0, color="#333333", linewidth=0.8, linestyle="--")
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("Avg RMSE gain (w14 − w30)   [positive = w30 better]")
    ax.set_title("Window ablation: RMSE gain from 14-day to 30-day lookback")
    _style_axes(ax)

    out = plots_dir / "bar_window_ablation.png"
    ensure_parent_dir(out)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_regime_ablation(ablation_df: pd.DataFrame, plots_dir: Path) -> Path:
    """Bar chart: NSE gain from context→hydro per (model, generation)."""
    col = "ctx_to_hyd_nse_gain"
    if ablation_df.empty or col not in ablation_df.columns:
        out = plots_dir / "bar_regime_ablation.png"
        return out

    sub = ablation_df.dropna(subset=[col]).copy()
    labels = sub.apply(lambda r: f"{r['model']}/{r['generation']}", axis=1)
    vals = sub[col].values
    colors = np.where(vals >= 0, "#2ca02c", "#d62728")

    fig, ax = plt.subplots(figsize=(max(8, len(sub) * 0.5), 5), constrained_layout=True)
    ax.bar(range(len(vals)), vals, color=colors)
    ax.axhline(0, color="#333333", linewidth=0.8, linestyle="--")
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_ylabel("NSE gain (hydro − context)   [positive = hydro better]")
    ax.set_title("Feature regime ablation: NSE gain from context-only to hydro-weather")
    _style_axes(ax)

    out = plots_dir / "bar_regime_ablation.png"
    ensure_parent_dir(out)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_scatter_nse_r2(df: pd.DataFrame, plots_dir: Path) -> Path:
    """Scatter: avg NSE vs avg R2 for all (model, regime, generation) combinations."""
    t = _test_micro(df)
    agg = (
        t.groupby(["generation", "model", "regime"], dropna=False)[["nse", "r2"]]
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(6, 5), constrained_layout=True)
    for regime in ("context", "weather", "hydro"):
        for gen in ("w14", "w30"):
            sub = agg.loc[(agg["regime"] == regime) & (agg["generation"] == gen)]
            if sub.empty:
                continue
            ax.scatter(
                sub["nse"], sub["r2"],
                label=f"{regime}/{gen}",
                color=_REGIME_COLORS.get(regime, "#888888"),
                marker=_GEN_MARKERS.get(gen, "o"),
                alpha=0.85,
                s=60,
            )
            for _, row in sub.iterrows():
                ax.annotate(
                    row["model"],
                    (row["nse"], row["r2"]),
                    fontsize=6,
                    textcoords="offset points",
                    xytext=(4, 2),
                )

    # Reference line NSE == R2
    lims = [min(agg["nse"].min(), agg["r2"].min()) - 0.05, max(agg["nse"].max(), agg["r2"].max()) + 0.05]
    ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.5, label="NSE = R²")
    ax.set_xlabel("Avg NSE")
    ax.set_ylabel("Avg R²")
    ax.set_title("NSE vs R² (test, micro, avg h1–h3)")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=7)
    _style_axes(ax)

    out = plots_dir / "scatter_nse_r2.png"
    ensure_parent_dir(out)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_line_horizon_rmse(df: pd.DataFrame, plots_dir: Path) -> list[Path]:
    """Line plot: RMSE by horizon (1→2→3) per model, one figure per (regime, generation)."""
    t = _test_micro(df)
    out_paths: list[Path] = []

    for (regime, gen), grp in t.groupby(["regime", "generation"], dropna=False):
        if grp.empty:
            continue
        models = sorted(grp["model"].unique())
        fig, ax = plt.subplots(figsize=(6, 4.5), constrained_layout=True)

        cmap = plt.get_cmap("tab10")
        for i, model in enumerate(models):
            mdf = grp.loc[grp["model"] == model].sort_values("horizon")
            if mdf.empty:
                continue
            ax.plot(mdf["horizon"], mdf["rmse"], marker="o", label=model, color=cmap(i % 10))

        ax.set_xticks([1, 2, 3])
        ax.set_xlabel("Forecast horizon")
        ax.set_ylabel("RMSE (test, micro)")
        ax.set_title(f"RMSE by horizon — {regime} / {gen}")
        ax.legend(fontsize=7)
        _style_axes(ax)

        out = plots_dir / f"line_horizon_rmse_{regime}_{gen}.png"
        ensure_parent_dir(out)
        fig.savefig(out, dpi=200, bbox_inches="tight")
        plt.close(fig)
        out_paths.append(out)

    return out_paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logger = get_logger("compare_all_results")
    logger.info("Loading all artifact runs...")

    df = _load_all_runs(logger)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, str] = {}

    # --- Tables ---
    logger.info("Building Table 01: full test micro master table")
    p = make_table_01(df, OUTPUT_DIR)
    manifest["table_01"] = str(p)

    logger.info("Building Table 02: pivot by horizon")
    p, ptex = make_table_02(df, OUTPUT_DIR)
    manifest["table_02"] = str(p)
    manifest["table_02_latex"] = str(ptex)

    logger.info("Building Table 03: avg across horizons")
    p, ptex = make_table_03(df, OUTPUT_DIR)
    manifest["table_03"] = str(p)
    manifest["table_03_latex"] = str(ptex)

    logger.info("Building Table 04: per-regime rankings")
    paths = make_table_04(df, OUTPUT_DIR)
    for i, p in enumerate(paths):
        manifest[f"table_04_{i}"] = str(p)

    logger.info("Building Table 05: window ablation (w14 vs w30)")
    p = make_table_05(df, OUTPUT_DIR)
    manifest["table_05"] = str(p)

    logger.info("Building Table 06: feature regime ablation")
    p = make_table_06(df, OUTPUT_DIR)
    manifest["table_06"] = str(p)

    logger.info("Building Table 07: best model per metric")
    p = make_table_07(df, OUTPUT_DIR)
    manifest["table_07"] = str(p)

    logger.info("Building Table 08: per-horizon ranking")
    p = make_table_08(df, OUTPUT_DIR)
    manifest["table_08"] = str(p)

    logger.info("Building Table 09: training metadata")
    p = make_table_09(df, OUTPUT_DIR)
    manifest["table_09"] = str(p)

    logger.info("Building Table 10: macro vs micro")
    p = make_table_10(df, OUTPUT_DIR)
    manifest["table_10"] = str(p)

    logger.info("Building Table 11: full 11-metric supplementary table")
    p = make_table_11(df, OUTPUT_DIR)
    manifest["table_11"] = str(p)

    # --- Plots ---
    logger.info("Generating heatmap plots")
    for p in plot_heatmaps(df, PLOTS_DIR):
        manifest[f"plot_{p.stem}"] = str(p)

    logger.info("Generating bar chart: avg NSE by regime")
    p = plot_bar_avg_nse_by_regime(df, PLOTS_DIR)
    manifest["plot_bar_avg_nse"] = str(p)

    # Load T05 dataframe for ablation plots
    t05_path = OUTPUT_DIR / "table_05_window_ablation.csv"
    t05_df = pd.read_csv(t05_path) if t05_path.exists() else pd.DataFrame()
    logger.info("Generating bar chart: window ablation")
    p = plot_window_ablation(t05_df, PLOTS_DIR)
    manifest["plot_window_ablation"] = str(p)

    t06_path = OUTPUT_DIR / "table_06_regime_ablation.csv"
    t06_df = pd.read_csv(t06_path) if t06_path.exists() else pd.DataFrame()
    logger.info("Generating bar chart: regime ablation")
    p = plot_regime_ablation(t06_df, PLOTS_DIR)
    manifest["plot_regime_ablation"] = str(p)

    logger.info("Generating scatter: NSE vs R2")
    p = plot_scatter_nse_r2(df, PLOTS_DIR)
    manifest["plot_scatter_nse_r2"] = str(p)

    logger.info("Generating line plots: RMSE by horizon")
    for p in plot_line_horizon_rmse(df, PLOTS_DIR):
        manifest[f"plot_{p.stem}"] = str(p)

    # --- Manifest ---
    save_json(manifest, OUTPUT_DIR / "manifest.json")
    logger.info(
        "Done. %d outputs saved under %s",
        len(manifest),
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()
