"""Generate comparison plots across all completed benchmark artifacts.

This script scans an artifacts root for completed model runs, loads their saved
metrics and optional prediction files, and writes the cross-model plots used in
the benchmark report. It is the highest-level plotting entry point in the
repository.

Use this script after training several models and you want one shared benchmark
folder with scorecards, ranking plots, progression plots, and diagnostics.

Usage
-----
    .venv/Scripts/python scripts/plot_benchmark_results.py
    .venv/Scripts/python scripts/plot_benchmark_results.py --artifacts artifacts/advanced_seq --out plots/benchmark
    .venv/Scripts/python scripts/plot_benchmark_results.py --skip-predictions
    .venv/Scripts/python scripts/plot_benchmark_results.py --only 0 1 9 10 18
    .venv/Scripts/python scripts/plot_benchmark_results.py --exclude lstm:weather tft:hydro_weather

Outputs
-------
    Each run writes into a timestamped subdirectory of --out, e.g.:
        plots/benchmark/20260320_143022/00_scorecard_heatmap.png

    plots are numbered:
        00  scorecard_heatmap
        01  nse_ranked_by_horizon
        02  rmse_ranked_by_horizon
        03  data_level_progression
        04  station_rank_curves
        06  loss_curves
        08  best_worst_stations  (single file, champion model only)
        09  horizon_degradation
        10  multi_metric_radar
        12  bias_rmse_scatter
        13  composite_ranking
        16  training_dynamics
        18  station_level_progression

Notes
-----
    Prediction-based plots are slower because they load predictions.parquet.
    Use --skip-predictions when you only need metric-based summary plots.

    Use --exclude to drop a specific model variant, e.g.:
        --exclude lstm:weather tft:hydro_weather
"""

from __future__ import annotations

import argparse
import json
import warnings
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_ORDER = [
    "ann", "lstm", "nhits", "patchtst", "tft",
    "xlstm", "mamba", "hybrid", "xgboost", "flownet",
]
VARIANT_ORDER  = ["context", "weather", "hydro_weather"]
VARIANT_COLORS = {
    "context":       "#4C72B0",  # blue
    "weather":       "#DD8452",  # orange
    "hydro_weather": "#55A868",  # green
}
VARIANT_MARKERS = {"context": "o", "weather": "s", "hydro_weather": "^"}
VARIANT_LS      = {"context": "-", "weather": "--", "hydro_weather": ":"}
VARIANT_LABELS  = {"context": "Context only", "weather": "+ Weather", "hydro_weather": "+ Weather + Soil moisture"}

# direction: +1 = higher is better, -1 = lower is better, 0 = absolute
METRIC_DIRECTION = {
    "nse": 1, "r2": 1,
    "rmse": -1, "mae": -1, "mase": -1, "rmsse": -1, "smape": -1, "wape": -1,
}
METRIC_LABELS = {
    "nse":   "NSE",
    "rmse":  "RMSE (m³/s)",
    "mae":   "MAE (m³/s)",
    "r2":    "R²",
    "mase":  "MASE",
    "rmsse": "RMSSE",
    "smape": "SMAPE",
    "wape":  "WAPE",
    "bias":  "Bias (m³/s)",
}

# ── Style helpers ──────────────────────────────────────────────────────────────

def _style(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis=grid_axis, alpha=0.22, linewidth=0.7)
    ax.tick_params(labelsize=8)


def _savefig(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved {path.name}")


def _variant_legend_handles(records: list[dict], *, extra: list | None = None) -> list[mpatches.Patch]:
    patches = [
        mpatches.Patch(color=VARIANT_COLORS[v], label=VARIANT_LABELS[v])
        for v in VARIANT_ORDER
        if any(r["variant"] == v for r in records)
    ]
    if extra:
        patches.extend(extra)
    return patches


# ── Data discovery & loading ───────────────────────────────────────────────────

def _discover_completed(artifacts_root: Path) -> list[dict]:
    """Return metadata dicts for every artifact dir that has metrics_summary.csv.

    Handles variants: context, weather, hydro_weather.
    """
    records = []
    if not artifacts_root.exists():
        return records
    for d in sorted(artifacts_root.iterdir()):
        if not d.is_dir() or not (d / "metrics_summary.csv").exists():
            continue
        parts = d.name.split("_")
        # Detect hydro_weather first (consecutive "hydro" + "weather")
        variant, variant_start = "context", len(parts)
        for i in range(len(parts) - 1):
            if parts[i] == "hydro" and parts[i + 1] == "weather":
                variant, variant_start = "hydro_weather", i
                break
        if variant == "context":
            for i, p in enumerate(parts):
                if p == "weather":
                    variant, variant_start = "weather", i
                    break
                if p == "context":
                    variant, variant_start = "context", i
                    break
        model = "_".join(parts[:variant_start])
        records.append({"model": model, "variant": variant, "dir": d})
    return records


def _load_metrics(records: list[dict]) -> pd.DataFrame:
    rows = []
    for rec in records:
        try:
            df = pd.read_csv(rec["dir"] / "metrics_summary.csv")
            df["model"]   = rec["model"]
            df["variant"] = rec["variant"]
            rows.append(df)
        except Exception:
            pass
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _load_station_metrics(records: list[dict]) -> pd.DataFrame:
    rows = []
    for rec in records:
        path = rec["dir"] / "metrics_by_station.csv"
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path)
            df["model"]   = rec["model"]
            df["variant"] = rec["variant"]
            rows.append(df)
        except Exception:
            pass
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def _load_loss_histories(records: list[dict]) -> dict[str, pd.DataFrame]:
    out = {}
    for rec in records:
        path = rec["dir"] / "loss_history.csv"
        if not path.exists():
            continue
        try:
            label = f"{rec['model']}:{rec['variant']}"
            out[label] = pd.read_csv(path)
        except Exception:
            pass
    return out


def _load_epoch_metrics(records: list[dict]) -> dict[str, pd.DataFrame]:
    out = {}
    for rec in records:
        path = rec["dir"] / "epoch_metrics.csv"
        if not path.exists():
            continue
        try:
            label = f"{rec['model']}:{rec['variant']}"
            out[label] = pd.read_csv(path)
        except Exception:
            pass
    return out


def _load_training_summaries(records: list[dict]) -> dict[str, dict]:
    out = {}
    for rec in records:
        path = rec["dir"] / "training_summary.json"
        if not path.exists():
            continue
        try:
            with open(path) as f:
                out[f"{rec['model']}:{rec['variant']}"] = json.load(f)
        except Exception:
            pass
    return out


def _sorted_model_keys(records: list[dict]) -> list[tuple[str, str]]:
    seen: list[tuple[str, str]] = []
    for m in MODEL_ORDER:
        for v in VARIANT_ORDER:
            if any(r["model"] == m and r["variant"] == v for r in records):
                seen.append((m, v))
    for r in records:
        key = (r["model"], r["variant"])
        if key not in seen:
            seen.append(key)
    return seen


def _mv_label(model: str, variant: str) -> str:
    suffix = {"context": "ctx", "weather": "wthr", "hydro_weather": "hydro"}
    return f"{model.upper()} ({suffix.get(variant, variant)})"


def _test_macro(metrics_df: pd.DataFrame) -> pd.DataFrame:
    return metrics_df[
        (metrics_df["split"] == "test") & (metrics_df["aggregation"] == "macro")
    ]


def _normalize_metric_scores(values: dict[tuple[str, str], float], metric_name: str) -> dict[tuple[str, str], float]:
    finite_values = {key: float(value) for key, value in values.items() if np.isfinite(value)}
    if not finite_values:
        return {}

    lo = min(finite_values.values())
    hi = max(finite_values.values())
    if hi <= lo:
        return {key: 1.0 for key in finite_values}

    direction = METRIC_DIRECTION.get(metric_name, 1)
    normalized: dict[tuple[str, str], float] = {}
    for key, value in finite_values.items():
        raw_score = (value - lo) / (hi - lo)
        normalized[key] = raw_score if direction >= 0 else 1.0 - raw_score
    return normalized


def _radar_metric_label(metric_name: str) -> str:
    if METRIC_DIRECTION.get(metric_name, 1) >= 0:
        return METRIC_LABELS.get(metric_name, metric_name.upper())
    base_label = METRIC_LABELS.get(metric_name, metric_name.upper()).split("(", 1)[0].strip()
    return f"{base_label} skill"


def _marker_size_from_value(value: float, minimum: float, maximum: float) -> float:
    if not np.isfinite(value):
        return 60.0
    if maximum <= minimum:
        return 90.0
    return 55.0 + 95.0 * ((value - minimum) / (maximum - minimum))


# ── Plot 00: Scorecard heatmap ─────────────────────────────────────────────────

def plot_scorecard_heatmap(metrics_df: pd.DataFrame, records: list[dict], out_dir: Path) -> None:
    if metrics_df.empty:
        return
    tm = _test_macro(metrics_df)
    if tm.empty:
        return

    horizons         = sorted(tm["horizon"].unique())
    available_metrics = [m for m in METRIC_DIRECTION if m in tm.columns]
    if not available_metrics:
        return

    keys       = _sorted_model_keys(records)
    row_labels = [_mv_label(m, v) for m, v in keys]
    col_labels = [
        f"{METRIC_LABELS.get(m, m.upper())}\nh={h}d"
        for m in available_metrics for h in horizons
    ]
    n_rows, n_cols = len(keys), len(col_labels)

    mat = np.full((n_rows, n_cols), np.nan)
    c = 0
    for m in available_metrics:
        for h in horizons:
            for r_idx, (model, variant) in enumerate(keys):
                row = tm[
                    (tm["model"] == model) & (tm["variant"] == variant) & (tm["horizon"] == h)
                ]
                if not row.empty:
                    mat[r_idx, c] = float(row[m].iloc[0])
            c += 1

    norm_mat = np.full_like(mat, np.nan)
    c = 0
    for m in available_metrics:
        direction = METRIC_DIRECTION[m]
        for _h in horizons:
            col  = mat[:, c]
            valid = col[np.isfinite(col)]
            if len(valid) > 1:
                lo, hi = valid.min(), valid.max()
                if hi > lo:
                    norm_mat[:, c] = (col - lo) / (hi - lo) if direction == 1 else (hi - col) / (hi - lo)
                else:
                    norm_mat[:, c] = np.where(np.isfinite(col), 0.5, np.nan)
            elif len(valid) == 1:
                norm_mat[:, c] = np.where(np.isfinite(col), 0.5, np.nan)
            c += 1

    fig, ax = plt.subplots(figsize=(max(12, n_cols * 1.05), max(4, n_rows * 0.55)))
    im = ax.imshow(norm_mat, aspect="auto", cmap="RdYlGn", vmin=0, vmax=1, interpolation="none")

    c = 0
    for m in available_metrics:
        for _h in horizons:
            for r_idx in range(n_rows):
                val = mat[r_idx, c]
                if np.isfinite(val):
                    nv = norm_mat[r_idx, c]
                    txt_color = "black" if 0.25 < nv < 0.82 else "white"
                    fmt = ".3f" if m in ("nse", "r2") else ".2f" if m in ("mase", "rmsse", "smape", "wape") else ".1f"
                    ax.text(c, r_idx, f"{val:{fmt}}", ha="center", va="center",
                            fontsize=6.5, color=txt_color)
            c += 1

    n_h = len(horizons)
    for i in range(1, len(available_metrics)):
        ax.axvline(i * n_h - 0.5, color="white", linewidth=2)
    # Horizontal separators between variant groups
    prev_variant = None
    for r_idx, (_, v) in enumerate(keys):
        if prev_variant is not None and v != prev_variant:
            ax.axhline(r_idx - 0.5, color="white", linewidth=1.5)
        prev_variant = v

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(col_labels, fontsize=6.5)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(row_labels, fontsize=8)
    ax.set_title(
        "Benchmark Scorecard — all models × metrics × horizons  (test set, macro-averaged, green = best)",
        fontsize=10, pad=10,
    )
    plt.colorbar(im, ax=ax, label="Normalized score (1 = best)", shrink=0.55, pad=0.02)
    fig.tight_layout()
    _savefig(fig, out_dir / "00_scorecard_heatmap.png")


# ── Plot 01/02: Ranked dot plots ──────────────────────────────────────────────

def plot_ranked_dots(
    metrics_df: pd.DataFrame,
    records: list[dict],
    out_dir: Path,
    metric: str,
    fname: str,
) -> None:
    if metrics_df.empty or metric not in metrics_df.columns:
        return
    tm = _test_macro(metrics_df)
    if tm.empty:
        return

    horizons  = sorted(tm["horizon"].unique())
    direction = METRIC_DIRECTION.get(metric, 1)
    all_keys  = _sorted_model_keys(records)
    n_h       = len(horizons)

    fig, axes = plt.subplots(
        1, n_h,
        figsize=(5 * n_h, max(4, len(all_keys) * 0.45 + 1.5)),
        sharey=False,
    )
    if n_h == 1:
        axes = [axes]

    for ax, h in zip(axes, horizons):
        h_df = tm[tm["horizon"] == h]
        vals = {}
        for model, variant in all_keys:
            row = h_df[(h_df["model"] == model) & (h_df["variant"] == variant)]
            if not row.empty:
                vals[(model, variant)] = float(row[metric].iloc[0])
        if not vals:
            ax.set_visible(False)
            continue

        sorted_keys = sorted(vals.keys(), key=lambda k: direction * vals[k])
        y_labels = [_mv_label(m, v) for m, v in sorted_keys]
        y_vals   = [vals[k] for k in sorted_keys]
        colors   = [VARIANT_COLORS.get(v, "#888") for _, v in sorted_keys]
        markers  = [VARIANT_MARKERS.get(v, "o") for _, v in sorted_keys]
        y_pos    = list(range(len(sorted_keys)))

        for yp, yv, c, mk in zip(y_pos, y_vals, colors, markers):
            ax.scatter(yv, yp, c=c, marker=mk, s=70, zorder=3, edgecolors="white", linewidths=0.5)
        x_left = 0.0 if metric == "nse" else min(y_vals) * 0.95
        ax.hlines(y_pos, x_left, y_vals, colors=colors, alpha=0.4, linewidth=1.5)

        if metric == "nse":
            ax.axvline(0.75, color="green", linestyle="--", linewidth=1, alpha=0.7, label="target=0.75")
            ax.axvline(0.0,  color="red",   linestyle=":",  linewidth=0.8, alpha=0.5, label="NSE=0")
            ax.set_xlim(left=min(0.0, min(y_vals) - 0.05))
            ax.legend(fontsize=7)

        ax.set_yticks(y_pos)
        ax.set_yticklabels(y_labels, fontsize=8)
        ax.set_title(f"h = {h} day", fontsize=9)
        ax.set_xlabel(METRIC_LABELS.get(metric, metric.upper()), fontsize=9)
        _style(ax, grid_axis="x")

    handles = _variant_legend_handles(records)
    fig.legend(handles=handles, fontsize=8, loc="lower center",
               ncol=len(handles), bbox_to_anchor=(0.5, -0.03))
    fig.suptitle(
        f"{METRIC_LABELS.get(metric, metric.upper())} — ranked by horizon  (test set, macro-averaged)",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    _savefig(fig, out_dir / fname)


# ── Plot 03: Data-level progression (context → weather → hydro) ───────────────

def plot_data_level_progression(metrics_df: pd.DataFrame, records: list[dict], out_dir: Path) -> None:
    """Grouped bar chart showing NSE improvement across data levels per model & horizon."""
    if metrics_df.empty:
        return
    tm = _test_macro(metrics_df)
    if tm.empty:
        return

    variants_present = [v for v in VARIANT_ORDER if any(r["variant"] == v for r in records)]
    if len(variants_present) < 2:
        print("  Skipping 03_data_level_progression: fewer than 2 data variants found.")
        return

    horizons      = sorted(tm["horizon"].unique())
    models_sorted = [m for m in MODEL_ORDER if any(r["model"] == m for r in records)]
    models_sorted += [r["model"] for r in records if r["model"] not in MODEL_ORDER and
                      r["model"] not in models_sorted]
    n_h, n_m, n_v = len(horizons), len(models_sorted), len(variants_present)

    fig, axes = plt.subplots(
        n_h, 1,
        figsize=(max(10, n_m * 1.6), 3.5 * n_h),
        squeeze=False,
    )

    bar_w = 0.7 / n_v
    offsets = np.linspace(-(n_v - 1) / 2, (n_v - 1) / 2, n_v) * bar_w
    x = np.arange(n_m)

    for h_idx, h in enumerate(horizons):
        ax = axes[h_idx][0]
        for v_idx, variant in enumerate(variants_present):
            nse_vals = []
            for model in models_sorted:
                row = tm[
                    (tm["model"] == model) & (tm["variant"] == variant) & (tm["horizon"] == h)
                ]
                nse_vals.append(float(row["nse"].iloc[0]) if not row.empty else np.nan)

            bars = ax.bar(
                x + offsets[v_idx], nse_vals, bar_w,
                color=VARIANT_COLORS[variant], alpha=0.85,
                label=VARIANT_LABELS[variant], edgecolor="white", linewidth=0.5,
            )
            # Annotate bar tops
            for bar, val in zip(bars, nse_vals):
                if np.isfinite(val):
                    ax.text(
                        bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=5.5, rotation=90,
                    )

        ax.axhline(0.75, color="green", linestyle="--", linewidth=1, alpha=0.6, label="NSE=0.75 target")
        ax.axhline(0.0,  color="red",   linestyle=":",  linewidth=0.8, alpha=0.4)
        ax.set_xticks(x)
        ax.set_xticklabels([m.upper() for m in models_sorted], fontsize=9)
        ax.set_ylabel(f"NSE  (h={h}d, test)", fontsize=9)
        ax.set_ylim(bottom=min(0.0, ax.get_ylim()[0]))
        ax.legend(fontsize=7, loc="lower right")
        _style(ax)

    fig.suptitle(
        "Data Level Progression — NSE by Model, Data Variant, and Forecast Horizon  "
        "(context → +weather → +soil moisture)",
        fontsize=11,
    )
    fig.tight_layout()
    _savefig(fig, out_dir / "03_data_level_progression.png")


# ── Plot 04: Station rank curves ──────────────────────────────────────────────

def plot_station_rank_curves(station_df: pd.DataFrame, records: list[dict], out_dir: Path) -> None:
    if station_df.empty:
        return
    test_h1 = station_df[
        (station_df["split"] == "test") &
        (station_df["horizon"] == 1) &
        station_df["nse"].notna()
    ]
    if test_h1.empty:
        return

    keys = _sorted_model_keys(records)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    cmap = matplotlib.colormaps.get_cmap("tab10").resampled(max(len(keys), 10))

    for i, (model, variant) in enumerate(keys):
        sub = test_h1[
            (test_h1["model"] == model) & (test_h1["variant"] == variant)
        ]["nse"].values
        if len(sub) == 0:
            continue
        sorted_nse = np.sort(sub)
        percentile = np.linspace(0, 100, len(sorted_nse))
        ax.plot(
            percentile, sorted_nse,
            linestyle=VARIANT_LS.get(variant, "-"),
            marker=VARIANT_MARKERS.get(variant, "o"),
            markevery=max(1, len(sorted_nse) // 8),
            markersize=4,
            color=cmap(MODEL_ORDER.index(model) if model in MODEL_ORDER else i),
            linewidth=1.6,
            label=_mv_label(model, variant),
            alpha=0.85,
        )

    ax.axhline(0.75, color="green", linestyle=":", linewidth=1, alpha=0.7, label="NSE=0.75 target")
    ax.axhline(0.0,  color="red",   linestyle=":", linewidth=0.8, alpha=0.5, label="NSE=0")
    ax.set_xlabel("Station percentile  (0 = worst, 100 = best)", fontsize=10)
    ax.set_ylabel("NSE  (test, h=1 day)", fontsize=10)
    ax.set_title("Station Rank Curves — sorted per-station NSE  (h=1 day)", fontsize=11)
    ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
    _style(ax)
    fig.tight_layout()
    _savefig(fig, out_dir / "04_station_rank_curves.png")


# ── Plot 06: Loss curves ──────────────────────────────────────────────────────

def plot_loss_curves(loss_histories: dict[str, pd.DataFrame], out_dir: Path) -> None:
    if not loss_histories:
        return
    n     = len(loss_histories)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)

    for idx, (label, df) in enumerate(sorted(loss_histories.items())):
        ax    = axes[idx // ncols][idx % ncols]
        train = df[df["split"] == "train"].sort_values("epoch")
        val   = df[df["split"] == "validation"].sort_values("epoch")

        ax.plot(train["epoch"], train["loss"], label="train", color="#4C72B0", linewidth=1.5)
        ax.plot(val["epoch"],   val["loss"],   label="val",   color="#DD8452", linewidth=1.5)

        if not val.empty:
            best_idx   = val["loss"].idxmin()
            best_epoch = val.loc[best_idx, "epoch"]
            best_loss  = val.loc[best_idx, "loss"]
            ax.axvline(best_epoch, color="#DD8452", linestyle=":", linewidth=1, alpha=0.7)
            ax.scatter([best_epoch], [best_loss], color="#DD8452", s=40, zorder=5,
                       label=f"best e={int(best_epoch)}")

        ax.set_title(label, fontsize=8)
        ax.set_xlabel("Epoch", fontsize=7)
        ax.set_ylabel("Loss", fontsize=7)
        ax.legend(fontsize=6)
        _style(ax)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("Training and Validation Loss Curves", fontsize=12, y=1.01)
    fig.tight_layout()
    _savefig(fig, out_dir / "06_loss_curves.png")


# ── Plot 08: Best / worst station time-series ─────────────────────────────────

def plot_best_worst_stations(
    records: list[dict],
    station_df: pd.DataFrame,
    out_dir: Path,
    n_each: int = 3,
) -> None:
    """Single-file best/worst station time series for the top-ranked model variant.

    Picks the model+variant with the highest mean station NSE at h=1, then plots
    the n_each worst and n_each best stations across all forecast horizons.
    Output: 08_best_worst_stations.png  (one file regardless of how many models exist).
    """
    if station_df.empty:
        return

    # Find the best model+variant by mean per-station NSE at h=1
    best_rec = None
    best_mean_nse = -np.inf
    for rec in records:
        st = station_df[
            (station_df["model"]   == rec["model"]) &
            (station_df["variant"] == rec["variant"]) &
            (station_df["split"]   == "test") &
            (station_df["horizon"] == 1)
        ]["nse"].dropna()
        if st.empty:
            continue
        mean_nse = float(st.mean())
        if mean_nse > best_mean_nse:
            best_mean_nse = mean_nse
            best_rec = rec

    if best_rec is None:
        print("  Skipping 08_best_worst_stations: no station metrics found.")
        return

    pred_path = best_rec["dir"] / "predictions.parquet"
    if not pred_path.exists():
        print(f"  Skipping 08_best_worst_stations: no predictions.parquet for "
              f"{best_rec['model']}:{best_rec['variant']}")
        return

    try:
        preds = pd.read_parquet(pred_path)
    except Exception:
        return

    test = preds[preds["split"] == "test"].copy()
    if test.empty:
        return
    test["target_ds"] = pd.to_datetime(test["target_ds"])

    st_nse = station_df[
        (station_df["model"]   == best_rec["model"]) &
        (station_df["variant"] == best_rec["variant"]) &
        (station_df["split"]   == "test") &
        (station_df["horizon"] == 1)
    ].set_index("unique_id")["nse"].dropna()

    if st_nse.empty:
        return

    sorted_st = st_nse.sort_values()
    worst    = list(sorted_st.head(n_each).index)
    best     = list(sorted_st.tail(n_each).index)
    selected = worst + best

    horizons = sorted(test["horizon"].unique())
    t_max    = test["target_ds"].max()
    test_w   = test[test["target_ds"] >= t_max - pd.DateOffset(years=2)]

    n_s, n_h = len(selected), len(horizons)
    fig, axes = plt.subplots(n_s, n_h, figsize=(5.5 * n_h, 2.8 * n_s), squeeze=False)
    model_label = f"{best_rec['model'].upper()} ({best_rec['variant']})"
    fig.suptitle(
        f"Best / Worst Stations — {model_label}  "
        f"(bottom {n_each} worst + top {n_each} best by h=1 NSE, last 2 yrs shown)",
        fontsize=10, y=1.002,
    )

    for row_idx, station in enumerate(selected):
        group  = "WORST" if row_idx < n_each else "BEST"
        nse_h1 = float(st_nse.get(station, np.nan))
        bg     = "#fff5f5" if group == "WORST" else "#f5fff7"
        title_color = "#a00000" if group == "WORST" else "#006400"

        for col_idx, h in enumerate(horizons):
            ax = axes[row_idx][col_idx]
            ax.set_facecolor(bg)
            sub = (
                test_w[(test_w["unique_id"] == station) & (test_w["horizon"] == h)]
                .sort_values("target_ds")
                .drop_duplicates("target_ds")
            )
            if sub.empty:
                ax.set_visible(False)
                continue

            ax.plot(sub["target_ds"], sub["y_true"], color="#2166AC", linewidth=1.0,
                    label="Observed", alpha=0.9)
            ax.plot(sub["target_ds"], sub["y_pred"], color="#D6604D", linewidth=0.9,
                    linestyle="--", label="Predicted", alpha=0.85)
            ax.fill_between(sub["target_ds"], sub["y_true"], sub["y_pred"],
                            alpha=0.08, color="#D6604D")

            nse_str = f"  NSE={nse_h1:.3f}" if np.isfinite(nse_h1) else ""
            ax.set_title(f"[{group}] {station}{nse_str}  h={h}d",
                         fontsize=7.5, pad=2, color=title_color)
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=4))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=6)
            ax.set_ylabel("Q (m³/s)", fontsize=6)
            _style(ax, grid_axis="both")

            if row_idx == 0 and col_idx == 0:
                ax.legend(fontsize=6, loc="upper right")

    fig.tight_layout()
    _savefig(fig, out_dir / "08_best_worst_stations.png", dpi=130)


# ── Plot 09: Horizon degradation curves ──────────────────────────────────────

def plot_horizon_degradation(metrics_df: pd.DataFrame, records: list[dict], out_dir: Path) -> None:
    """Line plots showing how NSE and RMSE degrade as horizon increases."""
    if metrics_df.empty:
        return
    tm = _test_macro(metrics_df)
    if tm.empty:
        return

    horizons = sorted(tm["horizon"].unique())
    if len(horizons) < 2:
        return
    keys = _sorted_model_keys(records)

    cmap = matplotlib.colormaps.get_cmap("tab10").resampled(max(len(MODEL_ORDER), 10))

    fig, (ax_nse, ax_rmse) = plt.subplots(1, 2, figsize=(14, 5.5))

    for model, variant in keys:
        model_color = cmap(MODEL_ORDER.index(model) if model in MODEL_ORDER else 0)
        ls = VARIANT_LS.get(variant, "-")
        mk = VARIANT_MARKERS.get(variant, "o")

        nse_vals  = []
        rmse_vals = []
        for h in horizons:
            row = tm[(tm["model"] == model) & (tm["variant"] == variant) & (tm["horizon"] == h)]
            nse_vals.append(float(row["nse"].iloc[0])  if not row.empty and "nse"  in row.columns else np.nan)
            rmse_vals.append(float(row["rmse"].iloc[0]) if not row.empty and "rmse" in row.columns else np.nan)

        lbl = _mv_label(model, variant)
        ax_nse.plot(horizons, nse_vals,   color=model_color, ls=ls, marker=mk, ms=6, linewidth=1.8, label=lbl, alpha=0.85)
        ax_rmse.plot(horizons, rmse_vals, color=model_color, ls=ls, marker=mk, ms=6, linewidth=1.8, label=lbl, alpha=0.85)

    for ax in (ax_nse, ax_rmse):
        ax.set_xticks(horizons)
        ax.set_xticklabels([f"h={h}d" for h in horizons], fontsize=9)
        ax.set_xlabel("Forecast horizon", fontsize=10)
        _style(ax)

    ax_nse.axhline(0.75, color="green", linestyle=":", linewidth=1, alpha=0.6, label="NSE=0.75")
    ax_nse.set_ylabel("NSE  (test, macro)", fontsize=10)
    ax_nse.set_title("NSE Degradation with Forecast Horizon", fontsize=10)

    ax_rmse.set_ylabel("RMSE  (m³/s, test, macro)", fontsize=10)
    ax_rmse.set_title("RMSE Increase with Forecast Horizon", fontsize=10)

    # Legend per variant (line style) + per model (color) — compact
    handles, labels_seen = [], []
    for model, variant in keys:
        lbl = _mv_label(model, variant)
        if lbl not in labels_seen:
            model_color = cmap(MODEL_ORDER.index(model) if model in MODEL_ORDER else 0)
            handles.append(plt.Line2D([0], [0], color=model_color,
                                      ls=VARIANT_LS.get(variant, "-"),
                                      marker=VARIANT_MARKERS.get(variant, "o"),
                                      ms=5, linewidth=1.5, label=lbl))
            labels_seen.append(lbl)

    fig.legend(handles=handles, fontsize=7, loc="lower center",
               ncol=min(6, len(handles)), bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Forecast Skill Degradation by Horizon  (test set, macro-averaged)", fontsize=12)
    fig.tight_layout(rect=[0, 0.08, 1, 0.97])
    _savefig(fig, out_dir / "09_horizon_degradation.png")


# ── Plot 10: Multi-metric radar charts ────────────────────────────────────────

def plot_radar_charts(metrics_df: pd.DataFrame, records: list[dict], out_dir: Path) -> None:
    """Spider/radar chart: one subplot per model, lines for each variant, h=1 metrics."""
    if metrics_df.empty:
        return
    tm = _test_macro(metrics_df)
    if tm.empty:
        return

    # Use h=1 for radar — focus on key metrics (need [0,1] normalization)
    radar_metrics = [m for m in ("nse", "rmse", "mae", "mase", "rmsse", "smape") if m in tm.columns]
    if len(radar_metrics) < 3:
        return

    tm_h1     = tm[tm["horizon"] == 1]
    models    = [m for m in MODEL_ORDER if any(r["model"] == m for r in records)]
    variants  = [v for v in VARIANT_ORDER if any(r["variant"] == v for r in records)]
    n_models  = len(models)
    if n_models == 0:
        return

    norm_vals: dict[str, dict[tuple, float]] = {met: {} for met in radar_metrics}
    for met in radar_metrics:
        col_vals: dict[tuple[str, str], float] = {}
        for model in models:
            for variant in variants:
                row = tm_h1[(tm_h1["model"] == model) & (tm_h1["variant"] == variant)]
                if not row.empty and met in row.columns:
                    col_vals[(model, variant)] = float(row[met].iloc[0])
        norm_vals[met] = _normalize_metric_scores(col_vals, met)

    N     = len(radar_metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close polygon

    ncols = min(4, n_models)
    nrows = (n_models + ncols - 1) // ncols
    fig   = plt.figure(figsize=(4.5 * ncols, 4.2 * nrows))

    for m_idx, model in enumerate(models):
        ax = fig.add_subplot(nrows, ncols, m_idx + 1, projection="polar")
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(
            [_radar_metric_label(met) for met in radar_metrics],
            fontsize=7,
        )
        ax.set_ylim(0, 1)
        ax.set_yticks([0.25, 0.5, 0.75, 1.0])
        ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"], fontsize=5)
        ax.grid(alpha=0.3)

        for variant in variants:
            vals = [norm_vals[met].get((model, variant), np.nan) for met in radar_metrics]
            if all(np.isnan(v) for v in vals):
                continue
            vals_closed = vals + vals[:1]
            ax.plot(angles, vals_closed,
                    color=VARIANT_COLORS[variant], linewidth=1.8,
                    linestyle=VARIANT_LS[variant], label=VARIANT_LABELS[variant])
            ax.fill(angles, vals_closed, color=VARIANT_COLORS[variant], alpha=0.12)

        ax.set_title(model.upper(), fontsize=10, pad=12)

    # Hide unused axes
    for idx in range(n_models, nrows * ncols):
        fig.add_subplot(nrows, ncols, idx + 1).set_visible(False)

    handles = _variant_legend_handles(records)
    fig.legend(handles=handles, fontsize=8, loc="lower center",
               ncol=len(handles), bbox_to_anchor=(0.5, 0.0))
    fig.suptitle(
        "Multi-Metric Radar Charts — h=1 day, test set  (outer edge = best per metric)",
        fontsize=11, y=1.01,
    )
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    _savefig(fig, out_dir / "10_multi_metric_radar.png")


# ── Plot 11: Station NSE correlation heatmap ──────────────────────────────────

def plot_station_nse_correlation(station_df: pd.DataFrame, records: list[dict], out_dir: Path) -> None:
    """Pearson correlation of per-station NSE between all model-variant pairs (h=1)."""
    if station_df.empty:
        return

    test_h1 = station_df[
        (station_df["split"]   == "test") &
        (station_df["horizon"] == 1) &
        station_df["nse"].notna()
    ]
    if test_h1.empty:
        return

    keys = _sorted_model_keys(records)
    # Build wide table: rows = unique_id, columns = model-variant
    wide = {}
    for model, variant in keys:
        sub = test_h1[(test_h1["model"] == model) & (test_h1["variant"] == variant)]
        if sub.empty:
            continue
        wide[_mv_label(model, variant)] = sub.set_index("unique_id")["nse"]

    if len(wide) < 2:
        return

    wide_df = pd.DataFrame(wide)
    corr    = wide_df.corr(method="pearson")

    n    = len(corr)
    fig, ax = plt.subplots(figsize=(max(6, n * 0.7), max(5, n * 0.65)))
    im   = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")

    for i in range(n):
        for j in range(n):
            val = corr.values[i, j]
            txt_color = "black" if abs(val) < 0.85 else "white"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=7 if n <= 15 else 5.5, color=txt_color)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(corr.columns, rotation=40, ha="right", fontsize=7.5)
    ax.set_yticklabels(corr.index,   fontsize=7.5)
    plt.colorbar(im, ax=ax, label="Pearson r  (per-station NSE, h=1)", shrink=0.7)
    ax.set_title(
        "Inter-Model Station NSE Correlation  (test, h=1)\n"
        "High correlation = models fail/succeed at the same stations",
        fontsize=10,
    )
    fig.tight_layout()
    _savefig(fig, out_dir / "11_station_nse_correlation.png")


# ── Plot 12: Bias–RMSE error decomposition scatter ───────────────────────────

def plot_bias_rmse_scatter(metrics_df: pd.DataFrame, records: list[dict], out_dir: Path) -> None:
    """Scatter: |bias| (systematic) vs RMSE (total error), per model-variant-horizon."""
    if metrics_df.empty or "bias" not in metrics_df.columns or "rmse" not in metrics_df.columns:
        return
    tm = _test_macro(metrics_df)
    if tm.empty:
        return

    horizons = sorted(tm["horizon"].unique())
    keys     = _sorted_model_keys(records)
    cmap     = matplotlib.colormaps.get_cmap("tab10").resampled(max(len(MODEL_ORDER), 10))

    h_markers = {1: "o", 2: "s", 3: "^"}

    fig, ax = plt.subplots(figsize=(9, 6))

    for model, variant in keys:
        model_color = cmap(MODEL_ORDER.index(model) if model in MODEL_ORDER else 0)
        for h in horizons:
            row = tm[(tm["model"] == model) & (tm["variant"] == variant) & (tm["horizon"] == h)]
            if row.empty:
                continue
            bias_val = abs(float(row["bias"].iloc[0]))
            rmse_val = float(row["rmse"].iloc[0])
            ax.scatter(
                bias_val, rmse_val,
                color=model_color,
                marker=h_markers.get(h, "o"),
                s=80,
                alpha=0.8,
                edgecolors=VARIANT_COLORS.get(variant, "gray"),
                linewidths=2,
                label=f"{_mv_label(model, variant)} h={h}d" if h == 1 else None,
                zorder=3,
            )
            ax.annotate(
                f"{model[:3]}{variant[0]}h{h}",
                (bias_val, rmse_val),
                fontsize=5.5, alpha=0.6, ha="left", va="bottom",
            )

    # Diagonal guide: RMSE = |Bias| (pure systematic)
    lim = max(ax.get_xlim()[1], ax.get_ylim()[1]) * 1.05
    ax.plot([0, lim], [0, lim], color="gray", linestyle="--", linewidth=1, alpha=0.5,
            label="RMSE = |Bias|  (pure systematic)")
    ax.set_xlim(left=0)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("|Bias|  (m³/s, systematic error)", fontsize=10)
    ax.set_ylabel("RMSE  (m³/s, total error)", fontsize=10)
    ax.set_title(
        "Error Decomposition — |Bias| vs RMSE  (test set, macro)\n"
        "Left = low systematic bias; Bottom = low total error; Edge color = data level",
        fontsize=10,
    )

    # Legend for horizons
    h_handles = [
        plt.scatter([], [], marker=h_markers.get(h, "o"), s=60, color="gray", label=f"h={h}d")
        for h in horizons
    ]
    variant_handles = _variant_legend_handles(records)
    ax.legend(handles=h_handles + variant_handles, fontsize=7.5, loc="upper left")
    _style(ax)
    fig.tight_layout()
    _savefig(fig, out_dir / "12_bias_rmse_scatter.png")


def plot_bias_balance_chart(metrics_df: pd.DataFrame, records: list[dict], out_dir: Path) -> None:
    """Show signed bias as a share of RMSE so bias is comparable across models and horizons."""
    if metrics_df.empty or "bias" not in metrics_df.columns or "rmse" not in metrics_df.columns:
        return

    tm = _test_macro(metrics_df).copy()
    tm = tm.loc[tm["bias"].notna() & tm["rmse"].notna() & (tm["rmse"].abs() > 1e-8)].copy()
    if tm.empty:
        return

    tm["label"] = [_mv_label(model, variant) for model, variant in zip(tm["model"], tm["variant"])]
    tm["bias_share_pct"] = 100.0 * tm["bias"] / tm["rmse"]

    label_strength: dict[str, float] = {}
    for label, values in tm.groupby("label", dropna=False)["bias_share_pct"]:
        finite_values = values.dropna().abs()
        if not finite_values.empty:
            label_strength[str(label)] = float(finite_values.mean())
    if not label_strength:
        return

    label_order = sorted(label_strength, key=label_strength.get, reverse=True)
    horizons = sorted(tm["horizon"].unique())
    max_abs_share = max(float(np.nanmax(np.abs(tm["bias_share_pct"]))), 5.0)
    rmse_min = float(tm["rmse"].min())
    rmse_max = float(tm["rmse"].max())
    y_positions = np.arange(len(label_order))

    fig, axes = plt.subplots(
        1,
        len(horizons),
        figsize=(max(10, 4.2 * len(horizons)), max(6.5, len(label_order) * 0.34 + 1.2)),
        sharey=True,
        constrained_layout=True,
    )
    axes_array = np.atleast_1d(axes)

    for axis, horizon in zip(axes_array, horizons):
        horizon_df = tm.loc[tm["horizon"] == horizon].set_index("label").reindex(label_order).reset_index()
        axis.axvspan(-max_abs_share, 0.0, color="#fae7e6", alpha=0.55)
        axis.axvspan(0.0, max_abs_share, color="#e8f3e8", alpha=0.55)
        axis.axvline(0.0, color="#333333", linewidth=1.2)

        for y_pos, row in zip(y_positions, horizon_df.itertuples(index=False)):
            if not np.isfinite(row.bias_share_pct):
                continue
            marker_size = _marker_size_from_value(float(row.rmse), rmse_min, rmse_max)
            color = VARIANT_COLORS.get(row.variant, "#666666")
            axis.hlines(y_pos, 0.0, float(row.bias_share_pct), color=color, alpha=0.45, linewidth=2.0)
            axis.scatter(
                float(row.bias_share_pct),
                y_pos,
                s=marker_size,
                color=color,
                edgecolors="white",
                linewidths=0.8,
                zorder=3,
            )

        axis.set_xlim(-max_abs_share * 1.08, max_abs_share * 1.08)
        axis.set_title(f"h = {horizon} day", fontsize=10)
        axis.set_xlabel("Signed bias as % of RMSE", fontsize=10)
        axis.xaxis.set_major_formatter(mticker.StrMethodFormatter("{x:.0f}%"))
        _style(axis, grid_axis="x")

        median_values = horizon_df["bias_share_pct"].dropna()
        if not median_values.empty:
            axis.text(
                0.98,
                0.97,
                f"median = {median_values.median():+.1f}%",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none", "pad": 2.5},
            )

    axes_array[0].set_yticks(y_positions)
    axes_array[0].set_yticklabels(label_order, fontsize=7)
    axes_array[0].set_ylabel("Model / data level", fontsize=10)
    axes_array[0].invert_yaxis()
    for axis in axes_array[1:]:
        axis.tick_params(labelleft=False)

    fig.suptitle(
        "Bias balance chart (test, macro)  Left = underprediction, Right = overprediction, marker size tracks RMSE",
        fontsize=11,
        y=1.01,
    )
    _savefig(fig, out_dir / "12_bias_rmse_scatter.png")


# ── Plot 13: Composite ranking ────────────────────────────────────────────────

def plot_composite_ranking(metrics_df: pd.DataFrame, records: list[dict], out_dir: Path) -> None:
    """Rank models across all metrics & horizons; show average rank as horizontal bar."""
    if metrics_df.empty:
        return
    tm = _test_macro(metrics_df)
    if tm.empty:
        return

    horizons         = sorted(tm["horizon"].unique())
    rank_metrics     = [m for m in METRIC_DIRECTION if m in tm.columns]
    if not rank_metrics:
        return

    keys = _sorted_model_keys(records)

    # Build a (model-variant) × (metric × horizon) rank matrix
    rank_data = {}  # (model, variant) -> list of ranks
    for met in rank_metrics:
        direction = METRIC_DIRECTION[met]
        for h in horizons:
            col_vals = {}
            for model, variant in keys:
                row = tm[(tm["model"] == model) & (tm["variant"] == variant) & (tm["horizon"] == h)]
                if not row.empty:
                    col_vals[(model, variant)] = float(row[met].iloc[0])
            if not col_vals:
                continue
            sorted_keys = sorted(col_vals.keys(), key=lambda k: direction * col_vals[k], reverse=True)
            for rank, key in enumerate(sorted_keys, start=1):
                rank_data.setdefault(key, []).append(rank)

    if not rank_data:
        return

    avg_ranks = {k: np.mean(v) for k, v in rank_data.items()}
    sorted_by_rank = sorted(avg_ranks.keys(), key=lambda k: avg_ranks[k])

    labels  = [_mv_label(m, v) for m, v in sorted_by_rank]
    values  = [avg_ranks[k] for k in sorted_by_rank]
    colors  = [VARIANT_COLORS.get(v, "#888") for _, v in sorted_by_rank]
    std_vals = [np.std(rank_data[k]) for k in sorted_by_rank]

    fig, (ax_main, ax_heat) = plt.subplots(
        1, 2,
        figsize=(14, max(5, len(labels) * 0.45 + 2)),
        gridspec_kw={"width_ratios": [1.4, 1]},
    )

    y_pos = list(range(len(labels)))
    bars  = ax_main.barh(y_pos, values, color=colors, alpha=0.85, edgecolor="white")
    ax_main.errorbar(values, y_pos, xerr=std_vals, fmt="none", color="black",
                     linewidth=1, capsize=3)
    for bar, val in zip(bars, values):
        ax_main.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height() / 2,
                     f"{val:.2f}", va="center", fontsize=8)
    ax_main.set_yticks(y_pos)
    ax_main.set_yticklabels(labels, fontsize=8)
    ax_main.set_xlabel("Average rank  (1 = best, error bar = std across metrics × horizons)", fontsize=9)
    ax_main.set_title("Composite Model Ranking", fontsize=10)
    ax_main.invert_xaxis()
    _style(ax_main, grid_axis="x")

    handles = _variant_legend_handles(records)
    ax_main.legend(handles=handles, fontsize=8, loc="lower right")

    # Mini heatmap: per-metric-horizon ranks
    col_names = [f"{m}\nh={h}" for m in rank_metrics for h in horizons]
    heat_mat  = np.full((len(sorted_by_rank), len(col_names)), np.nan)
    for r_idx, key in enumerate(sorted_by_rank):
        c = 0
        for met in rank_metrics:
            direction = METRIC_DIRECTION[met]
            for h in horizons:
                row = tm[(tm["model"] == key[0]) & (tm["variant"] == key[1]) & (tm["horizon"] == h)]
                if not row.empty and met in row.columns:
                    # Compute rank relative to all models for this metric-horizon
                    col_vals = {}
                    for model2, variant2 in keys:
                        row2 = tm[(tm["model"] == model2) & (tm["variant"] == variant2) & (tm["horizon"] == h)]
                        if not row2.empty:
                            col_vals[(model2, variant2)] = float(row2[met].iloc[0])
                    s_keys = sorted(col_vals, key=lambda k: direction * col_vals[k], reverse=True)
                    if key in col_vals:
                        heat_mat[r_idx, c] = s_keys.index(key) + 1
                c += 1

    # Normalize heat_mat for coloring (1=best=green)
    n_models_total = len(sorted_by_rank)
    norm_heat = (n_models_total - heat_mat + 1) / n_models_total if n_models_total > 1 else heat_mat * 0 + 0.5

    im = ax_heat.imshow(norm_heat, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto", interpolation="none")
    for i in range(len(sorted_by_rank)):
        for j in range(len(col_names)):
            v = heat_mat[i, j]
            if np.isfinite(v):
                nv = norm_heat[i, j]
                ax_heat.text(j, i, f"{int(v)}", ha="center", va="center",
                             fontsize=6, color="black" if 0.2 < nv < 0.85 else "white")

    ax_heat.set_xticks(range(len(col_names)))
    ax_heat.set_xticklabels(col_names, fontsize=5.5, rotation=40, ha="right")
    ax_heat.set_yticks(range(len(labels)))
    ax_heat.set_yticklabels(labels, fontsize=7)
    ax_heat.set_title("Rank per metric × horizon", fontsize=9)
    plt.colorbar(im, ax=ax_heat, label="Rank (green=1st)", shrink=0.7)

    n_h = len(horizons)
    for i in range(1, len(rank_metrics)):
        ax_heat.axvline(i * n_h - 0.5, color="white", linewidth=1.5)

    fig.suptitle(
        "Composite Performance Ranking  (test set, macro-averaged, lower rank = better)",
        fontsize=11,
    )
    fig.tight_layout()
    _savefig(fig, out_dir / "13_composite_ranking.png")


# ── Plot 15: Flow regime performance ──────────────────────────────────────────

def plot_flow_regime_performance(records: list[dict], out_dir: Path) -> None:
    """NSE stratified by flow quantile regime (low/medium/high/flood flows), h=1."""
    regime_results = []
    quantile_bins  = [0.0, 0.1, 0.5, 0.9, 1.0]
    regime_labels  = ["Low\n(Q<10%)", "Medium\n(Q10-50%)", "High\n(Q50-90%)", "Flood\n(Q>90%)"]

    for rec in records:
        pred_path = rec["dir"] / "predictions.parquet"
        if not pred_path.exists():
            continue
        try:
            preds = pd.read_parquet(pred_path, columns=["split", "horizon", "y_true", "y_pred", "unique_id"])
        except Exception:
            try:
                preds = pd.read_parquet(pred_path)
            except Exception:
                continue

        test = preds[(preds["split"] == "test") & (preds["horizon"] == 1)].copy()
        if test.empty or "y_true" not in test.columns:
            continue

        # Compute per-station quantile thresholds to avoid size bias
        test = test.dropna(subset=["y_true", "y_pred"])
        if test.empty:
            continue

        # Global quantile thresholds (simpler, still informative)
        global_qs = np.nanquantile(test["y_true"].values, quantile_bins)

        for b_idx in range(len(quantile_bins) - 1):
            lo = global_qs[b_idx]
            hi = global_qs[b_idx + 1]
            mask = (test["y_true"] >= lo) & (test["y_true"] <= hi)
            sub  = test[mask]
            if len(sub) < 20:
                continue
            yt = sub["y_true"].values
            yp = sub["y_pred"].values
            ss_res = np.sum((yt - yp) ** 2)
            ss_tot = np.sum((yt - yt.mean()) ** 2)
            nse    = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
            regime_results.append({
                "model":   rec["model"],
                "variant": rec["variant"],
                "regime":  regime_labels[b_idx],
                "regime_idx": b_idx,
                "nse":     nse,
            })

    if not regime_results:
        print("  Skipping 15_flow_regime_performance: no predictions.parquet found.")
        return

    rdf   = pd.DataFrame(regime_results)
    keys  = [(r["model"], r["variant"]) for _, r in rdf.drop_duplicates(["model", "variant"]).iterrows()]
    keys  = [k for k in _sorted_model_keys([{"model": m, "variant": v} for m, v in keys]) if k in set(keys)]
    cmap  = matplotlib.colormaps.get_cmap("tab10").resampled(max(len(MODEL_ORDER), 10))

    fig, ax = plt.subplots(figsize=(10, 5.5))

    x       = np.arange(len(regime_labels))
    bar_w   = 0.7 / len(keys)
    offsets = np.linspace(-(len(keys) - 1) / 2, (len(keys) - 1) / 2, len(keys)) * bar_w

    for i, (model, variant) in enumerate(keys):
        nse_by_regime = []
        for r_lbl in regime_labels:
            row = rdf[(rdf["model"] == model) & (rdf["variant"] == variant) & (rdf["regime"] == r_lbl)]
            nse_by_regime.append(float(row["nse"].iloc[0]) if not row.empty else np.nan)

        model_color = cmap(MODEL_ORDER.index(model) if model in MODEL_ORDER else 0)
        ax.bar(
            x + offsets[i], nse_by_regime, bar_w,
            color=model_color, alpha=0.82,
            edgecolor=VARIANT_COLORS.get(variant, "gray"), linewidth=1.5,
            label=_mv_label(model, variant),
        )

    ax.axhline(0.0, color="red",   linestyle=":", linewidth=0.8, alpha=0.5)
    ax.axhline(0.75, color="green", linestyle="--", linewidth=1, alpha=0.6, label="NSE=0.75")
    ax.set_xticks(x)
    ax.set_xticklabels(regime_labels, fontsize=9)
    ax.set_ylabel("NSE  (test, h=1, all stations pooled)", fontsize=10)
    ax.set_title(
        "Flow Regime Performance — NSE by Discharge Quantile  "
        "(h=1 day, edge color = data level)",
        fontsize=10,
    )
    ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
    _style(ax)
    fig.tight_layout()
    _savefig(fig, out_dir / "15_flow_regime_performance.png")


# ── Plot 16: Training dynamics (val NSE per epoch) ────────────────────────────

def plot_training_dynamics(epoch_metrics: dict[str, pd.DataFrame], out_dir: Path) -> None:
    """Validation NSE per epoch — shows learning speed and convergence."""
    if not epoch_metrics:
        return

    n     = len(epoch_metrics)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)

    for idx, (label, df) in enumerate(sorted(epoch_metrics.items())):
        ax  = axes[idx // ncols][idx % ncols]
        val = df[
            (df["split"] == "validation") & (df["aggregation"] == "macro") &
            (df["horizon"] == 1) & df["nse"].notna()
        ].sort_values("epoch")

        if val.empty:
            ax.set_visible(False)
            continue

        ax.plot(val["epoch"], val["nse"], color="#4C72B0", linewidth=1.8, label="Val NSE (h=1, macro)")

        # Also show h=2, h=3 if available
        for h, col in [(2, "#DD8452"), (3, "#55A868")]:
            val_h = df[
                (df["split"] == "validation") & (df["aggregation"] == "macro") &
                (df["horizon"] == h) & df["nse"].notna()
            ].sort_values("epoch")
            if not val_h.empty:
                ax.plot(val_h["epoch"], val_h["nse"], color=col, linewidth=1.2,
                        linestyle="--", label=f"Val NSE (h={h})", alpha=0.8)

        best_idx = val["nse"].idxmax()
        best_ep  = val.loc[best_idx, "epoch"]
        best_nse = val.loc[best_idx, "nse"]
        ax.axvline(best_ep, color="gray", linestyle=":", linewidth=1, alpha=0.7)
        ax.scatter([best_ep], [best_nse], color="#4C72B0", s=40, zorder=5,
                   label=f"best e={int(best_ep)} NSE={best_nse:.3f}")

        ax.axhline(0.75, color="green", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.set_title(label, fontsize=8)
        ax.set_xlabel("Epoch", fontsize=7)
        ax.set_ylabel("NSE", fontsize=7)
        ax.legend(fontsize=6)
        _style(ax)

    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("Training Dynamics — Validation NSE per Epoch  (macro h=1,2,3)", fontsize=12, y=1.01)
    fig.tight_layout()
    _savefig(fig, out_dir / "16_training_dynamics.png")


# ── Plot 18: Per-station data-level scatter ────────────────────────────────────

def plot_station_level_progression_quality(station_df: pd.DataFrame, records: list[dict], out_dir: Path) -> None:
    """Improved per-station progression chart with consistent limits and Delta NSE coloring."""
    if station_df.empty:
        return

    test_h1 = station_df[
        (station_df["split"] == "test") &
        (station_df["horizon"] == 1) &
        station_df["nse"].notna()
    ]
    if test_h1.empty:
        return

    models = [m for m in MODEL_ORDER if any(r["model"] == m for r in records)]
    models += [r["model"] for r in records if r["model"] not in MODEL_ORDER and r["model"] not in models]

    compare_pairs = [
        ("context", "weather", "Context vs +Weather"),
        ("context", "hydro_weather", "Context vs +Weather+Soil"),
        ("weather", "hydro_weather", "+Weather vs +Weather+Soil"),
    ]
    compare_pairs = [
        (va, vb, label) for va, vb, label in compare_pairs
        if any(r["variant"] == va for r in records) and any(r["variant"] == vb for r in records)
    ]
    if not compare_pairs:
        return

    panel_data: dict[tuple[int, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    row_limits: dict[int, tuple[float, float]] = {}
    all_deltas: list[np.ndarray] = []

    for row_idx, (va, vb, _label) in enumerate(compare_pairs):
        pair_values: list[np.ndarray] = []
        for model in models:
            sub_a = test_h1[(test_h1["model"] == model) & (test_h1["variant"] == va)].set_index("unique_id")["nse"]
            sub_b = test_h1[(test_h1["model"] == model) & (test_h1["variant"] == vb)].set_index("unique_id")["nse"]
            common = sub_a.index.intersection(sub_b.index)
            if common.empty:
                continue

            xa = sub_a.loc[common].to_numpy(dtype=float)
            xb = sub_b.loc[common].to_numpy(dtype=float)
            delta = xb - xa
            panel_data[(row_idx, model)] = (xa, xb, delta)
            pair_values.extend([xa, xb])
            all_deltas.append(delta)

        if not pair_values:
            continue

        low = min(float(np.min(values)) for values in pair_values)
        high = max(float(np.max(values)) for values in pair_values)
        padding = max((high - low) * 0.08, 0.05)
        row_limits[row_idx] = (low - padding, high + padding)

    if not panel_data:
        return

    delta_extent = max(float(np.max(np.abs(np.concatenate(all_deltas)))), 0.05)
    delta_norm = matplotlib.colors.TwoSlopeNorm(vmin=-delta_extent, vcenter=0.0, vmax=delta_extent)

    n_rows_fig = len(compare_pairs)
    n_cols_fig = len(models)
    fig, axes = plt.subplots(
        n_rows_fig,
        n_cols_fig,
        figsize=(4.5 * n_cols_fig + 1.0, 4.2 * n_rows_fig + 0.6),
        squeeze=False,
        sharex="row",
        sharey="row",
    )

    scatter_handle = None
    for row_idx, (va, vb, label) in enumerate(compare_pairs):
        low, high = row_limits.get(row_idx, (-0.2, 1.0))
        for col_idx, model in enumerate(models):
            axis = axes[row_idx][col_idx]
            payload = panel_data.get((row_idx, model))
            if payload is None:
                axis.set_visible(False)
                continue

            xa, xb, delta = payload
            scatter_handle = axis.scatter(
                xa,
                xb,
                c=delta,
                cmap="PiYG",           # pink = worse, white = neutral, green = better
                norm=delta_norm,
                s=55,
                alpha=0.88,
                edgecolors="white",
                linewidths=0.5,
                zorder=3,
            )
            axis.plot([low, high], [low, high], color="#444444", linestyle="--",
                      linewidth=1.2, alpha=0.6, zorder=2)
            axis.fill_between([low, high], [low, high], [high, high],
                              color="#d5f0d5", alpha=0.18, zorder=1)
            axis.fill_between([low, high], [low, high], [low, low],
                              color="#f5d5d5", alpha=0.14, zorder=1)
            axis.set_xlim(low, high)
            axis.set_ylim(low, high)
            axis.set_aspect("equal", adjustable="box")
            _style(axis, grid_axis="both")

            improved_share = 100.0 * float(np.mean(delta > 0))
            median_delta   = float(np.median(delta))
            axis.text(
                0.03, 0.97,
                f"Δ median = {median_delta:+.2f}\n{improved_share:.0f}% improved",
                transform=axis.transAxes,
                ha="left", va="top", fontsize=7.5,
                bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#cccccc",
                      "boxstyle": "round,pad=0.3"},
            )

            if row_idx == 0:
                axis.set_title(model.upper(), fontsize=10, fontweight="bold", pad=6)
            if row_idx == n_rows_fig - 1:
                axis.set_xlabel(f"NSE  ({VARIANT_LABELS.get(va, va)})", fontsize=8)
            if col_idx == 0:
                axis.set_ylabel(f"NSE  ({VARIANT_LABELS.get(vb, vb)})", fontsize=8)

    # Row labels via fig.text so they are never clipped
    for row_idx, (_va, _vb, label) in enumerate(compare_pairs):
        # Map axis position to figure coordinates
        ax0 = axes[row_idx][0]
        bbox = ax0.get_position()
        y_center = (bbox.y0 + bbox.y1) / 2
        fig.text(
            0.01, y_center, label,
            ha="left", va="center", rotation=90,
            fontsize=8.5, fontweight="bold", color="#333333",
        )

    fig.tight_layout(rect=[0.04, 0, 0.92, 0.96])

    if scatter_handle is not None:
        cbar = fig.colorbar(
            scatter_handle,
            ax=axes.ravel().tolist(),
            shrink=0.72,
            pad=0.02,
            aspect=28,
            label="ΔNSE  (target − baseline)",
        )
        cbar.ax.tick_params(labelsize=8)

    fig.suptitle(
        "Per-station NSE progression across data levels  (test, h=1)\n"
        "Points above diagonal / greener = added data helped that station",
        fontsize=11, y=0.99,
    )
    _savefig(fig, out_dir / "18_station_level_progression.png")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Comprehensive benchmark results plotting.")
    parser.add_argument(
        "--artifacts",
        default=str(PROJECT_ROOT / "artifacts" / "advanced_seq"),
        help="Root directory containing model artifact subdirs  (default: artifacts/advanced_seq)",
    )
    parser.add_argument(
        "--out",
        default=str(PROJECT_ROOT / "plots" / "benchmark"),
        help="Output directory for plots  (default: plots/benchmark/)",
    )
    parser.add_argument(
        "--skip-predictions",
        action="store_true",
        default=False,
        help="Skip plots that require loading predictions.parquet (faster)",
    )
    parser.add_argument(
        "--only", nargs="+", type=int,
        help="Only generate the specified plot numbers (e.g. --only 0 1 9 13)",
    )
    parser.add_argument(
        "--exclude", nargs="+", metavar="MODEL:VARIANT",
        help="Exclude one or more model variants, e.g. --exclude lstm:weather tft:hydro_weather",
    )
    args = parser.parse_args()

    artifacts_root = Path(args.artifacts)
    run_ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.out) / run_ts
    out_dir.mkdir(parents=True, exist_ok=True)

    records = _discover_completed(artifacts_root)
    if not records:
        print(f"No completed models found in {artifacts_root}. Nothing to plot.")
        return

    if args.exclude:
        exclude_set: set[tuple[str, str]] = set()
        for token in args.exclude:
            if ":" in token:
                m_ex, v_ex = token.split(":", 1)
                exclude_set.add((m_ex.strip(), v_ex.strip()))
            else:
                print(f"  Warning: --exclude token '{token}' ignored (expected model:variant format)")
        before = len(records)
        records = [r for r in records if (r["model"], r["variant"]) not in exclude_set]
        print(f"Excluded {before - len(records)} variant(s) via --exclude.")

    print(f"Found {len(records)} completed model(s) in {artifacts_root}:")
    for r in sorted(records, key=lambda x: (x["model"], VARIANT_ORDER.index(x["variant"]) if x["variant"] in VARIANT_ORDER else 99)):
        print(f"  {r['model']:20s}  {r['variant']}")

    only = set(args.only) if args.only else None

    def _want(n: int) -> bool:
        return only is None or n in only

    print("\nLoading aggregated metrics...")
    metrics_df     = _load_metrics(records)
    station_df     = _load_station_metrics(records)
    loss_histories = _load_loss_histories(records)
    epoch_metrics  = _load_epoch_metrics(records)

    skip_pred = args.skip_predictions

    print(f"\nGenerating plots -> {out_dir}/")

    if _want(0):
        plot_scorecard_heatmap(metrics_df, records, out_dir)
    if _want(1):
        plot_ranked_dots(metrics_df, records, out_dir, metric="nse",  fname="01_nse_ranked_by_horizon.png")
    if _want(2):
        plot_ranked_dots(metrics_df, records, out_dir, metric="rmse", fname="02_rmse_ranked_by_horizon.png")
    if _want(3):
        plot_data_level_progression(metrics_df, records, out_dir)
    if _want(4):
        plot_station_rank_curves(station_df, records, out_dir)
    if _want(6):
        plot_loss_curves(loss_histories, out_dir)
    if _want(8) and not skip_pred:
        plot_best_worst_stations(records, station_df, out_dir)
    if _want(9):
        plot_horizon_degradation(metrics_df, records, out_dir)
    if _want(10):
        plot_radar_charts(metrics_df, records, out_dir)
    if _want(12):
        plot_bias_balance_chart(metrics_df, records, out_dir)
    if _want(13):
        plot_composite_ranking(metrics_df, records, out_dir)
    if _want(16):
        plot_training_dynamics(epoch_metrics, out_dir)
    if _want(18):
        plot_station_level_progression_quality(station_df, records, out_dir)

    print("\nDone. Test macro-NSE summary (h=1):")
    if not metrics_df.empty:
        show_cols = [c for c in ["nse", "rmse", "mae", "mase", "smape"] if c in metrics_df.columns]
        if show_cols:
            summary = (
                metrics_df[
                    (metrics_df["split"]       == "test") &
                    (metrics_df["aggregation"] == "macro") &
                    (metrics_df["horizon"]     == 1)
                ]
                .assign(mv=lambda d: d["model"] + " (" + d["variant"] + ")")
                .set_index("mv")[show_cols]
                .sort_values("nse", ascending=False)
            )
            print(summary.to_string(float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
