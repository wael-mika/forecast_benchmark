"""Plot benchmark results for all completed models.

Usage
-----
    python scripts/plot_benchmark_results.py
    python scripts/plot_benchmark_results.py --out plots/custom_dir

Output (saved to plots/benchmark/ by default)
------
    comparison_nse.png       — macro-NSE bar chart by model × horizon × variant
    comparison_rmse.png      — macro-RMSE bar chart
    station_nse_box.png      — per-station NSE distribution (box plots) for all models
    loss_curves.png          — train/val loss curves for all completed models
    horizon_degradation.png  — NSE vs forecast horizon (1,2,3-day) per model
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS_ROOT = PROJECT_ROOT / "artifacts" / "advanced_seq"

# Model display order (subset that might have finished)
MODEL_ORDER = ["ann", "lstm", "nhits", "patchtst", "tft", "xlstm", "mamba", "hybrid"]
VARIANT_COLORS = {"context": "#4C72B0", "weather": "#DD8452"}
HORIZON_MARKERS = {1: "o", 2: "s", 3: "^"}


# ── Data loading ────────────────────────────────────────────────────────────

def _discover_completed() -> list[dict]:
    """Return metadata dicts for every artifact dir that has metrics_summary.csv."""
    records = []
    if not ARTIFACTS_ROOT.exists():
        return records
    for d in sorted(ARTIFACTS_ROOT.iterdir()):
        summary_path = d / "metrics_summary.csv"
        if not summary_path.exists():
            continue
        # Parse model name and variant from directory name, e.g. nhits_context_w14_h3
        parts = d.name.split("_")
        # variant is "context" or "weather"
        variant = next((p for p in parts if p in ("context", "weather")), "context")
        # model name is everything before the variant token
        variant_idx = parts.index(variant)
        model = "_".join(parts[:variant_idx])
        records.append({"model": model, "variant": variant, "dir": d})
    return records


def _load_metrics(records: list[dict]) -> pd.DataFrame:
    rows = []
    for rec in records:
        df = pd.read_csv(rec["dir"] / "metrics_summary.csv")
        df["model"] = rec["model"]
        df["variant"] = rec["variant"]
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _load_station_metrics(records: list[dict]) -> pd.DataFrame:
    rows = []
    for rec in records:
        path = rec["dir"] / "metrics_by_station.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["model"] = rec["model"]
        df["variant"] = rec["variant"]
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def _load_loss_histories(records: list[dict]) -> dict[str, pd.DataFrame]:
    out = {}
    for rec in records:
        path = rec["dir"] / "loss_history.csv"
        if not path.exists():
            continue
        label = f"{rec['model']}:{rec['variant']}"
        out[label] = pd.read_csv(path)
    return out


# ── Plotting helpers ─────────────────────────────────────────────────────────

def _model_label(model: str, variant: str) -> str:
    return f"{model.upper()}\n({variant})"


def _sorted_model_keys(records: list[dict]) -> list[tuple[str, str]]:
    """Return (model, variant) pairs in canonical order."""
    seen = []
    for m in MODEL_ORDER:
        for v in ("context", "weather"):
            if any(r["model"] == m and r["variant"] == v for r in records):
                seen.append((m, v))
    # Append any unknown models not in MODEL_ORDER
    for r in records:
        key = (r["model"], r["variant"])
        if key not in seen:
            seen.append(key)
    return seen


# ── Plot 1: Macro-NSE comparison bar chart ───────────────────────────────────

def plot_comparison(metrics_df: pd.DataFrame, records: list[dict], out_dir: Path) -> None:
    if metrics_df.empty:
        return

    test_macro = metrics_df[
        (metrics_df["split"] == "test") &
        (metrics_df["aggregation"] == "macro")
    ].copy()

    keys = _sorted_model_keys(records)
    horizons = sorted(test_macro["horizon"].unique())
    n_keys = len(keys)
    n_horizons = len(horizons)

    for metric, ylabel, fname in [
        ("nse",  "Macro-averaged NSE (↑ better)",  "comparison_nse.png"),
        ("rmse", "Macro-averaged RMSE m³/s (↓ better)", "comparison_rmse.png"),
    ]:
        fig, ax = plt.subplots(figsize=(max(8, n_keys * 1.4), 5))
        x = np.arange(n_keys)
        width = 0.22
        offsets = np.linspace(-(n_horizons - 1) / 2, (n_horizons - 1) / 2, n_horizons) * width

        for h_idx, h in enumerate(horizons):
            vals = []
            colors = []
            for model, variant in keys:
                row = test_macro[
                    (test_macro["model"] == model) &
                    (test_macro["variant"] == variant) &
                    (test_macro["horizon"] == h)
                ]
                vals.append(float(row[metric].iloc[0]) if not row.empty else float("nan"))
                colors.append(VARIANT_COLORS.get(variant, "#888888"))

            bars = ax.bar(x + offsets[h_idx], vals, width,
                          color=colors, alpha=0.85,
                          label=f"h={h}d", edgecolor="white", linewidth=0.5)
            # Value labels
            for bar, v in zip(bars, vals):
                if np.isfinite(v):
                    ax.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + (0.005 if metric == "nse" else 0.5),
                            f"{v:.2f}", ha="center", va="bottom", fontsize=6.5)

        ax.set_xticks(x)
        ax.set_xticklabels([_model_label(m, v) for m, v in keys], fontsize=8)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f"Benchmark Comparison — {metric.upper()} (test set, macro-averaged)", fontsize=11)

        # Variant legend patches
        import matplotlib.patches as mpatches
        handles = [mpatches.Patch(color=c, label=v)
                   for v, c in VARIANT_COLORS.items()
                   if any(r["variant"] == v for r in records)]
        horizon_handles = [plt.Line2D([0], [0], color="gray", lw=8, alpha=0.5, label=f"h={h}d",
                                      solid_capstyle="butt") for h in horizons]
        ax.legend(handles=handles + horizon_handles, fontsize=8, loc="upper right")

        if metric == "nse":
            ax.set_ylim(0, 1.05)
            ax.axhline(0.75, color="green", linestyle="--", linewidth=1, alpha=0.6, label="target NSE=0.75")
        ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

        fig.tight_layout()
        fig.savefig(out_dir / fname, dpi=150)
        plt.close(fig)
        print(f"  Saved {fname}")


# ── Plot 2: Per-station NSE box plots ────────────────────────────────────────

def plot_station_boxes(station_df: pd.DataFrame, records: list[dict], out_dir: Path) -> None:
    if station_df.empty:
        return

    test_st = station_df[
        (station_df["split"] == "test") & (station_df["horizon"] == 1)
    ]
    keys = _sorted_model_keys(records)
    labels = [_model_label(m, v) for m, v in keys]
    data = [
        test_st[(test_st["model"] == m) & (test_st["variant"] == v)]["nse"].dropna().values
        for m, v in keys
    ]
    data = [d for d in data if len(d) > 0]
    labels = [l for l, d in zip(labels, [
        test_st[(test_st["model"] == m) & (test_st["variant"] == v)]["nse"].dropna().values
        for m, v in keys
    ]) if len(d) > 0]

    if not data:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(data) * 1.3), 5))
    bp = ax.boxplot(data, patch_artist=True, notch=False,
                    medianprops=dict(color="black", linewidth=2))

    colors = [VARIANT_COLORS.get(v, "#888") for m, v in keys
              if len(test_st[(test_st["model"] == m) & (test_st["variant"] == v)]["nse"].dropna()) > 0]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("NSE per station (test set, h=1d)", fontsize=10)
    ax.set_title("Per-Station NSE Distribution — h=1 day ahead", fontsize=11)
    ax.axhline(0, color="red", linewidth=0.8, linestyle="--", alpha=0.5, label="NSE=0 (mean baseline)")
    ax.axhline(0.75, color="green", linewidth=0.8, linestyle="--", alpha=0.5, label="target NSE=0.75")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "station_nse_box.png", dpi=150)
    plt.close(fig)
    print("  Saved station_nse_box.png")


# ── Plot 3: Loss curves ───────────────────────────────────────────────────────

def plot_loss_curves(loss_histories: dict[str, pd.DataFrame], out_dir: Path) -> None:
    if not loss_histories:
        return

    n = len(loss_histories)
    ncols = min(3, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)

    for idx, (label, df) in enumerate(sorted(loss_histories.items())):
        ax = axes[idx // ncols][idx % ncols]
        train = df[df["split"] == "train"]
        val = df[df["split"] == "validation"]
        ax.plot(train["epoch"], train["loss"], label="train", color="#4C72B0", linewidth=1.5)
        ax.plot(val["epoch"], val["loss"], label="val", color="#DD8452", linewidth=1.5)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel("Loss", fontsize=8)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    # Hide unused subplots
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle("Training & Validation Loss Curves", fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(out_dir / "loss_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved loss_curves.png")


# ── Plot 4: Horizon degradation ───────────────────────────────────────────────

def plot_horizon_degradation(metrics_df: pd.DataFrame, records: list[dict], out_dir: Path) -> None:
    if metrics_df.empty:
        return

    test_macro = metrics_df[
        (metrics_df["split"] == "test") & (metrics_df["aggregation"] == "macro")
    ]
    keys = _sorted_model_keys(records)
    horizons = sorted(test_macro["horizon"].unique())

    fig, ax = plt.subplots(figsize=(7, 4.5))
    cmap = matplotlib.colormaps.get_cmap("tab10").resampled(len(keys))

    for i, (model, variant) in enumerate(keys):
        nse_vals = []
        for h in horizons:
            row = test_macro[
                (test_macro["model"] == model) &
                (test_macro["variant"] == variant) &
                (test_macro["horizon"] == h)
            ]
            nse_vals.append(float(row["nse"].iloc[0]) if not row.empty else float("nan"))

        linestyle = "-" if variant == "context" else "--"
        ax.plot(horizons, nse_vals, marker="o", linestyle=linestyle,
                color=cmap(i), linewidth=1.8, markersize=6,
                label=f"{model.upper()} ({variant})")

    ax.set_xticks(horizons)
    ax.set_xticklabels([f"{h}-day" for h in horizons])
    ax.set_xlabel("Forecast Horizon", fontsize=10)
    ax.set_ylabel("Macro-NSE (test)", fontsize=10)
    ax.set_title("NSE Degradation with Forecast Horizon", fontsize=11)
    ax.axhline(0.75, color="green", linestyle=":", linewidth=1, alpha=0.6)
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8, bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(alpha=0.3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / "horizon_degradation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  Saved horizon_degradation.png")


# ── Plot 5: Test-period time series per station ───────────────────────────────

def plot_timeseries(records: list[dict], station_df: pd.DataFrame, out_dir: Path) -> None:
    """For each completed model, plot observed vs predicted for every station.

    Stations are sorted worst→best NSE so the most problematic ones appear first.
    Each page = one model. Within the page: one row per station, columns = horizons.
    Saves one PNG per model: timeseries_<model>_<variant>.png
    """
    # Pick a 2-year window at the end of the test period for readability
    WINDOW_YEARS = 2

    for rec in records:
        pred_path = rec["dir"] / "predictions.parquet"
        if not pred_path.exists():
            continue

        preds = pd.read_parquet(pred_path)
        test = preds[preds["split"] == "test"].copy()
        if test.empty:
            continue

        # Use target_ds as the time axis (the day being predicted)
        test["target_ds"] = pd.to_datetime(test["target_ds"])
        t_max = test["target_ds"].max()
        t_min = t_max - pd.DateOffset(years=WINDOW_YEARS)
        test = test[test["target_ds"] >= t_min]

        label = f"{rec['model']}:{rec['variant']}"
        horizons = sorted(test["horizon"].unique())
        stations = sorted(test["unique_id"].unique())

        # Sort stations worst→best by h=1 NSE for this model
        st_nse = station_df[
            (station_df["model"] == rec["model"]) &
            (station_df["variant"] == rec["variant"]) &
            (station_df["split"] == "test") &
            (station_df["horizon"] == 1)
        ].set_index("unique_id")["nse"]
        stations = sorted(
            stations,
            key=lambda s: float(st_nse.get(s, 0.0))
        )  # worst first

        n_stations = len(stations)
        n_horizons = len(horizons)
        fig, axes = plt.subplots(
            n_stations, n_horizons,
            figsize=(6 * n_horizons, 2.8 * n_stations),
            squeeze=False,
            sharex=False,
        )
        fig.suptitle(
            f"Test-Period Forecast vs Observed — {label.upper()}\n"
            f"(last {WINDOW_YEARS} years of test set, stations ordered worst→best NSE)",
            fontsize=11, y=1.002,
        )

        for row_idx, station in enumerate(stations):
            for col_idx, h in enumerate(horizons):
                ax = axes[row_idx][col_idx]
                sub = (
                    test[(test["unique_id"] == station) & (test["horizon"] == h)]
                    .sort_values("target_ds")
                    .drop_duplicates("target_ds")
                )
                if sub.empty:
                    ax.set_visible(False)
                    continue

                nse_val = float(st_nse.get(station, float("nan"))) if h == 1 else float("nan")

                ax.plot(sub["target_ds"], sub["y_true"],
                        color="#2C7BB6", linewidth=1.0, label="Observed", alpha=0.9)
                ax.plot(sub["target_ds"], sub["y_pred"],
                        color="#D7191C", linewidth=0.9, linestyle="--", label="Predicted", alpha=0.85)

                # Shade error area
                ax.fill_between(
                    sub["target_ds"], sub["y_true"], sub["y_pred"],
                    alpha=0.12, color="#D7191C",
                )

                # Title with NSE badge
                nse_str = f"NSE={nse_val:.3f}" if (h == 1 and np.isfinite(nse_val)) else f"h={h}d"
                ax.set_title(f"Station {station}  |  {nse_str}  |  h={h}d",
                             fontsize=8, pad=3)

                ax.xaxis.set_major_formatter(
                    matplotlib.dates.DateFormatter("%Y-%m")
                )
                ax.xaxis.set_major_locator(matplotlib.dates.MonthLocator(interval=4))
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha="right", fontsize=6)
                ax.set_ylabel("Discharge (m³/s)", fontsize=7)
                ax.grid(alpha=0.25)
                ax.spines[["top", "right"]].set_visible(False)

                if row_idx == 0 and col_idx == 0:
                    ax.legend(fontsize=7, loc="upper right")

        fig.tight_layout()
        fname = f"timeseries_{rec['model']}_{rec['variant']}.png"
        fig.savefig(out_dir / fname, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved {fname}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Plot benchmark results for all completed models.")
    parser.add_argument("--out", default=str(PROJECT_ROOT / "plots" / "benchmark"),
                        help="Output directory for plots (default: plots/benchmark/)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    records = _discover_completed()
    if not records:
        print("No completed models found in artifacts/advanced_seq/. Nothing to plot.")
        return

    print(f"Found {len(records)} completed model(s):")
    for r in records:
        print(f"  {r['model']:12s} {r['variant']}")

    print("\nLoading data...")
    metrics_df = _load_metrics(records)
    station_df = _load_station_metrics(records)
    loss_histories = _load_loss_histories(records)

    print(f"\nGenerating plots → {out_dir}/")
    plot_comparison(metrics_df, records, out_dir)
    plot_station_boxes(station_df, records, out_dir)
    plot_loss_curves(loss_histories, out_dir)
    plot_horizon_degradation(metrics_df, records, out_dir)
    plot_timeseries(records, station_df, out_dir)

    print("\nDone. Summary of test macro-NSE:")
    if not metrics_df.empty:
        summary = (
            metrics_df[
                (metrics_df["split"] == "test") &
                (metrics_df["aggregation"] == "macro") &
                (metrics_df["horizon"] == 1)
            ]
            .groupby(["model", "variant"])[["nse", "rmse"]]
            .first()
            .sort_values("nse", ascending=False)
        )
        print(summary.to_string())


if __name__ == "__main__":
    main()
