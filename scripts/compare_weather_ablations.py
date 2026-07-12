"""Compare context-only, weather-enabled, and hydro-weather advanced model runs.

This script can discover runs in two ways:

1. **Auto-discover from a directory** (recommended):
   Scans a directory for subdirs whose names encode the model name and tier
   using the pattern  ``{model}_{tier}[_{suffix}]``  where tier is one of
   ``context``, ``weather``, or ``hydro_weather``.  Each subdir must contain
   a ``metrics_summary.csv`` file to be included.

2. **Config-based** (legacy / default when no --dir given):
   Reads artifact paths from the YAML configs checked into the repo.

Three comparisons are produced for every model that has at least two tiers:
    weather_ablation  : weather  − context       (gain from adding weather)
    hydro_ablation    : hydro    − weather        (gain from adding hydro on top)
    full_ablation     : hydro    − context        (total gain over context-only)

Outputs
-------
    {out_dir}/weather_ablation/
    {out_dir}/hydro_ablation/
    {out_dir}/full_ablation/
        *_by_horizon.csv
        *_average.csv
        avg_{rmse,mae,r2,nse}_gain.png
        plot_manifest.json

Usage
-----
    # auto-discover all runs under a directory
    .venv/Scripts/python scripts/compare_weather_ablations.py --dir artifacts/advanced_seq

    # specify a custom output location
    .venv/Scripts/python scripts/compare_weather_ablations.py --dir runs/my_exp --out results/my_exp

    # filter to specific models
    .venv/Scripts/python scripts/compare_weather_ablations.py --dir artifacts/advanced_seq ann lstm flownet

    # legacy config-based mode (no --dir)
    .venv/Scripts/python scripts/compare_weather_ablations.py
    .venv/Scripts/python scripts/compare_weather_ablations.py ann lstm tft
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_yaml_config
from src.utils.io import ensure_parent_dir, save_csv, save_json
from src.utils.logging import get_logger


# ---------------------------------------------------------------------------
# Config-based artifact maps (legacy / default mode)
# ---------------------------------------------------------------------------

CONTEXT_CONFIGS = {
    "xgboost": PROJECT_ROOT / "configs" / "xgboost_advanced_context.yaml",
    "ann":     PROJECT_ROOT / "configs" / "ann_advanced_context.yaml",
    "lstm":    PROJECT_ROOT / "configs" / "lstm_advanced_context.yaml",
    "nhits":   PROJECT_ROOT / "configs" / "nhits_advanced_context.yaml",
    "patchtst":PROJECT_ROOT / "configs" / "patchtst_advanced_context.yaml",
    "tft":     PROJECT_ROOT / "configs" / "tft_advanced_context.yaml",
    "xlstm":   PROJECT_ROOT / "configs" / "xlstm_advanced_context.yaml",
    "mamba":   PROJECT_ROOT / "configs" / "mamba_advanced_context.yaml",
    "hybrid":  PROJECT_ROOT / "configs" / "hybrid_context.yaml",
    "flownet": PROJECT_ROOT / "configs" / "flownet_context.yaml",
}

WEATHER_CONFIGS = {
    "xgboost": PROJECT_ROOT / "configs" / "xgboost_advanced_weather.yaml",
    "ann":     PROJECT_ROOT / "configs" / "ann_advanced_weather.yaml",
    "lstm":    PROJECT_ROOT / "configs" / "lstm_advanced_weather.yaml",
    "nhits":   PROJECT_ROOT / "configs" / "nhits_advanced_weather.yaml",
    "patchtst":PROJECT_ROOT / "configs" / "patchtst_advanced_weather.yaml",
    "tft":     PROJECT_ROOT / "configs" / "tft_advanced_weather.yaml",
    "xlstm":   PROJECT_ROOT / "configs" / "xlstm_advanced_weather.yaml",
    "mamba":   PROJECT_ROOT / "configs" / "mamba_advanced_weather.yaml",
    "hybrid":  PROJECT_ROOT / "configs" / "hybrid_weather.yaml",
    "flownet": PROJECT_ROOT / "configs" / "flownet_weather.yaml",
}

HYDRO_CONFIGS = {
    "xgboost": PROJECT_ROOT / "configs" / "xgboost_hydro_weather.yaml",
    "ann":     PROJECT_ROOT / "configs" / "ann_hydro_weather.yaml",
    "lstm":    PROJECT_ROOT / "configs" / "lstm_hydro_weather.yaml",
    "nhits":   PROJECT_ROOT / "configs" / "nhits_hydro_weather.yaml",
    "patchtst":PROJECT_ROOT / "configs" / "patchtst_hydro_weather.yaml",
    "tft":     PROJECT_ROOT / "configs" / "tft_hydro_weather.yaml",
    "xlstm":   PROJECT_ROOT / "configs" / "xlstm_hydro_weather.yaml",
    "mamba":   PROJECT_ROOT / "configs" / "mamba_hydro_weather.yaml",
    "hybrid":  PROJECT_ROOT / "configs" / "hybrid_hydro_weather.yaml",
    "flownet": PROJECT_ROOT / "configs" / "flownet_hydro_weather.yaml",
}

ALL_MODELS = list(CONTEXT_CONFIGS)
GAIN_METRICS = ["rmse_gain", "mae_gain", "r2_gain", "nse_gain"]

# Regex to extract (model_name, tier) from a directory name like
#   ann_context_w14_h3  /  flownet_hydro_weather_w14_h3
_DIR_RE = re.compile(
    r"^(?P<model>.+?)_(?P<tier>hydro_weather|weather|context)(?:_|$)"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _resolve_artifact_path(config_path: Path) -> Path:
    return (PROJECT_ROOT / load_yaml_config(config_path)["artifact_dir"]).resolve()


def _load_test_micro_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.loc[
        (df["split"] == "test") & (df["aggregation"] == "micro")
    ].copy()


def _plot_gain_bars(
    avg_gain_df: pd.DataFrame,
    *,
    metric: str,
    output_path: Path,
    title: str,
) -> None:
    ordered = avg_gain_df.sort_values(metric, ascending=False, kind="stable")
    fig, ax = plt.subplots(
        figsize=(9.0, max(4.5, len(ordered) * 0.4)),
        constrained_layout=True,
    )
    colors = np.where(ordered[metric] >= 0.0, "#2ca02c", "#d62728")
    ax.barh(ordered["model_name"], ordered[metric], color=colors)
    ax.axvline(0.0, color="#111111", linestyle="--", linewidth=1.2)
    ax.set_xlabel(metric.upper())
    ax.set_title(title)
    ax.grid(True, axis="x", alpha=0.25)
    ensure_parent_dir(output_path)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _compute_gains(
    base_path: Path,
    plus_path: Path,
    model_name: str,
    base_suffix: str,
    plus_suffix: str,
) -> pd.DataFrame | None:
    """Merge two metrics CSVs and return per-horizon gain rows, or None if either is missing."""
    if not base_path.exists() or not plus_path.exists():
        return None

    base_df = _load_test_micro_metrics(base_path).add_suffix(f"_{base_suffix}")
    plus_df = _load_test_micro_metrics(plus_path).add_suffix(f"_{plus_suffix}")
    merged = base_df.merge(
        plus_df,
        left_on=f"horizon_{base_suffix}",
        right_on=f"horizon_{plus_suffix}",
        how="inner",
    )
    merged["model_name"] = model_name
    merged["horizon"] = merged[f"horizon_{base_suffix}"]
    merged["rmse_gain"] = merged[f"rmse_{base_suffix}"] - merged[f"rmse_{plus_suffix}"]
    merged["mae_gain"]  = merged[f"mae_{base_suffix}"]  - merged[f"mae_{plus_suffix}"]
    merged["r2_gain"]   = merged[f"r2_{plus_suffix}"]   - merged[f"r2_{base_suffix}"]
    merged["nse_gain"]  = merged[f"nse_{plus_suffix}"]  - merged[f"nse_{base_suffix}"]
    return merged.loc[
        :,
        [
            "model_name", "horizon",
            f"rmse_{base_suffix}", f"rmse_{plus_suffix}", "rmse_gain",
            f"mae_{base_suffix}",  f"mae_{plus_suffix}",  "mae_gain",
            f"r2_{base_suffix}",   f"r2_{plus_suffix}",   "r2_gain",
            f"nse_{base_suffix}",  f"nse_{plus_suffix}",  "nse_gain",
        ],
    ]


def _save_ablation(
    rows: list[pd.DataFrame],
    output_dir: Path,
    file_stem: str,
    plot_titles: dict[str, str],
    logger,
) -> None:
    effect_df = pd.concat(rows, ignore_index=True).sort_values(
        ["model_name", "horizon"], kind="stable"
    )
    average_df = (
        effect_df.groupby("model_name", dropna=False)[GAIN_METRICS]
        .mean()
        .reset_index()
        .sort_values("rmse_gain", ascending=False, kind="stable")
    )

    save_csv(effect_df,  output_dir / f"{file_stem}_by_horizon.csv")
    save_csv(average_df, output_dir / f"{file_stem}_average.csv")

    plot_paths: dict[str, str] = {
        f"{file_stem}_by_horizon": str(output_dir / f"{file_stem}_by_horizon.csv"),
        f"{file_stem}_average":    str(output_dir / f"{file_stem}_average.csv"),
    }
    for metric, title in plot_titles.items():
        out_path = output_dir / f"avg_{metric}.png"
        _plot_gain_bars(average_df, metric=metric, output_path=out_path, title=title)
        plot_paths[f"avg_{metric}"] = str(out_path)

    save_json(plot_paths, output_dir / "plot_manifest.json")
    logger.info("Saved ablation outputs under %s", output_dir)


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _discover_runs(scan_dir: Path) -> dict[str, dict[str, Path]]:
    """Return {model_name: {tier: metrics_summary.csv path}} for all runs found."""
    runs: dict[str, dict[str, Path]] = {}
    for subdir in sorted(scan_dir.iterdir()):
        if not subdir.is_dir():
            continue
        csv = subdir / "metrics_summary.csv"
        if not csv.exists():
            continue
        m = _DIR_RE.match(subdir.name)
        if not m:
            continue
        model, tier = m.group("model"), m.group("tier")
        runs.setdefault(model, {})[tier] = csv
    return runs


def _runs_from_configs(
    requested_models: list[str],
) -> dict[str, dict[str, Path]]:
    """Build the same {model: {tier: path}} structure from config files."""
    runs: dict[str, dict[str, Path]] = {}
    for model in requested_models:
        entry: dict[str, Path] = {}
        if model in CONTEXT_CONFIGS:
            entry["context"] = _resolve_artifact_path(CONTEXT_CONFIGS[model]) / "metrics_summary.csv"
        if model in WEATHER_CONFIGS:
            entry["weather"] = _resolve_artifact_path(WEATHER_CONFIGS[model]) / "metrics_summary.csv"
        if model in HYDRO_CONFIGS:
            entry["hydro_weather"] = _resolve_artifact_path(HYDRO_CONFIGS[model]) / "metrics_summary.csv"
        if entry:
            runs[model] = entry
    return runs


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Ablation comparison: context / weather / hydro-weather runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--dir",
        metavar="PATH",
        default=None,
        help=(
            "Directory to scan for artifact subdirs.  Subdirs must be named "
            "'{model}_{context|weather|hydro_weather}[_suffix]' and contain "
            "a metrics_summary.csv.  When omitted, artifact paths are read "
            "from the repo YAML configs."
        ),
    )
    parser.add_argument(
        "--out",
        metavar="PATH",
        default=None,
        help=(
            "Output base directory for the three ablation sub-folders.  "
            "Defaults to  {--dir}/ablation  when --dir is given, or "
            "artifacts/advanced_seq  otherwise."
        ),
    )
    parser.add_argument(
        "models",
        nargs="*",
        metavar="MODEL",
        help="Optional list of model names to include (default: all discovered).",
    )
    args = parser.parse_args(argv if argv is not None else sys.argv[1:])

    logger = get_logger("compare_weather_ablations")

    # --- resolve scan dir and collect runs -----------------------------------
    if args.dir is not None:
        scan_dir = Path(args.dir)
        if not scan_dir.is_absolute():
            scan_dir = PROJECT_ROOT / scan_dir
        if not scan_dir.is_dir():
            raise NotADirectoryError(f"--dir does not exist or is not a directory: {scan_dir}")
        all_runs = _discover_runs(scan_dir)
        default_out = scan_dir / "ablation"
    else:
        requested = args.models or ALL_MODELS
        unknown = [m for m in requested if m not in CONTEXT_CONFIGS]
        if unknown:
            raise ValueError(
                f"Unknown model(s): {', '.join(unknown)}.  "
                f"Known models: {', '.join(ALL_MODELS)}"
            )
        all_runs = _runs_from_configs(requested)
        default_out = PROJECT_ROOT / "artifacts" / "advanced_seq"

    # --- apply model filter if --dir was used --------------------------------
    if args.dir is not None and args.models:
        missing = [m for m in args.models if m not in all_runs]
        if missing:
            logger.warning("Models not found in %s: %s", scan_dir, ", ".join(missing))
        all_runs = {m: v for m, v in all_runs.items() if m in args.models}

    if not all_runs:
        raise FileNotFoundError("No runs with metrics_summary.csv were found.")

    out_base = Path(args.out) if args.out else default_out
    if not out_base.is_absolute():
        out_base = PROJECT_ROOT / out_base

    logger.info(
        "Comparing %d model(s): %s", len(all_runs), ", ".join(sorted(all_runs))
    )

    # --- build gain rows for each ablation type ------------------------------
    weather_rows:  list[pd.DataFrame] = []
    hydro_rows:    list[pd.DataFrame] = []
    full_rows:     list[pd.DataFrame] = []
    weather_skipped: list[str] = []
    hydro_skipped:   list[str] = []
    full_skipped:    list[str] = []

    for model_name, tiers in sorted(all_runs.items()):
        ctx_path = tiers.get("context")
        wth_path = tiers.get("weather")
        hyd_path = tiers.get("hydro_weather")

        # weather ablation: context → weather
        df = _compute_gains(ctx_path, wth_path, model_name, "context", "weather") \
            if ctx_path and wth_path else None
        if df is not None:
            weather_rows.append(df)
        elif ctx_path or wth_path:   # at least one tier existed but not both
            weather_skipped.append(model_name)

        # hydro ablation: weather → hydro_weather
        df = _compute_gains(wth_path, hyd_path, model_name, "weather", "hydro") \
            if wth_path and hyd_path else None
        if df is not None:
            hydro_rows.append(df)
        elif wth_path or hyd_path:
            hydro_skipped.append(model_name)

        # full ablation: context → hydro_weather
        df = _compute_gains(ctx_path, hyd_path, model_name, "context", "hydro") \
            if ctx_path and hyd_path else None
        if df is not None:
            full_rows.append(df)
        elif ctx_path or hyd_path:
            full_skipped.append(model_name)

    # --- save results --------------------------------------------------------
    any_saved = False

    if weather_rows:
        if weather_skipped:
            logger.warning("Incomplete weather pairs skipped: %s", ", ".join(sorted(weather_skipped)))
        _save_ablation(
            weather_rows,
            output_dir=out_base / "weather_ablation",
            file_stem="weather_effect",
            plot_titles={
                "rmse_gain": "Average RMSE Gain From Weather Variables",
                "mae_gain":  "Average MAE Gain From Weather Variables",
                "r2_gain":   "Average R\u00b2 Gain From Weather Variables",
                "nse_gain":  "Average NSE Gain From Weather Variables",
            },
            logger=logger,
        )
        any_saved = True
    else:
        logger.warning("No context/weather pairs found — weather ablation skipped.")

    if hydro_rows:
        if hydro_skipped:
            logger.warning("Incomplete hydro pairs skipped: %s", ", ".join(sorted(hydro_skipped)))
        _save_ablation(
            hydro_rows,
            output_dir=out_base / "hydro_ablation",
            file_stem="hydro_effect",
            plot_titles={
                "rmse_gain": "Average RMSE Gain From Hydro Variables (over Weather)",
                "mae_gain":  "Average MAE Gain From Hydro Variables (over Weather)",
                "r2_gain":   "Average R\u00b2 Gain From Hydro Variables (over Weather)",
                "nse_gain":  "Average NSE Gain From Hydro Variables (over Weather)",
            },
            logger=logger,
        )
        any_saved = True
    else:
        logger.warning("No weather/hydro pairs found — hydro ablation skipped.")

    if full_rows:
        if full_skipped:
            logger.warning("Incomplete full pairs skipped: %s", ", ".join(sorted(full_skipped)))
        _save_ablation(
            full_rows,
            output_dir=out_base / "full_ablation",
            file_stem="full_effect",
            plot_titles={
                "rmse_gain": "Total RMSE Gain: Hydro+Weather vs Context-Only",
                "mae_gain":  "Total MAE Gain: Hydro+Weather vs Context-Only",
                "r2_gain":   "Total R\u00b2 Gain: Hydro+Weather vs Context-Only",
                "nse_gain":  "Total NSE Gain: Hydro+Weather vs Context-Only",
            },
            logger=logger,
        )
        any_saved = True
    else:
        logger.warning("No context/hydro pairs found — full ablation skipped.")

    if not any_saved:
        raise FileNotFoundError(
            "No completed model pairs were found for any ablation comparison."
        )


if __name__ == "__main__":
    main()
