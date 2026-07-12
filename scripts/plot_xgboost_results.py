"""Generate the standard plot bundle for one XGBoost artifact directory.

This script reads a completed XGBoost artifact directory, reloads the feature
frame referenced by its saved config snapshot, and writes the evaluation plots
defined in src.evaluation.plots.

Use this script after training has finished when you want to regenerate plots
without rerunning the model.

Usage
-----
    .venv/Scripts/python scripts/plot_xgboost_results.py
    .venv/Scripts/python scripts/plot_xgboost_results.py artifacts/advanced_seq/xgboost_advanced_context_w30_h3

Inputs
------
    artifact_dir/config_snapshot.json
    artifact_dir/training_summary.json
    The feature parquet referenced by the saved config snapshot

Outputs
-------
    artifact_dir/plots/
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.plots import generate_xgboost_plot_bundle, load_json
from src.utils.config import load_yaml_config
from src.utils.logging import get_logger


def _resolve_artifact_dir(argv: list[str]) -> Path:
    if len(argv) > 1:
        return (PROJECT_ROOT / argv[1]).resolve()

    config = load_yaml_config(PROJECT_ROOT / "configs" / "xgboost.yaml")
    return (PROJECT_ROOT / config["artifact_dir"]).resolve()


def main(argv: list[str] | None = None) -> None:
    """Regenerate plots for one saved XGBoost artifact directory."""
    active_argv = argv or sys.argv
    logger = get_logger("plot_xgboost_results")

    artifact_dir = _resolve_artifact_dir(active_argv)
    if not artifact_dir.exists():
        raise FileNotFoundError(f"Artifact directory does not exist: {artifact_dir}")

    config_snapshot = load_json(artifact_dir / "config_snapshot.json")
    training_summary = load_json(artifact_dir / "training_summary.json")
    feature_df = pd.read_parquet(PROJECT_ROOT / config_snapshot["feature_frame_path"])
    manifest = generate_xgboost_plot_bundle(
        artifact_dir=artifact_dir,
        feature_df=feature_df,
        feature_columns=training_summary["feature_columns"],
        target_column=config_snapshot.get("target_column", "target"),
        split_column=config_snapshot.get("split_column", "split"),
    )

    logger.info("Generated %s plot outputs under %s", len(manifest), artifact_dir / "plots")


if __name__ == "__main__":
    main()
