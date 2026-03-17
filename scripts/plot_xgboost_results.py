"""Generate plots for a completed XGBoost experiment artifact directory."""

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
    """Generate round-by-round and summary plots for one XGBoost run."""
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
