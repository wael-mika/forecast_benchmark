"""Generate plots for ANN or LSTM experiment artifact directories."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.plots import generate_neural_plot_bundle, load_json
from src.utils.config import load_yaml_config
from src.utils.logging import get_logger


def _resolve_config_path(argv: list[str]) -> Path:
    if len(argv) > 1:
        return (PROJECT_ROOT / argv[1]).resolve()
    return PROJECT_ROOT / "configs" / "lstm.yaml"


def main(argv: list[str] | None = None) -> None:
    """Generate summary plots for a trained ANN or LSTM run."""
    active_argv = argv or sys.argv
    logger = get_logger("plot_neural_results")

    config_path = _resolve_config_path(active_argv)
    config = load_yaml_config(config_path)
    artifact_dir = (PROJECT_ROOT / config["artifact_dir"]).resolve()
    if not artifact_dir.exists():
        raise FileNotFoundError(f"Artifact directory does not exist: {artifact_dir}")

    training_summary = load_json(artifact_dir / "training_summary.json")
    model_label = str(training_summary.get("model_name", config.get("model_name", "neural"))).upper()
    manifest = generate_neural_plot_bundle(artifact_dir=artifact_dir, model_label=model_label)
    logger.info("Generated %s plot outputs under %s", len(manifest), artifact_dir / "plots")


if __name__ == "__main__":
    main()
