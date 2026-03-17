"""Run the matched advanced benchmark suite with and without weather covariates."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.config import load_yaml_config
from src.utils.logging import get_logger


DATA_CONFIGS = [
    "configs/advanced_data_context.yaml",
    "configs/advanced_data_weather.yaml",
]

MODEL_CONFIGS = {
    "xgboost": [
        "configs/xgboost_advanced_context.yaml",
        "configs/xgboost_advanced_weather.yaml",
    ],
    "ann": [
        "configs/ann_advanced_context.yaml",
        "configs/ann_advanced_weather.yaml",
    ],
    "lstm": [
        "configs/lstm_advanced_context.yaml",
        "configs/lstm_advanced_weather.yaml",
    ],
    "nhits": [
        "configs/nhits_advanced_context.yaml",
        "configs/nhits_advanced_weather.yaml",
    ],
    "patchtst": [
        "configs/patchtst_advanced_context.yaml",
        "configs/patchtst_advanced_weather.yaml",
    ],
    "tft": [
        "configs/tft_advanced_context.yaml",
        "configs/tft_advanced_weather.yaml",
    ],
    "xlstm": [
        "configs/xlstm_advanced_context.yaml",
        "configs/xlstm_advanced_weather.yaml",
    ],
    "mamba": [
        "configs/mamba_advanced_context.yaml",
        "configs/mamba_advanced_weather.yaml",
    ],
    "hybrid": [
        "configs/hybrid_context.yaml",
        "configs/hybrid_weather.yaml",
    ],
}


def _base_environment() -> dict[str, str]:
    environment = os.environ.copy()
    environment.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    environment.setdefault("KMP_INIT_AT_FORK", "FALSE")
    environment.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
    environment.setdefault("OMP_NUM_THREADS", "1")
    return environment


def _run_command(command: list[str], *, logger, environment: dict[str, str]) -> None:
    logger.info("Running: %s", " ".join(command))
    subprocess.run(command, cwd=PROJECT_ROOT, check=True, env=environment)


def _has_completed_training(artifact_dir: Path) -> bool:
    return (artifact_dir / "metrics_summary.csv").exists() and (artifact_dir / "training_summary.json").exists()


def _has_completed_plot_bundle(artifact_dir: Path) -> bool:
    return (artifact_dir / "plots" / "plot_manifest.json").exists()


def main(argv: list[str] | None = None) -> None:
    """Prepare data, train paired context/weather models, and generate comparison plots."""
    active_argv = argv or sys.argv
    logger = get_logger("run_advanced_suite")
    force = "--force" in active_argv[1:]
    requested_models = [argument for argument in active_argv[1:] if argument != "--force"] or list(MODEL_CONFIGS)
    invalid_models = [model_name for model_name in requested_models if model_name not in MODEL_CONFIGS]
    if invalid_models:
        raise ValueError(f"Unknown advanced suite models: {invalid_models}")

    environment = _base_environment()
    python_executable = sys.executable

    for config_path in DATA_CONFIGS:
        config = load_yaml_config(PROJECT_ROOT / config_path)
        feature_frame_path = (PROJECT_ROOT / config["feature_frame_path"]).resolve()
        if force or not feature_frame_path.exists():
            _run_command(
                [python_executable, str(PROJECT_ROOT / "scripts" / "prepare_xgboost_data.py"), config_path],
                logger=logger,
                environment=environment,
            )
        else:
            logger.info("Skipping data prep because %s already exists", feature_frame_path)

    for model_name in requested_models:
        for config_path in MODEL_CONFIGS[model_name]:
            config = load_yaml_config(PROJECT_ROOT / config_path)
            artifact_dir = (PROJECT_ROOT / config["artifact_dir"]).resolve()

            if force or not _has_completed_training(artifact_dir):
                _run_command(
                    [python_executable, str(PROJECT_ROOT / "scripts" / "run_experiment.py"), config_path],
                    logger=logger,
                    environment=environment,
                )
            else:
                logger.info("Skipping completed training run under %s", artifact_dir)

            if not force and _has_completed_plot_bundle(artifact_dir):
                logger.info("Skipping completed plot bundle under %s", artifact_dir / "plots")
                continue

            _run_command(
                [python_executable, str(PROJECT_ROOT / "scripts" / "plot_xgboost_results.py"), config["artifact_dir"]]
                if model_name == "xgboost"
                else [python_executable, str(PROJECT_ROOT / "scripts" / "plot_neural_results.py"), config_path],
                logger=logger,
                environment=environment,
            )

    _run_command(
        [python_executable, str(PROJECT_ROOT / "scripts" / "compare_advanced_results.py"), *requested_models],
        logger=logger,
        environment=environment,
    )
    _run_command(
        [python_executable, str(PROJECT_ROOT / "scripts" / "compare_weather_ablations.py"), *requested_models],
        logger=logger,
        environment=environment,
    )


if __name__ == "__main__":
    main()
