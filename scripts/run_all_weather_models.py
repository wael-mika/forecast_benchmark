"""Run the full weather-aware benchmark suite and regenerate comparison artifacts."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CONFIG_SEQUENCE = [
    "configs/seasonal_naive.yaml",
    "configs/xgboost_weather.yaml",
    "configs/ann_weather.yaml",
    "configs/lstm_weather.yaml",
    "configs/nhits.yaml",
    "configs/patchtst.yaml",
    "configs/tft.yaml",
    "configs/xlstm.yaml",
    "configs/mamba.yaml",
]


def main(argv: list[str] | None = None) -> None:
    """Run each configured weather-aware model sequentially."""
    active_argv = argv or sys.argv
    selected_configs = CONFIG_SEQUENCE if len(active_argv) == 1 else active_argv[1:]

    for config_path in selected_configs:
        subprocess.run(
            [sys.executable, str(PROJECT_ROOT / "scripts" / "run_experiment.py"), config_path],
            check=True,
            cwd=PROJECT_ROOT,
        )

    subprocess.run(
        [sys.executable, str(PROJECT_ROOT / "scripts" / "compare_model_results.py")],
        check=True,
        cwd=PROJECT_ROOT,
    )


if __name__ == "__main__":
    main()
