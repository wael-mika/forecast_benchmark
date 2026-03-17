from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.plots import discover_round_model_paths
from src.training.train import prepare_inference_frame


def test_prepare_inference_frame_rebuilds_station_id_feature() -> None:
    feature_df = pd.DataFrame(
        {
            "unique_id": ["station_a", "station_b"],
            "target": [1.0, 2.0],
        }
    )

    model_frame = prepare_inference_frame(feature_df, ["target", "station_id_feature"])

    assert "station_id_feature" in model_frame.columns
    assert str(model_frame["station_id_feature"].dtype) == "category"
    assert model_frame["station_id_feature"].astype(str).tolist() == ["station_a", "station_b"]


def test_discover_round_model_paths_uses_checkpoints_and_final_model(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "artifacts"
    checkpoint_dir = artifact_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)

    (checkpoint_dir / "xgboost_1.ubj").write_text("checkpoint", encoding="utf-8")
    (checkpoint_dir / "xgboost_2.ubj").write_text("checkpoint", encoding="utf-8")
    (artifact_dir / "model_rounds_0003.json").write_text("final", encoding="utf-8")

    round_paths = discover_round_model_paths(artifact_dir, final_round=3)

    assert [round_number for round_number, _ in round_paths] == [1, 2, 3]
    assert round_paths[-1][1].name == "model_rounds_0003.json"
