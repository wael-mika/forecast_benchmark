"""Unified training runner for the Slovak river discharge benchmark.

This script is the main command-line entry point for benchmark training. It
maps each requested model/data-level pair to the correct YAML config, prepares
missing feature parquets when needed, launches training, and stores outputs in
run-specific folders so repeated suites do not overwrite each other.

Use this script when you want to train:
- one model,
- a subset of models across one or more data levels, or
- the full benchmark suite in one command.

Outputs
-------
    runs/{run_name}/{model}_{level}/
        model.pt
        predictions.parquet
        metrics_summary.csv
        metrics_by_station.csv
        loss_history.csv
        epoch_metrics.csv
        training_summary.json
        plots/

    logs/{run_name}/
        data_prep_<level>.log
        <model>_<level>.log
        <model>_<level>_plots.log

Usage
-----
    # Full benchmark: all configured models across all data levels
    .venv/Scripts/python scripts/run_train.py

    # Train a named subset
    .venv/Scripts/python scripts/run_train.py --run-name w30_v2 --models ann lstm --levels weather

    # Override the loss and force CUDA
    .venv/Scripts/python scripts/run_train.py --loss mse --loss-weights 1.0 1.2 1.5 --device cuda

    # Resume a named suite and skip already completed artifacts
    .venv/Scripts/python scripts/run_train.py --run-name full_suite

    # Force retraining even if outputs already exist
    .venv/Scripts/python scripts/run_train.py --run-name full_suite --force

Key Options
-----------
    --models
        Subset of model names to run.
    --levels
        Data levels to run: context, weather, hydro_weather.
    --run-name
        Folder name used under runs/ and logs/.
    --force
        Retrain even if a model-level artifact directory already looks complete.
    --skip-plots
        Skip post-training plot generation.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Project root
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# All valid model x level -> YAML config mappings.
# Absence of a key = that combination has no config (handled gracefully).
MODEL_LEVEL_CONFIGS: dict[str, dict[str, str]] = {
    "xgboost": {
        "context":       "configs/xgboost_advanced_context.yaml",
        "weather":       "configs/xgboost_advanced_weather.yaml",
        "hydro_weather": "configs/xgboost_hydro_weather.yaml",
    },
    "ann": {
        "context":       "configs/ann_advanced_context.yaml",
        "weather":       "configs/ann_advanced_weather.yaml",
        "hydro_weather": "configs/ann_hydro_weather.yaml",
    },
    "lstm": {
        "context":       "configs/lstm_advanced_context.yaml",
        "weather":       "configs/lstm_advanced_weather.yaml",
        "hydro_weather": "configs/lstm_hydro_weather.yaml",
    },
    "nhits": {
        "context":       "configs/nhits_advanced_context.yaml",
        "weather":       "configs/nhits_advanced_weather.yaml",
        "hydro_weather": "configs/nhits_hydro_weather.yaml",
    },
    "patchtst": {
        "context":       "configs/patchtst_advanced_context.yaml",
        "weather":       "configs/patchtst_advanced_weather.yaml",
        "hydro_weather": "configs/patchtst_hydro_weather.yaml",
    },
    "tft": {
        "context":       "configs/tft_advanced_context.yaml",
        "weather":       "configs/tft_advanced_weather.yaml",
        "hydro_weather": "configs/tft_hydro_weather.yaml",
    },
    "xlstm": {
        "context":       "configs/xlstm_advanced_context.yaml",
        "weather":       "configs/xlstm_advanced_weather.yaml",
        "hydro_weather": "configs/xlstm_hydro_weather.yaml",
    },
    "mamba": {
        "context":       "configs/mamba_advanced_context.yaml",
        "weather":       "configs/mamba_advanced_weather.yaml",
        "hydro_weather": "configs/mamba_hydro_weather.yaml",
    },
    "hybrid": {
        # NOTE: hybrid / flownet use non-_advanced_ naming in context & weather
        "context":       "configs/hybrid_context.yaml",
        "weather":       "configs/hybrid_weather.yaml",
        "hydro_weather": "configs/hybrid_hydro_weather.yaml",
    },
    "flownet": {
        "context":       "configs/flownet_context.yaml",
        "weather":       "configs/flownet_weather.yaml",
        "hydro_weather": "configs/flownet_hydro_weather.yaml",
    },
}

# Data preparation: level → (script, optional_config_arg)
DATA_PREP: dict[str, tuple[str, str | None]] = {
    "context":       ("scripts/prepare_features_w30.py", None),    # builds both context+weather
    "weather":       ("scripts/prepare_features_w30.py", None),    # same script, parquet cached
    "hydro_weather": ("scripts/prepare_hydro_features.py",  None),
}

# Known output parquet paths — used to decide whether data prep can be skipped.
DATA_PARQUETS: dict[str, str] = {
    "context":       "data/processed/xgboost/features_context_w30_h3.parquet",
    "weather":       "data/processed/xgboost/features_weather_plus_w30_h3.parquet",
    "hydro_weather": "data/processed/xgboost/features_hydro_weather_w30_h3.parquet",
}

# Files whose presence signals a completed training run.
COMPLETION_SENTINELS = ("metrics_summary.csv", "training_summary.json")

ALL_MODELS  = list(MODEL_LEVEL_CONFIGS)
ALL_LEVELS  = ["context", "weather", "hydro_weather"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_python() -> str:
    """Return the Python executable that has the project's dependencies."""
    candidates = [
        PROJECT_ROOT / ".venv" / "Scripts" / "python.exe",   # Windows venv
        PROJECT_ROOT / ".venv" / "Scripts" / "python3.exe",
        PROJECT_ROOT / ".venv" / "bin" / "python",            # Unix venv
        PROJECT_ROOT / ".venv" / "bin" / "python3",
    ]
    for c in candidates:
        if c.exists():
            return str(c)
    return sys.executable


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_ts()}] {msg}", flush=True)


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("KMP_INIT_AT_FORK",     "FALSE")
    env.setdefault("OMP_NUM_THREADS",      "1")
    env.setdefault("MPLCONFIGDIR",         str(PROJECT_ROOT / ".matplotlib"))
    # Ensure src.* packages are importable in all subprocesses (data prep + training).
    root_str = str(PROJECT_ROOT)
    existing  = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (root_str + os.pathsep + existing) if existing else root_str
    return env


def _is_training_complete(artifact_dir: Path) -> bool:
    return all((artifact_dir / s).exists() for s in COMPLETION_SENTINELS)


def _is_plot_complete(artifact_dir: Path) -> bool:
    return (artifact_dir / "plots" / "plot_manifest.json").exists()


def _cleanup_epoch_checkpoints(artifact_dir: Path) -> None:
    """Remove model_epoch_*.pt files — only model.pt (best) is kept."""
    removed = list(artifact_dir.glob("model_epoch_*.pt"))
    for p in removed:
        p.unlink(missing_ok=True)
    if removed:
        _log(f"  cleaned {len(removed)} epoch checkpoint(s) from {artifact_dir.name}")


def _run(
    command: list[str],
    *,
    log_path: Path,
    label: str,
    header_lines: list[str] | None = None,
) -> tuple[bool, float]:
    """Run *command* as a subprocess, tee output to *log_path* and stdout.

    Returns (success, elapsed_seconds).
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _log(f"START  {label}")
    _log(f"       log -> {log_path.relative_to(PROJECT_ROOT)}")
    start = time.time()

    with log_path.open("w", encoding="utf-8", errors="replace") as fh:
        # --- header block ------------------------------------------------
        fh.write(f"{'=' * 70}\n")
        fh.write(f"Command : {' '.join(command)}\n")
        fh.write(f"Started : {_ts()}\n")
        if header_lines:
            for line in header_lines:
                fh.write(line + "\n")
        fh.write(f"{'=' * 70}\n\n")
        fh.flush()

        # --- stream subprocess output ------------------------------------
        proc = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            fh.write(line)
            fh.flush()
            sys.stdout.write(f"  | {line}")
            sys.stdout.flush()
        proc.wait()

    elapsed = time.time() - start
    mm, ss  = divmod(int(elapsed), 60)
    dur     = f"{mm}m{ss:02d}s"

    if proc.returncode == 0:
        _log(f"OK     {label}  ({dur})")
        return True, elapsed
    else:
        _log(f"FAILED {label}  (exit={proc.returncode}, {dur})")
        _log(f"       see: {log_path.relative_to(PROJECT_ROOT)}")
        return False, elapsed


def _apply_overrides(
    config: dict,
    args: argparse.Namespace,
    model: str,
    level: str,
    run_dir: Path,
) -> tuple[dict, bool]:
    """Mutate *config* in-place with all CLI overrides.  Returns (config, was_modified)."""
    modified = False

    # Output directory — always redirected to runs/{run_name}/{model}_{level}/.
    # Must stay relative to PROJECT_ROOT so run_experiment.py resolves correctly.
    rel = run_dir.relative_to(PROJECT_ROOT)
    config["artifact_dir"] = str(rel).replace("\\", "/")
    modified = True

    # Best-checkpoint-only for neural models (0 = no intermediate checkpoints).
    # XGBoost uses checkpoint_interval differently; leave it at its YAML value.
    if model != "xgboost":
        config["checkpoint_interval"] = 0
        modified = True

    # Device override
    if args.device:
        config["device"] = args.device
        modified = True

    # Loss overrides
    if args.loss:
        config["loss_name"] = args.loss
        modified = True
    if args.loss_weights:
        config["loss_horizon_weights"] = list(args.loss_weights)
        modified = True
    if args.loss_diff_weight is not None:
        config["loss_diff_weight"] = args.loss_diff_weight
        modified = True
    if args.loss_curve_weight is not None:
        config["loss_curvature_weight"] = args.loss_curve_weight
        modified = True

    return config, modified


def _write_temp_config(config: dict) -> Path:
    """Write *config* to a temp YAML file and return its Path (caller must delete)."""
    fd, tmp = tempfile.mkstemp(suffix=".yaml", prefix="run_train_")
    os.close(fd)
    Path(tmp).write_text(
        yaml.dump(config, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )
    return Path(tmp)


def _build_log_header(
    key: str,
    canonical_path: Path,
    artifact_dir: Path,
    config: dict,
) -> list[str]:
    lines = [
        f"RUN     : {key}",
        f"CONFIG  : {canonical_path.relative_to(PROJECT_ROOT)}",
        f"OUT DIR : {artifact_dir}",
        f"DEVICE  : {config.get('device', 'auto')}",
        f"LOSS    : {config.get('loss_name', 'n/a')}",
        "",
        "--- CONFIG DUMP ---",
        yaml.dump(config, default_flow_style=False, allow_unicode=True).rstrip(),
        "--- END CONFIG ---",
    ]
    return lines


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unified training runner — all models × all data levels.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=None,
    )

    # Model / level selection
    p.add_argument(
        "--models", nargs="+", metavar="MODEL", default=None,
        choices=ALL_MODELS,
        help=f"Models to train (default: all {len(ALL_MODELS)}). "
             f"Choices: {ALL_MODELS}",
    )
    p.add_argument(
        "--levels", nargs="+", metavar="LEVEL", default=None,
        choices=ALL_LEVELS,
        help="Data levels to train on (default: all 3). "
             "Choices: context, weather, hydro_weather",
    )

    # Execution control
    p.add_argument("--force", action="store_true",
                   help="Retrain even if run_dir already has completed artifacts.")
    p.add_argument("--skip-plots", action="store_true",
                   help="Skip per-model plot generation after training.")

    # Output
    p.add_argument(
        "--run-name", metavar="NAME", default=None,
        help="Run identifier used as directory name under runs/ and logs/. "
             "Default: YYYYMMDD_HHMMSS timestamp.",
    )

    # Device
    p.add_argument(
        "--device", choices=["auto", "cuda", "cpu", "mps"], default=None,
        help="Override device selection (default: value from each model's YAML, "
             "which is 'auto' — picks CUDA when available).",
    )

    # Loss overrides
    p.add_argument("--loss", choices=["trajectory", "mse", "mae", "smooth_l1"],
                   default=None, help="Override loss_name.")
    p.add_argument("--loss-weights", nargs=3, type=float, metavar="W", default=None,
                   help="Override loss_horizon_weights (exactly 3 values).")
    p.add_argument("--loss-diff-weight", type=float, default=None,
                   metavar="F", help="Override loss_diff_weight.")
    p.add_argument("--loss-curve-weight", type=float, default=None,
                   metavar="F", help="Override loss_curvature_weight.")

    # Logging
    p.add_argument("--log-dir", default="logs", metavar="DIR",
                   help="Root directory for per-model log files (default: logs/).")

    return p.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    # Apply defaults
    models = args.models or ALL_MODELS
    levels = args.levels or ALL_LEVELS
    run_name = args.run_name or datetime.now().strftime("%Y%m%d_%H%M%S")

    runs_root = PROJECT_ROOT / "runs"  / run_name
    log_dir   = PROJECT_ROOT / args.log_dir / run_name
    runs_root.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    python      = _find_python()
    suite_start = time.time()
    results: dict[str, str]   = {}
    timings: dict[str, float] = {}
    data_prep_ok: dict[str, bool] = {}

    _log("=" * 70)
    _log("Slovak River Discharge — Unified Training Runner")
    _log(f"Run name : {run_name}")
    _log(f"Models   : {models}")
    _log(f"Levels   : {levels}")
    _log(f"Output   : {runs_root.relative_to(PROJECT_ROOT)}")
    _log(f"Python   : {python}")
    _log("=" * 70)

    # ------------------------------------------------------------------
    # Step 1: Data preparation (automatic — skipped if parquet exists)
    # ------------------------------------------------------------------
    _log("\n--- Data preparation ---")
    needed_levels = set(levels)
    for level in ALL_LEVELS:
        if level not in needed_levels:
            data_prep_ok[level] = True
            continue

        parquet_path = PROJECT_ROOT / DATA_PARQUETS[level]
        if parquet_path.exists():
            _log(f"  data:{level}  parquet exists -> skip")
            data_prep_ok[level] = True
            continue

        script, cfg_arg = DATA_PREP[level]
        command = [python, script] + ([cfg_arg] if cfg_arg else [])
        prep_label = f"data_prep:{level}"
        ok, elapsed = _run(
            command,
            log_path=log_dir / f"data_prep_{level}.log",
            label=prep_label,
        )
        data_prep_ok[level] = ok
        timings[prep_label] = elapsed
        results[prep_label] = "ok" if ok else "fail"
        if not ok:
            _log(f"  WARNING: data prep FAILED for level '{level}'. "
                 f"Models using this level will be skipped.")

    # ------------------------------------------------------------------
    # Step 2: Training loop (model × level)
    # ------------------------------------------------------------------
    _log(f"\n--- Training: {len(models)} model(s) x {len(levels)} level(s) ---")

    for model in models:
        for level in levels:
            key = f"{model}:{level}"

            # Guard 1: config exists for this combination?
            if level not in MODEL_LEVEL_CONFIGS[model]:
                _log(f"SKIP   {key}  (no config for this model/level combination)")
                results[key] = "skip(no config)"
                continue

            # Guard 2: data for this level is ready?
            if not data_prep_ok.get(level, False):
                _log(f"SKIP   {key}  (data prep failed for level '{level}')")
                results[key] = "skip(data prep failed)"
                continue

            canonical_path = PROJECT_ROOT / MODEL_LEVEL_CONFIGS[model][level]
            if not canonical_path.exists():
                _log(f"SKIP   {key}  (config file missing: {canonical_path})")
                results[key] = "skip(config file missing)"
                continue

            config   = yaml.safe_load(canonical_path.read_text(encoding="utf-8"))
            run_dir  = runs_root / f"{model}_{level}"
            config, was_modified = _apply_overrides(config, args, model, level, run_dir)

            temp_path: Path | None = None
            try:
                if was_modified:
                    temp_path = _write_temp_config(config)
                cfg_to_use  = str(temp_path) if temp_path else str(canonical_path)
                artifact_dir = PROJECT_ROOT / config["artifact_dir"]

                # Guard 3: already complete?
                if not args.force and _is_training_complete(artifact_dir):
                    _log(f"SKIP   {key}  (artifacts already complete in {run_dir.name})")
                    results[key] = "skip(done)"
                    continue

                # Build structured log header
                header = _build_log_header(key, canonical_path, artifact_dir, config)

                # Train
                ok, elapsed = _run(
                    [python, "scripts/run_experiment.py", cfg_to_use],
                    log_path=log_dir / f"{model}_{level}.log",
                    label=key,
                    header_lines=header,
                )
                results[key] = "ok" if ok else "fail"
                timings[key] = elapsed

                # Remove intermediate best-epoch snapshots — keep only model.pt
                if ok and model != "xgboost":
                    _cleanup_epoch_checkpoints(artifact_dir)

                # Per-model plots
                if ok and not args.skip_plots:
                    if not args.force and _is_plot_complete(artifact_dir):
                        _log(f"  plots:{key}  already complete -> skip")
                    else:
                        if model == "xgboost":
                            # plot_xgboost_results.py takes the artifact_dir string
                            plot_cmd = [python, "scripts/plot_xgboost_results.py",
                                        config["artifact_dir"]]
                        else:
                            # plot_neural_results.py takes the config path
                            # temp_path is still alive here (inside the try block)
                            plot_cmd = [python, "scripts/plot_neural_results.py", cfg_to_use]

                        plot_ok, _ = _run(
                            plot_cmd,
                            log_path=log_dir / f"{model}_{level}_plots.log",
                            label=f"{key}:plots",
                        )
                        if not plot_ok:
                            _log(f"  WARNING: plot generation failed for {key} "
                                 f"(training result kept as OK)")

            except Exception as exc:
                _log(f"FAILED {key}  unexpected exception: {exc}")
                results[key] = "fail"

            finally:
                if temp_path and temp_path.exists():
                    temp_path.unlink(missing_ok=True)

    # ------------------------------------------------------------------
    # Step 3: Summary
    # ------------------------------------------------------------------
    total_elapsed = time.time() - suite_start
    hh, rem = divmod(int(total_elapsed), 3600)
    mm, ss  = divmod(rem, 60)
    total_dur = f"{hh}h {mm}m {ss:02d}s" if hh else f"{mm}m {ss:02d}s"

    _log("\n" + "=" * 70)
    _log("TRAINING SUMMARY")
    _log("=" * 70)

    ok_count   = 0
    skip_count = 0
    fail_count = 0
    failed_keys: list[str] = []

    # Print data prep results first
    for level in ALL_LEVELS:
        prep_key = f"data_prep:{level}"
        if prep_key in results:
            status  = results[prep_key]
            icon    = "OK  " if status == "ok" else "FAIL"
            elapsed_s = timings.get(prep_key, 0.0)
            mm2, ss2  = divmod(int(elapsed_s), 60)
            dur_str   = f"  ({mm2}m{ss2:02d}s)" if elapsed_s else ""
            _log(f"  {icon}  {prep_key:40s}  {status.upper()}{dur_str}")

    _log("")

    # Print training results in model × level order
    for model in models:
        for level in levels:
            key    = f"{model}:{level}"
            status = results.get(key, "not_run")
            if status.startswith("ok"):
                icon = "OK  "; ok_count += 1
            elif status.startswith("skip"):
                icon = "SKIP"; skip_count += 1
            else:
                icon = "FAIL"; fail_count += 1
                failed_keys.append(key)

            elapsed_s = timings.get(key, 0.0)
            mm2, ss2  = divmod(int(elapsed_s), 60)
            dur_str   = f"  ({mm2}m{ss2:02d}s)" if elapsed_s else ""
            _log(f"  {icon}  {key:40s}  {status.upper()}{dur_str}")

    _log("")
    _log(f"Total: {ok_count + skip_count + fail_count}  "
         f"|  OK: {ok_count}  |  Skipped: {skip_count}  |  Failed: {fail_count}")
    _log(f"Total elapsed: {total_dur}")
    _log(f"Run artifacts: runs/{run_name}/")
    _log(f"Log files    : {args.log_dir}/{run_name}/")

    if failed_keys:
        _log("")
        _log("FAILED combinations:")
        for fk in failed_keys:
            log_name = fk.replace(":", "_") + ".log"
            _log(f"  FAIL  {fk:40s}  -> {args.log_dir}/{run_name}/{log_name}")
        _log("")
        return 1

    _log("")
    _log("All done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
