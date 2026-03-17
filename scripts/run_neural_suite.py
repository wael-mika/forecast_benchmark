"""Run all neural benchmark models sequentially with per-model log files.

Usage
-----
# Foreground (see output in terminal):
    python scripts/run_neural_suite.py

# Background (detached from terminal, all output to logs/):
    nohup python scripts/run_neural_suite.py > logs/master.log 2>&1 &
    echo "PID: $!"          # note the PID so you can kill it if needed
    tail -f logs/master.log  # follow progress from another terminal

Flags
-----
--force         Re-run all models even if their artifact dirs already exist.
--no-data-prep  Skip regenerating feature parquets (use existing ones).
--weather-only  Run only the weather-augmented variants.
--context-only  Run only the context-only (no weather) variants.
--models X Y Z  Run only the listed model names (ann lstm nhits patchtst tft xlstm mamba hybrid).

Examples
--------
    # Re-run everything from scratch in the background:
    nohup python scripts/run_neural_suite.py --force > logs/master.log 2>&1 &

    # Re-run only LSTM and Mamba (weather variants) in foreground:
    python scripts/run_neural_suite.py --models lstm mamba --weather-only --force
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def _find_python() -> str:
    """Return the Python executable that has the project's dependencies installed.

    Priority:
      1. The .venv inside the project root (created by e.g. `python -m venv .venv`)
      2. The current interpreter (sys.executable) if it already has numpy
      3. Raise with a helpful message
    """
    candidates = [
        PROJECT_ROOT / ".venv" / "bin" / "python",
        PROJECT_ROOT / ".venv" / "bin" / "python3",
        PROJECT_ROOT / "venv" / "bin" / "python",
        PROJECT_ROOT / "venv" / "bin" / "python3",
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate)
    # Fall back to the current interpreter and let it fail loudly if packages are missing
    return sys.executable

# ---------------------------------------------------------------------------
# Configuration: model names → (context config, weather config)
# ---------------------------------------------------------------------------
NEURAL_MODEL_CONFIGS: dict[str, dict[str, str]] = {
    "ann":      {"context": "configs/ann_advanced_context.yaml",      "weather": "configs/ann_advanced_weather.yaml"},
    "lstm":     {"context": "configs/lstm_advanced_context.yaml",     "weather": "configs/lstm_advanced_weather.yaml"},
    "nhits":    {"context": "configs/nhits_advanced_context.yaml",    "weather": "configs/nhits_advanced_weather.yaml"},
    "patchtst": {"context": "configs/patchtst_advanced_context.yaml", "weather": "configs/patchtst_advanced_weather.yaml"},
    "tft":      {"context": "configs/tft_advanced_context.yaml",      "weather": "configs/tft_advanced_weather.yaml"},
    "xlstm":    {"context": "configs/xlstm_advanced_context.yaml",    "weather": "configs/xlstm_advanced_weather.yaml"},
    "mamba":    {"context": "configs/mamba_advanced_context.yaml",    "weather": "configs/mamba_advanced_weather.yaml"},
    "hybrid":   {"context": "configs/hybrid_context.yaml",            "weather": "configs/hybrid_weather.yaml"},
}

DATA_PREP_CONFIGS = [
    "configs/advanced_data_context.yaml",
    "configs/advanced_data_weather.yaml",
]

LOGS_DIR = PROJECT_ROOT / "logs"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(message: str) -> None:
    print(f"[{_ts()}] {message}", flush=True)


def _env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    env.setdefault("KMP_INIT_AT_FORK", "FALSE")
    env.setdefault("OMP_NUM_THREADS", "1")
    env.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib"))
    return env


def _is_completed(config_path: str) -> bool:
    """Return True if this model's artifact dir already has final results."""
    try:
        # Parse the YAML manually to avoid importing yaml from the wrong Python
        import re
        text = (PROJECT_ROOT / config_path).read_text()
        match = re.search(r"^artifact_dir\s*:\s*(.+)$", text, re.MULTILINE)
        if not match:
            return False
        artifact_dir = PROJECT_ROOT / match.group(1).strip()
        return (artifact_dir / "metrics_summary.csv").exists() and \
               (artifact_dir / "training_summary.json").exists()
    except Exception:
        return False


def _run(command: list[str], *, log_path: Path, label: str) -> bool:
    """Run a subprocess, tee output to log_path, return True on success."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    _log(f"START  {label}")
    _log(f"       Log → {log_path.relative_to(PROJECT_ROOT)}")
    start = time.time()

    with log_path.open("w") as log_fh:
        # Write header
        log_fh.write(f"{'='*70}\n")
        log_fh.write(f"Command : {' '.join(command)}\n")
        log_fh.write(f"Started : {_ts()}\n")
        log_fh.write(f"{'='*70}\n\n")
        log_fh.flush()

        proc = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            env=_env(),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        # Stream output: write to log file AND forward to our stdout
        assert proc.stdout is not None
        for line in proc.stdout:
            log_fh.write(line)
            log_fh.flush()
            # Print to master stdout so `tail -f logs/master.log` shows it too
            sys.stdout.write(f"  | {line}")
            sys.stdout.flush()

        proc.wait()

    elapsed = time.time() - start
    minutes, seconds = divmod(int(elapsed), 60)
    duration_str = f"{minutes}m{seconds:02d}s"

    if proc.returncode == 0:
        _log(f"OK     {label}  ({duration_str})")
        return True
    else:
        _log(f"FAILED {label}  (exit={proc.returncode}, {duration_str})")
        _log(f"       See full output in {log_path.relative_to(PROJECT_ROOT)}")
        return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]

    force = "--force" in args
    skip_data_prep = "--no-data-prep" in args
    weather_only = "--weather-only" in args
    context_only = "--context-only" in args

    # --models X Y Z: pick subset
    if "--models" in args:
        idx = args.index("--models")
        requested_models = []
        for token in args[idx + 1:]:
            if token.startswith("--"):
                break
            requested_models.append(token)
        invalid = [m for m in requested_models if m not in NEURAL_MODEL_CONFIGS]
        if invalid:
            _log(f"ERROR: unknown model names: {invalid}. Valid: {list(NEURAL_MODEL_CONFIGS)}")
            return 1
    else:
        requested_models = list(NEURAL_MODEL_CONFIGS)

    variants = []
    if not weather_only:
        variants.append("context")
    if not context_only:
        variants.append("weather")
    if not variants:
        _log("ERROR: --weather-only and --context-only cannot both be set.")
        return 1

    python = _find_python()
    run_exp = str(PROJECT_ROOT / "scripts" / "run_experiment.py")
    prep_script = str(PROJECT_ROOT / "scripts" / "prepare_xgboost_data.py")

    _log("=" * 70)
    _log("Slovak River Discharge — Neural Benchmark Suite")
    _log(f"Models  : {requested_models}")
    _log(f"Variants: {variants}")
    _log(f"Force   : {force}")
    _log("=" * 70)

    results: dict[str, str] = {}  # label → "ok" | "skip" | "fail"

    # ------------------------------------------------------------------
    # Step 1: Data preparation
    # ------------------------------------------------------------------
    if not skip_data_prep:
        _log("\n--- Data preparation ---")
        for data_cfg in DATA_PREP_CONFIGS:
            # Determine which variant this config covers
            variant_tag = "weather" if "weather" in data_cfg else "context"
            if variant_tag not in variants:
                continue

            label = f"data_prep:{variant_tag}"
            log_path = LOGS_DIR / f"data_prep_{variant_tag}.log"

            # Re-generate if forced or if either the context or weather parquet is missing
            try:
                import re
                text = (PROJECT_ROOT / data_cfg).read_text()
                match = re.search(r"^feature_frame_path\s*:\s*(.+)$", text, re.MULTILINE)
                parquet_path = PROJECT_ROOT / match.group(1).strip() if match else None
                already_exists = parquet_path is not None and parquet_path.exists()
            except Exception:
                already_exists = False

            if not force and already_exists:
                _log(f"SKIP   {label}  (parquet exists; pass --force to rebuild)")
                results[label] = "skip"
                continue

            ok = _run([python, prep_script, data_cfg], log_path=log_path, label=label)
            results[label] = "ok" if ok else "fail"
            if not ok:
                _log("Data preparation failed — aborting to avoid training on stale features.")
                return 1
    else:
        _log("Skipping data preparation (--no-data-prep).")

    # ------------------------------------------------------------------
    # Step 2: Train each model
    # ------------------------------------------------------------------
    _log(f"\n--- Training {len(requested_models)} models × {len(variants)} variant(s) ---")

    for model_name in requested_models:
        for variant in variants:
            config_path = NEURAL_MODEL_CONFIGS[model_name][variant]
            label = f"{model_name}:{variant}"
            log_path = LOGS_DIR / f"{model_name}_{variant}.log"

            if not force and _is_completed(config_path):
                _log(f"SKIP   {label}  (artifacts exist; pass --force to retrain)")
                results[label] = "skip"
                continue

            ok = _run([python, run_exp, config_path], log_path=log_path, label=label)
            results[label] = "ok" if ok else "fail"

    # ------------------------------------------------------------------
    # Step 3: Summary
    # ------------------------------------------------------------------
    _log("\n" + "=" * 70)
    _log("SUMMARY")
    _log("=" * 70)
    total = len(results)
    ok_count = sum(1 for v in results.values() if v == "ok")
    skip_count = sum(1 for v in results.values() if v == "skip")
    fail_count = sum(1 for v in results.values() if v == "fail")

    for label, status in results.items():
        icon = "✓" if status == "ok" else ("→" if status == "skip" else "✗")
        _log(f"  {icon}  {label:35s} {status.upper()}")

    _log("")
    _log(f"Total: {total}  |  OK: {ok_count}  |  Skipped: {skip_count}  |  Failed: {fail_count}")

    if fail_count > 0:
        _log("\nSome runs FAILED. Check the individual log files in logs/ for details.")
        return 1

    _log("\nAll done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
