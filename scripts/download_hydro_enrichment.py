"""Download and assemble additional hydro reanalysis features.

This script extends the base weather reanalysis with hydrology-relevant ERA5
and ERA5-Land variables such as radiation, evapotranspiration, soil
temperature, soil moisture, and snow water equivalent. It manages the raw
cache files and writes one processed parquet in the same long format as
reanalysis_daily.parquet.

Use this script after the canonical discharge parquet and station metadata are
available. It is the highest-level entry point for hydro enrichment.

Inputs
------
    configs/reanalysis.yaml
    The canonical parquet referenced by that config
    The station metadata referenced by that config

Outputs
-------
    data/raw/reanalysis_open_meteo_hydro/
    data/processed/reanalysis_hydro_daily.parquet

Usage
-----
    .venv/Scripts/python scripts/download_hydro_enrichment.py

Notes
-----
    Daily variables are downloaded in batches. Hourly soil moisture and snow
    variables are downloaded per station and aggregated to daily values before
    the final save.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import urllib.parse
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.reanalysis import build_station_request_specs, StationRequestSpec
from src.utils.config import load_yaml_config
from src.utils.io import ensure_parent_dir, save_parquet
from src.utils.logging import get_logger

logger = get_logger("download_hydro_enrichment")

ARCHIVE_BASE = "https://archive-api.open-meteo.com/v1/archive"

# ---------------------------------------------------------------------------
# Variable definitions (empirically verified against the API)
# ---------------------------------------------------------------------------

# --- daily endpoint variables ---
ERA5_SEAMLESS_DAILY_EXTRA = (
    "shortwave_radiation_sum",
    "wind_speed_10m_mean",
    "et0_fao_evapotranspiration",
)

ERA5_LAND_DAILY_EXTRA = (
    "soil_temperature_0_to_7cm_mean",
    "soil_temperature_7_to_28cm_mean",
    "soil_temperature_28_to_100cm_mean",
    "soil_temperature_100_to_255cm_mean",
)

# --- hourly endpoint variables (will be averaged to daily) ---
ERA5_LAND_HOURLY = (
    "soil_moisture_0_to_7cm",
    "soil_moisture_7_to_28cm",
    "soil_moisture_28_to_100cm",
    "soil_moisture_100_to_255cm",
    "snow_depth_water_equivalent",
)

DAILY_MODEL_SPECS: list[tuple[str, tuple[str, ...], str]] = [
    ("era5_seamless", ERA5_SEAMLESS_DAILY_EXTRA, "era5_"),
    ("era5_land", ERA5_LAND_DAILY_EXTRA, "era5l_"),
]

# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

RATE_LIMIT_WAIT = 3700  # seconds — open-meteo free tier resets at the top of each hour


def _curl_json(url: str, *, max_attempts: int = 3, wait: int = 65) -> Any:
    for attempt in range(1, max_attempts + 1):
        raw = subprocess.check_output(["curl", "-sSL", url], text=True)
        payload = json.loads(raw)
        if isinstance(payload, dict) and payload.get("error"):
            reason = payload.get("reason", "API error")
            if "limit exceeded" in reason.lower():
                # Rate limit: wait for the next hourly bucket reset
                logger.warning(
                    "Rate limit hit (attempt %d/%d). Sleeping %d s for reset…",
                    attempt, max_attempts, RATE_LIMIT_WAIT,
                )
                if attempt == max_attempts:
                    raise ValueError(reason)
                time.sleep(RATE_LIMIT_WAIT)
            else:
                logger.warning("API error (attempt %d/%d): %s", attempt, max_attempts, reason)
                if attempt == max_attempts:
                    raise ValueError(reason)
                time.sleep(wait)
            continue
        return payload
    raise ValueError("All download attempts failed.")


def _build_batch_daily_url(
    specs: list[StationRequestSpec],
    *,
    variables: tuple[str, ...],
    model: str,
) -> str:
    start = min(s.start_date for s in specs)
    end = max(s.end_date for s in specs)
    q = urllib.parse.urlencode({
        "latitude": ",".join(f"{s.latitude:.4f}" for s in specs),
        "longitude": ",".join(f"{s.longitude:.4f}" for s in specs),
        "start_date": start,
        "end_date": end,
        "daily": ",".join(variables),
        "timezone": "Europe/Bratislava",
        "models": model,
    })
    return f"{ARCHIVE_BASE}?{q}"


def _build_single_hourly_url(
    spec: StationRequestSpec,
    *,
    variables: tuple[str, ...],
    model: str,
) -> str:
    q = urllib.parse.urlencode({
        "latitude": f"{spec.latitude:.4f}",
        "longitude": f"{spec.longitude:.4f}",
        "start_date": spec.start_date,
        "end_date": spec.end_date,
        "hourly": ",".join(variables),
        "timezone": "Europe/Bratislava",
        "models": model,
    })
    return f"{ARCHIVE_BASE}?{q}"


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def _load_json_cache(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as fh:
        p = json.load(fh)
    if isinstance(p, dict) and not p.get("error"):
        return p
    return None


def _save_json_cache(path: Path, payload: dict[str, Any]) -> None:
    ensure_parent_dir(path)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh)


# ---------------------------------------------------------------------------
# Daily download (batched across all stations in one API call)
# ---------------------------------------------------------------------------

BATCH_SIZE = 5  # stations per API call — keeps requests well within free-tier limits
INTER_BATCH_SLEEP = 5  # seconds between batches


def _download_daily_model(
    specs: list[StationRequestSpec],
    *,
    raw_dir: Path,
    model: str,
    variables: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    """Return {unique_id: payload}, using per-station cache where available.

    Downloads in small batches of BATCH_SIZE stations to stay within the
    open-meteo free-tier API rate limit.
    """
    cached: dict[str, dict[str, Any]] = {}
    missing: list[StationRequestSpec] = []

    for spec in specs:
        p = _load_json_cache(raw_dir / model / f"{spec.unique_id}.json")
        if p is not None:
            cached[spec.unique_id] = p
        else:
            missing.append(spec)

    if not missing:
        logger.info("[%s daily] All %d stations from cache.", model, len(cached))
        return cached

    batches = [missing[i:i + BATCH_SIZE] for i in range(0, len(missing), BATCH_SIZE)]
    logger.info("[%s daily] Downloading %d stations in %d batch(es)…",
                model, len(missing), len(batches))

    for batch_idx, batch in enumerate(batches, 1):
        if batch_idx > 1:
            time.sleep(INTER_BATCH_SLEEP)
        logger.info("[%s daily] Batch %d/%d (%d stations)…",
                    model, batch_idx, len(batches), len(batch))
        url = _build_batch_daily_url(batch, variables=variables, model=model)
        payloads = _curl_json(url)
        if not isinstance(payloads, list):
            payloads = [payloads]

        if len(payloads) != len(batch):
            raise ValueError(f"Expected {len(batch)} payloads, got {len(payloads)}.")

        for spec, payload in zip(batch, payloads, strict=True):
            cache_path = raw_dir / model / f"{spec.unique_id}.json"
            _save_json_cache(cache_path, payload)
            cached[spec.unique_id] = payload
            logger.info("  cached %s", spec.unique_id)

    return cached


def _daily_payload_to_frame(
    payload: dict[str, Any], *, unique_id: str, prefix: str
) -> pd.DataFrame:
    daily = payload.get("daily", {})
    if "time" not in daily:
        raise ValueError(f"No 'time' in daily payload for {unique_id}.")
    frame = pd.DataFrame({"ds": pd.to_datetime(daily["time"]), "unique_id": unique_id})
    for k, v in daily.items():
        if k != "time":
            frame[f"{prefix}{k}"] = v
    return frame


# ---------------------------------------------------------------------------
# Hourly download (per-station) → aggregate to daily mean
# ---------------------------------------------------------------------------

def _download_hourly_station(
    spec: StationRequestSpec,
    *,
    raw_dir: Path,
    model: str,
    variables: tuple[str, ...],
) -> dict[str, Any]:
    cache_path = raw_dir / f"{model}_hourly" / f"{spec.unique_id}.json"
    cached = _load_json_cache(cache_path)
    if cached is not None:
        return cached

    logger.info("[%s hourly] Downloading station %s (%s → %s)…",
                model, spec.unique_id, spec.start_date, spec.end_date)
    url = _build_single_hourly_url(spec, variables=variables, model=model)
    payload = _curl_json(url)
    _save_json_cache(cache_path, payload)
    return payload


def _hourly_payload_to_daily_frame(
    payload: dict[str, Any], *, unique_id: str, prefix: str
) -> pd.DataFrame:
    hourly = payload.get("hourly", {})
    if "time" not in hourly:
        raise ValueError(f"No 'time' in hourly payload for {unique_id}.")
    frame = pd.DataFrame({"ds": pd.to_datetime(hourly["time"]), "unique_id": unique_id})
    for k, v in hourly.items():
        if k != "time":
            frame[f"{prefix}{k}"] = v
    # Aggregate hourly → daily mean (soil moisture is a state variable, mean is appropriate)
    frame["ds"] = frame["ds"].dt.normalize()
    agg_cols = [c for c in frame.columns if c not in ("ds", "unique_id")]
    daily = (
        frame.groupby(["unique_id", "ds"], sort=False)[agg_cols]
        .mean()
        .reset_index()
    )
    return daily


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Download hydro enrichment variables and save the combined daily parquet."""
    config = load_yaml_config(PROJECT_ROOT / "configs" / "reanalysis.yaml")

    canonical_df = pd.read_parquet(PROJECT_ROOT / config["canonical_data_path"])
    station_specs = build_station_request_specs(
        canonical_df,
        PROJECT_ROOT / config["station_metadata_path"],
        min_start_date=str(config.get("min_start_date", "1984-01-01")),
    )
    logger.info("Processing %d stations.", len(station_specs))

    raw_dir = PROJECT_ROOT / "data" / "raw" / "reanalysis_open_meteo_hydro"
    out_path = PROJECT_ROOT / "data" / "processed" / "reanalysis_hydro_daily.parquet"

    frames_by_station: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    # 1. Daily variables (batched)
    # ------------------------------------------------------------------
    for model, variables, prefix in DAILY_MODEL_SPECS:
        payloads = _download_daily_model(
            station_specs, raw_dir=raw_dir, model=model, variables=variables
        )
        for spec in station_specs:
            uid = spec.unique_id
            if uid not in payloads:
                logger.warning("Missing payload for %s / %s — skipping.", uid, model)
                continue
            frame = _daily_payload_to_frame(payloads[uid], unique_id=uid, prefix=prefix)
            if uid not in frames_by_station:
                frames_by_station[uid] = frame
            else:
                frames_by_station[uid] = frames_by_station[uid].merge(
                    frame, on=["unique_id", "ds"], how="outer"
                )

    # ------------------------------------------------------------------
    # 2. Hourly → daily: soil moisture + snow depth water equivalent
    # ------------------------------------------------------------------
    logger.info("Sleeping 10 s before hourly downloads to avoid rate limits…")
    time.sleep(10)
    for spec in station_specs:
        uid = spec.unique_id
        payload = _download_hourly_station(
            spec,
            raw_dir=raw_dir,
            model="era5_land",
            variables=ERA5_LAND_HOURLY,
        )
        frame = _hourly_payload_to_daily_frame(payload, unique_id=uid, prefix="era5l_")
        if uid not in frames_by_station:
            frames_by_station[uid] = frame
        else:
            frames_by_station[uid] = frames_by_station[uid].merge(
                frame, on=["unique_id", "ds"], how="outer"
            )

    # ------------------------------------------------------------------
    # 3. Combine and save
    # ------------------------------------------------------------------
    result = (
        pd.concat(list(frames_by_station.values()), ignore_index=True)
        .sort_values(["unique_id", "ds"], kind="stable")
        .reset_index(drop=True)
    )

    save_parquet(result, out_path)
    logger.info(
        "Saved %d rows × %d columns → %s",
        len(result), result.shape[1], out_path,
    )
    logger.info("New columns: %s", [c for c in result.columns if c not in ("unique_id", "ds")])


if __name__ == "__main__":
    main()
