"""Download daily ERA5-Land soil temperature and soil moisture caches.

This is a low-level helper for the hydro enrichment workflow. It reads station
locations from the existing era5_seamless cache, batches requests against the
Open-Meteo archive API, and saves one raw JSON file per station under the
era5_land cache directory.

Use this script when you want the raw daily hydro cache only. If you want the
full processed hydro parquet in one step, prefer
scripts/download_hydro_enrichment.py.

Inputs
------
    data/raw/reanalysis_open_meteo_hydro/era5_seamless/*.json

Outputs
-------
    data/raw/reanalysis_open_meteo_hydro/era5_land/<unique_id>.json

Usage
-----
    .venv/Scripts/python scripts/download_hydro_era5l_daily.py
"""

from __future__ import annotations

import json
import subprocess
import time
import urllib.parse
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

ARCHIVE_BASE = "https://archive-api.open-meteo.com/v1/archive"
MODEL = "era5_land"
VARIABLES = (
    "soil_temperature_0_to_7cm_mean",
    "soil_temperature_7_to_28cm_mean",
    "soil_temperature_28_to_100cm_mean",
    "soil_temperature_100_to_255cm_mean",
    "soil_moisture_0_to_7cm_mean",
    "soil_moisture_7_to_28cm_mean",
    "soil_moisture_28_to_100cm_mean",
    "soil_moisture_100_to_255cm_mean",
)

SEAMLESS_CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "reanalysis_open_meteo_hydro" / "era5_seamless"
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "reanalysis_open_meteo_hydro" / "era5_land"

BATCH_SIZE = 5          # stations per API call
INTER_BATCH_SLEEP = 6   # seconds between batches
RATE_LIMIT_WAIT = 3700  # seconds — free tier resets at the top of each hour


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_stations() -> list[dict]:
    """Read station id, lat, lon, start_date, end_date from seamless cache."""
    stations = []
    for path in sorted(SEAMLESS_CACHE_DIR.glob("*.json")):
        with path.open(encoding="utf-8") as fh:
            p = json.load(fh)
        if p.get("error"):
            continue
        times = p.get("daily", {}).get("time", [])
        stations.append({
            "unique_id": path.stem,
            "latitude": p["latitude"],
            "longitude": p["longitude"],
            "start_date": times[0] if times else "1984-01-01",
            "end_date": times[-1] if times else "2024-12-31",
        })
    return stations


def _curl_json(url: str, *, max_attempts: int = 3, wait: int = 65):
    for attempt in range(1, max_attempts + 1):
        raw = subprocess.check_output(["curl", "-sSL", url], text=True)
        payload = json.loads(raw)
        if isinstance(payload, dict) and payload.get("error"):
            reason = payload.get("reason", "API error")
            print(f"  [attempt {attempt}/{max_attempts}] API error: {reason}")
            if "limit exceeded" in reason.lower():
                if attempt == max_attempts:
                    raise ValueError(reason)
                print(f"  Rate limit — sleeping {RATE_LIMIT_WAIT} s …")
                time.sleep(RATE_LIMIT_WAIT)
            else:
                if attempt == max_attempts:
                    raise ValueError(reason)
                time.sleep(wait)
            continue
        return payload
    raise ValueError("All download attempts failed.")


def _build_batch_url(batch: list[dict]) -> str:
    start = min(s["start_date"] for s in batch)
    end = max(s["end_date"] for s in batch)
    q = urllib.parse.urlencode({
        "latitude":   ",".join(f"{s['latitude']:.4f}" for s in batch),
        "longitude":  ",".join(f"{s['longitude']:.4f}" for s in batch),
        "start_date": start,
        "end_date":   end,
        "daily":      ",".join(VARIABLES),
        "timezone":   "Europe/Bratislava",
        "models":     MODEL,
    })
    return f"{ARCHIVE_BASE}?{q}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Download missing daily ERA5-Land cache files for each station."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stations = _load_stations()
    print(f"Total stations: {len(stations)}")

    missing = [s for s in stations if not (OUTPUT_DIR / f"{s['unique_id']}.json").exists()]
    cached_count = len(stations) - len(missing)
    print(f"Already cached: {cached_count}  |  To download: {len(missing)}")

    if not missing:
        print("Nothing to download — all stations already cached.")
        return

    batches = [missing[i:i + BATCH_SIZE] for i in range(0, len(missing), BATCH_SIZE)]
    print(f"Downloading in {len(batches)} batch(es) of up to {BATCH_SIZE} stations each.\n")

    for batch_idx, batch in enumerate(batches, 1):
        if batch_idx > 1:
            print(f"  Sleeping {INTER_BATCH_SLEEP} s between batches …")
            time.sleep(INTER_BATCH_SLEEP)

        ids = [s["unique_id"] for s in batch]
        print(f"Batch {batch_idx}/{len(batches)}: {ids}")

        url = _build_batch_url(batch)
        payloads = _curl_json(url)
        if not isinstance(payloads, list):
            payloads = [payloads]

        if len(payloads) != len(batch):
            raise ValueError(f"Expected {len(batch)} payloads, got {len(payloads)}.")

        for station, payload in zip(batch, payloads):
            out_path = OUTPUT_DIR / f"{station['unique_id']}.json"
            with out_path.open("w", encoding="utf-8") as fh:
                json.dump(payload, fh)
            print(f"  saved {station['unique_id']}")

    print(f"\nDone. {len(missing)} stations written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
