"""Download hourly ERA5-Land snow water equivalent caches.

This helper downloads the hourly variable that is not available from the daily
endpoint. It reads the station list from the existing era5_seamless cache and
stores one raw JSON payload per station under the hourly cache directory.

Use this script when you need the raw hourly snow cache. If you want the
processed hydro parquet directly, prefer scripts/download_hydro_enrichment.py.

Inputs
------
    data/raw/reanalysis_open_meteo_hydro/era5_seamless/*.json

Outputs
-------
    data/raw/reanalysis_open_meteo_hydro/era5_land_hourly/<unique_id>.json

Usage
-----
    .venv/Scripts/python scripts/download_hydro_era5l_hourly.py
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
    "snow_depth_water_equivalent",
)

SEAMLESS_CACHE_DIR = PROJECT_ROOT / "data" / "raw" / "reanalysis_open_meteo_hydro" / "era5_seamless"
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "reanalysis_open_meteo_hydro" / "era5_land_hourly"

INTER_STATION_SLEEP = 4  # seconds between per-station requests
RATE_LIMIT_WAIT = 3700   # seconds — free tier resets at the top of each hour


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


def _build_station_url(station: dict) -> str:
    q = urllib.parse.urlencode({
        "latitude":   f"{station['latitude']:.4f}",
        "longitude":  f"{station['longitude']:.4f}",
        "start_date": station["start_date"],
        "end_date":   station["end_date"],
        "hourly":     ",".join(VARIABLES),
        "timezone":   "Europe/Bratislava",
        "models":     MODEL,
    })
    return f"{ARCHIVE_BASE}?{q}"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Download missing hourly ERA5-Land cache files for each station."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    stations = _load_stations()
    print(f"Total stations: {len(stations)}")

    missing = [s for s in stations if not (OUTPUT_DIR / f"{s['unique_id']}.json").exists()]
    cached_count = len(stations) - len(missing)
    print(f"Already cached: {cached_count}  |  To download: {len(missing)}\n")

    if not missing:
        print("Nothing to download — all stations already cached.")
        return

    for idx, station in enumerate(missing, 1):
        uid = station["unique_id"]
        print(f"[{idx}/{len(missing)}] Downloading {uid} ({station['start_date']} → {station['end_date']}) …")

        url = _build_station_url(station)
        payload = _curl_json(url)

        out_path = OUTPUT_DIR / f"{uid}.json"
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        print(f"  saved {uid}")

        if idx < len(missing):
            time.sleep(INTER_STATION_SLEEP)

    print(f"\nDone. {len(missing)} stations written to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
