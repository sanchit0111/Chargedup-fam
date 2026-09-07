"""GEE NDVI ingestion — execution.md Phase 3. Contract frozen in skills.md §2.

BLOCKED on credentials: requires GEE_SERVICE_ACCOUNT_JSON (see .env.example), a Google Earth
Engine service account — an account-creation step only the user can complete (non-instant
approval, see execution.md §1). This module is implemented against the real `earthengine-api`
(no mocking) but has NOT been live-tested against actual imagery — flagged per skills.md §4
rather than silently assumed working. Once credentials exist, run this file directly to smoke
test it against the 3 known points below.
"""
from __future__ import annotations

import json
import os
from datetime import date, timedelta
from functools import lru_cache

import ee

_initialized = False


class NDVIUnavailableError(Exception):
    """Raised when Earth Engine has no usable NDVI series for the given point/window."""


def _ensure_initialized() -> None:
    global _initialized
    if _initialized:
        return
    creds_raw = os.environ.get("GEE_SERVICE_ACCOUNT_JSON")
    if not creds_raw:
        raise NDVIUnavailableError(
            "GEE_SERVICE_ACCOUNT_JSON not set — see .env.example and execution.md §1"
        )
    info = json.loads(creds_raw)
    credentials = ee.ServiceAccountCredentials(info["client_email"], key_data=creds_raw)
    ee.Initialize(credentials)
    _initialized = True


def get_ndvi_anomaly(lat: float, lon: float, season_year: int, cutoff_date: date) -> float:
    """NDVI anomaly vs the historical baseline, using only observations dated on or before
    cutoff_date. Never averages in data after cutoff_date (execution.md §0 constraint 2).

    Raises NDVIUnavailableError rather than returning a default — a silent 0 here is a
    silently wrong risk feature (skills.md §7).
    """
    from src.config import load_config, sowing_date

    _ensure_initialized()
    config = load_config()
    sowing = sowing_date(season_year, config)
    if cutoff_date < sowing:
        raise ValueError("cutoff_date is before the season's sowing date")
    window_days = (cutoff_date - sowing).days

    current = _mean_ndvi(lat, lon, sowing, cutoff_date)
    if current is None:
        raise NDVIUnavailableError(f"no MODIS NDVI observations for {lat},{lon} in current window")

    baseline_years = config["ndvi"]["historical_baseline_years"]
    baseline_values = []
    for y in range(season_year - baseline_years, season_year):
        b_sowing = sowing_date(y, config)
        b_cutoff = b_sowing + timedelta(days=window_days)
        value = _mean_ndvi(lat, lon, b_sowing, b_cutoff)
        if value is not None:
            baseline_values.append(value)

    if not baseline_values:
        raise NDVIUnavailableError(
            f"no historical baseline NDVI available for {lat},{lon} before {season_year}"
        )

    baseline_mean = sum(baseline_values) / len(baseline_values)
    return current - baseline_mean


@lru_cache(maxsize=None)
def _mean_ndvi(lat: float, lon: float, start: date, end: date) -> float | None:
    """Mean NDVI (scaled) from MODIS/061/MOD13Q1 over [start, end] at (lat, lon), or None if the
    collection has no images in that window (e.g. before MODIS coverage begins, ~2000). Cached:
    train_survival.py's baseline windows overlap heavily across consecutive season_years, same
    reasoning as weather.py's _daily_rainfall cache.
    """
    point = ee.Geometry.Point([lon, lat])
    collection = (
        ee.ImageCollection("MODIS/061/MOD13Q1")
        .filterDate(start.isoformat(), (end + timedelta(days=1)).isoformat())
        .select("NDVI")
    )
    if collection.size().getInfo() == 0:
        return None
    mean_image = collection.mean()
    result = mean_image.reduceRegion(
        reducer=ee.Reducer.mean(), geometry=point, scale=250
    ).getInfo()
    ndvi_raw = result.get("NDVI")
    if ndvi_raw is None:
        return None
    return ndvi_raw * 0.0001  # MOD13Q1 NDVI scale factor


if __name__ == "__main__":
    from src.config import district_centroid, scoring_cutoff_date

    lat, lon = district_centroid()
    for year in [2002, 2011, 2015]:
        cutoff = scoring_cutoff_date(year)
        anomaly = get_ndvi_anomaly(lat, lon, year, cutoff)
        print(year, "cutoff", cutoff, "NDVI anomaly:", round(anomaly, 4))
