"""Fertility sub-model — execution.md Phase 6a. Contract frozen in SKILL.md §2.

"What would this land produce in a normal year." Location-agnostic: fit once
(train_fertility.py) from a panel pooled across configs/district.yaml's training_district_pool
(soil organic carbon + a shared time trend as predictors), then evaluated at *any* lat/lon via
those same predictors — not a lookup of one district's own remembered yield history, which was
the original design and couldn't generalize beyond Bellary (see data/raw/PHASE1_FINDINGS.md).

FEATURE_FUNCTIONS signature is (lat, lon, season_year) -> float, not weather.py's
(lat, lon, season_year, cutoff_date) — these features (soil, calendar year) aren't time-of-season
dependent the way weather is, so there's no cutoff to respect here.
"""
import json
from pathlib import Path

from src.ingestion.soil import SoilDataUnavailableError, _topsoil_value, get_organic_carbon

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def coefficients_path(crop: str) -> Path:
    """Rice keeps the original unprefixed filename (backward compat with the already-validated
    file); every other crop gets its own file — see train_fertility.py's per-crop training."""
    if crop == "Rice":
        return REPO_ROOT / "models" / "fertility_coefficients.json"
    slug = crop.lower().replace(" ", "_")
    return REPO_ROOT / "models" / f"fertility_coefficients_{slug}.json"

FEATURE_FUNCTIONS = {
    "organic_carbon": lambda lat, lon, season_year: _topsoil_value(get_organic_carbon(lat, lon)),
    "year_trend": lambda lat, lon, season_year: float(season_year),
}


class FertilityModelUnavailableError(Exception):
    """Raised when the fitted coefficients file is missing — run train_fertility.py first."""


class FertilityUnavailableError(Exception):
    """Raised when a required feature (e.g. soil data) is unavailable for this point, or the
    fitted formula produces a non-positive/implausible yield (usually extrapolation past the
    training pool's range — see EXTRAPOLATION_MARGIN below)."""


def _load_coefficients(crop: str) -> dict:
    path = coefficients_path(crop)
    if not path.exists():
        raise FertilityModelUnavailableError(
            f"{path} not found — run `python -m src.model.train_fertility` first (see --crop)"
        )
    with open(path) as f:
        return json.load(f)


# The ICRISAT panel only runs through 2015; scoring a 2025/2026 loan means the fitted year_trend
# term is extrapolating a decade+ past its training range. Rather than let that extrapolate
# silently, clip to a margin around the observed training yield range (saved by train_fertility.py)
# — an honesty guard, not a scientific bound.
EXTRAPOLATION_MARGIN = 0.5  # allow up to 50% beyond the observed min/max before flagging


def reference_yield(
    lat: float, lon: float, season_year: int, config: dict, crop: str | None = None
) -> float:
    """Baseline "normal year" yield (kg/ha) at (lat, lon) for season_year — a fitted function of
    location (soil) and time (a trend shared across the training pool), evaluated at any point,
    not looked up from a single district's own history. `crop` defaults to config["crop"].
    """
    crop = crop or config["crop"]
    coeffs = _load_coefficients(crop)
    try:
        raw = coeffs["intercept"]
        for feature_name in coeffs["features"]:
            value = FEATURE_FUNCTIONS[feature_name](lat, lon, season_year)
            raw += coeffs[f"coef_{feature_name}"] * value
    except SoilDataUnavailableError as e:
        raise FertilityUnavailableError(str(e)) from e

    y_min, y_max = coeffs["observed_yield_min"], coeffs["observed_yield_max"]
    lower = y_min * (1 - EXTRAPOLATION_MARGIN)
    upper = y_max * (1 + EXTRAPOLATION_MARGIN)
    if raw < lower or raw > upper:
        raise FertilityUnavailableError(
            f"fitted reference_yield ({raw:.0f} kg/ha) is outside a {EXTRAPOLATION_MARGIN:.0%} "
            f"margin around the training pool's observed range [{y_min:.0f}, {y_max:.0f}] for "
            f"{lat},{lon},{season_year} — likely extrapolating past what the model was fit on"
        )
    return raw


if __name__ == "__main__":
    from src.config import district_centroid, load_config

    config = load_config()
    lat, lon = district_centroid(config)
    for year in [2010, 2015, 2020]:
        print(year, "reference_yield (kg/ha):", round(reference_yield(lat, lon, year, config), 1))
