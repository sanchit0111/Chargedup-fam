"""Survival sub-model — execution.md Phase 6b. Contract frozen in SKILL.md §2.

Loads the fitted regression (OLS or ridge — see train_survival.py) and applies it to a new
lat/lon/season. Deliberately linear, not a tree ensemble — SKILL.md §7 small-sample discipline
(~25 labeled ICRISAT seasons can't support more model capacity without overfitting).

FEATURE_FUNCTIONS is the single source of truth for feature-name -> function mapping —
train_survival.py imports it too, so the two never drift out of sync on what "rainfall_deficit"
etc. actually means. Composite/interaction features (e.g. dry_heat_interaction) live here rather
than in src/ingestion/weather.py since they're model-level feature engineering, not raw ingestion.
"""
import json
from datetime import date
from pathlib import Path

from src.ingestion.satellite import get_ndvi_anomaly
from src.ingestion.weather import (
    get_dry_spell_days,
    get_dry_spell_days_critical_stage,
    get_dry_spell_days_vegetative,
    get_evapotranspiration_anomaly,
    get_excess_rainfall_days,
    get_excess_rainfall_days_critical_stage,
    get_growing_degree_days,
    get_heat_stress_days,
    get_heat_stress_days_critical_stage,
    get_max_daily_rainfall,
    get_monsoon_onset_delay,
    get_rainfall_deficit,
    get_rainfall_deficit_critical_stage,
    get_rainfall_deficit_vegetative,
    get_water_balance,
    get_water_balance_critical_stage,
)

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
COEFFICIENTS_PATH = REPO_ROOT / "models" / "survival_coefficients.json"


def _dry_heat_interaction(lat: float, lon: float, season_year: int, cutoff_date: date) -> float:
    """A dry spell during a heat wave compounds worse than either alone — a simple product
    interaction, not a new ingestion call (both underlying fetches are already cached)."""
    return (
        get_dry_spell_days(lat, lon, season_year, cutoff_date)
        * get_heat_stress_days(lat, lon, season_year, cutoff_date)
    )


FEATURE_FUNCTIONS = {
    "rainfall_deficit": get_rainfall_deficit,
    "dry_spell_days": get_dry_spell_days,
    "ndvi_anomaly": get_ndvi_anomaly,
    "evapotranspiration_anomaly": get_evapotranspiration_anomaly,
    "heat_stress_days": get_heat_stress_days,
    "growing_degree_days": get_growing_degree_days,
    "water_balance": get_water_balance,
    "monsoon_onset_delay": get_monsoon_onset_delay,
    "dry_heat_interaction": _dry_heat_interaction,
    # Stage-specific (growth_stages in configs/district.yaml — crop-agnostic "vegetative" /
    # "critical_stage" keys) — a dry spell during the critical stage matters far more than the
    # same dry spell during germination; these test whether splitting the whole-window aggregates
    # above by phenological stage recovers signal they wash out.
    "rainfall_deficit_vegetative": get_rainfall_deficit_vegetative,
    "rainfall_deficit_critical_stage": get_rainfall_deficit_critical_stage,
    "dry_spell_days_vegetative": get_dry_spell_days_vegetative,
    "dry_spell_days_critical_stage": get_dry_spell_days_critical_stage,
    "heat_stress_days_critical_stage": get_heat_stress_days_critical_stage,
    "water_balance_critical_stage": get_water_balance_critical_stage,
    # Asymmetric excess-rain features — irrigation (canal/tubewell, the dominant water source for
    # much of Indian agriculture) buffers deficit but not destructive excess rain (waterlogging,
    # lodging), so the risk relationship may be one-sided in a way rainfall_deficit's signed value
    # can't capture with a single linear coefficient.
    "excess_rainfall_days": get_excess_rainfall_days,
    "excess_rainfall_days_critical_stage": get_excess_rainfall_days_critical_stage,
    "max_daily_rainfall": get_max_daily_rainfall,
}

# yield_ratio outside this range means the linear fit is being extrapolated somewhere it has no
# business making claims — clip rather than return a nonsensical multiplier.
MIN_SURVIVAL_FACTOR = 0.1
MAX_SURVIVAL_FACTOR = 1.5


class SurvivalModelUnavailableError(Exception):
    """Raised when the fitted coefficients file is missing — run train_survival.py first."""


def _load_coefficients() -> dict:
    if not COEFFICIENTS_PATH.exists():
        raise SurvivalModelUnavailableError(
            f"{COEFFICIENTS_PATH} not found — run `python -m src.model.train_survival` first"
        )
    with open(COEFFICIENTS_PATH) as f:
        return json.load(f)


def survival_factor(lat: float, lon: float, season_year: int, cutoff_date: date) -> float:
    """Multiplier on reference_yield reflecting this season's shocks so far. 0.6 means this
    season's conditions cut expected yield to 60% of Fertility's baseline potential.
    """
    coeffs = _load_coefficients()

    raw = coeffs["intercept"]
    for feature_name in coeffs["features"]:
        value = FEATURE_FUNCTIONS[feature_name](lat, lon, season_year, cutoff_date)
        raw += coeffs[f"coef_{feature_name}"] * value

    return max(MIN_SURVIVAL_FACTOR, min(MAX_SURVIVAL_FACTOR, raw))


if __name__ == "__main__":
    from src.config import district_centroid, scoring_cutoff_date

    lat, lon = district_centroid()
    for year in [2002, 2011, 2015]:
        cutoff = scoring_cutoff_date(year)
        print(year, "survival_factor:", round(survival_factor(lat, lon, year, cutoff), 3))
