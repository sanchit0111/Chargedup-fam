"""NASA POWER weather ingestion — execution.md Phase 4. Contract frozen in SKILL.md §2.

No auth required — verified live in data/raw/PHASE1_FINDINGS.md. Covers rainfall, dry-spell
length, heat-stress days, evapotranspiration, GDD, and monsoon timing — all from the same free
API, just different `parameters=` values, so one generic cached fetcher backs all of them.

Stage-aware variants (get_*_by_stage) compute the same signals restricted to one phenological
growth stage (configs/district.yaml `growth_stages` — crop-agnostic keys, "vegetative" and
"critical_stage") instead of the whole [sowing, cutoff] window — a dry spell during the
reproductive/critical stage matters far more for yield than the same dry spell during germination,
and a single whole-window aggregate washes that difference out.
"""
from datetime import date, datetime, timedelta
from functools import lru_cache

import requests

POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"
REQUEST_TIMEOUT = 30
RETRIES = 3


class RainfallUnavailableError(Exception):
    """Raised when NASA POWER has no usable series for the given point/window/parameter —
    covers every function in this module, not just rainfall (kept this name to avoid an
    unnecessary rename churn; it's the one exception class this whole module raises)."""


@lru_cache(maxsize=None)
def _daily_power_values(parameter: str, lat: float, lon: float, start: date, end: date) -> dict:
    # cached: train_survival.py's/train_fertility.py's pooled multi-district runs make many calls
    # per district, and baseline windows overlap heavily across consecutive season_years within
    # one district — callers only read this dict, never mutate it, so caching is safe.
    params = {
        "parameters": parameter,
        "community": "AG",
        "longitude": lon,
        "latitude": lat,
        "start": start.strftime("%Y%m%d"),
        "end": end.strftime("%Y%m%d"),
        "format": "JSON",
    }
    resp = None
    last_error = None
    for _ in range(RETRIES):
        try:
            resp = requests.get(POWER_URL, params=params, timeout=REQUEST_TIMEOUT)
            break
        except requests.exceptions.RequestException as e:
            last_error = e
            resp = None
    if resp is None:
        raise RainfallUnavailableError(f"NASA POWER unreachable after retries: {last_error}")
    if resp.status_code != 200:
        raise RainfallUnavailableError(f"NASA POWER returned {resp.status_code} for {lat},{lon}")
    try:
        return resp.json()["properties"]["parameter"][parameter]
    except KeyError as e:
        raise RainfallUnavailableError(
            f"unexpected NASA POWER response shape: {resp.text[:200]}"
        ) from e


def _daily_rainfall(lat: float, lon: float, start: date, end: date) -> dict:
    return _daily_power_values("PRECTOTCORR", lat, lon, start, end)


def _sowing_date(season_year: int, config: dict) -> date:
    return date.fromisoformat(config["season"]["sowing_date_template"].format(year=season_year))


def _offset_window_anomaly(parameter: str, lat: float, lon: float, season_year: int,
                            start_offset_days: int, end_offset_days: int) -> float:
    """Cumulative `parameter` over [sowing+start_offset, sowing+end_offset] minus the historical
    normal for the same offset window in the prior ndvi.historical_baseline_years seasons.
    Generalizes the old whole-window-only version to an arbitrary phenological sub-window; whole-
    window callers just pass start_offset_days=0. Never uses data past whatever end_offset_days
    the caller already clipped to cutoff_date (execution.md §0 constraint 2) — this function
    trusts its caller on that, see get_*_by_stage below for the clipping.
    """
    from src.config import load_config

    config = load_config()
    sowing = _sowing_date(season_year, config)
    start = sowing + timedelta(days=start_offset_days)
    end = sowing + timedelta(days=end_offset_days)

    current_total = sum(_daily_power_values(parameter, lat, lon, start, end).values())

    baseline_years = config["ndvi"]["historical_baseline_years"]
    baseline_totals = []
    for y in range(season_year - baseline_years, season_year):
        b_sowing = _sowing_date(y, config)
        b_start = b_sowing + timedelta(days=start_offset_days)
        b_end = b_sowing + timedelta(days=end_offset_days)
        try:
            baseline_totals.append(
                sum(_daily_power_values(parameter, lat, lon, b_start, b_end).values())
            )
        except RainfallUnavailableError:
            continue  # a missing single historical year shouldn't block the whole calculation

    if not baseline_totals:
        raise RainfallUnavailableError(
            f"no historical baseline {parameter} available for {lat},{lon} before {season_year}"
        )

    normal = sum(baseline_totals) / len(baseline_totals)
    return current_total - normal


def get_rainfall_deficit(lat: float, lon: float, season_year: int, cutoff_date: date) -> float:
    """Positive = surplus rainfall vs. normal, negative = deficit."""
    from src.config import load_config

    config = load_config()
    sowing = _sowing_date(season_year, config)
    if cutoff_date < sowing:
        raise ValueError("cutoff_date is before the season's sowing date")
    return _offset_window_anomaly(
        "PRECTOTCORR", lat, lon, season_year, 0, (cutoff_date - sowing).days
    )


def get_evapotranspiration_anomaly(
    lat: float, lon: float, season_year: int, cutoff_date: date
) -> float:
    """Positive = more evapotranspiration than normal — the atmosphere is pulling more moisture
    out of the crop than usual, a truer water-stress signal than rainfall alone since it reflects
    what the crop actually needed, not just what fell. NASA POWER's EVPTRNS parameter (there is
    no "PET" parameter in this API despite it being a common name elsewhere — verified live).
    """
    from src.config import load_config

    config = load_config()
    sowing = _sowing_date(season_year, config)
    if cutoff_date < sowing:
        raise ValueError("cutoff_date is before the season's sowing date")
    return _offset_window_anomaly(
        "EVPTRNS", lat, lon, season_year, 0, (cutoff_date - sowing).days
    )


def _stage_offsets(season_year: int, stage_name: str, config: dict, cutoff_date: date) -> tuple:
    """(start_offset_days, end_offset_days) for a growth stage, end clipped to cutoff_date so a
    stage-aware feature never reaches past the scoring cutoff (execution.md §0 constraint 2)."""
    sowing = _sowing_date(season_year, config)
    start_offset, end_offset = config["growth_stages"][stage_name]
    end_offset = min(end_offset, (cutoff_date - sowing).days)
    return start_offset, end_offset


def get_rainfall_deficit_by_stage(
    lat: float, lon: float, season_year: int, cutoff_date: date, stage_name: str
) -> float:
    """get_rainfall_deficit, restricted to one growth stage's window instead of the whole season
    so far — see module docstring for why that distinction matters."""
    from src.config import load_config

    config = load_config()
    start_offset, end_offset = _stage_offsets(season_year, stage_name, config, cutoff_date)
    if end_offset <= start_offset:
        raise ValueError(f"stage {stage_name!r} window is empty at or before cutoff_date")
    return _offset_window_anomaly("PRECTOTCORR", lat, lon, season_year, start_offset, end_offset)


DRY_DAY_THRESHOLD_MM = 1.0  # a day with less than this much rain counts as "dry"


def _longest_dry_streak(daily: dict) -> int:
    longest = current = 0
    for _, mm in sorted(daily.items()):
        if mm < DRY_DAY_THRESHOLD_MM:
            current += 1
            longest = max(longest, current)
        else:
            current = 0
    return longest


def get_dry_spell_days(lat: float, lon: float, season_year: int, cutoff_date: date) -> int:
    """Longest run of consecutive dry days (< DRY_DAY_THRESHOLD_MM rainfall) in
    [sowing, cutoff_date]. A normal seasonal rainfall total can hide a multi-week dry spell during
    a critical growth stage that a cumulative-deficit feature alone would miss (execution.md
    Phase 4/6b).
    """
    from src.config import load_config

    config = load_config()
    sowing = _sowing_date(season_year, config)
    if cutoff_date < sowing:
        raise ValueError("cutoff_date is before the season's sowing date")
    return _longest_dry_streak(_daily_rainfall(lat, lon, sowing, cutoff_date))


def get_dry_spell_days_by_stage(
    lat: float, lon: float, season_year: int, cutoff_date: date, stage_name: str
) -> int:
    """get_dry_spell_days, restricted to one growth stage's window instead of the whole season
    so far — see module docstring for why that distinction matters."""
    from src.config import load_config

    config = load_config()
    sowing = _sowing_date(season_year, config)
    start_offset, end_offset = _stage_offsets(season_year, stage_name, config, cutoff_date)
    if end_offset <= start_offset:
        raise ValueError(f"stage {stage_name!r} window is empty at or before cutoff_date")
    start = sowing + timedelta(days=start_offset)
    end = sowing + timedelta(days=end_offset)
    return _longest_dry_streak(_daily_rainfall(lat, lon, start, end))


# IMD's standard 24-hour rainfall-intensity classification: light 2.5-15.5mm, moderate
# 15.6-64.4mm, heavy 64.5-115.5mm, very heavy 115.6-204.4mm, extremely heavy >204.5mm. Started at
# the "heavy" threshold (64.5mm) but that produced a near-always-zero feature at a single point
# over a 15-day critical-stage window (verified: 0 across nearly every sampled year) — too strict
# to have any variance to test against at this sample size. Using "moderate" (15.6mm) instead,
# still a recognized IMD tier, not an arbitrary number picked to force a result.
HEAVY_RAIN_THRESHOLD_MM = 15.6


def _excess_rain_days(daily: dict) -> int:
    return sum(1 for mm in daily.values() if mm >= HEAVY_RAIN_THRESHOLD_MM)


def _max_daily_rain(daily: dict) -> float:
    return max(daily.values()) if daily else 0.0


def get_excess_rainfall_days(lat: float, lon: float, season_year: int, cutoff_date: date) -> int:
    """Count of days with rainfall >= HEAVY_RAIN_THRESHOLD_MM in [sowing, cutoff_date] — captures
    waterlogging/lodging/flowering-damage risk from too much rain, which rainfall_deficit's signed
    total can't isolate: a linear coefficient on a symmetric deficit/surplus feature forces one
    slope to fit both "too dry" and "too wet," which washes out an asymmetric relationship where
    irrigation (canal/tubewell — the dominant water source for much of Indian agriculture) buffers
    deficit far more than it protects against destructive excess rain.
    """
    from src.config import load_config

    config = load_config()
    sowing = _sowing_date(season_year, config)
    if cutoff_date < sowing:
        raise ValueError("cutoff_date is before the season's sowing date")
    return _excess_rain_days(_daily_rainfall(lat, lon, sowing, cutoff_date))


def get_excess_rainfall_days_critical_stage(
    lat: float, lon: float, season_year: int, cutoff_date: date
) -> int:
    """get_excess_rainfall_days, restricted to the critical stage — heavy rain during flowering
    (lodging, pollen washout, panicle damage for rice) is more destructive than the same event
    during vegetative growth."""
    from src.config import load_config

    config = load_config()
    sowing = _sowing_date(season_year, config)
    start_offset, end_offset = _stage_offsets(season_year, "critical_stage", config, cutoff_date)
    if end_offset <= start_offset:
        raise ValueError("critical_stage window is empty at or before cutoff_date")
    start = sowing + timedelta(days=start_offset)
    end = sowing + timedelta(days=end_offset)
    return _excess_rain_days(_daily_rainfall(lat, lon, start, end))


def get_max_daily_rainfall(lat: float, lon: float, season_year: int, cutoff_date: date) -> float:
    """Single wettest day's rainfall (mm) in [sowing, cutoff_date] — a cumulative total or a
    day-count-over-threshold can both understate risk from one genuinely extreme event (flash
    flooding, a single storm that lodges standing rice); this captures peak intensity directly.
    """
    from src.config import load_config

    config = load_config()
    sowing = _sowing_date(season_year, config)
    if cutoff_date < sowing:
        raise ValueError("cutoff_date is before the season's sowing date")
    return _max_daily_rain(_daily_rainfall(lat, lon, sowing, cutoff_date))


# Named per-stage wrappers — src/model/survival.py's FEATURE_FUNCTIONS needs plain
# (lat, lon, season_year, cutoff_date) -> float callables, not a stage_name kwarg. Stage keys
# ("vegetative", "critical_stage") come from configs/district.yaml growth_stages and are
# deliberately crop-agnostic — switching crops means changing those boundaries, not this code.
def get_rainfall_deficit_vegetative(lat: float, lon: float, season_year: int, cutoff_date: date) -> float:
    return get_rainfall_deficit_by_stage(lat, lon, season_year, cutoff_date, "vegetative")


def get_rainfall_deficit_critical_stage(lat: float, lon: float, season_year: int, cutoff_date: date) -> float:
    return get_rainfall_deficit_by_stage(lat, lon, season_year, cutoff_date, "critical_stage")


def get_dry_spell_days_vegetative(lat: float, lon: float, season_year: int, cutoff_date: date) -> int:
    return get_dry_spell_days_by_stage(lat, lon, season_year, cutoff_date, "vegetative")


def get_dry_spell_days_critical_stage(lat: float, lon: float, season_year: int, cutoff_date: date) -> int:
    return get_dry_spell_days_by_stage(lat, lon, season_year, cutoff_date, "critical_stage")


HEAT_STRESS_THRESHOLD_C = 35.0  # general reproductive-stage heat-damage threshold, documented
# and adjustable (same pattern as combine.py's SAFE_COVERAGE_RATIO) — not a crop-specific
# literature figure verified for groundnut specifically, treat as a reasonable default.


def get_heat_stress_days(lat: float, lon: float, season_year: int, cutoff_date: date) -> int:
    """Count of days with max temperature (T2M_MAX) at or above HEAT_STRESS_THRESHOLD_C in
    [sowing, cutoff_date]."""
    from src.config import load_config

    config = load_config()
    sowing = _sowing_date(season_year, config)
    if cutoff_date < sowing:
        raise ValueError("cutoff_date is before the season's sowing date")

    daily = _daily_power_values("T2M_MAX", lat, lon, sowing, cutoff_date)
    return sum(1 for temp in daily.values() if temp >= HEAT_STRESS_THRESHOLD_C)


def get_heat_stress_days_critical_stage(lat: float, lon: float, season_year: int, cutoff_date: date) -> int:
    """get_heat_stress_days, restricted to the critical (reproductive) stage — heat stress during
    reproductive development is more damaging than the same heat during vegetative growth."""
    from src.config import load_config

    config = load_config()
    sowing = _sowing_date(season_year, config)
    start_offset, end_offset = _stage_offsets(season_year, "critical_stage", config, cutoff_date)
    if end_offset <= start_offset:
        raise ValueError("critical_stage window is empty at or before cutoff_date")
    start = sowing + timedelta(days=start_offset)
    end = sowing + timedelta(days=end_offset)
    daily = _daily_power_values("T2M_MAX", lat, lon, start, end)
    return sum(1 for temp in daily.values() if temp >= HEAT_STRESS_THRESHOLD_C)


GDD_BASE_TEMP_C = 10.0  # common general base temperature for field-crop GDD calcs


def get_growing_degree_days(lat: float, lon: float, season_year: int, cutoff_date: date) -> float:
    """Cumulative growing degree days (sum of max(T2M_avg - GDD_BASE_TEMP_C, 0)) over
    [sowing, cutoff_date] — standard agronomic measure of accumulated heat driving crop
    developmental stage, distinct from a raw heat-stress-day count (this captures the whole
    distribution's contribution, not just days over a damage threshold).
    """
    from src.config import load_config

    config = load_config()
    sowing = _sowing_date(season_year, config)
    if cutoff_date < sowing:
        raise ValueError("cutoff_date is before the season's sowing date")

    daily = _daily_power_values("T2M", lat, lon, sowing, cutoff_date)
    return sum(max(t - GDD_BASE_TEMP_C, 0) for t in daily.values())


def get_water_balance(lat: float, lon: float, season_year: int, cutoff_date: date) -> float:
    """Cumulative rainfall minus cumulative evapotranspiration (P-ET) over [sowing, cutoff_date]
    — the standard soil-moisture-balance proxy, more principled than rainfall or ET alone since it
    reflects net water actually available to the crop. Positive = surplus, negative = deficit.
    """
    from src.config import load_config

    config = load_config()
    sowing = _sowing_date(season_year, config)
    if cutoff_date < sowing:
        raise ValueError("cutoff_date is before the season's sowing date")

    rain = sum(_daily_rainfall(lat, lon, sowing, cutoff_date).values())
    et = sum(_daily_power_values("EVPTRNS", lat, lon, sowing, cutoff_date).values())
    return rain - et


def get_water_balance_critical_stage(lat: float, lon: float, season_year: int, cutoff_date: date) -> float:
    """get_water_balance, restricted to the critical (reproductive) stage."""
    from src.config import load_config

    config = load_config()
    sowing = _sowing_date(season_year, config)
    start_offset, end_offset = _stage_offsets(season_year, "critical_stage", config, cutoff_date)
    if end_offset <= start_offset:
        raise ValueError("critical_stage window is empty at or before cutoff_date")
    start = sowing + timedelta(days=start_offset)
    end = sowing + timedelta(days=end_offset)
    rain = sum(_daily_rainfall(lat, lon, start, end).values())
    et = sum(_daily_power_values("EVPTRNS", lat, lon, start, end).values())
    return rain - et


MONSOON_SEARCH_START_OFFSET_DAYS = 45  # start searching this many days before template sowing
ONSET_RAIN_THRESHOLD_MM = 15.0  # 3-day cumulative rainfall marking "monsoon has arrived"


def get_monsoon_onset_delay(lat: float, lon: float, season_year: int, cutoff_date: date) -> int:
    """Days between the template sowing date and the actual monsoon onset (first 3-day window
    with >= ONSET_RAIN_THRESHOLD_MM cumulative rainfall, searched from
    MONSOON_SEARCH_START_OFFSET_DAYS before the template date through cutoff_date). Positive =
    late onset, negative = early. Distinct from cumulative rainfall deficit — captures *when* the
    season effectively started, not just how much fell by the cutoff.
    """
    from src.config import load_config

    config = load_config()
    template_sowing = _sowing_date(season_year, config)
    search_start = template_sowing - timedelta(days=MONSOON_SEARCH_START_OFFSET_DAYS)
    if cutoff_date < search_start:
        raise ValueError("cutoff_date is before the monsoon search window starts")

    daily = _daily_rainfall(lat, lon, search_start, cutoff_date)
    dates_sorted = sorted(daily.keys())
    values = [daily[d] for d in dates_sorted]

    onset_index = None
    for i in range(len(values) - 2):
        if sum(values[i:i + 3]) >= ONSET_RAIN_THRESHOLD_MM:
            onset_index = i
            break

    if onset_index is None:
        onset_date = cutoff_date  # no clear onset found yet — as late as possible, honestly
    else:
        onset_date = datetime.strptime(dates_sorted[onset_index], "%Y%m%d").date()

    return (onset_date - template_sowing).days
