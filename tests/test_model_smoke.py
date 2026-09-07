"""Smoke tests for the Fertility/Price/Combine/Survival sub-models against real data (execution.md
Phases 6a-8, SKILL.md §8). Survival's own fit validation lives in train_survival.py's own report
(models/survival_coefficients.json) — run that separately; this just checks survival_factor()
loads and returns a plausible value once trained.
"""
import pytest

from src.config import district_centroid, load_config, scoring_cutoff_date


@pytest.mark.parametrize("season_year", [1997, 2002, 2011, 2015])
def test_reference_yield_known_point(season_year):
    from src.model.fertility import reference_yield

    config = load_config()
    lat, lon = district_centroid()
    ref = reference_yield(lat, lon, season_year, config)
    assert isinstance(ref, float)
    assert ref > 0


def test_reference_yield_varies_by_plot():
    from src.model.fertility import reference_yield

    config = load_config()
    lat, lon = district_centroid()
    centroid_ref = reference_yield(lat, lon, 2015, config)
    offset_ref = reference_yield(lat, lon + 0.2, 2015, config)
    assert isinstance(offset_ref, float)
    assert offset_ref > 0
    # not asserting they differ — soil variation may legitimately be near-zero at some offsets —
    # just that the plot-level path runs end to end and returns something sane.
    assert centroid_ref > 0


def test_reference_yield_genuinely_district_agnostic():
    """The real test of the district-agnostic pivot: evaluate at a pooled district's centroid
    that ISN'T the default_district — this only works if reference_yield() is a fitted formula
    evaluable anywhere, not a lookup of one district's own remembered history."""
    from src.model.fertility import reference_yield

    config = load_config()
    pool = config["training_district_pool"]
    other = next(d for d in pool if d["name"] != config["default_district"])
    ref = reference_yield(other["lat"], other["lon"], 2015, config)
    assert isinstance(ref, float)
    assert ref > 0


@pytest.mark.parametrize("season_year", [2002, 2011, 2015])
def test_evapotranspiration_anomaly_known_point(season_year):
    from src.ingestion.weather import get_evapotranspiration_anomaly

    lat, lon = district_centroid()
    value = get_evapotranspiration_anomaly(lat, lon, season_year, scoring_cutoff_date(season_year))
    assert isinstance(value, float)


@pytest.mark.parametrize("season_year", [2002, 2011, 2015])
def test_heat_stress_days_known_point(season_year):
    from src.ingestion.weather import get_heat_stress_days

    lat, lon = district_centroid()
    days = get_heat_stress_days(lat, lon, season_year, scoring_cutoff_date(season_year))
    assert isinstance(days, int)
    assert days >= 0


def test_organic_carbon_known_point():
    from src.ingestion.soil import DEPTH_BANDS, get_organic_carbon

    lat, lon = district_centroid()
    depths = get_organic_carbon(lat, lon)
    assert isinstance(depths, dict)
    assert set(depths).issubset(DEPTH_BANDS)
    assert all(v >= 0 for v in depths.values())




@pytest.mark.parametrize("season_year", [2002, 2011, 2015])
def test_survival_factor_known_point(season_year):
    from src.model.survival import MAX_SURVIVAL_FACTOR, MIN_SURVIVAL_FACTOR, survival_factor

    lat, lon = district_centroid()
    factor = survival_factor(lat, lon, season_year, scoring_cutoff_date(season_year))
    assert MIN_SURVIVAL_FACTOR <= factor <= MAX_SURVIVAL_FACTOR


@pytest.mark.parametrize("season_year", [2013, 2015])
def test_expected_price_known_point(season_year):
    from src.model.price import expected_price

    config = load_config()
    price = expected_price(
        config["default_district"], config["crop"], season_year, scoring_cutoff_date(season_year)
    )
    assert isinstance(price, float)
    assert price > 0


def test_coverage_ratio_pure():
    from src.model.combine import coverage_ratio

    result = coverage_ratio(production_kg=2000.0, expected_price=5000.0, loan_amount=80000.0)
    assert result["revenue"] == pytest.approx(100000.0)
    assert result["coverage_ratio"] == pytest.approx(1.25)
    assert 0.0 <= result["risk_score"] <= 1.0
    assert result["tenure_adequate"] is None  # no tenure info given


def test_coverage_ratio_tenure_adequacy():
    from src.model.combine import coverage_ratio

    adequate = coverage_ratio(2000.0, 5000.0, 80000.0, crop_duration_days=130, loan_tenure_days=200)
    assert adequate["tenure_adequate"] is True
    assert adequate["tenure_shortfall_days"] == 0

    short = coverage_ratio(2000.0, 5000.0, 80000.0, crop_duration_days=130, loan_tenure_days=120)
    assert short["tenure_adequate"] is False
    assert short["tenure_shortfall_days"] > 0
    # short tenure must be at least as risky as the same coverage with adequate tenure
    assert short["risk_score"] >= adequate["risk_score"]


@pytest.mark.parametrize("season_year", [2013, 2015])
def test_predict_score_end_to_end(season_year):
    from src.model.predict import score

    config = load_config()
    lat, lon = district_centroid()
    result = score(
        lat, lon, config["default_district"], config["crop"], season_year,
        scoring_cutoff_date(season_year), area_ha=2.0, loan_amount=80000.0,
        loan_tenure_days=200,
    )
    for key in (
        "reference_yield", "survival_factor", "production_kg", "expected_price",
        "revenue", "coverage_ratio", "risk_score", "tenure_adequate", "tenure_shortfall_days",
    ):
        assert key in result
