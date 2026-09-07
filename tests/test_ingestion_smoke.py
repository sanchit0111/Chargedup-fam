"""Smoke tests against real APIs/known points — the acceptance test for each ingestion
module's Definition of Done (execution.md, skills.md §8). Not mocked.
"""
import pytest

from src.config import district_centroid, scoring_cutoff_date


@pytest.mark.parametrize("season_year", [2002, 2011, 2015])
def test_ndvi_anomaly_known_point(season_year):
    from src.ingestion.satellite import get_ndvi_anomaly

    lat, lon = district_centroid()
    anomaly = get_ndvi_anomaly(lat, lon, season_year, scoring_cutoff_date(season_year))
    assert isinstance(anomaly, float)


@pytest.mark.parametrize("season_year", [2002, 2011, 2015])
def test_rainfall_deficit_known_point(season_year):
    from src.ingestion.weather import get_rainfall_deficit

    lat, lon = district_centroid()
    deficit = get_rainfall_deficit(lat, lon, season_year, scoring_cutoff_date(season_year))
    assert isinstance(deficit, float)


@pytest.mark.parametrize("season_year", [2002, 2011, 2015])
def test_dry_spell_days_known_point(season_year):
    from src.ingestion.weather import get_dry_spell_days

    lat, lon = district_centroid()
    days = get_dry_spell_days(lat, lon, season_year, scoring_cutoff_date(season_year))
    assert isinstance(days, int)
    assert days >= 0


@pytest.mark.parametrize("season_year", [2002, 2011, 2015])
def test_growing_degree_days_known_point(season_year):
    from src.ingestion.weather import get_growing_degree_days

    lat, lon = district_centroid()
    gdd = get_growing_degree_days(lat, lon, season_year, scoring_cutoff_date(season_year))
    assert isinstance(gdd, float)
    assert gdd >= 0


@pytest.mark.parametrize("season_year", [2002, 2011, 2015])
def test_water_balance_known_point(season_year):
    from src.ingestion.weather import get_water_balance

    lat, lon = district_centroid()
    balance = get_water_balance(lat, lon, season_year, scoring_cutoff_date(season_year))
    assert isinstance(balance, float)


@pytest.mark.parametrize("season_year", [2002, 2011, 2015])
def test_monsoon_onset_delay_known_point(season_year):
    from src.ingestion.weather import get_monsoon_onset_delay

    lat, lon = district_centroid()
    delay = get_monsoon_onset_delay(lat, lon, season_year, scoring_cutoff_date(season_year))
    assert isinstance(delay, int)


@pytest.mark.parametrize("season_year", [2002, 2005, 2015])
def test_excess_rainfall_days_known_point(season_year):
    from src.ingestion.weather import get_excess_rainfall_days

    lat, lon = district_centroid()
    days = get_excess_rainfall_days(lat, lon, season_year, scoring_cutoff_date(season_year))
    assert isinstance(days, int)
    assert days >= 0


@pytest.mark.parametrize("season_year", [2002, 2005, 2015])
def test_max_daily_rainfall_known_point(season_year):
    from src.ingestion.weather import get_max_daily_rainfall

    lat, lon = district_centroid()
    value = get_max_daily_rainfall(lat, lon, season_year, scoring_cutoff_date(season_year))
    assert isinstance(value, float)
    assert value >= 0


@pytest.mark.parametrize("season_year,stage", [
    (2002, "vegetative"), (2002, "critical_stage"),
    (2011, "vegetative"), (2011, "critical_stage"),
])
def test_rainfall_deficit_by_stage_known_point(season_year, stage):
    from src.ingestion.weather import get_rainfall_deficit_by_stage

    lat, lon = district_centroid()
    value = get_rainfall_deficit_by_stage(
        lat, lon, season_year, scoring_cutoff_date(season_year), stage
    )
    assert isinstance(value, float)


@pytest.mark.parametrize("season_year,stage", [
    (2002, "vegetative"), (2002, "critical_stage"),
    (2011, "vegetative"), (2011, "critical_stage"),
])
def test_dry_spell_days_by_stage_known_point(season_year, stage):
    from src.ingestion.weather import get_dry_spell_days_by_stage

    lat, lon = district_centroid()
    days = get_dry_spell_days_by_stage(
        lat, lon, season_year, scoring_cutoff_date(season_year), stage
    )
    assert isinstance(days, int)
    assert days >= 0


@pytest.mark.parametrize("season_year", [2010, 2013, 2015])
def test_price_volatility_known_point(season_year):
    from src.config import load_config
    from src.ingestion.market import get_price_volatility

    config = load_config()
    volatility = get_price_volatility(
        config["default_district"], config["crop"], season_year, scoring_cutoff_date(season_year)
    )
    assert isinstance(volatility, float)
    assert volatility >= 0


@pytest.mark.parametrize("season_year", [2010, 2013, 2015])
def test_price_level_known_point(season_year):
    from src.config import load_config
    from src.ingestion.market import get_price_level

    config = load_config()
    level = get_price_level(
        config["default_district"], config["crop"], season_year, scoring_cutoff_date(season_year)
    )
    assert isinstance(level, float)
    assert level > 0
