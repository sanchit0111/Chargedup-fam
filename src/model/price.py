"""Price sub-model — execution.md Phase 6c. Contract frozen in SKILL.md §2.

Builds on market.get_price_level() (current trend), adjusted for the historical harvest-time
supply-glut dip observed in Agmarknet's own multi-year series for this crop. Validated entirely
independent of ICRISAT/yield data (SKILL.md architecture note) — this pathway never touches
Fertility/Survival's ground truth.
"""
from datetime import date, timedelta

from src.ingestion.market import PriceDataUnavailableError, get_price_level


def _harvest_date(season_year: int, config: dict) -> date:
    from src.config import sowing_date

    sowing = sowing_date(season_year, config)
    return sowing + timedelta(days=config["season"]["season_length_days"])


def expected_price(district: str, crop: str, season_year: int, cutoff_date: date) -> float:
    """Expected price (INR/quintal) at harvest: the current pre-cutoff trend, adjusted for the
    historical harvest-vs-mid-season price ratio averaged over the prior
    ndvi.historical_baseline_years complete seasons. Falls back to the unadjusted current trend
    if no historical seasonality signal is available (never fabricates a seasonality factor).

    `crop` is the canonical name (configs/district.yaml's `crops` registry key) — translated to
    Agmarknet's own commodity string internally, since the two providers don't always agree
    (verified live per crop, 2026-09-07: Chickpea and Pearl Millet differ, Rice and Groundnut
    happen to match). `null` agmarknet_commodity (Sugarcane) means verified unavailable.
    """
    from src.config import load_config, scoring_cutoff_date

    config = load_config()
    crop_entry = config["crops"].get(crop)
    if crop_entry is None or crop_entry.get("agmarknet_commodity") is None:
        raise PriceDataUnavailableError(
            f"crop={crop!r} has no Agmarknet commodity mapping — see configs/district.yaml crops.{crop}"
        )
    commodity = crop_entry["agmarknet_commodity"]
    current_level = get_price_level(district, commodity, season_year, cutoff_date)

    baseline_years = config["ndvi"]["historical_baseline_years"]
    ratios = []
    for y in range(season_year - baseline_years, season_year):
        try:
            y_mid_level = get_price_level(district, commodity, y, scoring_cutoff_date(y, config))
            y_harvest_level = get_price_level(district, commodity, y, _harvest_date(y, config))
        except PriceDataUnavailableError:
            continue  # a missing single historical year shouldn't block the whole calculation
        if y_mid_level > 0:
            ratios.append(y_harvest_level / y_mid_level)

    if not ratios:
        return current_level

    seasonality_factor = sum(ratios) / len(ratios)
    return current_level * seasonality_factor


if __name__ == "__main__":
    from src.config import load_config, scoring_cutoff_date

    config = load_config()
    district, crop = config["default_district"], config["crop"]
    for year in [2013, 2014, 2015]:
        cutoff = scoring_cutoff_date(year, config)
        price = expected_price(district, crop, year, cutoff)
        print(year, "expected_price (INR/quintal):", round(price, 1))
