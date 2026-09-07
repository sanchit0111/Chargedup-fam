"""Per-farmer scoring layer — execution.md Phase 8. Contract frozen in SKILL.md §2.

Orchestrates Fertility (6a) + Survival (6b) + Price (6c) + Combine (7) for a live application, at
any lat/lon — Fertility and Survival are fit once (train_fertility.py/train_survival.py) across
configs/district.yaml's training_district_pool, not pinned to one district (see
data/raw/PHASE1_FINDINGS.md for why that changed). Returns the full breakdown, not just a final
number — the intermediate values are what makes the LLM narrative (Phase 9) legible.
"""
from datetime import date

from src.ingestion.geocoding import geo_sanity_check
from src.ingestion.land_cover import land_use_check
from src.model.combine import coverage_ratio
from src.model.confidence import production_confidence_report
from src.model.fertility import reference_yield
from src.model.mitigants import apply_mitigants
from src.model.price import expected_price
from src.model.risk_tier import risk_tier
from src.model.survival import survival_factor


def _production_estimate(
    lat: float, lon: float, season_year: int, cutoff_date: date, config: dict, crop: str
) -> tuple:
    """(reference_yield, survival_factor, survival_mode) for `crop` at this plot/season.

    Only Rice has a fitted Survival model (models/survival_coefficients.json) — every other crop
    gets a normal-year baseline (survival_factor fixed at 1.0), labeled via survival_mode. Shared
    by score() and compare_crops() so the two can't silently diverge on which crop's production
    they're actually computing (a real bug this project had until 2026-09-07: score() ignored its
    own `crop` argument when calling reference_yield(), always computing Rice's production but
    whatever crop's price — caught by demo-scenario verification, not by inspection).
    """
    ref_yield = reference_yield(lat, lon, season_year, config, crop=crop)
    if crop == "Rice":
        return ref_yield, survival_factor(lat, lon, season_year, cutoff_date), "this-season-adjusted"
    return ref_yield, 1.0, "normal-year baseline (no fitted Survival model for this crop yet)"


def score(
    lat: float,
    lon: float,
    district: str,
    crop: str,
    season_year: int,
    cutoff_date: date,
    area_ha: float,
    loan_amount: float,
    loan_tenure_days: float,
    irrigation_source: str | None = None,
    pmfby_enrolled: bool | None = None,
    fpo_member: bool | None = None,
    existing_annual_debt_service: float | None = None,
    off_farm_annual_income: float | None = None,
) -> dict:
    """Farmer-context params (irrigation_source, pmfby_enrolled, ...) are the "ask the lendee"
    factors that can't be derived from lat/lon alone — see src/model/mitigants.py. All optional:
    a loan officer can score a plot before that intake is complete.
    """
    from src.config import load_config

    config = load_config()

    land_use = land_use_check(lat, lon)

    ref_yield, survival, survival_mode = _production_estimate(
        lat, lon, season_year, cutoff_date, config, crop
    )
    production_kg = ref_yield * survival * area_ha

    price = expected_price(district, crop, season_year, cutoff_date)
    combined = coverage_ratio(
        production_kg, price, loan_amount,
        crop_duration_days=config["season"]["season_length_days"],
        loan_tenure_days=loan_tenure_days,
    )
    combined = apply_mitigants(
        combined, loan_amount,
        irrigation_source=irrigation_source, pmfby_enrolled=pmfby_enrolled,
        fpo_member=fpo_member, existing_annual_debt_service=existing_annual_debt_service,
        off_farm_annual_income=off_farm_annual_income,
    )
    combined.update(risk_tier(combined["risk_score"], land_use_plausible=land_use["agricultural_plausible"]))

    return {
        "reference_yield": ref_yield,
        "survival_factor": survival,
        "survival_mode": survival_mode,
        "production_kg": production_kg,
        "expected_price": price,
        **combined,  # revenue, coverage_ratio, risk_score, tenure_adequate, tenure_shortfall_days,
                     # mitigants_applied, scale_of_finance_compliant, scale_of_finance_max_loan,
                     # risk_tier, recommended_action
        "confidence": production_confidence_report(crop),
        "geo_sanity": geo_sanity_check(lat, lon, district, config.get("default_state", "Karnataka")),
        "land_use": land_use,
    }


def compare_crops(
    lat: float,
    lon: float,
    district: str,
    requested_crop: str,
    season_year: int,
    cutoff_date: date,
    area_ha: float,
    loan_amount: float,
    loan_tenure_days: float,
    irrigation_source: str | None = None,
    pmfby_enrolled: bool | None = None,
    fpo_member: bool | None = None,
    existing_annual_debt_service: float | None = None,
    off_farm_annual_income: float | None = None,
) -> dict:
    """For the SAME plot/loan terms, scores every crop in configs/district.yaml's
    comparable_crops — the "can approve X for crop A, not for crop B the farmer requested" view.

    Only Rice has a fitted Survival model (models/survival_coefficients.json) — its entry gets
    the full this-season-adjusted treatment. Every other crop, including the requested one if it
    isn't Rice, gets a normal-year baseline (survival_factor fixed at 1.0), and `survival_mode`
    says so explicitly per entry — a real, stated scope limit, not glossed over. Sugarcane is
    excluded entirely (see configs/district.yaml's crops registry: no Agmarknet price available).
    A single crop's failure (e.g. missing price data) doesn't block the others — see `error`.
    """
    from src.config import load_config

    config = load_config()
    land_use = land_use_check(lat, lon)
    crops = []
    for crop in config["comparable_crops"]:
        entry = {"crop": crop, "requested": crop == requested_crop, "error": None}
        try:
            ref_yield, survival, survival_mode = _production_estimate(
                lat, lon, season_year, cutoff_date, config, crop
            )
            production_kg = ref_yield * survival * area_ha
            price = expected_price(district, crop, season_year, cutoff_date)
            combined = coverage_ratio(
                production_kg, price, loan_amount,
                crop_duration_days=config["season"]["season_length_days"],
                loan_tenure_days=loan_tenure_days,
            )
            combined = apply_mitigants(
                combined, loan_amount,
                irrigation_source=irrigation_source, pmfby_enrolled=pmfby_enrolled,
                fpo_member=fpo_member, existing_annual_debt_service=existing_annual_debt_service,
                off_farm_annual_income=off_farm_annual_income,
            )
            combined.update(
                risk_tier(combined["risk_score"], land_use_plausible=land_use["agricultural_plausible"])
            )

            entry.update({
                "reference_yield": ref_yield,
                "survival_factor": survival,
                "survival_mode": survival_mode,
                "production_kg": production_kg,
                "expected_price": price,
                **combined,
                "confidence": production_confidence_report(crop),
            })
        except Exception as e:
            entry["error"] = str(e)
        crops.append(entry)

    # Best-first among crops that scored successfully; failures sink to the bottom.
    crops.sort(key=lambda c: (c["error"] is not None, -(c.get("scale_of_finance_max_loan") or 0)))

    return {
        "requested_crop": requested_crop,
        "geo_sanity": geo_sanity_check(lat, lon, district, config.get("default_state", "Karnataka")),
        "land_use": land_use,
        "crops": crops,
    }


if __name__ == "__main__":
    from src.config import load_config, scoring_cutoff_date

    config = load_config()
    centroid_lat, centroid_lon = config["centroid"]["lat"], config["centroid"]["lon"]
    default_district = config["default_district"]
    # Agmarknet's Bellary/Rice history only starts 2007 (data/raw/PHASE1_FINDINGS.md) — the Price
    # pathway can't be demoed on earlier years even though Fertility/Survival can.
    # Two plots (centroid vs. a point ~20km east) to show the soil modifier actually differs, and
    # a short-tenure loan to show the tenure-adequacy check firing.
    scenarios = [
        ("centroid, adequate tenure", centroid_lat, centroid_lon, 200),
        ("east plot, adequate tenure", centroid_lat, centroid_lon + 0.2, 200),
        ("centroid, short tenure", centroid_lat, centroid_lon, 120),
    ]
    for label, lat, lon, tenure in scenarios:
        for year in [2013, 2015]:
            cutoff = scoring_cutoff_date(year, config)
            result = score(
                lat, lon, default_district, config["crop"], year, cutoff,
                area_ha=2.0, loan_amount=80000.0, loan_tenure_days=tenure,
            )
            print(label, year, result)
