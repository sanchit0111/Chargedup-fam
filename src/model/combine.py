"""Combine sub-model — execution.md Phase 7. Contract frozen in SKILL.md §2.

Pure function, no I/O — the seam where Production (Fertility x Survival) and Price meet Loan
Amount and Tenure. `risk_score` is a documented formula, not a learned black box, so it stays
auditable back to arithmetic a credit committee can check (product_idea.md §8).
"""
from __future__ import annotations

QUINTAL_KG = 100  # Agmarknet prices are INR/quintal — don't silently drop this unit conversion

# Coverage ratio at/above which the coverage-driven part of risk_score bottoms out at 0. A
# placeholder threshold, not a calibrated figure — real calibration needs partner loan-default
# data (product_idea.md §7C). Set here, once, deliberately not learned, so it stays inspectable.
SAFE_COVERAGE_RATIO = 1.5

# Tenure adequacy is a DISTINCT risk factor from coverage: a loan whose tenure ends before the
# crop is harvested and sold is a cash-flow-timing mismatch even if the crop succeeds and the
# price is good. TENURE_SALE_BUFFER_DAYS is a general default for harvest-to-sale timing, not a
# verified crop-specific figure (same honesty caveat as weather.py's HEAT_STRESS_THRESHOLD_C).
TENURE_SALE_BUFFER_DAYS = 30
TENURE_RISK_PENALTY_PER_DAY = 0.01  # each day short of adequate tenure adds this to risk_score


def coverage_ratio(
    production_kg: float,
    expected_price: float,
    loan_amount: float,
    crop_duration_days: float | None = None,
    loan_tenure_days: float | None = None,
) -> dict:
    """revenue = (production_kg / 100) * expected_price (INR/quintal).
    coverage_ratio = revenue / loan_amount.
    base risk = clip((SAFE_COVERAGE_RATIO - coverage_ratio) / SAFE_COVERAGE_RATIO, 0, 1).

    If crop_duration_days and loan_tenure_days are both given, also checks tenure adequacy
    (sowing -> harvest -> TENURE_SALE_BUFFER_DAYS to sell) and adds a linear penalty for any
    shortfall, capped at 1.0 total — a loan that's too short for the crop cycle is risky even at
    a comfortable coverage ratio, since the farmer may not have sold the harvest by repayment.
    Both tenure params are optional so this stays usable wherever tenure isn't known yet.
    """
    if loan_amount <= 0:
        raise ValueError("loan_amount must be positive")

    revenue = (production_kg / QUINTAL_KG) * expected_price
    ratio = revenue / loan_amount
    base_risk = max(0.0, min(1.0, (SAFE_COVERAGE_RATIO - ratio) / SAFE_COVERAGE_RATIO))

    result = {"revenue": revenue, "coverage_ratio": ratio}

    if crop_duration_days is None or loan_tenure_days is None:
        result["risk_score"] = base_risk
        result["tenure_adequate"] = None
        result["tenure_shortfall_days"] = None
        return result

    required_tenure_days = crop_duration_days + TENURE_SALE_BUFFER_DAYS
    shortfall = max(0.0, required_tenure_days - loan_tenure_days)
    result["tenure_adequate"] = shortfall == 0
    result["tenure_shortfall_days"] = shortfall
    result["risk_score"] = min(1.0, base_risk + shortfall * TENURE_RISK_PENALTY_PER_DAY)
    return result


if __name__ == "__main__":
    # hand-checkable examples (execution.md Phase 7 DoD)
    cases = [
        (2000.0, 5000.0, 80000.0, None, None),  # comfortable coverage, no tenure info
        (2000.0, 5000.0, 80000.0, 130.0, 200.0),  # comfortable coverage, adequate tenure
        (2000.0, 5000.0, 80000.0, 130.0, 120.0),  # comfortable coverage, but tenure too short
    ]
    for production_kg, price, loan, duration, tenure in cases:
        result = coverage_ratio(production_kg, price, loan, duration, tenure)
        print(f"production={production_kg}kg price={price} loan={loan} "
              f"duration={duration} tenure={tenure} -> {result}")
