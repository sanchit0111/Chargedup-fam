"""Farmer-context mitigant layer — applies to combine.py's coverage_ratio() output.

Extends the loan officer's intake beyond what geo-coordinates alone can tell us (production,
weather, price — already covered by Fertility/Survival/Price). These are the factors a lender
would ask the *farmer* about directly (see the loan-underwriting factor research this maps to),
and none of them are learned from data — same discipline as combine.py's TENURE_RISK_PENALTY_PER_DAY:
small, named, policy-set adjustments that stay auditable back to a stated reason, not fit to
loan-outcome data (none exists yet — product_idea.md §7C). Every adjustment is capped and
documented so a credit committee can see exactly why a number moved.

Guardrail (product_idea.md §9): this module only adjusts risk_score and flags compliance — it
never changes production_kg, expected_price, or revenue, which stay pure outputs of Fertility/
Survival/Price.
"""
from __future__ import annotations

# Irrigated land buffers a single bad monsoon from translating into crop failure — this project's
# own finding (data/raw/PHASE1_FINDINGS.md) is that canal-irrigated Rice's weather-driven Survival
# signal is much weaker than rainfed Groundnut's, i.e. irrigation genuinely displaces weather risk.
# These are policy-set caps expressing that direction, not coefficients fit to outcomes.
IRRIGATION_RISK_ADJUSTMENT = {
    "canal": -0.05,
    "borewell": -0.03,
    "tank": -0.02,
    "rainfed": 0.0,
}

# PMFBY enrollment is a real payout backstop against the exact tail risk (crop failure) Survival
# estimates — reduces effective default risk without changing the underlying production estimate.
PMFBY_ENROLLED_ADJUSTMENT = -0.08

# FPO/SHG/JLG membership correlates with better market access and repayment discipline in the
# agri-finance literature this project's factor research drew on. Small and capped deliberately.
FPO_MEMBER_ADJUSTMENT = -0.02

# If existing debt obligations eat too much of *this crop's* expected revenue, that's a real
# repayment-capacity risk distinct from coverage ratio (which only looks at this loan vs. this
# crop's revenue). Threshold and per-step penalty are policy choices, not fitted.
DEBT_SERVICE_RISK_THRESHOLD = 0.3  # existing annual debt service as a fraction of this crop's revenue
DEBT_SERVICE_RISK_PENALTY_PER_STEP = 0.05  # per 10 percentage points over the threshold
DEBT_SERVICE_STEP = 0.10

# Off-farm income is a household-level buffer against a bad season on this specific plot.
OFF_FARM_BUFFER_THRESHOLD = 0.25  # off-farm income as a fraction of this crop's expected revenue
OFF_FARM_BUFFER_ADJUSTMENT = -0.03

# RBI/NABARD guideline: scale of finance (and by extension, loan amount for a single crop loan)
# should not exceed roughly half of gross expected returns. This is a COMPLIANCE FLAG, not a
# risk_score adjustment — it's a distinct, named regulatory check a credit committee needs to see.
SCALE_OF_FINANCE_MAX_RATIO = 0.5


def apply_mitigants(
    base: dict,
    loan_amount: float,
    irrigation_source: str | None = None,
    pmfby_enrolled: bool | None = None,
    fpo_member: bool | None = None,
    existing_annual_debt_service: float | None = None,
    off_farm_annual_income: float | None = None,
) -> dict:
    """Takes combine.coverage_ratio()'s output dict and optional farmer-reported context.

    Every field is optional and defaults to "unknown" (no adjustment) — this stays usable at
    intake time before a loan officer has collected the full picture, and never penalizes a
    farmer for a question that wasn't asked yet.

    Returns base's dict, plus: risk_score (overridden with mitigants applied), mitigants_applied
    (list of {factor, adjustment, reason} — the audit trail), scale_of_finance_compliant,
    scale_of_finance_max_loan (both None if revenue isn't known, i.e. base has no "revenue" key).
    """
    result = dict(base)
    adjustments: list[dict] = []

    if irrigation_source is not None:
        adj = IRRIGATION_RISK_ADJUSTMENT.get(irrigation_source)
        if adj is not None and adj != 0.0:
            adjustments.append({
                "factor": f"irrigation_source={irrigation_source}",
                "adjustment": adj,
                "reason": "irrigated land buffers weather-driven production risk",
            })

    if pmfby_enrolled:
        adjustments.append({
            "factor": "pmfby_enrolled",
            "adjustment": PMFBY_ENROLLED_ADJUSTMENT,
            "reason": "crop insurance backstops the production-risk tail this score estimates",
        })

    if fpo_member:
        adjustments.append({
            "factor": "fpo_member",
            "adjustment": FPO_MEMBER_ADJUSTMENT,
            "reason": "FPO/SHG/JLG membership correlates with market access and repayment discipline",
        })

    revenue = base.get("revenue")

    if existing_annual_debt_service is not None and revenue:
        ratio = existing_annual_debt_service / revenue
        if ratio > DEBT_SERVICE_RISK_THRESHOLD:
            steps = (ratio - DEBT_SERVICE_RISK_THRESHOLD) / DEBT_SERVICE_STEP
            penalty = steps * DEBT_SERVICE_RISK_PENALTY_PER_STEP
            adjustments.append({
                "factor": f"existing_debt_service_ratio={ratio:.2f}",
                "adjustment": penalty,
                "reason": (
                    f"existing debt obligations exceed "
                    f"{DEBT_SERVICE_RISK_THRESHOLD:.0%} of this crop's expected revenue"
                ),
            })

    if off_farm_annual_income is not None and revenue:
        ratio = off_farm_annual_income / revenue
        if ratio >= OFF_FARM_BUFFER_THRESHOLD:
            adjustments.append({
                "factor": f"off_farm_income_ratio={ratio:.2f}",
                "adjustment": OFF_FARM_BUFFER_ADJUSTMENT,
                "reason": "off-farm household income buffers a bad season on this plot",
            })

    total_adjustment = sum(a["adjustment"] for a in adjustments)
    result["risk_score"] = max(0.0, min(1.0, base["risk_score"] + total_adjustment))
    result["mitigants_applied"] = adjustments

    if revenue:
        max_loan = SCALE_OF_FINANCE_MAX_RATIO * revenue
        result["scale_of_finance_max_loan"] = max_loan
        result["scale_of_finance_compliant"] = loan_amount <= max_loan
    else:
        result["scale_of_finance_max_loan"] = None
        result["scale_of_finance_compliant"] = None

    return result


if __name__ == "__main__":
    # hand-checkable examples, same pattern as combine.py's __main__
    from src.model.combine import coverage_ratio

    base = coverage_ratio(2000.0, 5000.0, 80000.0, crop_duration_days=130.0, loan_tenure_days=200.0)
    print("no context:", apply_mitigants(base, loan_amount=80000.0))
    print(
        "rainfed, no insurance, high existing debt:",
        apply_mitigants(
            base, loan_amount=80000.0, irrigation_source="rainfed",
            pmfby_enrolled=False, existing_annual_debt_service=45000.0,
        ),
    )
    print(
        "canal-irrigated, PMFBY-enrolled, FPO member:",
        apply_mitigants(
            base, loan_amount=80000.0, irrigation_source="canal",
            pmfby_enrolled=True, fpo_member=True,
        ),
    )
