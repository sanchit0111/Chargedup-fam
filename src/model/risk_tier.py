"""Risk tier — maps the final risk_score (combine.py + mitigants.py) to a recommended action band.

Deliberately NOT a yes/no auto-decision (product_idea.md §11: "decision-support score, not an
auto-reject gate") — priority-sector agri lending needs a human in the loop, and with both
Fertility and Survival currently showing negative cv_r2 (src/model/confidence.py), auto-declining
a real farmer off this formula would be indefensible. Bands are documented policy thresholds, same
discipline as combine.py's SAFE_COVERAGE_RATIO — not fit to outcomes (none exist yet, product_idea.md
§7C).
"""
from __future__ import annotations

# (max risk_score for this tier, tier label, recommended action)
TIER_BANDS = [
    (0.15, "Low", "Approve"),
    (0.35, "Moderate", "Approve with conditions"),
    (0.60, "Elevated", "Refer to manual review"),
    (1.01, "High", "Decline-recommend"),
]


def risk_tier(risk_score: float, land_use_plausible: bool | None = None) -> dict:
    """land_use_plausible=False (src/ingestion/land_cover.py: satellite land cover at this exact
    point is structurally non-agricultural — built-up, water, snow/ice) overrides the score-based
    tier entirely. This is a hard override, not a mitigant-style adjustment: no coverage ratio or
    mitigant stack can talk a residential building's coordinates into an "Approve" — the arithmetic
    might look fine on a claimed crop/area that was never actually verified against the plot.
    """
    if land_use_plausible is False:
        return {
            "risk_tier": "High",
            "recommended_action": "Decline-recommend (satellite land cover at this location is not agricultural)",
        }

    for max_score, tier, action in TIER_BANDS:
        if risk_score <= max_score:
            return {"risk_tier": tier, "recommended_action": action}
    return {"risk_tier": "High", "recommended_action": "Decline-recommend"}


if __name__ == "__main__":
    for s in [0.0, 0.1, 0.2, 0.4, 0.5, 0.7, 1.0]:
        print(s, "->", risk_tier(s))
    print("0.05 (Low by score) with land_use_plausible=False ->", risk_tier(0.05, land_use_plausible=False))
