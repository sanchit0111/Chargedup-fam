"""Confidence disclosure — reads the fitted models' own cv_r2 and states it plainly instead of
letting a production_kg/risk_score number imply more certainty than the fit actually supports.

Bands are documented policy, not fitted — same discipline as combine.py's SAFE_COVERAGE_RATIO and
mitigants.py's adjustments. cv_r2 < 0 means the fitted formula explains *less* variance than just
predicting the mean would — that's the honest, current state of both Fertility and Survival's
deployed coefficients (see models/*.json's own cv_r2, leave-one-district-out CV) and this module
exists specifically so that fact reaches the API response and narrative, not just a stale doc.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SURVIVAL_COEFFICIENTS_PATH = REPO_ROOT / "models" / "survival_coefficients.json"

# cv_r2 bands -> a plain-language confidence label. A model with cv_r2 < 0 (worse than the mean)
# is genuinely "low" — this isn't false modesty, it's what leave-one-district-out CV measured.
CONFIDENCE_BANDS = [
    (0.3, "high"),
    (0.0, "moderate"),
    (float("-inf"), "low"),
]


def _band(cv_r2: float) -> str:
    for threshold, label in CONFIDENCE_BANDS:
        if cv_r2 >= threshold:
            return label
    return "low"


def _model_confidence(path: Path, label: str) -> dict:
    if not path.exists():
        return {"label": label, "confidence": "unknown", "reason": f"{path.name} not found"}
    with open(path) as f:
        coeffs = json.load(f)
    cv_r2 = coeffs.get("cv_r2")
    if cv_r2 is None:
        return {"label": label, "confidence": "unknown", "reason": "no cv_r2 recorded"}
    return {
        "label": label,
        "confidence": _band(cv_r2),
        "cv_r2": cv_r2,
        "validation_method": coeffs.get("validation_method"),
        "n_rows": coeffs.get("n_rows_total") or coeffs.get("n_district_years"),
        "n_districts": coeffs.get("n_districts_pool") or coeffs.get("n_districts"),
    }


def production_confidence_report(crop: str = "Rice") -> dict:
    """Confidence for the *production estimate* (reference_yield x survival_factor) — the part
    of the pipeline actually built on a small, honestly-weak-fitting regression. Price is a live
    trend read off Agmarknet, not a small-sample fit, so it isn't included here; combine.py's
    formula is pure arithmetic, not a fit at all.

    Survival is only fitted for Rice today (models/survival_coefficients.json) — for any other
    crop, survival_factor is a fixed 1.0 baseline (predict.compare_crops()), not a fitted model,
    so its confidence is reported as "n/a (not modeled)" rather than silently reusing Rice's.
    """
    from src.model.fertility import coefficients_path

    fertility = _model_confidence(coefficients_path(crop), "fertility")
    if crop == "Rice":
        survival = _model_confidence(SURVIVAL_COEFFICIENTS_PATH, "survival")
    else:
        survival = {"label": "survival", "confidence": "n/a (not modeled — normal-year baseline used)"}

    order = {"low": 0, "moderate": 1, "high": 2, "unknown": 1}
    scored = [m for m in (fertility, survival) if m["confidence"] in order]
    overall = min(scored, key=lambda m: order[m["confidence"]])["confidence"] if scored else "low"

    return {"overall": overall, "fertility": fertility, "survival": survival}


if __name__ == "__main__":
    import json as _json

    for crop in ["Rice", "Groundnut", "Chickpea", "Pearl Millet"]:
        print(crop, "->")
        print(_json.dumps(production_confidence_report(crop), indent=2))
