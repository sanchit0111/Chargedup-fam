"""Prepares the 2-3 fixed demo scenarios (execution.md Phase 12 DoD) — runs each live against
the real pipeline (predict.score() + predict.compare_crops()), caches the full output to
demo/cached/*.json, and prints a plain-English summary of what each one demonstrates.

Why cached: a live demo depending on GEE/NASA POWER/Agmarknet/Bedrock all responding within a
pitch's time window is a real, accepted risk (deferred, not solved, per this project's own
scope decisions). Caching these three specific, real runs ahead of time means a network hiccup
during judging doesn't sink the demo — the cached JSON is a genuine prior run, not a fabricated
fallback. Re-run this script to refresh the cache; it always hits real APIs, never mocks.

Scenario years are chosen from data/processed/labels.csv (build_labels()'s real ICRISAT-derived
stress labels for Bellary Rice) — not picked to look good, picked because they're documented,
verified good/bad seasons.
"""
import json
from pathlib import Path

from src.config import load_config, scoring_cutoff_date
from src.model.predict import compare_crops, score

REPO_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = REPO_ROOT / "demo" / "cached"

SCENARIOS = [
    {
        "id": "a_clean_approval",
        "story": (
            "Clean approval — a real non-stress Rice season (2013, labels.csv label=0) with "
            "irrigation + insurance context provided, well within NABARD's scale-of-finance "
            "guideline and adequate tenure. Shows the pipeline's happy path end to end, and the "
            "mitigants layer actively lowering an already-comfortable risk score."
        ),
        "lat": 15.14, "lon": 76.92, "district": "Bellary", "crop": "Rice",
        "season_year": 2013, "area_ha": 2.0, "loan_amount": 50000.0, "loan_tenure_days": 200,
        "irrigation_source": "canal", "pmfby_enrolled": True, "fpo_member": True,
    },
    {
        "id": "b_elevated_needs_review",
        "story": (
            "High risk, decline-recommend — 2009, a REAL documented stress season for Bellary "
            "Rice (labels.csv label=1: yield >5% below its trailing 5-year trend, verified from "
            "ICRISAT, not picked to look bad). No farmer-context intake done yet (all mitigant "
            "fields unknown — the realistic 'just applied' state), a loan amount that breaches "
            "the NABARD scale-of-finance guideline once the weaker season's revenue is known, "
            "and inadequate tenure. Shows the honest confidence disclosure and the tier system "
            "doing real work, not rubber-stamping. Note: exact coverage ratio has shown minor "
            "run-to-run variance (observed 1.06x-1.21x across two live runs) — tier stays "
            "Elevated-to-High either way; traced to data.gov.in pagination/rate-limit behavior "
            "under repeated rapid calls, a known deferred reliability item, not this fix."
        ),
        "lat": 15.14, "lon": 76.92, "district": "Bellary", "crop": "Rice",
        "season_year": 2009, "area_ha": 2.0, "loan_amount": 80000.0, "loan_tenure_days": 120,
    },
    {
        "id": "c_counter_proposal",
        "story": (
            "Multi-crop counter-proposal — the farmer requests a Chickpea loan; on this exact "
            "land, Chickpea's own economics can't support the amount requested, but Rice and "
            "Groundnut can. This is the compare_crops() flagship view: 'we can approve X for "
            "crop A, not the crop you asked about, and here's a real alternative on your own "
            "land' — the differentiator that came directly out of this project's conversation "
            "with the user, not a generic feature."
        ),
        "lat": 15.14, "lon": 76.92, "district": "Bellary", "crop": "Chickpea",
        "season_year": 2013, "area_ha": 2.0, "loan_amount": 80000.0, "loan_tenure_days": 200,
    },
]


def run_all() -> None:
    config = load_config()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    for s in SCENARIOS:
        cutoff = scoring_cutoff_date(s["season_year"], config)
        kwargs = dict(
            lat=s["lat"], lon=s["lon"], district=s["district"], season_year=s["season_year"],
            cutoff_date=cutoff, area_ha=s["area_ha"], loan_amount=s["loan_amount"],
            loan_tenure_days=s["loan_tenure_days"],
            irrigation_source=s.get("irrigation_source"), pmfby_enrolled=s.get("pmfby_enrolled"),
            fpo_member=s.get("fpo_member"),
            existing_annual_debt_service=s.get("existing_annual_debt_service"),
            off_farm_annual_income=s.get("off_farm_annual_income"),
        )
        print(f"\n=== {s['id']} ===\n{s['story']}\n")

        score_result = score(crop=s["crop"], **kwargs)
        print(
            f"score: risk_tier={score_result['risk_tier']} "
            f"({score_result['recommended_action']}), "
            f"coverage_ratio={score_result['coverage_ratio']:.2f}x, "
            f"NABARD_compliant={score_result['scale_of_finance_compliant']}, "
            f"tenure_adequate={score_result['tenure_adequate']}"
        )

        compare_result = compare_crops(requested_crop=s["crop"], **kwargs)
        for c in compare_result["crops"]:
            if c["error"]:
                print(f"  compare[{c['crop']}]: ERROR {c['error']}")
            else:
                print(
                    f"  compare[{c['crop']}]: tier={c['risk_tier']}, "
                    f"max_loan=INR {c['scale_of_finance_max_loan']:,.0f}"
                )

        out = {"scenario": s, "score": score_result, "compare_crops": compare_result}
        out_path = CACHE_DIR / f"{s['id']}.json"
        with open(out_path, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"cached -> {out_path}")


if __name__ == "__main__":
    run_all()
