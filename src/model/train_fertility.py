"""Fits the Fertility sub-model — execution.md Phase 6a.

Location-agnostic by design: instead of "this district's own trailing yield mean" (the original,
single-district version that couldn't generalize past Bellary — see
data/raw/PHASE1_FINDINGS.md), this fits yield ~ f(soil organic carbon, a shared time trend)
across the pooled district-year panel (configs/district.yaml training_district_pool), so it
evaluates at any lat/lon via that point's own soil reading, not a lookup of pre-seen history.

**Leave-one-district-out CV, not leave-one-row-out** (src/model/regression_utils.py) — rows from
the same district share unobserved local factors (soil quality beyond what organic_carbon
captures, local practices, microclimate) that a naive row-level split would leak across
train/test, overstating how well this generalizes to a genuinely new district.
"""
import json
from pathlib import Path

import numpy as np

from src.config import load_config
from src.ingestion.ground_truth import pooled_yield_series
from src.ingestion.soil import SoilDataUnavailableError, _topsoil_value, get_organic_carbon
from src.model.fertility import coefficients_path
from src.model.regression_utils import fit_ols, fit_ridge, leave_one_group_out_cv

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RIDGE_ALPHAS = [1.0, 10.0, 100.0]

FEATURE_SETS = {
    "soc_only": ["organic_carbon"],
    "trend_only": ["year_trend"],
    "soc_and_trend": ["organic_carbon", "year_trend"],
}


def _build_dataset(config: dict, crop: str) -> list:
    df = pooled_yield_series(config, crop=crop)
    rows = []
    for _, r in df.iterrows():
        try:
            soc = _topsoil_value(get_organic_carbon(r["lat"], r["lon"]))
        except SoilDataUnavailableError as e:
            print(f"  {r['district']} {int(r['season_year'])}: soil unavailable ({e})")
            continue
        rows.append({
            "district": r["district"], "season_year": int(r["season_year"]),
            "organic_carbon": soc, "year_trend": float(r["season_year"]),
            "yield": float(r["yield"]),
        })
    for d in sorted({r["district"] for r in rows}):
        d_rows = [r for r in rows if r["district"] == d]
        print(f"  {d}: n={len(d_rows)} soc={d_rows[0]['organic_carbon']:.2f} "
              f"yield_range=({min(r['yield'] for r in d_rows):.0f},"
              f"{max(r['yield'] for r in d_rows):.0f})")
    return rows


def train(crop: str | None = None) -> None:
    config = load_config()
    crop = crop or config["crop"]
    pool = config["training_district_pool"]
    print(f"Building pooled Fertility dataset for {crop!r} across {len(pool)} districts...")
    rows = _build_dataset(config, crop)

    districts = np.array([r["district"] for r in rows])
    y = np.array([r["yield"] for r in rows], dtype=float)

    candidates = []
    for set_name, feature_names in FEATURE_SETS.items():
        X = np.array([[r[f] for f in feature_names] for r in rows], dtype=float)
        methods = {"ols": fit_ols}
        for alpha in RIDGE_ALPHAS:
            methods[f"ridge_a{alpha:g}"] = (lambda Xt, yt, a=alpha: fit_ridge(Xt, yt, a))
        for method_name, fit_fn in methods.items():
            cv = leave_one_group_out_cv(X, y, districts, fit_fn)
            candidates.append({
                "feature_set": set_name, "method": method_name, "features": feature_names,
                "cv_r2": cv["r2"], "cv_mae": cv["mae"], "n": cv["n"], "X": X,
            })
        best_for_set = max(
            (c for c in candidates if c["feature_set"] == set_name), key=lambda c: c["cv_r2"]
        )
        print(f"{set_name} (n={best_for_set['n']}): best={best_for_set['method']} "
              f"leave-one-district-out R²={best_for_set['cv_r2']:.3f} MAE={best_for_set['cv_mae']:.1f}")

    best = max(candidates, key=lambda c: c["cv_r2"])
    print(f"\nOverall winner: {best['feature_set']} / {best['method']} "
          f"(R²={best['cv_r2']:.3f}) — honest number, not filtered, see SKILL.md §4")

    fit_fn = fit_ols if best["method"] == "ols" else (
        lambda Xt, yt: fit_ridge(Xt, yt, float(best["method"].removeprefix("ridge_a")))
    )
    final_coeffs = fit_fn(best["X"], y)

    result = {
        "intercept": float(final_coeffs[0]),
        **{
            f"coef_{name}": float(final_coeffs[i + 1])
            for i, name in enumerate(best["features"])
        },
        "n_district_years": len(rows),
        "n_districts": len(set(districts)),
        "observed_yield_min": float(y.min()),
        "observed_yield_max": float(y.max()),
        "validation_method": "leave_one_district_out_cv",
        "feature_set_name": best["feature_set"],
        "fit_method": best["method"],
        "cv_r2": best["cv_r2"],
        "cv_mae": best["cv_mae"],
        "features": best["features"],
        "candidates_compared": [
            {
                "feature_set": c["feature_set"], "method": c["method"],
                "cv_r2": c["cv_r2"], "cv_mae": c["cv_mae"], "n": c["n"],
            }
            for c in sorted(candidates, key=lambda c: -c["cv_r2"])
        ],
    }
    path = coefficients_path(crop)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {path}")
    print(json.dumps({k: v for k, v in result.items() if k != "candidates_compared"}, indent=2))


if __name__ == "__main__":
    import sys

    # `python -m src.model.train_fertility` trains config["crop"] (backward compat).
    # `python -m src.model.train_fertility --all-comparable` trains every crop in
    # configs/district.yaml's comparable_crops (the multi-crop comparison's Fertility side).
    if "--all-comparable" in sys.argv:
        for crop_name in load_config()["comparable_crops"]:
            train(crop_name)
            print()
    else:
        train()
