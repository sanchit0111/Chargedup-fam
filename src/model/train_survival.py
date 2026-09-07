"""Fits the Survival sub-model — execution.md Phase 6b.

Target: actual_yield / reference_yield for each (district, year) row in the pooled panel
(configs/district.yaml training_district_pool — see data/raw/PHASE1_FINDINGS.md for why this
moved from a single Bellary time series to a multi-district panel: per-district feature
engineering was exhausted without ever beating R²=0, because the real constraint was N=25 for one
district, not the choice of feature). Weather/NDVI features are computed at each row's own
district centroid, not a single pinned location — reference_yield now comes from the
location-agnostic Fertility model (src/model/fertility.py), not a district-specific trailing mean.

**Leave-one-district-out CV, not leave-one-row-out** (src/model/regression_utils.py) — same
reasoning as train_fertility.py: rows from the same district share unobserved local factors that
a naive row-level split would leak across train/test.

Ridge regression alongside plain OLS, same as before — with many candidate features and limited
rows per district, ridge's shrinkage lets a larger feature set coexist with the sample without
guaranteed overfitting (SKILL.md §7).
"""
import json
from pathlib import Path

import numpy as np

from src.config import load_config, scoring_cutoff_date
from src.ingestion.ground_truth import pooled_yield_series
from src.model.fertility import FertilityUnavailableError, reference_yield
from src.model.regression_utils import fit_ols, fit_ridge, leave_one_group_out_cv
from src.model.survival import FEATURE_FUNCTIONS

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
COEFFICIENTS_PATH = REPO_ROOT / "models" / "survival_coefficients.json"
RIDGE_ALPHAS = [1.0, 10.0, 100.0]

FEATURE_SETS = {
    "rainfall_only": ["rainfall_deficit"],
    "dry_spell_only": ["dry_spell_days"],
    "et_only": ["evapotranspiration_anomaly"],
    "heat_stress_only": ["heat_stress_days"],
    "gdd_only": ["growing_degree_days"],
    "water_balance_only": ["water_balance"],
    "monsoon_onset_only": ["monsoon_onset_delay"],
    "ndvi_only": ["ndvi_anomaly"],
    "dry_heat_interaction_only": ["dry_heat_interaction"],
    "excess_rainfall_only": ["excess_rainfall_days"],
    "excess_rainfall_critical_stage_only": ["excess_rainfall_days_critical_stage"],
    "max_daily_rainfall_only": ["max_daily_rainfall"],
    "rainfall_critical_stage_only": ["rainfall_deficit_critical_stage"],
    "rainfall_vegetative_only": ["rainfall_deficit_vegetative"],
    "dry_spell_critical_stage_only": ["dry_spell_days_critical_stage"],
    "dry_spell_vegetative_only": ["dry_spell_days_vegetative"],
    "heat_stress_critical_stage_only": ["heat_stress_days_critical_stage"],
    "water_balance_critical_stage_only": ["water_balance_critical_stage"],
    "weather_core": ["rainfall_deficit", "dry_spell_days"],
    "stage_split_rainfall": ["rainfall_deficit_vegetative", "rainfall_deficit_critical_stage"],
    "stage_split_dry_spell": ["dry_spell_days_vegetative", "dry_spell_days_critical_stage"],
    "excess_rain_full": [
        "excess_rainfall_days", "excess_rainfall_days_critical_stage", "max_daily_rainfall",
    ],
    "weather_full": [
        "rainfall_deficit", "dry_spell_days", "evapotranspiration_anomaly", "heat_stress_days",
        "growing_degree_days", "water_balance", "monsoon_onset_delay", "dry_heat_interaction",
    ],
    "weather_ndvi": ["rainfall_deficit", "dry_spell_days", "ndvi_anomaly"],
    "everything": [
        "rainfall_deficit", "dry_spell_days", "evapotranspiration_anomaly", "heat_stress_days",
        "growing_degree_days", "water_balance", "monsoon_onset_delay", "dry_heat_interaction",
        "ndvi_anomaly", "excess_rainfall_days", "max_daily_rainfall",
    ],
}


def _build_dataset(config: dict) -> list:
    df = pooled_yield_series(config)

    rows = []
    for _, r in df.iterrows():
        district, lat, lon = r["district"], r["lat"], r["lon"]
        year = int(r["season_year"])
        try:
            ref = reference_yield(lat, lon, year, config)
        except FertilityUnavailableError as e:
            print(f"  {district} {year}: fertility unavailable ({e})")
            continue

        cutoff = scoring_cutoff_date(year, config)
        features = {}
        for name, fn in FEATURE_FUNCTIONS.items():
            try:
                features[name] = fn(lat, lon, year, cutoff)
            except Exception as e:
                features[name] = None
                print(f"  {district} {year}: {name} unavailable ({e})")

        if features["rainfall_deficit"] is None or features["dry_spell_days"] is None:
            continue  # every candidate feature set needs at least these

        yield_ratio = r["yield"] / ref
        rows.append((district, year, features, yield_ratio))
        print(f"  {district} {year}: ref_yield={ref:.0f} actual={r['yield']:.0f} "
              f"ratio={yield_ratio:.2f}")

    return rows


def train() -> None:
    config = load_config()
    pool = config["training_district_pool"]
    print(f"Building pooled Survival dataset across {len(pool)} districts...")
    rows = _build_dataset(config)

    all_results = []
    best = None

    for set_name, feature_names in FEATURE_SETS.items():
        subset = [r for r in rows if all(r[2][f] is not None for f in feature_names)]
        districts_in_subset = {r[0] for r in subset}
        if len(subset) < 10 or len(districts_in_subset) < 3:
            print(f"{set_name}: only {len(subset)} rows / {len(districts_in_subset)} districts "
                  f"— skipping (too few for leave-one-district-out CV)")
            continue
        X = np.array([[r[2][f] for f in feature_names] for r in subset], dtype=float)
        y = np.array([r[3] for r in subset], dtype=float)
        districts = np.array([r[0] for r in subset])

        methods = {"ols": fit_ols}
        for alpha in RIDGE_ALPHAS:
            methods[f"ridge_a{alpha:g}"] = (lambda Xt, yt, a=alpha: fit_ridge(Xt, yt, a))

        set_best = None
        for method_name, fit_fn in methods.items():
            cv = leave_one_group_out_cv(X, y, districts, fit_fn)
            entry = {
                "feature_set": set_name, "method": method_name, "features": feature_names,
                "cv_r2": cv["r2"], "cv_mae": cv["mae"], "n": cv["n"], "X": X, "y": y,
            }
            all_results.append(entry)
            if set_best is None or cv["r2"] > set_best["cv_r2"]:
                set_best = entry
            if best is None or cv["r2"] > best["cv_r2"]:
                best = entry

        print(f"{set_name} (n={set_best['n']}, districts={len(districts_in_subset)}): "
              f"best={set_best['method']} leave-one-district-out R²={set_best['cv_r2']:.3f} "
              f"MAE={set_best['cv_mae']:.3f}")

    if best is None:
        raise RuntimeError("no candidate feature set had enough usable rows/districts for CV")

    print(f"\nOverall winner: {best['feature_set']} / {best['method']} "
          f"(leave-one-district-out R²={best['cv_r2']:.3f}) — honest number, not filtered, "
          f"see SKILL.md §4")

    fit_fn = fit_ols if best["method"] == "ols" else (
        lambda Xt, yt: fit_ridge(Xt, yt, float(best["method"].removeprefix("ridge_a")))
    )
    final_coeffs = fit_fn(best["X"], best["y"])

    result = {
        "intercept": float(final_coeffs[0]),
        **{
            f"coef_{name}": float(final_coeffs[i + 1])
            for i, name in enumerate(best["features"])
        },
        "n_rows_total": best["n"],
        "n_districts_pool": len(pool),
        "validation_method": "leave_one_district_out_cv",
        "feature_set_name": best["feature_set"],
        "fit_method": best["method"],
        "cv_r2": best["cv_r2"],
        "cv_mae": best["cv_mae"],
        "features": best["features"],
        "candidates_compared": [
            {
                "feature_set": e["feature_set"], "method": e["method"],
                "cv_r2": e["cv_r2"], "cv_mae": e["cv_mae"], "n": e["n"],
            }
            for e in sorted(all_results, key=lambda e: -e["cv_r2"])
        ],
    }
    COEFFICIENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(COEFFICIENTS_PATH, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nWrote {COEFFICIENTS_PATH}")
    print(json.dumps({k: v for k, v in result.items() if k != "candidates_compared"}, indent=2))


if __name__ == "__main__":
    train()
