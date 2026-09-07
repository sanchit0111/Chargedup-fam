"""ICRISAT yield-series loader + proxy label builder — execution.md Phase 2.

`yield_series()` is the canonical ICRISAT-file loader (skills.md §3) — `fertility.py`'s
reference_yield() imports it too. Don't fork a second ICRISAT parser anywhere else.
"""
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class GroundTruthUnavailableError(Exception):
    """Raised when the configured ground_truth.source has no usable rows for this district."""


def yield_series(district: str, config: dict, crop: str | None = None) -> pd.DataFrame:
    """Returns columns [season_year, yield] (kg/ha) for `district`, sorted ascending by year,
    with zero/near-zero-area artifact years already dropped. `crop` defaults to config["crop"]
    (the single-crop demo path); pass it explicitly for the multi-crop comparison
    (predict.compare_crops()) — see configs/district.yaml's `crops` registry.
    """
    gt = config["ground_truth"]
    if gt["source"] != "icrisat_yield_shortfall":
        raise GroundTruthUnavailableError(f"unsupported ground_truth.source: {gt['source']!r}")

    path = REPO_ROOT / gt["file"]
    if not path.exists():
        raise GroundTruthUnavailableError(
            f"{path} not found — see data/raw/PHASE1_FINDINGS.md for the download URL"
        )

    crop = crop or config["crop"]
    crop_entry = config["crops"].get(crop)
    if crop_entry is None:
        raise GroundTruthUnavailableError(f"crop={crop!r} not in configs/district.yaml's crops registry")
    yield_col = crop_entry["icrisat_yield_column"]

    df = pd.read_excel(path, engine="xlrd")
    district_upper = district.strip().upper()
    df = df[df["Dist Name"].astype(str).str.strip().str.upper() == district_upper]
    if df.empty:
        raise GroundTruthUnavailableError(f"no ICRISAT rows found for district={district!r}")

    df = df[["Year", yield_col]].dropna()
    df = df[df[yield_col] > 0].sort_values("Year")  # drop zero/near-zero-area artifacts
    df["Year"] = df["Year"].astype(int)
    return df.rename(columns={"Year": "season_year", yield_col: "yield"}).reset_index(drop=True)


def pooled_yield_series(config: dict, crop: str | None = None) -> pd.DataFrame:
    """Returns columns [district, lat, lon, season_year, yield] across every district in
    `training_district_pool` — the panel src/model/train_fertility.py and train_survival.py fit
    against, instead of one district's 25 rows. Each row carries its own district's centroid so
    weather/soil features get computed at the right point, not a single pinned location.
    """
    frames = []
    for entry in config["training_district_pool"]:
        df = yield_series(entry["name"], config, crop=crop)
        df["district"] = entry["name"]
        df["lat"] = entry["lat"]
        df["lon"] = entry["lon"]
        frames.append(df)
    return pd.concat(frames, ignore_index=True)[["district", "lat", "lon", "season_year", "yield"]]


def build_labels(config: dict) -> pd.DataFrame:
    """Returns columns [district, season_year, label] for the default/demo district only.
    label=1 marks a season where yield fell more than ground_truth.stress_threshold below the
    trailing mean of the prior ndvi.historical_baseline_years seasons. Requires both classes
    present.

    Reporting/sanity-check use only (SKILL.md architecture note) — no longer the training target
    for a classifier; Phase 6b's survival regression trains against the continuous
    actual/reference yield ratio instead, pooled across training_district_pool.
    """
    district = config["default_district"]
    df = yield_series(district, config)

    baseline_years = config["ndvi"]["historical_baseline_years"]
    df["trend"] = df["yield"].rolling(window=baseline_years, min_periods=1).mean().shift(1)
    df = df.dropna(subset=["trend"])  # first year has no prior trend, can't be labeled

    threshold = config["ground_truth"]["stress_threshold"]
    df["label"] = (df["yield"] < df["trend"] * (1 - threshold)).astype(int)
    df["district"] = district

    labels = df[["district", "season_year", "label"]].reset_index(drop=True)
    if labels["label"].nunique() < 2:
        raise GroundTruthUnavailableError(
            "labels contain only one class — stress_threshold or district choice needs revisiting"
        )
    return labels


if __name__ == "__main__":
    from src.config import load_config

    labels = build_labels(load_config())
    out_path = REPO_ROOT / "data" / "processed" / "labels.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    labels.to_csv(out_path, index=False)
    print(f"wrote {len(labels)} labeled seasons to {out_path}")
    print(labels["label"].value_counts().to_dict())
