"""Single loader for configs/district.yaml — the shared source of truth (skills.md §3)."""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()  # every module that needs a secret imports from here first (skills.md §3)

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "district.yaml"


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def sowing_date(
    season_year: int, config: dict | None = None, actual_sowing_date: date | None = None
) -> date:
    """`actual_sowing_date` overrides the config template. The template
    (`season.sowing_date_template`, a fixed "{year}-06-15") is a generic Kharif-rice default
    applied uniformly across the whole training pool — real sowing dates vary by farmer and by
    season, and a live loan application should use what was actually reported, not the template.
    Training/backtesting (train_fertility.py, train_survival.py, demo/run_scenarios.py) correctly
    keep using the template, since there's no per-farmer sowing date in the historical ICRISAT
    panel — this override only matters at the live-scoring boundary (src/api/main.py).
    """
    if actual_sowing_date is not None:
        return actual_sowing_date
    config = config or load_config()
    return date.fromisoformat(config["season"]["sowing_date_template"].format(year=season_year))


def scoring_cutoff_date(
    season_year: int, config: dict | None = None, actual_sowing_date: date | None = None
) -> date:
    """The hard temporal cutoff (execution.md §0 constraint 2) for a given season.

    Every ingestion function must be called with this value and must not use data dated
    after it — see the frozen contracts in skills.md §2. See sowing_date() for `actual_sowing_date`.
    """
    config = config or load_config()
    sowing = sowing_date(season_year, config, actual_sowing_date)
    return sowing + timedelta(days=config["scoring_cutoff_days_after_sowing"])


class CutoffInFutureError(Exception):
    """Raised when a scoring cutoff hasn't been reached yet — not enough of the season has
    elapsed to compute this season's features. Only meaningful for a live application scored
    against today's real calendar date; historical backtesting (tests, demo scenarios, training)
    deliberately scores past season_years and should never hit this — see src/api/main.py, the
    only caller that checks it, for why it isn't baked into scoring_cutoff_date() itself.
    """


def ensure_cutoff_reached(cutoff_date: date, as_of: date | None = None) -> None:
    as_of = as_of or date.today()
    if cutoff_date > as_of:
        days_remaining = (cutoff_date - as_of).days
        raise CutoffInFutureError(
            f"scoring cutoff ({cutoff_date.isoformat()}) is {days_remaining} day(s) in the future "
            f"— not enough of this season has elapsed yet to score this application; check back "
            f"on or after {cutoff_date.isoformat()}"
        )


def district_centroid(config: dict | None = None) -> tuple[float, float]:
    """(lat, lon) for the configured district — a convenience for scripts/tests; the
    per-farmer scoring path (Phase 8) takes an actual plot lat/lon, not this centroid.
    """
    config = config or load_config()
    return config["centroid"]["lat"], config["centroid"]["lon"]
