"""Agmarknet mandi price ingestion — execution.md Phase 5. Contract frozen in skills.md §2.

Resource id + demo API key verified live in data/raw/PHASE1_FINDINGS.md. Register a dedicated
DATA_GOV_IN_API_KEY for anything beyond hackathon-demo use — see .env.example.
"""
import os
from datetime import date, datetime, timedelta
from functools import lru_cache
from statistics import pstdev

import requests

AGMARKNET_URL_TEMPLATE = "https://api.data.gov.in/resource/{resource_id}"
PAGE_SIZE = 500
MAX_PAGES = 30  # bounds worst-case fetch to 15,000 rows for one district+commodity
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0"}  # data.gov.in is flaky/throttles bare requests-UA
REQUEST_TIMEOUT = 60
RETRIES_PER_PAGE = 3


class PriceDataUnavailableError(Exception):
    """Raised when data.gov.in has no usable price series for the given district/crop/window."""


def _parse_arrival_date(raw: str) -> date:
    return datetime.strptime(raw, "%d/%m/%Y").date()


@lru_cache(maxsize=None)
def _all_prices(district: str, crop: str) -> tuple:
    """All (date, Modal_Price) pairs ever recorded for district+crop, fetched once and cached —
    every window request (get_price_volatility, get_price_level, price.py's expected_price across
    several historical seasons) filters this in memory instead of re-paging the API per window.
    The API only supports exact-match filters (no server-side date range), so there's no cheaper
    way to ask for "just this window" than paging through everything once and slicing locally.
    """
    from src.config import load_config

    config = load_config()
    resource_id = config["data_gov_in"]["agmarknet_resource_id"]
    api_key = os.environ.get("DATA_GOV_IN_API_KEY")
    if not api_key:
        raise PriceDataUnavailableError("DATA_GOV_IN_API_KEY not set — see .env.example")

    url = AGMARKNET_URL_TEMPLATE.format(resource_id=resource_id)

    pairs = []
    for page in range(MAX_PAGES):
        params = {
            "api-key": api_key,
            "format": "json",
            "limit": PAGE_SIZE,
            "offset": page * PAGE_SIZE,
            "filters[District]": district,
            "filters[Commodity]": crop,
        }
        resp = None
        last_error = None
        for _ in range(RETRIES_PER_PAGE):
            try:
                resp = requests.get(
                    url, params=params, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT
                )
                break
            except requests.exceptions.RequestException as e:
                last_error = e
                resp = None
        if resp is None:
            raise PriceDataUnavailableError(f"data.gov.in unreachable after retries: {last_error}")
        if resp.status_code != 200:
            raise PriceDataUnavailableError(f"data.gov.in returned {resp.status_code}")
        records = resp.json().get("records", [])
        if not records:
            break
        for r in records:
            try:
                d = _parse_arrival_date(r["Arrival_Date"])
                price = float(r["Modal_Price"])
            except (KeyError, ValueError, TypeError):
                continue
            pairs.append((d, price))
        if len(records) < PAGE_SIZE:
            break

    return tuple(pairs)


def _fetch_prices_in_window(district: str, crop: str, start: date, end: date) -> list:
    """Modal_Price values for district+crop dated in [start, end], sliced from the cached
    full series (see _all_prices)."""
    return [price for d, price in _all_prices(district, crop) if start <= d <= end]


def get_price_volatility(district: str, crop: str, season_year: int, cutoff_date: date) -> float:
    """Std dev of Modal_Price for district+crop over the whole season so far
    [sowing_date, cutoff_date] (execution.md §0 constraint 2)."""
    from src.config import sowing_date

    sowing = sowing_date(season_year)
    prices = _fetch_prices_in_window(district, crop, sowing, cutoff_date)
    if len(prices) < 2:
        raise PriceDataUnavailableError(
            f"fewer than 2 usable price points for {district}/{crop} in "
            f"[{sowing}, {cutoff_date}] — got {len(prices)}"
        )
    return pstdev(prices)


RECENT_PRICE_WINDOW_DAYS = 30


def get_price_level(district: str, crop: str, season_year: int, cutoff_date: date) -> float:
    """Mean Modal_Price (INR/quintal) over the RECENT_PRICE_WINDOW_DAYS before cutoff_date —
    "where is the market right now," feeding price.py's expected_price(). Never uses data dated
    after cutoff_date (execution.md §0 constraint 2).
    """
    from src.config import sowing_date

    sowing = sowing_date(season_year)
    window_start = max(sowing, cutoff_date - timedelta(days=RECENT_PRICE_WINDOW_DAYS))
    prices = _fetch_prices_in_window(district, crop, window_start, cutoff_date)
    if not prices:
        raise PriceDataUnavailableError(
            f"no usable price points for {district}/{crop} in [{window_start}, {cutoff_date}]"
        )
    return sum(prices) / len(prices)
