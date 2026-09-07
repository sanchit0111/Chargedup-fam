"""District geocoding + a coarse plot/district sanity check — the cheap half of a fraud-integrity
control: does the submitted plot lat/lon actually sit near the district the application claims?

This is NOT a real cadastral/boundary check (no district polygon data is wired in) — it's a
straight-line distance from the district's geocoded centroid, flagged past a generous threshold.
A false negative (fraud that stays within threshold) is expected and fine for a hackathon-scope
control; a false positive here should be rare given the threshold's size relative to a typical
Indian district. Said plainly rather than silently presented as verified identity — same honesty
pattern as everything else in this codebase.
"""
from __future__ import annotations

import math
from functools import lru_cache

import requests

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
# Nominatim's usage policy requires a real identifying User-Agent — anonymous/default UAs get
# blocked. 1 req/sec max; @lru_cache means each district is only ever looked up once per process.
REQUEST_HEADERS = {"User-Agent": "AgriRisk-hackathon-demo/1.0 (agri-lending risk scoring)"}
REQUEST_TIMEOUT = 15

# A coarse sanity net, not a boundary check (see module docstring) — most Indian districts span
# well under this radius; a few large/desert districts (e.g. Kutch) don't, which is a known,
# accepted false-negative gap at this scope.
GEO_SANITY_THRESHOLD_KM = 80.0


class GeocodingUnavailableError(Exception):
    """Raised when Nominatim has no result for the given district/state, or is unreachable."""


@lru_cache(maxsize=256)
def geocode_district(district: str, state: str = "Karnataka", country: str = "India") -> tuple:
    """(lat, lon) for a district's centroid via Nominatim/OpenStreetMap — live, not a static
    lookup table, so it works for any district name, not just configs/district.yaml's pool.
    """
    params = {"q": f"{district}, {state}, {country}", "format": "json", "limit": 1}
    resp = requests.get(NOMINATIM_URL, params=params, headers=REQUEST_HEADERS, timeout=REQUEST_TIMEOUT)
    resp.raise_for_status()
    results = resp.json()
    if not results:
        raise GeocodingUnavailableError(f"no geocoding result for {district!r}, {state!r}")
    return float(results[0]["lat"]), float(results[0]["lon"])


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 6371.0  # Earth radius, km
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def geo_sanity_check(lat: float, lon: float, district: str, state: str = "Karnataka") -> dict:
    """Returns {plausible, distance_km, district_centroid} — or {plausible: None, reason: ...} if
    geocoding itself failed, so a network hiccup here degrades to "unchecked", not a hard error
    (this is a mitigant-tier signal, not a blocking one — see predict.py's mitigants call site).
    """
    try:
        centroid_lat, centroid_lon = geocode_district(district, state)
    except (GeocodingUnavailableError, requests.exceptions.RequestException) as e:
        return {"plausible": None, "reason": str(e)}

    distance = haversine_km(lat, lon, centroid_lat, centroid_lon)
    return {
        "plausible": distance <= GEO_SANITY_THRESHOLD_KM,
        "distance_km": round(distance, 1),
        "district_centroid": [centroid_lat, centroid_lon],
    }


if __name__ == "__main__":
    print("Bellary centroid:", geocode_district("Bellary"))
    print("plot at Bellary's own centroid:", geo_sanity_check(15.14, 76.92, "Bellary"))
    print("plot 300km away claiming Bellary:", geo_sanity_check(12.97, 77.59, "Bellary"))  # Bengaluru
