"""Soil ingestion — OpenLandMap organic carbon via GEE (execution.md Phase 6a).

ISRIC's SoilGrids REST API (rest.isric.org) was the original plan but returned null for our exact
district centroid on 3 repeated tries, even though nearby points a few km away returned real
values — a real, reproducible gap in that specific legacy endpoint's coverage, not a transient
error (see data/raw/PHASE1_FINDINGS.md). Using OpenLandMap's organic-carbon layer via GEE instead,
since GEE is infrastructure we already have reliable credentials for.

Note on depth labels: OpenLandMap's bands (b0/b10/b30/...) are point readings *at* that depth, not
layer averages over a range the way ISRIC's SoilGrids "0-5cm"/"5-15cm" convention is — so this
module labels them "0cm"/"10cm"/"30cm" rather than borrowing SoilGrids' range notation for data
that isn't actually binned that way. Don't relabel these as ISRIC-style ranges elsewhere.

Static — soil composition doesn't meaningfully change season to season, so per-point results are
cached rather than re-queried per scoring event.
"""
from functools import lru_cache

import ee

from src.ingestion.satellite import _ensure_initialized

SOC_ASSET = "OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02"
DEPTH_BANDS = {"0cm": "b0", "10cm": "b10", "30cm": "b30"}  # shallow -> deep, most relevant first


class SoilDataUnavailableError(Exception):
    """Raised when the soil layer has no value at this point."""


@lru_cache(maxsize=None)
def get_organic_carbon(lat: float, lon: float) -> dict:
    """Topsoil organic carbon by depth at (lat, lon): {"0cm": v, "10cm": v, "30cm": v} in
    OpenLandMap's native (relative) units. Missing depths are simply omitted from the dict rather
    than raising, unless every depth is missing.
    """
    _ensure_initialized()
    point = ee.Geometry.Point([lon, lat])
    image = ee.Image(SOC_ASSET).select(list(DEPTH_BANDS.values()))
    result = image.reduceRegion(reducer=ee.Reducer.mean(), geometry=point, scale=250).getInfo()

    values = {label: float(result[band]) for label, band in DEPTH_BANDS.items()
              if result.get(band) is not None}
    if not values:
        raise SoilDataUnavailableError(f"no organic carbon values for {lat},{lon}")
    return values


def _topsoil_value(depths: dict) -> float:
    """The shallowest available depth reading — what fertility.py's ratio modifier uses."""
    for label in DEPTH_BANDS:  # "0cm", "10cm", "30cm" — shallowest first
        if label in depths:
            return depths[label]
    raise SoilDataUnavailableError("no usable depth in organic-carbon result")


if __name__ == "__main__":
    from src.config import district_centroid

    lat, lon = district_centroid()
    print("centroid organic carbon by depth:", get_organic_carbon(lat, lon))
    print("offset plot organic carbon by depth:", get_organic_carbon(lat, lon + 0.2))
