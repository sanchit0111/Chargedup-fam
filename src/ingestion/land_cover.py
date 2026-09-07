"""Land-cover verification — is this actually agricultural land, not a building/road/water body?

A real gap this project had until 2026-09-07: geo_sanity_check (src/ingestion/geocoding.py) only
checks "is this roughly the right district" (an ~80km radius) — nothing verified the *specific*
lat/lon is farmland. Coordinates sitting on a residential building 2km from the district centroid
would pass every existing check and still get scored/approved. This closes that gap with a live
satellite classification, not a guess: ESA WorldCover v200 (2021, 10m resolution, global) via the
same Google Earth Engine service account already used for NDVI (satellite.py) — verified live
2026-09-07 against known points (Bellary centroid -> Cropland=40, Bengaluru city center ->
Built-up=50).

Deliberately not merged into satellite.py's GEE init — this module's failure mode (can't classify
land cover) is independent of NDVI's, and duplicating the ~10-line init is a smaller risk than
refactoring already-tested code.
"""
import json
import os
from functools import lru_cache

import ee

_initialized = False

WORLDCOVER_ASSET = "ESA/WorldCover/v200/2021"

# ESA WorldCover class codes -> label. https://esa-worldcover.org/en
LAND_COVER_CLASSES = {
    10: "Tree cover", 20: "Shrubland", 30: "Grassland", 40: "Cropland", 50: "Built-up",
    60: "Bare/sparse vegetation", 70: "Snow/ice", 80: "Water bodies", 90: "Wetland",
    95: "Mangroves", 100: "Moss/lichen",
}

# Structurally incompatible with agriculture — a loan application at one of these coordinates
# isn't farmland, regardless of what was claimed on the intake form.
NON_AGRICULTURAL_CLASSES = {50, 80, 70}  # Built-up, Water bodies, Snow/ice

CROPLAND_CLASS = 40


class LandCoverUnavailableError(Exception):
    """Raised when Earth Engine has no usable WorldCover classification for this point."""


def _ensure_initialized() -> None:
    global _initialized
    if _initialized:
        return
    creds_raw = os.environ.get("GEE_SERVICE_ACCOUNT_JSON")
    if not creds_raw:
        raise LandCoverUnavailableError("GEE_SERVICE_ACCOUNT_JSON not set — see .env.example")
    info = json.loads(creds_raw)
    credentials = ee.ServiceAccountCredentials(info["client_email"], key_data=creds_raw)
    ee.Initialize(credentials)
    _initialized = True


@lru_cache(maxsize=512)
def get_land_cover_class(lat: float, lon: float) -> tuple:
    """(class_code, label) at (lat, lon) — a single 10m pixel read, not a zonal average, since
    the question is "is this exact point farmland," not "is this area mostly farmland."
    """
    _ensure_initialized()
    point = ee.Geometry.Point([lon, lat])
    image = ee.Image(WORLDCOVER_ASSET)
    value = image.reduceRegion(ee.Reducer.first(), point, 10).get("Map").getInfo()
    if value is None:
        raise LandCoverUnavailableError(f"no WorldCover classification at {lat},{lon}")
    code = int(value)
    return code, LAND_COVER_CLASSES.get(code, f"unknown class {code}")


def land_use_check(lat: float, lon: float) -> dict:
    """Returns {agricultural_plausible, class_code, class_label, caveat}. `agricultural_plausible`
    is False (hard flag) for structurally non-agricultural classes, True for Cropland, None
    (soft flag, not a settled fact) for everything else — grassland/bare/shrubland/tree-cover
    could plausibly be fallow or newly-cleared farmland at 10m resolution.
    """
    try:
        code, label = get_land_cover_class(lat, lon)
    except LandCoverUnavailableError as e:
        return {"agricultural_plausible": None, "class_code": None, "class_label": None, "caveat": str(e)}

    if code in NON_AGRICULTURAL_CLASSES:
        return {
            "agricultural_plausible": False, "class_code": code, "class_label": label,
            "caveat": (
                f"Satellite land cover at this exact point is classified '{label}' "
                f"(ESA WorldCover, 10m, 2021) — this is not farmland, regardless of what was "
                f"claimed on intake."
            ),
        }
    if code == CROPLAND_CLASS:
        return {"agricultural_plausible": True, "class_code": code, "class_label": label, "caveat": None}
    return {
        "agricultural_plausible": None, "class_code": code, "class_label": label,
        "caveat": (
            f"Land cover classified '{label}', not 'Cropland' — plausibly fallow or "
            f"newly-cleared farmland at 10m resolution, but worth a field check, not a settled fact."
        ),
    }


if __name__ == "__main__":
    import src.config  # noqa: F401 — side effect: load_dotenv(), same pattern as this project's other __main__ blocks

    print("Bellary centroid (known farmland):", land_use_check(15.14, 76.92))
    print("Bengaluru city center (should NOT be agricultural):", land_use_check(12.9716, 77.5946))
