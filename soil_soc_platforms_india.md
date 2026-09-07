# Platforms/APIs for Soil Organic Carbon from Latitude/Longitude

Focus: services that can take latitude/longitude coordinates, directly or through point sampling, and return or estimate Soil Organic Carbon (SOC) for the Indian subcontinent.

## Recommended Options

| Platform | Lat/long point query | SOC variable | India coverage | Free/API status | Notes |
|---|---:|---|---|---|---|
| ISRIC SoilGrids | Yes, via REST when available; stable via WCS/GEE | `soc`, `ocd`, `ocs` | Global, includes India | Free, CC BY 4.0 | Best scientific default. REST API is beta and may be unstable; WCS/WebDAV/GEE are recommended for stable access. |
| Google Earth Engine + SoilGrids assets | Yes, via `sample()` / Earth Engine API | `projects/soilgrids-isric/soc_mean`, plus `ocd_mean`, `ocs_mean` | Global, includes India | Free for eligible noncommercial/research use; paid for commercial/operational use | Good for batch point queries over India. SoilGrids GEE assets have six depth bands. |
| OpenLandMap / OpenGeoHub STAC + COG | Yes, by sampling COG rasters from STAC URLs | SOC content, SOC density | Global, includes India | Public data, generally CC BY 4.0 | Good open alternative to SoilGrids. Supports self-hosted or server-side raster sampling workflows. |
| Google Earth Engine + OpenLandMap dataset | Yes, via Earth Engine API | Organic carbon content | Global, includes India | Same as Earth Engine access terms | Dataset ID: `OpenLandMap/SOL/SOL_ORGANIC-CARBON_USDA-6A1C_M/v02`; six standard depths, 250 m resolution. |
| Tarla.io Soil Analyze API | Yes, direct GET API | `soil_organic_carbon_content` | Claims coordinate-based global soil API; India should be verified with sample calls | API key required; free tier/limits unclear | Easiest direct JSON API shape, but less transparent scientifically than SoilGrids/OpenLandMap. |
| FAO GSOCmap | Map/web service, not ideal direct API | SOC stock, mainly 0-30 cm | Global, includes India | Public/CC BY | Useful reference layer, but less convenient for app integration unless consuming map services or tiles. |
| HWSD / hwsdr | Yes, through R package or local raster/database sampling | Organic carbon fields in soil database | Global, includes India | Free data | Coarser, older legacy dataset. Useful fallback, not preferred for modern SOC prediction. |
| BHOOMI Geoportal / ICAR-NBSS&LUP | Public geoportal; API not clearly documented | Soil Organic Carbon / Organic Carbon layers | India-focused | Public portal; API status unclear | Most India-specific source found, but appears oriented around visualization rather than a documented free API. |

## Best Practical Choices

1. For a quick hackathon prototype, use SoilGrids REST if it is live, with Google Earth Engine SoilGrids as fallback.

   Example SoilGrids REST pattern:

   ```text
   https://rest.isric.org/soilgrids/v2.0/properties/query?lat=LAT&lon=LON&property=soc&value=mean
   ```

2. For reliable batch querying, use Google Earth Engine and sample SoilGrids or OpenLandMap images at coordinate points.

3. For a fully open and self-hostable system, use OpenLandMap STAC/COGs or SoilGrids WCS/WebDAV and sample rasters server-side.

4. For India-specific authoritative context, inspect BHOOMI/ICAR-NBSS&LUP layers, but do not assume a public API without confirming service endpoints.

## Source Links

- ISRIC SoilGrids FAQ and access notes: https://docs.isric.org/globaldata/soilgrids/SoilGrids_faqs_02.html
- SoilGrids on Google Earth Engine: https://docs.isric.org/globaldata/soilgrids/access_on_gee.html
- OpenLandMap documentation: https://docs.openlandmap.org/
- Tarla Soil Analyze API docs: https://world.tarla.io/api/docs/
- BHOOMI Geoportal: https://bhoomigeoportal-nbsslup.in/BHOOMI_StateWise/
- FAO Global Soil Organic Carbon Map: https://www.fao.org/soils-portal/data-hub/soil-maps-and-databases/global-soil-organic-carbon-map-gsocmap/en/
