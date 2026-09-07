"""FastAPI /score endpoint — execution.md Phase 10. Wires predict.score() + narrative together."""
from datetime import date

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.config import CutoffInFutureError, ensure_cutoff_reached, load_config, scoring_cutoff_date
from src.llm.narrative import NarrativeUnavailableError, generate_narrative
from src.model.predict import compare_crops as predict_compare_crops
from src.model.predict import score as predict_score

app = FastAPI(title="AgriRisk") 


class ScoreRequest(BaseModel):
    lat: float
    lon: float
    district: str
    crop: str
    loan_amount: float
    area_ha: float
    season_year: int
    loan_tenure_days: float

    # Real sowing date as reported by the farmer, overriding configs/district.yaml's generic
    # "{year}-06-15" template — see src/config.py's sowing_date() docstring for why this matters:
    # the template was always a uniform simplification across the training pool, never a
    # substitute for what actually happened on this specific farmer's plot this season.
    actual_sowing_date: date | None = None

    # "Ask the lendee" factors — not derivable from lat/lon, see src/model/mitigants.py.
    # All optional: a plot can be scored before this intake is complete.
    irrigation_source: str | None = None  # "canal" | "borewell" | "tank" | "rainfed"
    pmfby_enrolled: bool | None = None
    fpo_member: bool | None = None
    existing_annual_debt_service: float | None = None
    off_farm_annual_income: float | None = None


@app.post("/score")
def score_endpoint(req: ScoreRequest) -> dict:
    config = load_config()
    cutoff = scoring_cutoff_date(req.season_year, config, actual_sowing_date=req.actual_sowing_date)
    try:
        ensure_cutoff_reached(cutoff)
    except CutoffInFutureError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    try:
        result = predict_score(
            req.lat, req.lon, req.district, req.crop, req.season_year, cutoff,
            req.area_ha, req.loan_amount, req.loan_tenure_days,
            irrigation_source=req.irrigation_source, pmfby_enrolled=req.pmfby_enrolled,
            fpo_member=req.fpo_member,
            existing_annual_debt_service=req.existing_annual_debt_service,
            off_farm_annual_income=req.off_farm_annual_income,
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    loan = {
        "district": req.district, "crop": req.crop, "season_year": req.season_year,
        "loan_amount": req.loan_amount, "area_ha": req.area_ha,
    }
    try:
        result["narrative"] = generate_narrative(result, loan)
    except NarrativeUnavailableError as e:
        result["narrative"] = None
        result["narrative_error"] = str(e)  # score itself still returned — narrative is additive

    return result


@app.post("/compare-crops")
def compare_crops_endpoint(req: ScoreRequest) -> dict:
    """Same plot/loan terms, scored across every crop in configs/district.yaml's
    comparable_crops — "can approve X for crop A, not for the crop the farmer requested" (req.crop
    is the requested crop). No LLM narrative here by design: a 4-crop comparison is a table/chart
    the dashboard renders directly, not prose — src/model/predict.py's compare_crops() docstring
    has the full scope note (only Rice gets this-season-adjusted Survival; Sugarcane excluded).
    """
    config = load_config()
    cutoff = scoring_cutoff_date(req.season_year, config, actual_sowing_date=req.actual_sowing_date)
    try:
        ensure_cutoff_reached(cutoff)
    except CutoffInFutureError as e:
        raise HTTPException(status_code=422, detail=str(e)) from e

    try:
        return predict_compare_crops(
            req.lat, req.lon, req.district, req.crop, req.season_year, cutoff,
            req.area_ha, req.loan_amount, req.loan_tenure_days,
            irrigation_source=req.irrigation_source, pmfby_enrolled=req.pmfby_enrolled,
            fpo_member=req.fpo_member,
            existing_annual_debt_service=req.existing_annual_debt_service,
            off_farm_annual_income=req.off_farm_annual_income,
        )
    except Exception as e:
        raise HTTPException(status_code=422, detail=str(e)) from e
