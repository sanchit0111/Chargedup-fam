# AgriRisk — Execution Runbook

Purpose: a build-order playbook for the agent implementing AgriRisk (see `product_idea.md` for the
product shape). Optimized for one thing — **a tight, real, end-to-end pipeline** (real APIs, real
data, no mocked stages) by demo time, not breadth of features. Build in the phase order below;
each phase has a **Definition of Done (DoD)** that must pass before moving to the next. Do not
parallelize phases 1–7 — each is a real dependency of the next, not a nice-to-have ordering.

**Architecture note (supersedes the original single-classifier plan):** Phases 6–8 below implement
three independently-validated pathways — Fertility × Survival → Production; Production × Price →
Revenue; Revenue vs. Loan → Coverage Ratio → risk — instead of one gradient-boosted classifier over
a merged feature vector. This changed after Phase 2/5 turned up a real constraint: ICRISAT yield
history (1990–2015) and Agmarknet price history (2010–present, for the chosen district+crop) don't
overlap, so a single jointly-trained model would fit the price signal to zero co-occurring positive
examples. See `.claude/agent_collab_playbook/SKILL.md`'s architecture note for the full reasoning.

## 0. Non-negotiable constraints (read before building anything)

1. **No mocked data in the final pipeline.** Notebooks/scratch scripts can use fixtures while
   iterating, but the pipeline that runs at demo time must hit real APIs / real downloaded datasets
   end to end. If a phase can't get real data in time, that's a scope-cut decision to surface to the
   user immediately — don't silently stub it and keep going.
2. **Temporal leakage is the single biggest correctness risk in this system — treat it as a hard
   constraint, not a modeling nicety.** The model must only ever see data available *as of the
   scoring point* (loan origination, or a mid-season check-in). Concretely:
   - Ground-truth yield/claims data is only known at harvest — never use full-season NDVI or final
     yield figures as a *feature*, only as the *label*.
   - Pick one explicit scoring cutoff for the MVP (recommended: **60 days after monsoon onset /
     sowing**) and only ever build features from data available before that cutoff. Write this
     cutoff into `configs/district.yaml` so it's enforced in one place, not re-decided per script.
   - When you train/test split, split **by year (out-of-time)**, never randomly — random splits
     leak season-level weather patterns across train/test and will produce a fake-good AUC.
3. **One district, hardcoded, for the whole build.** Do not generalize to multi-district or
   multi-crop until the single-district pipeline is proven end to end. Generalizing early is how
   hackathon projects run out of time with nothing demoable.
4. **Every ingestion script is runnable standalone** (`python -m src.ingestion.satellite --help`)
   and prints/returns real output for a real lat/lon — this is both a testability requirement and
   what "tight end to end" means in practice: each link in the chain independently provably works.

## 1. Environment Setup

```
python -m venv .venv && source .venv/bin/activate
pip install earthengine-api geopandas rasterio xarray requests pandas numpy xlrd \
            fastapi uvicorn streamlit folium pydeck anthropic python-dotenv pyyaml
```

No `xgboost`/`shap` — the small-sample discipline in `SKILL.md` §7 rules out a tree ensemble for
Survival (~25 labeled ICRISAT seasons), and a documented coverage-ratio formula doesn't need
post-hoc explainability. `numpy` (already a `pandas` dependency) covers the linear regression.

Accounts / credentials needed (`.env`, never commit):

| Var | Source | Notes |
|---|---|---|
| `GEE_SERVICE_ACCOUNT_JSON` | Google Earth Engine — register a Cloud project, enable Earth Engine API, create a service account | non-instant approval, **do this first, day 1** |
| `ANTHROPIC_API_KEY` | Anthropic Console | for the LLM narrative layer |
| `DATA_GOV_IN_API_KEY` | data.gov.in → register → API key | for Agmarknet mandi price resource |

NASA POWER and SoilGrids need no auth/key.

## 2. Repo Structure

```
hackathon/
  configs/
    district.yaml          # chosen district, scoring cutoff, crop, date ranges — single source of truth
  data/
    raw/                    # untouched downloads (ICRISAT/PMFBY csv, etc.)
    interim/                 # per-modality pulled series, keyed by geo+season
    processed/                # joined features.csv with label column
  src/
    ingestion/
      satellite.py            # GEE NDVI pull + anomaly calc
      weather.py               # NASA POWER rainfall/dry-spell pull
      market.py                  # Agmarknet price level + volatility pull
      ground_truth.py             # ICRISAT yield_series() loader + build_labels() proxy label
    model/
      fertility.py                  # reference_yield() — trailing ICRISAT baseline
      survival.py                    # survival_factor() — regression, weather -> yield ratio
      train_survival.py               # fits + validates survival.py against ICRISAT, saves coefficients
      price.py                          # expected_price() — Agmarknet trend + harvest-dip adjustment
      combine.py                         # coverage_ratio() — Production x Price vs Loan -> risk_score
      predict.py                          # orchestrates fertility+survival+price+combine
    llm/
      narrative.py                         # predict.score() breakdown -> Claude -> narrative text
    api/
      main.py                               # FastAPI /score endpoint, wires everything together
    dashboard/
      app.py                                 # Streamlit demo UI
  models/
    survival_coefficients.json              # fitted regression weights + fit metric, not a black-box artifact
  tests/
    test_ingestion_smoke.py              # each ingestion fn returns real, sane output for known points
  README.md                               # exact commands to reproduce, for the demo
```

## 3. Build Phases

### Phase 1 — District & data-availability spike
Pick 2–3 candidate districts. For each, check: MODIS/Sentinel coverage (trivial — global), NASA
POWER coverage (trivial — global), and **whether ICRISAT district yield data or PMFBY claims data
actually has usable rows for that district/crop/years**. This last check is the real constraint and
must be verified by actually pulling a sample, not assumed.

**DoD**: `configs/district.yaml` populated with district name, state, primary crop, season date
range, scoring cutoff date offset, and the chosen ground-truth source — before writing any other
code.

### Phase 2 — Ground truth
Download ICRISAT district-level yield series or PMFBY claims data for the chosen district. Compute
a proxy label per season/year: yield shortfall vs trailing N-year trend (e.g. >20% below trend =
stress=1) or claim-triggered=1. Land this in `data/raw/` (original) and `data/processed/labels.csv`
(district, year, label).

**DoD**: `labels.csv` with ≥8 years of labeled seasons for the district, both classes present (not
all-0 or all-1 — check this explicitly, it will break model training silently otherwise).

### Phase 3 — Satellite ingestion (`src/ingestion/satellite.py`)
GEE, `MODIS/061/MOD13Q1` NDVI collection (250m, 16-day composite — apply the 0.0001 scale factor).
Given a lat/lon + season, pull the NDVI time series **up to the scoring cutoff date only** (Phase 0
constraint #2), and compute anomaly vs the same window's 5-year historical mean at that point.

**DoD**: function `get_ndvi_anomaly(lat, lon, season_year) -> float`, tested against 3 known points
in the district, values sanity-checked by eye (e.g. a known drought year shows negative anomaly).

### Phase 4 — Weather ingestion (`src/ingestion/weather.py`)
NASA POWER daily API:
`https://power.larc.nasa.gov/api/temporal/daily/point?parameters=PRECTOTCORR&community=AG&longitude={lon}&latitude={lat}&start={YYYYMMDD}&end={YYYYMMDD}&format=JSON`
Sum rainfall up to the scoring cutoff, compare to the district's historical normal for that window.

**DoD**: `get_rainfall_deficit(lat, lon, season_year) -> float`, same 3-point sanity check as Phase 3.

### Phase 5 — Market ingestion (`src/ingestion/market.py`)
data.gov.in Agmarknet resource API (resource-id based; look up the correct resource id for the
mandi-prices dataset during Phase 1, don't guess it). Pull price series for the district's primary
crop, compute pre-cutoff volatility (std dev of daily/weekly price over the window).

**DoD**: `get_price_volatility(district, crop, season_year) -> float`, verified against at least one
season with a manually-checked known price spike/crash.

### Phase 6a — Fertility sub-model (`src/model/fertility.py`)
`reference_yield(season_year, config)`: trailing mean of `ground_truth.yield_series()` over the
prior `ndvi.historical_baseline_years` seasons. Move the ICRISAT-file-parsing logic that currently
lives inline in `ground_truth.build_labels()` into a shared `yield_series()` function that both
modules import — don't fork a second ICRISAT loader.

**DoD**: `reference_yield()` returns a sane kg/ha figure for at least 3 known seasons; both
`fertility.py` and `ground_truth.py` import the same `yield_series()`, verified by grep, not by eye.

### Phase 6b — Survival sub-model (`src/model/survival.py`, `src/model/train_survival.py`)
Regression target: `actual_yield / reference_yield` per ICRISAT season (from Phase 2/6a). Features:
`rainfall_deficit` + `dry_spell_days` (Phase 4) to start, NDVI anomaly added once GEE unblocks.
**Linear/ridge regression via `numpy.linalg.lstsq`, not a tree ensemble** — ~25 labeled seasons
can't support more model capacity than that without overfitting (SKILL.md §7). `train_survival.py`
fits the regression, reports R²/held-out error, and saves the coefficients (a small JSON, not a
black-box pickle) to `models/survival_coefficients.json`. `survival.py`'s `survival_factor()` loads
those coefficients and applies them to a new lat/lon/season.

**DoD**: `train_survival.py` run against real ICRISAT + NASA POWER data reports a fit metric (even
if weak — report it honestly, don't hide a bad R² by not printing it); `survival_factor()` runs
standalone for a real lat/lon and returns a plausible multiplier (roughly 0.5–1.2 range).

### Phase 6c — Price sub-model (`src/model/price.py`)
`expected_price(district, crop, season_year, cutoff_date)`: builds on `market.get_price_level()`
(pre-cutoff price trend), adjusted for the historical harvest-time supply-glut dip observed in the
Agmarknet series for this crop. Validated against Agmarknet's own multi-year seasonality — this
pathway never touches ICRISAT data.

**DoD**: `expected_price()` runs standalone for the configured district+crop across 3+ seasons,
values sanity-checked against the raw Agmarknet series by eye.

### Phase 7 — Combine (`src/model/combine.py`)
`coverage_ratio(production_kg, expected_price, loan_amount) -> dict`. `revenue = production_kg / 100
* expected_price` (Agmarknet prices are INR/quintal = 100kg — don't silently drop the unit
conversion), `coverage_ratio = revenue / loan_amount`. `risk_score` is a **documented formula** of
coverage ratio (e.g. a monotonic mapping like `risk_score = clip(1 - coverage_ratio, 0, 1)`,
tunable but written down in the code, not learned) — the point is that this number stays auditable
back to arithmetic a credit committee can check, not a model output to be trusted blindly.

**DoD**: unit-testable pure function (no I/O) — given a few hand-picked (production, price, loan)
triples, the output coverage ratios and risk scores match hand-calculated expectations.

### Phase 8 — Per-farmer scoring layer (`src/model/predict.py`)
Given (lat/lon, district, crop, loan amount, area_ha, season_year — the demo's "new application"),
call Phase 6a (fertility) + 6b (survival) to get `production_kg = reference_yield * survival_factor
* area_ha`, call Phase 6c (price) for `expected_price`, call Phase 7 (combine) for the coverage
ratio and risk score. Returns the full breakdown dict (see `SKILL.md` §2's `predict.score()`
contract), not just a final number — the intermediate values are what makes the narrative in Phase
9 legible.

**DoD**: given a real lat/lon inside the chosen district, returns the full breakdown dict end to
end, hitting real APIs (not a cached features file).

### Phase 9 — LLM narrative (`src/llm/narrative.py`)
Prompt template: inputs = the full `predict.score()` breakdown (reference yield, survival factor,
production, expected price, revenue, coverage ratio) + loan details. Output = 3–5 sentence
underwriting narrative that states the causal chain explicitly (see the example in
`product_idea.md` §9) + one concrete recommended mitigant. Use Claude via the `anthropic` SDK. Keep
every number in the narrative computed purely by Phases 6–7 — the LLM only explains/contextualizes
them (product_idea.md §9 guardrail).

**DoD**: tested on one clearly-high-coverage and one clearly-low-coverage example from Phase 8,
narratives name the actual reference yield / survival factor / price / coverage-ratio numbers they
were given, not generic boilerplate.

### Phase 10 — API (`src/api/main.py`)
FastAPI `/score` endpoint: input = {lat, lon, district, crop, loan_amount, area_ha, season_year},
calls Phase 8 then Phase 9, returns Phase 8's full breakdown dict plus the narrative string.

**DoD**: `curl -X POST localhost:8000/score -d '{...}'` returns valid JSON, full pipeline, real APIs,
under a few seconds (cache GEE/weather calls per lat/lon+season if latency is an issue).

### Phase 11 — Dashboard (`src/dashboard/app.py`)
Streamlit: map (folium/pydeck) centered on the plot, NDVI trend chart, rainfall chart, score gauge,
narrative text block. Calls the FastAPI `/score` endpoint (don't reimplement the pipeline in the
dashboard).

**DoD**: `streamlit run src/dashboard/app.py` shows a working scoring flow for at least one real
example plot in the chosen district.

### Phase 12 — Validation & demo prep
Run Phase 6b's held-out evaluation (Survival vs. actual ICRISAT yield) and Phase 6c's fit against
Agmarknet's own seasonality, reported separately per `SKILL.md` §4 (never blended into one number).
Prepare 2–3 fixed demo scenarios spanning a known bad season and a known good season for the
district (so the demo tells a legible story, not a random draw).

**DoD**: `README.md` has the exact commands to reproduce the pipeline from a clean clone, plus the
2–3 demo lat/lons + season_years to run live during the pitch.

## 4. If time runs out — cut order

Cut from the bottom up, not randomly, since each phase depends on the ones above it:
1. Phase 12 polish (calibration plot) — nice to have, not demo-critical.
2. Phase 11 dashboard → fall back to hitting the FastAPI endpoint with curl/Postman live.
3. Phase 6c (price) → if Agmarknet access breaks, report Production-side risk only (Fertility ×
   Survival vs. a fixed reference price) and say so explicitly in the pitch rather than silently
   shipping a 2-pathway system framed as 3.
4. Never cut Phase 6b's out-of-time split or Phase 0 constraint #2 (temporal leakage) to save time —
   a leaky demo number is worse than a smaller, honest one, especially in front of technical judges.

## 5. Definition of Done (whole system)

- One district, one crop, real satellite + weather + (ideally) market data, real proxy ground truth,
  out-of-time-validated model, live-scoring API, Claude narrative layer, working dashboard.
- Every ingestion function independently testable and tested against known points.
- README with exact reproduction commands and the specific demo scenarios to run live.
