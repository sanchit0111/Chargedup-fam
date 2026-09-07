# AgriRisk — Multimodal Agricultural Lending Risk Engine

Target market: **India**, agri-loan lending banks / NBFCs / RRBs / MFIs.

## 1. Problem Statement

Banks lending to farmers (crop loans, KCC, term loans for equipment/irrigation) currently underwrite
risk almost entirely off static, backward-looking data: farmer credit bureau score (often thin or
absent for smallholders), landholding papers, past repayment history, and self-reported crop plans.
This misses the dominant real driver of agri-credit risk — **whether the crop actually succeeds** —
which is a function of weather, soil, water availability, and market prices *during the loan tenure*,
not at disbursal time. This causes both over-lending into bad seasons and under-lending to genuinely
low-risk farmers who lack formal credit history. Agri-NPA rates in India are structurally higher than
other retail lending segments partly for this reason.

## 2. Product Vision

A **lending risk prediction module** that banks plug into their existing loan origination workflow.
Given a farmer + farm plot (geo-coordinates) + loan terms, it returns:

- A **risk score** (probability of default / stress) at application time, and a **dynamic risk
  trajectory** through the loan tenure as new satellite/weather data arrives.
- A **plain-language risk narrative** (LLM-generated) explaining *why*, in terms a loan officer or
  credit committee can act on and defend to an auditor/regulator.
- **Early-warning alerts** mid-season (e.g. NDVI drop, rainfall deficit) so the bank can proactively
  restructure or offer insurance linkage before default, not just score at origination.

The differentiator is fusing **satellite-observed ground truth about the farm** with traditional
financial/behavioral data — most incumbent agri-scoring tools use one or the other, rarely both.

## 3. Users

- **Loan officer / branch manager**: needs a fast score + a narrative they can put in a credit
  file, not a raw probability.
- **Credit risk / underwriting team**: needs explainability (feature attributions) and calibration,
  since agri lending is priority-sector and RBI-scrutinized.
- **Portfolio risk / collections team**: needs mid-season early-warning signals across their book,
  not just an origination-time score.

## 4. How It Works (Pipeline)

```
Farmer + farm geo-coordinates + loan application
        │
        ▼
┌───────────────────────────────────────────────────────┐
│ Data collection layer (per modality, keyed by geo+time) │
│  - Farmer/loan structured data (from bank's LOS)         │
│  - Satellite-derived agronomic signals                   │
│  - Weather / rainfall / drought indices                  │
│  - Market (mandi) price signals                          │
│  - Geospatial/infrastructure context                      │
└───────────────────────────────────────────────────────┘
        │
        ▼
Feature engineering (per-plot time series → tabular + embeddings)
        │
        ▼
Risk model (gradient-boosted trees on engineered features → prod: multimodal fusion model)
        │
        ├─► Risk score + calibrated probability + SHAP attributions
        │
        ▼
LLM layer (Claude): turns score + attributions + raw signals into
  - underwriting narrative
  - risk mitigants / recommendations (e.g. push crop insurance)
  - flags for human review
        │
        ▼
Loan officer dashboard (score, map, narrative, trend, alerts)
```

## 5. Data Modalities

| Modality | Signal | Example features |
|---|---|---|
| **Farmer/loan (structured)** | Existing LOS/bureau data | landholding size, crop pattern, irrigation type, past repayment, loan amount/tenure, CIBIL/bureau score if present |
| **Satellite / remote sensing** | Is the crop actually healthy? | NDVI/EVI level & anomaly vs 5-yr baseline, sowing-date estimate, land-cover/crop-type verification, waterlogging/drought stress flags |
| **Weather & climate** | Was the season good? | rainfall vs normal (deficit/surplus), monsoon onset delay, temperature extremes, historical drought frequency for the district |
| **Soil & water** | Structural farm quality | soil type/fertility, groundwater depth trend, distance to irrigation canal/reservoir |
| **Market/economic** | Will the harvest sell for enough? | mandi price level & volatility for the farmer's crop, MSP-vs-market gap, input cost trend (fertilizer/diesel) |
| **Geospatial/infrastructure** | Structural access risk | distance to nearest mandi/warehouse/cold storage, road connectivity, flood-plain proximity |
| **Alternative (prod-stage)** | Behavioral proxy | UPI/digital payment regularity, mobile recharge pattern, multi-year cropping consistency from satellite history |

## 6. Open-Source / Public Data Sources (India-focused)

**Satellite / remote sensing**
- **Google Earth Engine** — free for research/nonprofit use; single API surface over Sentinel-2,
  Landsat, MODIS; best choice for hackathon speed (no separate downloads/mosaicking needed).
- **Copernicus Sentinel-2 / Sentinel-1** (via Sentinel Hub or GEE) — 10m optical + SAR (SAR pierces
  cloud cover, useful during monsoon when optical imagery is unusable).
- **MODIS NDVI/EVI (MOD13Q1)** — 250m, 16-day composite, long historical baseline for anomaly
  detection; easiest starting point for a hackathon NDVI-anomaly feature.
- **Bhuvan (ISRO/NRSC)** — India-specific geoportal; has crop-type and cropping-intensity products
  tailored to Indian agriculture; API access is more limited than GEE but higher domain relevance.
- **Microsoft Planetary Computer** — alternative free STAC catalog + compute (Sentinel, Landsat,
  soil, land cover), Python-native, good GEE alternative if quota becomes an issue.

**Weather / climate**
- **NASA POWER API** — free, no auth for reasonable use, daily rainfall/temperature/solar by
  lat-lon, simplest weather API to integrate in a hackathon.
- **CHIRPS** (via GEE) — satellite+station-blended rainfall estimates, good resolution for India,
  standard input for drought/SPI indices.
- **IMD gridded rainfall data** — authoritative Indian rainfall data where available.

**Soil**
- **SoilGrids (ISRIC)** — global soil property maps (texture, organic carbon, pH) via REST API.

**Market prices**
- **Agmarknet (data.gov.in)** — daily mandi (wholesale market) prices by crop/state/district; open
  government data, direct proxy for farmer's expected sale price.
- **data.gov.in** broadly — crop production statistics, MSP data.

**Ground-truth / proxy label candidates** (see §7)
- **PMFBY (crop insurance) claims data** — district/block-level crop-failure claims, published by
  the Ministry of Agriculture; a real-world proxy for "the crop failed badly enough to trigger a
  payout."
- **ICRISAT District-Level Database** — long-run district crop yield time series, usable to compute
  yield-shortfall-vs-trend as a proxy label.
- **NDMA / state disaster management drought & flood declarations** — binary district-level
  stress events.

## 7. Ground Truth & Validation Strategy

Real loan-level default data is the hard constraint for a hackathon (no bank will hand over an NPA
book on short notice). Three options, not mutually exclusive — recommend starting with A, keeping B
as fallback, and pursuing C as the real path to product-market validation post-hackathon:

**A. Public proxy dataset (fastest to execute)**
Correlate satellite/weather-derived risk signals against **PMFBY crop-insurance claim rates** or
**ICRISAT yield-shortfall-vs-trend** at district/block level. This isn't loan default, but crop
failure is the dominant causal driver of agri loan default, so it's a defensible proxy for a
hackathon demo. Framing for judges: "we validate that our satellite risk signal predicts crop
failure; crop failure is the established leading indicator of agri default per RBI/NABARD NPA
literature."

**B. Synthetic/simulated ground truth (fallback if A's data proves too sparse/late in time)**
Rules-based simulator: `default_proxy = f(yield_shortfall, price_crash, existing_debt_burden)`.
Explicitly flagged in the demo as a stopgap, used only to sanity-check model plumbing end-to-end,
not as a claimed accuracy result.

**C. Partner bank/NBFC data (the real validation, post-hackathon)**
Anonymized historical loan book with actual repayment/default outcomes, geo-tagged to plots. This is
the only way to get a defensible AUC number for production. Should be framed in the doc/pitch as
"Phase 2: pilot with a partner lender," not assumed available for the hackathon itself.

## 8. Modeling Architecture

**Not a single black-box classifier.** The first design (one gradient-boosted classifier over a
merged feature vector) ran into a real problem during the hackathon build: Production-side ground
truth (ICRISAT district yield, 1990–2015) and Sale-side ground truth (Agmarknet mandi prices,
2010–present) don't overlap in time for the chosen district — training one model jointly on both
would have meant fitting the price signal to zero co-occurring positive examples. The fix is
structural, not statistical: split the problem along its actual causal seams, validate each piece
against *its own* ground truth independently, and combine only at inference time.

```
Fertility (baseline land quality)  ──┐
                                       ×──► Production (kg)
Survival (this season's shocks)   ──┘             │
                                                    ×──► Revenue (INR)
Price (expected at harvest)      ─────────────────┘             │
                                                                   vs. Loan Amount
                                                                   ──► Coverage Ratio ──► Risk Score
```

**Fertility** — "what would this land produce in a normal year." A trailing-mean reference yield
(kg/ha) from the ICRISAT district series, refinable to plot level with a static soil-quality signal
(SoilGrids organic carbon) once added. Slow-changing — computed once per district/plot, not per
scoring event. Validated against ICRISAT's full 1990–2015 window.

**Survival** — a multiplier (e.g. 0.6 = this season's shocks cut yield to 60% of baseline
potential) driven by this season's rainfall deficit, dry-spell length, and — once GEE credentials
land — NDVI anomaly. Given the small sample (~25 labeled ICRISAT seasons for one district), this is
a simple linear/ridge regression against the actual/reference yield ratio, not a tree ensemble —
more features than data points is how a hackathon model quietly overfits and produces a confident,
wrong number. Validated against the same ICRISAT window as Fertility.

**Price** — expected price at harvest (INR/quintal), from the Agmarknet trend adjusted for the
well-documented harvest-time supply-glut dip. Validated against Agmarknet's own multi-year price
series, entirely independent of the yield-side validation above — this is precisely the pathway
that doesn't need to share years with ICRISAT anymore.

**Combine** — `Production × Price = Revenue`; `Revenue / Loan Amount = Coverage Ratio`. This mirrors
how revenue-based crop insurance (e.g. USDA's Revenue Protection product) is actuarially designed —
not a novel methodology, an established one. The risk score is a documented function of coverage
ratio, not a learned black box, so it stays auditable back to a formula a credit committee can
actually inspect.

**Production-direction (roadmap, not hackathon scope)**
- Replace the linear Survival regression with embeddings from a **pretrained geospatial foundation
  model** (e.g. Clay, IBM/NASA Prithvi, SatMAE) once there's enough labeled seasons (multi-district,
  multi-year) to support it without overfitting — the small-sample constraint above is a hackathon
  reality, not a permanent architectural choice.
- Multimodal fusion inside the Survival pathway specifically: separate encoders per modality
  (time-series encoder for weather/NDVI, image embedding for satellite patch) → late-fusion — kept
  scoped to Survival rather than the whole system, since Fertility and Price have their own,
  separate validation stories that shouldn't be re-entangled.
- Mid-season **monitoring service**: re-scores active loans on a schedule as new Sentinel/MODIS
  passes and rainfall data arrive, re-running Survival (not the whole pipeline) and raising alerts
  on material coverage-ratio deviation.
- Feedback loop: actual repayment/default outcomes feed back into periodic retraining — and,
  critically, into recalibrating the coverage-ratio → risk-score function itself, which right now
  is a documented formula, not something fit to real defaults (see §7C).

## 9. LLM Layer

Explicitly acceptable/expected in production per product direction — this is where the model output
becomes usable by a non-data-scientist loan officer and defensible to a regulator.

- **Underwriting narrative generation**: given the full Fertility→Survival→Production→Price→Coverage
  breakdown from `predict.score()` (§8), Claude generates a short natural-language risk memo that
  states the actual causal chain, not a feature-attribution list — e.g. *"Expected production is 18
  quintals/acre against a fertility-baseline potential of 22; a 3-week dry spell during flowering cut
  this season's survival factor to 0.82. At the current price trend of ₹5,200/quintal, expected
  revenue is ₹93,600 against a loan request of ₹80,000 — 1.17× coverage."* This is legible to a
  credit committee in a way a SHAP value never is, and it's a direct readout of the model's actual
  arithmetic, not a post-hoc explanation of a black box.
- **Document extraction (prod-stage)**: vision-capable Claude models can extract structured data
  from scanned land records / KYC documents / physical loan applications, feeding the structured
  data modality without a manual data-entry step.
- **Compliance grounding (prod-stage)**: RAG over RBI priority-sector-lending and NABARD circulars
  so generated narratives stay within policy language the bank's compliance team can sign off on.
- **Guardrail**: the LLM explains and contextualizes a model score; it does not itself decide
  approve/reject. Score comes from the calibrated ML model, not from the LLM, to keep the
  quantitative decision auditable independent of prompt behavior.

## 10. Tech Stack Summary

| Layer | Hackathon choice | Notes |
|---|---|---|
| Satellite/geo data access | `earthengine-api` (Python) via Google Earth Engine | fastest path to NDVI/rainfall without hosting raster data yourself |
| Geospatial processing | `geopandas`, `rasterio`, `xarray` | plot-level zonal stats from raster time series |
| Weather API | NASA POWER REST API | no-auth, simple JSON |
| Market data | Agmarknet / data.gov.in APIs | mandi price series |
| ML | `numpy` linear/ridge regression for Survival | small-sample discipline (~25 ICRISAT seasons) — a tree ensemble would overfit; SHAP is unnecessary once the score is a documented formula, not a black box |
| LLM | Claude API (Sonnet/Haiku for narrative generation) | narrative + explainability layer |
| Backend | FastAPI | serve score + narrative endpoint |
| Dashboard | Streamlit + Folium/pydeck for map | fastest demo-able UI for a hackathon |
| Prod-stage additions | Prefect/Airflow (pipeline scheduling), a geospatial foundation model (Clay/Prithvi), vector DB for RAG, model monitoring | out of hackathon scope, listed for roadmap credibility |

## 11. Responsible AI / Regulatory Notes

Agricultural lending is RBI priority-sector lending — decisions need to be explainable and
non-discriminatory, and geography-based features carry real redlining risk (a village-level risk
score can proxy for caste/community composition in parts of India). For the product doc, not just
the hackathon demo:
- Model is a **decision-support score**, not an auto-reject gate.
- SHAP + LLM narrative required on every score, not optional — this is both a UX and a compliance
  requirement.
- Bias audit needed pre-production: check score distributions don't systematically disadvantage
  specific social groups via geographic proxies.
- Farmer/land-record data is sensitive PII — data handling/consent needs a real compliance review
  before any pilot with a bank.

## 12. Hackathon Scope Cut Line

In scope for the demo:
- 1 state / a handful of districts in India (pick one with good Sentinel/MODIS coverage and PMFBY
  claims data availability — validate data availability before committing to a specific district).
- Fertility (ICRISAT reference yield) × Survival (rainfall/dry-spell regression, NDVI once GEE
  unblocks) → Production; Production × Price (Agmarknet) → Revenue vs. Loan → Coverage Ratio (§8).
- Each pathway validated against its own ground truth independently — no forced joint training.
- LLM narrative generation on top of the coverage-ratio breakdown.
- Streamlit dashboard demo with a map + one example farmer.

Explicitly out of scope for the hackathon: geospatial foundation model fine-tuning, mid-season
monitoring service, real bank pilot data, RAG compliance grounding.

## 13. Success Metrics

- **Fertility + Survival**: yield prediction error (R² / MAE of `reference_yield × survival_factor`
  vs. actual ICRISAT yield) on held-out out-of-time seasons.
- **Price**: fit of `expected_price` against Agmarknet's own historical harvest-time price
  distribution, independent of the yield-side metric above.
- **Coverage Ratio (sanity check, not the primary metric)**: do low-coverage-ratio seasons line up
  with ICRISAT's historically-flagged stress years? This is qualitative corroboration, not a
  calibrated default-probability claim — that requires §7C's partner loan data.
- **Explainability**: loan officer (or judge) can articulate *why* a given score was produced from
  the narrative alone, without reading raw SHAP values.
- **Demo completeness**: live score for at least one real geo-located farm plot with real pulled
  satellite/weather/price data (not mocked), end to end.

## 14. Open Questions

- Which specific district(s) have both good satellite coverage *and* usable PMFBY/ICRISAT proxy
  data — needs a short data-availability spike before locking the demo scope.
- Whether Earth Engine access/quota can be provisioned in time (needs a Google account
  registration step, non-instant).
- Real bank/NBFC partner for Phase 2 validation — not needed for the hackathon, but worth flagging
  in the pitch as the path from proxy-validated to actually-validated.
