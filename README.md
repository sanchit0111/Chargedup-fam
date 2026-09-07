# AgriRisk

Multimodal agricultural lending risk prediction for Indian agri-lending banks/NBFCs. Three
independently-validated pathways — Fertility × Survival → Production; Production × Price →
Revenue; Revenue vs. Loan → Coverage Ratio → risk — combined at inference time, not one black-box
classifier (see `product_idea.md` §8 for why).

- Product shape & scope: `product_idea.md`
- Build order & Definitions of Done: `execution.md`
- Multi-agent collaboration contract: `.claude/agent_collab_playbook/SKILL.md`

## Setup

```
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in GEE_SERVICE_ACCOUNT_JSON, DATA_GOV_IN_API_KEY, and AWS credentials
                        # for the bedrock-role profile (default LLM_PROVIDER) — see .env.example
```

`configs/district.yaml` is already populated (Bellary, Karnataka / Rice — see
`data/raw/PHASE1_FINDINGS.md` for how that was verified, not guessed, including the crop pivot
from Groundnut to Rice and why Wheat was ruled out).

## Run

```
python -m src.ingestion.ground_truth   # builds data/processed/labels.csv from ICRISAT
python -m src.model.train_survival     # compares candidate feature sets via leave-one-district-out CV, writes models/survival_coefficients.json
python -m src.model.predict            # end-to-end score for a couple of known seasons/plots
uvicorn src.api.main:app --reload      # API
streamlit run src/dashboard/app.py     # dashboard (calls the API above)
pytest tests/                          # 60 real-data tests, all passing
python -m demo.run_scenarios           # runs + caches the 3 fixed demo scenarios below
```

## Demo scenarios (execution.md Phase 12)

Three fixed scenarios for the pitch — real inputs, real API calls, cached to `demo/cached/*.json`
so a network hiccup during judging doesn't sink the demo (the cache is a real prior run, not a
fabricated fallback). Re-run `python -m demo.run_scenarios` to refresh it. Full detail, including
each scenario's rationale, is in `demo/run_scenarios.py`.

| # | Scenario | Plot / season | Result |
|---|---|---|---|
| A | **Clean approval** — Rice, a real non-stress season (2013, `data/processed/labels.csv` label=0), irrigation + PMFBY context given | Bellary centroid, 2 ha, INR 50,000 loan, 200-day tenure | Low risk / Approve, 2.29x coverage, NABARD-compliant, tenure adequate |
| B | **High risk, decline-recommend** — Rice, a real documented stress season (2009, label=1: yield >5% below trailing trend, verified from ICRISAT, not picked to look bad), no farmer-context intake yet | Bellary centroid, 2 ha, INR 80,000 loan, 120-day tenure | High risk / Decline-recommend, ~1.1x coverage, NABARD breach, tenure inadequate |
| C | **Multi-crop counter-proposal** (the flagship differentiator) — farmer requests Chickpea; this land's Chickpea economics can't support the amount, but Rice and Groundnut on the *same plot* can | Bellary centroid, 2 ha, Chickpea requested, INR 80,000 loan | Requested crop (Chickpea): High risk / Decline-recommend, 0.37x coverage, NABARD breach. Compare-crops: Rice Low/INR 57k max loan, Groundnut Elevated/INR 33k |

Scenario B's exact coverage ratio has shown minor run-to-run variance (observed 1.06x-1.21x across
two live runs, tier stable at Elevated-to-High either way) — traced to data.gov.in's pagination/
rate-limit behavior under repeated rapid calls (the same live 429 error observed elsewhere in this
project's testing), a known, deferred data-reliability item, not a bug in the scoring logic itself.

## Status (current as of the last implementation pass)

| Phase | Status |
|---|---|
| 1 — district spike | done — Bellary/Karnataka pilot, verified live; crop pivoted Groundnut -> Rice; then pivoted from single-district to a pooled multi-district design (8 Karnataka districts, `training_district_pool`) so Fertility/Survival generalize to any lat/lon — see `data/raw/PHASE1_FINDINGS.md`'s district-agnostic-pivot addendum |
| 2 — ground truth | done — pooled ICRISAT panel across the 8-district pool (120-184 district-year rows depending on crop), both stress classes present |
| 3 — satellite (NDVI) | done — GEE credentials live, real MODIS data |
| 4 — weather | done — rainfall deficit, dry-spell days, heat-stress days, evapotranspiration anomaly, all live |
| 5 — market | done — price level + volatility, live-tested, cached |
| 6a — fertility | done — pooled panel fit (soil organic carbon + time trend), evaluable at any lat/lon; current `cv_r2 = -0.20` (leave-one-district-out) — see below |
| 6b — survival | done, but **honestly weak fit** — see below |
| 6c — price | done — Agmarknet trend + harvest-dip adjustment |
| 7 — combine | done — pure formula, hand-checked |
| 7b — mitigants | done — farmer-context factors (irrigation, PMFBY, FPO, debt service, off-farm income) as documented risk_score adjustments, plus a NABARD scale-of-finance compliance flag; see `src/model/mitigants.py` |
| 7c — risk tier / confidence / geo-sanity | done — documented risk-tier bands (not a yes/no auto-decision), an honest confidence disclosure sourced from the models' own `cv_r2`, and a geocoded plot-vs-district plausibility check; see `src/model/risk_tier.py`, `src/model/confidence.py`, `src/ingestion/geocoding.py` |
| 8 — predict (orchestrator) | done — full breakdown, live end-to-end, plot-specific |
| 9 — LLM narrative | done — live via AWS Bedrock (Claude Sonnet 5, `ap-south-1`, IAM role, no raw API key); Gemini Flash 2.5 and direct Anthropic kept as alternate providers behind the same interface (`LLM_PROVIDER` in `.env`) |
| 10 — API | done — live-tested end to end over real HTTP; `/compare-crops` added alongside `/score` |
| 10b — multi-crop comparison | done — `compare_crops()`/`/compare-crops`: same plot/loan terms scored across Rice, Groundnut, Chickpea, Pearl Millet (Sugarcane excluded, no Agmarknet price — see `configs/district.yaml`'s `crops` registry). Only Rice has a fitted Survival model; others get an explicitly labeled normal-year baseline |
| 11 — dashboard | done — `streamlit run src/dashboard/app.py`; score view + compare-crops view + narrative, verified via Streamlit's `AppTest` harness against the live API |
| 12 — validation/demo prep | done — PMFBY ground-truth spike: no usable district/crop-level dataset found after checking 3 real sources (see `data/raw/PMFBY_VALIDATION_FINDINGS.md`); ICRISAT remains the only working validation signal. 3 fixed demo scenarios picked, run live, and cached — see "Demo scenarios" above |

### Survival's fit, in full

**Current, deployed number** (what's actually loaded by `src/model/survival.py` right now):
`cv_r2 = -0.27` (single feature `ndvi_anomaly`, ridge, leave-one-district-out CV across the
8-district pool, 120 rows). Fertility's current `cv_r2 = -0.20`, same validation method. Both are
worse than the single-district numbers below, and that's expected, not a regression — see
`data/raw/PHASE1_FINDINGS.md`'s district-agnostic-pivot addendum for why leave-one-district-out is
a harder, more honest test than what produced the (now-historical) numbers just below. This is the
number the API actually reports via `src/model/confidence.py` ("low" confidence, both models) —
treat everything below as methodology history, not the current headline figure.

The modeling methodology (leave-one-out CV across many candidate feature sets, OLS vs. ridge,
growth-stage-aware features via `configs/district.yaml` `growth_stages` — crop-agnostic
`vegetative`/`critical_stage` keys, not hardcoded to one crop's biology) was developed against
Groundnut across three optimization rounds (feature expansion, ridge regularization, stage
splitting), each producing a small honest improvement that never crossed R²=0
(-0.14 -> -0.089 -> -0.072). Full history of that work is in git history / prior session context,
not repeated here to keep this section current rather than archival.

**After the crop pivot to Rice** (same methodology, one clean run — see PHASE1_FINDINGS.md for
why): best LOOCV R² = -0.090 (`heat_stress_days` alone, heavy ridge shrinkage), in the same honest
range as Groundnut's final result. The *absolute* error is much smaller (MAE~0.07 vs. Groundnut's
~0.29-0.32) because canal-irrigated Rice's yield_ratio target has far less variance to begin with
— but R² (which normalizes by that variance) told the same story: weather features didn't explain
Rice's remaining variance any better than the mean.

**Asymmetric excess-rain test**: user hypothesis — Indian irrigation is mostly canal/tubewell, so
deficit is buffered but *excess* rain (waterlogging, lodging) is actively destructive; a plain
signed `rainfall_deficit` forces one linear slope to fit both directions, which could wash out a
one-sided effect. Added `excess_rainfall_days`/`_critical_stage` (IMD rain-intensity thresholds)
and `max_daily_rainfall` as features isolating the excess side specifically. First attempt used
IMD's "heavy" threshold (64.5mm) — degenerate (near-always-zero at a single point over a 15-day
window, verified before trusting it), so the reported "winner" was actually just an intercept in
disguise (coefficient exactly 0.0), not evidence for or against the hypothesis. Lowered to IMD's
"moderate" threshold (15.6mm) for a fair retry: **best LOOCV R² improved to -0.022** — the closest
to zero of every candidate tried across both crops. But the coefficient came back **positive**
(+0.044): more moderate-rain-days during the critical stage predicts *higher* yield, not lower.
That's not confirmation of "excess rain destroys the crop" as stated — it reads more like "adequate
monsoon rainfall during the reproductive stage helps," a different (still sensible) finding. Genuine
destructive-flooding events (IMD "heavy"+) are too rare in this 25-year single-point sample to say
anything about at all — that part of the hypothesis is untested, not disproven.

Across all of this, one honest constant: the same structural causes apply regardless of crop or
feature (60-day cutoff withholding most of the season, thin/noisy trailing-mean labels, single-point
vs. district-aggregate mismatch), and irrigation buffering (canal/tubewell) is a real, now
twice-corroborated reason rainfed weather signals carry less information for this crop/district
than they did for rainfed Groundnut. Full comparison across 28 feature sets x 4 fit methods in
`models/survival_coefficients.json`'s `candidates_compared`.

The rest of the pipeline (Fertility's plot-level soil differentiation, Price's real Agmarknet
integration, the Coverage Ratio arithmetic) is solid and demonstrable end-to-end — Survival's
predictive quality is the one piece that's honestly weak, and should be framed that way in any
pitch rather than hidden. Per explicit scope direction, further Survival tuning is paused here;
next effort should go toward demo-readiness (Phases 9-12) rather than continued feature search.
