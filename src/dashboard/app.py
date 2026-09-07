"""Streamlit demo UI — execution.md Phase 11.

Calls the FastAPI /score and /compare-crops endpoints; does not reimplement the pipeline
(skills.md §1). Run `uvicorn src.api.main:app --reload` first, then `streamlit run
src/dashboard/app.py`.
"""
import os

import pandas as pd
import requests
import streamlit as st

API_BASE_URL = os.environ.get("AGRIRISK_API_URL", "http://localhost:8080")

st.set_page_config(page_title="AgriRisk", layout="wide")
st.title("AgriRisk — Agricultural Lending Risk")
st.caption(
    "Decision-support score for a loan officer, not an automated approval — see the confidence "
    "and risk-tier notes below before acting on any number here."
)

TIER_COLOR = {"Low": "🟢", "Moderate": "🟡", "Elevated": "🟠", "High": "🔴"}


def _call_api(path: str, payload: dict) -> dict:
    try:
        resp = requests.post(f"{API_BASE_URL}{path}", json=payload, timeout=90)
    except requests.exceptions.RequestException as e:
        st.error(f"Couldn't reach the API at {API_BASE_URL}{path} — is uvicorn running? ({e})")
        st.stop()
    if resp.status_code != 200:
        st.error(f"API returned {resp.status_code}: {resp.text}")
        st.stop()
    return resp.json()


with st.sidebar:
    st.header("Application")
    with st.form("intake"):
        st.subheader("Plot")
        lat = st.number_input("Latitude", value=15.14, format="%.4f")
        lon = st.number_input("Longitude", value=76.92, format="%.4f")
        district = st.text_input("District", value="Bellary")
        area_ha = st.number_input("Area (ha)", value=2.0, min_value=0.1)

        st.subheader("Loan")
        crop = st.selectbox("Crop requested", ["Rice", "Groundnut", "Chickpea", "Pearl Millet"])
        season_year = st.number_input("Season year", value=2013, step=1)
        use_actual_sowing_date = st.checkbox(
            "Farmer reported an actual sowing date (overrides the generic June 15 default)"
        )
        actual_sowing_date = (
            st.date_input("Actual sowing date", value=None) if use_actual_sowing_date else None
        )
        loan_amount = st.number_input("Loan amount requested (INR)", value=80000.0, step=1000.0)
        loan_tenure_days = st.number_input("Loan tenure (days)", value=150, step=10)

        st.subheader("Ask the lendee (optional — all improve the score's accuracy, none are required)")
        irrigation_source = st.selectbox(
            "Irrigation source", [None, "canal", "borewell", "tank", "rainfed"],
            format_func=lambda v: "Unknown" if v is None else v.title(),
        )
        pmfby_enrolled = st.checkbox("PMFBY crop insurance enrolled")
        fpo_member = st.checkbox("FPO / SHG / JLG member")
        existing_annual_debt_service = st.number_input(
            "Existing annual debt service (INR, 0 = none/unknown)", value=0.0, step=1000.0
        )
        off_farm_annual_income = st.number_input(
            "Off-farm annual household income (INR, 0 = none/unknown)", value=0.0, step=1000.0
        )

        submitted = st.form_submit_button("Score this application", width='stretch')

if submitted:
    payload = {
        "lat": lat, "lon": lon, "district": district, "crop": crop,
        "loan_amount": loan_amount, "area_ha": area_ha, "season_year": int(season_year),
        "loan_tenure_days": loan_tenure_days,
        "actual_sowing_date": actual_sowing_date.isoformat() if actual_sowing_date else None,
        "irrigation_source": irrigation_source,
        "pmfby_enrolled": pmfby_enrolled or None,
        "fpo_member": fpo_member or None,
        "existing_annual_debt_service": existing_annual_debt_service or None,
        "off_farm_annual_income": off_farm_annual_income or None,
    }
    with st.spinner("Scoring — pulling live satellite, weather, and market data..."):
        st.session_state["score_result"] = _call_api("/score", payload)
        st.session_state["compare_result"] = _call_api("/compare-crops", payload)
    st.session_state["last_plot"] = (lat, lon)

if "score_result" not in st.session_state:
    st.info("Fill in the application in the sidebar and click **Score this application**.")
    st.stop()

result = st.session_state["score_result"]
compare = st.session_state["compare_result"]

tab_score, tab_compare, tab_narrative = st.tabs(
    ["Score", f"Compare crops for this land", "Underwriting narrative"]
)

with tab_score:
    land_use = result.get("land_use", {})
    if land_use.get("agricultural_plausible") is False:
        st.error(
            f"🚫 **Land-use check: NOT AGRICULTURAL LAND.** {land_use['caveat']} "
            f"This overrides the score below — treat as Decline regardless of the arithmetic."
        )

    col_map, col_metrics = st.columns([1, 1.4])

    with col_map:
        st.map(pd.DataFrame([{"lat": lat, "lon": lon}]), size=200, zoom=9)
        if land_use.get("agricultural_plausible") is True:
            st.caption(f"✓ Land-use check: classified '{land_use['class_label']}' — plausible farmland.")
        elif land_use.get("agricultural_plausible") is None and land_use.get("class_label"):
            st.caption(f"⚠️ Land-use check: classified '{land_use['class_label']}' — {land_use.get('caveat', '')}")
        geo = result.get("geo_sanity", {})
        if geo.get("plausible") is False:
            st.warning(
                f"⚠️ Geo-sanity check: plot is {geo['distance_km']} km from {district}'s "
                f"geocoded centroid — worth manual verification, not a settled fraud finding."
            )
        elif geo.get("plausible") is True:
            st.caption(f"✓ Geo-sanity check: {geo['distance_km']} km from {district}'s centroid — plausible.")
        else:
            st.caption("Geo-sanity check unavailable (geocoding lookup failed).")

    with col_metrics:
        tier = result.get("risk_tier", "?")
        st.markdown(f"### {TIER_COLOR.get(tier, '⚪')} {tier} risk — {result.get('recommended_action', '?')}")

        m1, m2, m3 = st.columns(3)
        m1.metric("Coverage ratio", f"{result['coverage_ratio']:.2f}x")
        m2.metric("Risk score", f"{result['risk_score']:.2f}")
        m3.metric("Production", f"{result['production_kg']:.0f} kg")

        if not result.get("scale_of_finance_compliant", True):
            st.error(
                f"NABARD scale-of-finance breach: loan amount INR {loan_amount:,.0f} exceeds the "
                f"recommended maximum of INR {result['scale_of_finance_max_loan']:,.0f} "
                f"(50% of expected revenue)."
            )
        if result.get("tenure_adequate") is False:
            st.warning(
                f"Tenure shortfall: {result['tenure_shortfall_days']:.0f} days short of the crop "
                f"cycle plus time-to-sale."
            )

        confidence = result.get("confidence", {})
        overall = confidence.get("overall", "unknown")
        conf_style = st.error if overall == "low" else (st.warning if overall == "moderate" else st.success)
        fert = confidence.get("fertility", {})
        surv = confidence.get("survival", {})
        conf_style(
            f"**Model confidence: {overall}.** Fertility fit cv_r²={fert.get('cv_r2', 'n/a')}, "
            f"Survival fit cv_r²={surv.get('cv_r2', surv.get('confidence', 'n/a'))} "
            f"({fert.get('validation_method', 'n/a')} across {fert.get('n_districts', 'n/a')} districts). "
            f"Treat this score as decision-support input, not a certain number."
        )

        mitigants = result.get("mitigants_applied", [])
        st.markdown("**Mitigants applied**" if mitigants else "**Mitigants applied:** none")
        if mitigants:
            st.dataframe(
                pd.DataFrame(mitigants).rename(
                    columns={"factor": "Factor", "adjustment": "Risk adjustment", "reason": "Why"}
                ),
                hide_index=True, width='stretch',
            )

with tab_compare:
    st.caption(
        f"Same plot and loan terms, scored for every crop this pipeline has data for — "
        f"'{crop}' is what was requested."
    )
    rows = []
    for c in compare["crops"]:
        rows.append({
            "Crop": ("→ " if c["requested"] else "") + c["crop"],
            "Risk tier": c.get("risk_tier", "—") if not c["error"] else "error",
            "Coverage ratio": f"{c['coverage_ratio']:.2f}x" if not c["error"] else "—",
            "Max recommended loan (INR)": c.get("scale_of_finance_max_loan") if not c["error"] else None,
            "Survival treatment": c.get("survival_mode", "—") if not c["error"] else c["error"],
        })
    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True, width='stretch')

    chart_df = df.dropna(subset=["Max recommended loan (INR)"]).set_index("Crop")
    if not chart_df.empty:
        st.bar_chart(chart_df["Max recommended loan (INR)"])

    st.caption(
        "Only Rice has a fitted this-season Survival model — every other crop shown is a "
        "normal-year baseline (Fertility × Price only), labeled under 'Survival treatment'. "
        "Sugarcane isn't shown: it has ICRISAT yield data but no Agmarknet mandi price "
        "(see configs/district.yaml)."
    )

with tab_narrative:
    if result.get("narrative"):
        st.markdown(result["narrative"])
    else:
        st.warning(f"Narrative unavailable: {result.get('narrative_error', 'unknown error')}")
