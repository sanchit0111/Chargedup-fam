"""LLM narrative layer — execution.md Phase 9. Contract frozen in SKILL.md §2.

Provider is swappable via LLM_PROVIDER (.env): "bedrock" (default) calls Claude through AWS
Bedrock using an assumed IAM role (AWS_PROFILE) in ap-south-1 — no raw API key to manage, and the
Mumbai region matters for the data-residency question an Indian bank will actually ask. "gemini"
and "anthropic" (direct API) are kept behind the same interface as alternates. Live-tested against
Bedrock 2026-09-07 — see git history for the verification.

The LLM explains/contextualizes predict.score()'s breakdown; it never computes the numbers itself
(product_idea.md §9 guardrail) — every figure in the prompt comes from Fertility/Survival/Price/
Combine, so the narrative can state the causal chain directly instead of a SHAP-style attribution.
"""
import os

LLM_PROVIDER = os.environ.get("LLM_PROVIDER", "bedrock")
BEDROCK_MODEL_ID = os.environ.get("BEDROCK_MODEL_ID", "global.anthropic.claude-sonnet-5")
BEDROCK_AWS_PROFILE = os.environ.get("AWS_PROFILE", "bedrock-role")
BEDROCK_AWS_REGION = os.environ.get("AWS_REGION", "ap-south-1")
GEMINI_MODEL_ID = os.environ.get("GEMINI_MODEL_ID", "gemini-2.5-flash")
ANTHROPIC_DIRECT_MODEL = "claude-sonnet-5"

PROMPT_TEMPLATE = """You are writing a short underwriting risk memo for a loan officer at an Indian \
agricultural bank. Use ONLY the numbers given below — do not invent or estimate anything. State \
the causal chain explicitly: reference yield -> this season's survival factor -> production -> \
expected price -> revenue -> coverage ratio. Then address loan tenure adequacy separately from \
coverage — a short tenure is a cash-flow-timing risk even when coverage is comfortable. Mention \
any mitigants applied (listed below) and why each moved the risk score. Flag clearly if the loan \
amount exceeds the NABARD scale-of-finance guideline. State the recommended action (risk tier) \
plainly. If model confidence is "low", say so explicitly and note this is a decision-support \
score for a human loan officer, not an automated approval — do not let a low-risk-score number \
read as more certain than the confidence line supports. If the geo-sanity check is implausible, \
flag it as worth manual verification, not a settled fraud finding. If the land-use check says \
this location is not agricultural, lead with that — it overrides every other number in this memo, \
state it first and plainly, and do not soften it by discussing coverage ratio as if it still \
matters. End with one concrete, specific recommended mitigant if the final risk score is elevated \
or tenure is inadequate, or a one-line confirmation if the picture is clean.

District: {district}, Crop: {crop}, Season: {season_year}
Loan amount requested: INR {loan_amount:,.0f}
Farm area: {area_ha} ha

Reference yield (normal-year baseline): {reference_yield:.0f} kg/ha
This season's survival factor: {survival_factor:.2f} (1.0 = normal season)
Estimated production: {production_kg:.0f} kg
Expected price at harvest: INR {expected_price:.0f}/quintal
Estimated revenue: INR {revenue:,.0f}
Coverage ratio (revenue / loan amount): {coverage_ratio:.2f}x
Loan tenure adequate for crop cycle + time-to-sale: {tenure_adequate} (shortfall: {tenure_shortfall_days:.0f} days)
Final risk score after mitigants (0=lowest, 1=highest): {risk_score:.2f}
Risk tier / recommended action: {risk_tier} / {recommended_action}
Mitigants applied: {mitigants_summary}
NABARD scale-of-finance compliant (loan <= 50% of expected revenue): {scale_of_finance_compliant} (max recommended loan: INR {scale_of_finance_max_loan})
Model confidence in the production estimate: {confidence_summary}
Geo-sanity check (does the plot's location match the stated district?): {geo_sanity_summary}
Land-use check (is this exact point actually agricultural land?): {land_use_summary}

Write 5-7 sentences. Be specific with the numbers above, not generic."""


def _format_mitigants(mitigants_applied: list[dict]) -> str:
    if not mitigants_applied:
        return "none"
    return "; ".join(f"{m['factor']} ({m['adjustment']:+.2f}: {m['reason']})" for m in mitigants_applied)


def _format_land_use(land_use: dict) -> str:
    if not land_use or land_use.get("agricultural_plausible") is None:
        return f"unchecked or inconclusive ({land_use.get('caveat', 'unavailable')})" if land_use else "unchecked"
    if land_use["agricultural_plausible"]:
        return f"plausible (classified '{land_use['class_label']}')"
    return f"NOT AGRICULTURAL — {land_use['caveat']}"


def _format_confidence(confidence: dict) -> str:
    if not confidence:
        return "unknown"
    f, s = confidence.get("fertility", {}), confidence.get("survival", {})
    return (
        f"{confidence.get('overall', 'unknown')} overall "
        f"(fertility fit cv_r2={f.get('cv_r2', 'n/a')}, survival fit cv_r2={s.get('cv_r2', 'n/a')}, "
        f"both via {f.get('validation_method', 'n/a')} across {f.get('n_districts', 'n/a')} districts)"
    )


def _format_geo_sanity(geo_sanity: dict) -> str:
    if not geo_sanity or geo_sanity.get("plausible") is None:
        return "unchecked (geocoding unavailable)"
    if geo_sanity["plausible"]:
        return f"plausible (plot is {geo_sanity['distance_km']} km from the district centroid)"
    return f"IMPLAUSIBLE — plot is {geo_sanity['distance_km']} km from the stated district's centroid"


class NarrativeUnavailableError(Exception):
    """Raised when the configured LLM_PROVIDER has no usable credentials, or the call fails."""


def _extract_text(content_blocks) -> str:
    """response.content[0] isn't reliably the text block — a model with extended thinking
    enabled returns a ThinkingBlock first, then the TextBlock. Take the first block that
    actually has text rather than assuming position.
    """
    for block in content_blocks:
        text = getattr(block, "text", None)
        if text:
            return text
    raise NarrativeUnavailableError(f"no text block in response content: {content_blocks!r}")


def _call_bedrock(prompt: str) -> str:
    from anthropic import AnthropicBedrock

    try:
        client = AnthropicBedrock(aws_profile=BEDROCK_AWS_PROFILE, aws_region=BEDROCK_AWS_REGION)
        response = client.messages.create(
            model=BEDROCK_MODEL_ID,
            max_tokens=800,
            messages=[{"role": "user", "content": prompt}],
        )
        return _extract_text(response.content)
    except NarrativeUnavailableError:
        raise
    except Exception as e:
        raise NarrativeUnavailableError(
            f"Bedrock call failed (profile={BEDROCK_AWS_PROFILE!r}, region={BEDROCK_AWS_REGION!r}, "
            f"model={BEDROCK_MODEL_ID!r}): {e}"
        ) from e


def _call_gemini(prompt: str) -> str:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise NarrativeUnavailableError("GEMINI_API_KEY not set — see .env.example")
    try:
        import google.generativeai as genai

        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(GEMINI_MODEL_ID)
        response = model.generate_content(prompt)
        return response.text
    except NarrativeUnavailableError:
        raise
    except Exception as e:
        raise NarrativeUnavailableError(f"Gemini call failed (model={GEMINI_MODEL_ID!r}): {e}") from e


def _call_anthropic_direct(prompt: str) -> str:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise NarrativeUnavailableError("ANTHROPIC_API_KEY not set — see .env.example")
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model=ANTHROPIC_DIRECT_MODEL,
        max_tokens=800,
        messages=[{"role": "user", "content": prompt}],
    )
    return _extract_text(response.content)


_PROVIDERS = {
    "bedrock": _call_bedrock,
    "gemini": _call_gemini,
    "anthropic": _call_anthropic_direct,
}


def generate_narrative(score_result: dict, loan: dict) -> str:
    call = _PROVIDERS.get(LLM_PROVIDER)
    if call is None:
        raise NarrativeUnavailableError(
            f"unknown LLM_PROVIDER={LLM_PROVIDER!r} — expected one of {sorted(_PROVIDERS)}"
        )

    mitigants_summary = _format_mitigants(score_result.get("mitigants_applied", []))
    confidence_summary = _format_confidence(score_result.get("confidence", {}))
    geo_sanity_summary = _format_geo_sanity(score_result.get("geo_sanity", {}))
    land_use_summary = _format_land_use(score_result.get("land_use", {}))
    prompt = PROMPT_TEMPLATE.format(
        **loan, **score_result, mitigants_summary=mitigants_summary,
        confidence_summary=confidence_summary, geo_sanity_summary=geo_sanity_summary,
        land_use_summary=land_use_summary,
    )
    return call(prompt)


if __name__ == "__main__":
    from src.config import load_config, scoring_cutoff_date
    from src.model.predict import score

    config = load_config()
    lat, lon = config["centroid"]["lat"], config["centroid"]["lon"]
    district = config["default_district"]
    year = 2013
    cutoff = scoring_cutoff_date(year, config)
    result = score(
        lat, lon, district, config["crop"], year, cutoff,
        area_ha=2.0, loan_amount=80000.0, loan_tenure_days=150,
    )
    loan = {
        "district": district, "crop": config["crop"], "season_year": year,
        "loan_amount": 80000.0, "area_ha": 2.0,
    }
    print(generate_narrative(result, loan))
