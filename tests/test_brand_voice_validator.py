"""
Tests for brand voice validator on-brand validation.

bv_baseline has been removed (RFORGE-9). Two well-separated poles remain:
bv_warm_exploratory and bv_direct_clinical.

This file retains the regression guard for bv_warm_exploratory on-brand validation
to ensure the brand voice validator handles the dominant variant correctly.
"""
from __future__ import annotations

from resonantforge.profiles.lexicons import saas as saas_lex
from resonantforge.validators.extractors.brand_voice import extract_brand_voice_signals
from resonantforge.validators.rule_engine import validate_brand_voice

# ── Shared lexicon fixtures ───────────────────────────────────────────────────

_FEATURE_PROFILES = saas_lex.BRAND_VOICE_FEATURE_PROFILES

# Warm-exploratory prose with ≥2 hedging terms, ≥1 question, and ≥2 warm terms —
# all within bv_warm_exploratory ranges.
_WARM_EXPLORATORY_PROSE = (
    "I'm wondering if that might be connected to what you described earlier. "
    "Let's explore a couple of possibilities together — it could be a configuration issue. "
    "I'd love to help figure this out. "
    "Could you tell me more about when this started happening? "
    "I appreciate you flagging this, and I'm glad we're looking into it together."
)

# Direct-clinical prose with directive terms, clinical vocabulary, and formal register.
# Includes ≥2 clinical_terms (confirm, verify, configure, validate) to satisfy aligned_vocabulary check.
_CLINICAL_PROSE = (
    "Navigate to Settings and select the Integrations panel. "
    "Enter the API key and confirm the value matches what was provisioned. "
    "Click Save to configure your webhook endpoint. "
    "Verify the connection status in the dashboard and validate that events are received."
)


def test_bv_warm_exploratory_on_brand_passes():
    """
    Warm-exploratory prose with hedging terms and questions must pass on_brand
    validation for bv_warm_exploratory.
    """
    signals = extract_brand_voice_signals(
        agent_prose=_WARM_EXPLORATORY_PROSE,
        hedging_lexicon=saas_lex.HEDGING_LEXICON,
        directive_lexicon=saas_lex.DIRECTIVE_LEXICON,
        warm_terms=saas_lex.WARM_TERMS,
        clinical_terms=saas_lex.CLINICAL_TERMS,
        contraction_patterns=saas_lex.CONTRACTION_PATTERNS,
    )

    verdict = validate_brand_voice(
        signals=signals,
        target="on_brand",
        variant_id="bv_warm_exploratory",
        feature_profiles=_FEATURE_PROFILES,
    )

    assert verdict.verdict.value == "pass", (
        f"Warm-exploratory prose must pass on_brand for bv_warm_exploratory. "
        f"feature_checks={verdict.signals_summary.get('feature_checks')}"
    )


def test_bv_direct_clinical_on_brand_passes():
    """
    Direct-clinical prose with directive terms and formal structure must pass on_brand
    validation for bv_direct_clinical.
    """
    signals = extract_brand_voice_signals(
        agent_prose=_CLINICAL_PROSE,
        hedging_lexicon=saas_lex.HEDGING_LEXICON,
        directive_lexicon=saas_lex.DIRECTIVE_LEXICON,
        warm_terms=saas_lex.WARM_TERMS,
        clinical_terms=saas_lex.CLINICAL_TERMS,
        contraction_patterns=saas_lex.CONTRACTION_PATTERNS,
    )

    verdict = validate_brand_voice(
        signals=signals,
        target="on_brand",
        variant_id="bv_direct_clinical",
        feature_profiles=_FEATURE_PROFILES,
    )

    assert verdict.verdict.value == "pass", (
        f"Clinical prose must pass on_brand for bv_direct_clinical. "
        f"feature_checks={verdict.signals_summary.get('feature_checks')}"
    )
