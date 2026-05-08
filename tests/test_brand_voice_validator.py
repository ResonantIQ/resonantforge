"""
Tests for RFORGE-42: bv_baseline directive_range minimum fires false positive
for non-instructional conversations.

bv_baseline directive_range was [1, 8], requiring at least 1 directive term.
The directive lexicon contains UI/instruction verbs (click, check, navigate,
use, etc.). Consultative conversations with no step-by-step instructions
naturally produce 0 directive terms — valid on-brand behaviour for bv_baseline,
which says "provide complete, actionable responses" not "always give step-by-step
instructions."

Fix: relax the minimum from 1 to 0 so the range becomes [0, 8].
"""
from __future__ import annotations

from resonantforge.profiles.lexicons import saas as saas_lex
from resonantforge.validators.extractors.brand_voice import extract_brand_voice_signals
from resonantforge.validators.rule_engine import validate_brand_voice

# ── Shared lexicon fixtures ───────────────────────────────────────────────────

_FEATURE_PROFILES = saas_lex.BRAND_VOICE_FEATURE_PROFILES

# Consultative agent prose with zero directive terms.
# Sentence length, question count, and hedging all within bv_baseline ranges.
# No UI-instruction verbs (click, check, navigate, use, go to, …).
_CONSULTATIVE_PROSE = (
    "Great question — I'm glad you're thinking ahead about your capacity. "
    "I'm looking at your account right now and your health score is sitting at 0.92 this month. "
    "You've got plenty of room under your current plan limits for those new campaigns. "
    "Doubling your volume is solid growth. "
    "What kind of scale are we talking about? "
    "If you can share your growth projections, I can make sure you won't run into constraints. "
    "I've got availability tomorrow morning between 10 and noon. "
    "What works best for your schedule?"
)

# Step-by-step instructional prose with a small number of directive terms (3)
# that stays within bv_baseline directive_range=[1, 8].
_INSTRUCTIONAL_PROSE = (
    "To fix this, navigate to Settings and select the Integrations panel. "
    "Once there, use the API key shown on screen to update your webhook configuration. "
    "The connection should clear within a few minutes."
)

# Warm-exploratory prose with ≥2 hedging terms, ≥1 question, and ≥2 warm terms —
# all within bv_warm_exploratory ranges. Used for the warm_exploratory regression guard.
_WARM_EXPLORATORY_PROSE = (
    "I'm wondering if that might be connected to what you described earlier. "
    "Let's explore a couple of possibilities together — it could be a configuration issue. "
    "I'd love to help figure this out. "
    "Could you tell me more about when this started happening? "
    "I appreciate you flagging this, and I'm glad we're looking into it together."
)


# ── RFORGE-42: consultative prose with 0 directive terms passes on_brand ──────


def test_bv_baseline_on_brand_passes_with_zero_directive_terms():
    """
    Consultative prose (no UI instructions) with directive_terms_count=0 must
    pass on_brand validation against bv_baseline after the range is relaxed to [0, 8].
    """
    signals = extract_brand_voice_signals(
        agent_prose=_CONSULTATIVE_PROSE,
        hedging_lexicon=saas_lex.HEDGING_LEXICON,
        directive_lexicon=saas_lex.DIRECTIVE_LEXICON,
        warm_terms=saas_lex.WARM_TERMS,
        clinical_terms=saas_lex.CLINICAL_TERMS,
        contraction_patterns=saas_lex.CONTRACTION_PATTERNS,
    )

    assert signals.directive_terms_count == 0, (
        f"Expected 0 directive terms in consultative prose, got {signals.directive_terms_count}: "
        f"{signals.directive_terms}"
    )

    verdict = validate_brand_voice(
        signals=signals,
        target="on_brand",
        variant_id="bv_baseline",
        feature_profiles=_FEATURE_PROFILES,
    )

    assert verdict.verdict.value == "pass", (
        f"Consultative prose with 0 directive terms must pass on_brand for bv_baseline "
        f"after relaxing directive_range minimum to 0. "
        f"feature_checks={verdict.signals_summary.get('feature_checks')}"
    )


def test_bv_baseline_on_brand_still_passes_with_directive_terms():
    """
    Regression guard: instructional prose with directive terms must still pass
    on_brand validation after the range change.
    """
    signals = extract_brand_voice_signals(
        agent_prose=_INSTRUCTIONAL_PROSE,
        hedging_lexicon=saas_lex.HEDGING_LEXICON,
        directive_lexicon=saas_lex.DIRECTIVE_LEXICON,
        warm_terms=saas_lex.WARM_TERMS,
        clinical_terms=saas_lex.CLINICAL_TERMS,
        contraction_patterns=saas_lex.CONTRACTION_PATTERNS,
    )

    assert signals.directive_terms_count >= 1, (
        f"Instructional prose should have directive terms, got {signals.directive_terms_count}"
    )

    verdict = validate_brand_voice(
        signals=signals,
        target="on_brand",
        variant_id="bv_baseline",
        feature_profiles=_FEATURE_PROFILES,
    )

    assert verdict.verdict.value == "pass", (
        f"Instructional prose with directive terms must still pass on_brand for bv_baseline. "
        f"feature_checks={verdict.signals_summary.get('feature_checks')}"
    )


def test_bv_warm_exploratory_directive_range_unchanged():
    """
    Regression guard: bv_warm_exploratory directive_range [0, 3] is unaffected.
    Warm-exploratory prose with hedging terms must still pass on_brand.
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
        f"bv_warm_exploratory should not be affected by the bv_baseline fix. "
        f"feature_checks={verdict.signals_summary.get('feature_checks')}"
    )
