"""
Tests for RFORGE-6: guided resolution target, already_resolved signal,
pronoun-reference fallback, escalation/ownership lexicon extensions.

Spec: docs/superpowers/specs/2026-05-08-rforge-6-guided-resolution-target-design.md
"""
from __future__ import annotations

from resonantforge.profiles.lexicons import saas as saas_lex
from resonantforge.validators.extractors.resolution import extract_resolution_signals


def _extract(agent_prose: str, customer_prose: str = "", completion_verb_patterns=None, completion_resolution_patterns=None) -> object:
    return extract_resolution_signals(
        agent_prose=agent_prose,
        customer_prose=customer_prose,
        resolution_patterns=saas_lex.RESOLUTION_PATTERNS,
        deflection_patterns=saas_lex.DEFLECTION_PATTERNS,
        next_steps_patterns=saas_lex.NEXT_STEPS_PATTERNS,
        temporal_anchor_patterns=saas_lex.TEMPORAL_ANCHOR_PATTERNS,
        specific_actor_patterns=saas_lex.SPECIFIC_ACTOR_PATTERNS,
        ownership_patterns=saas_lex.OWNERSHIP_PATTERNS,
        issue_keywords=saas_lex.ISSUE_KEYWORDS,
        completion_verb_patterns=completion_verb_patterns if completion_verb_patterns is not None else saas_lex.COMPLETION_VERB_PATTERNS,
        completion_resolution_patterns=completion_resolution_patterns if completion_resolution_patterns is not None else saas_lex.COMPLETION_RESOLUTION_PATTERNS,
    )


# ── Task 1: Lexicon constants exist ──────────────────────────────────────────

def test_completion_verb_patterns_exists():
    assert isinstance(saas_lex.COMPLETION_VERB_PATTERNS, list)
    assert len(saas_lex.COMPLETION_VERB_PATTERNS) > 0


def test_completion_resolution_patterns_exists():
    assert isinstance(saas_lex.COMPLETION_RESOLUTION_PATTERNS, list)
    assert len(saas_lex.COMPLETION_RESOLUTION_PATTERNS) > 0


def test_completion_resolution_patterns_is_subset_of_resolution_patterns():
    """Every completion pattern must also be in the full RESOLUTION_PATTERNS."""
    for p in saas_lex.COMPLETION_RESOLUTION_PATTERNS:
        assert p in saas_lex.RESOLUTION_PATTERNS, (
            f"Pattern {p!r} is in COMPLETION_RESOLUTION_PATTERNS "
            f"but not in RESOLUTION_PATTERNS"
        )


def test_investigative_verbs_not_in_completion_resolution_patterns():
    """
    Investigative I'll-verbs must be in RESOLUTION_PATTERNS but NOT in
    COMPLETION_RESOLUTION_PATTERNS — they set solution_provided=True but
    must not trigger the pronoun-reference fallback to complete.
    """
    investigative = ["look into", "dig into", "dig in", "investigate"]
    for verb in investigative:
        in_completion = any(
            verb in p for p in saas_lex.COMPLETION_RESOLUTION_PATTERNS
        )
        assert not in_completion, (
            f"Investigative verb {verb!r} must not appear in "
            f"COMPLETION_RESOLUTION_PATTERNS"
        )


def test_completion_verb_alt_pattern_matches_completion_phrases():
    """The dynamically-built i've + verb pattern must match completion phrases."""
    import re
    from resonantforge.profiles.lexicons.saas import _COMPLETION_VERB_ALT
    pattern = rf"\bi'?ve\s+(?:{_COMPLETION_VERB_ALT})\b"
    assert re.search(pattern, "i've fixed it", re.IGNORECASE)
    assert re.search(pattern, "I've updated the record", re.IGNORECASE)
    assert re.search(pattern, "I've resolved your issue", re.IGNORECASE)
    assert not re.search(pattern, "i'll look into it", re.IGNORECASE)
    assert not re.search(pattern, "i've been looking at this", re.IGNORECASE)


# ── Task 2: ResolutionSignals already_resolved field ─────────────────────────


def test_resolution_signals_already_resolved_defaults_false():
    """ResolutionSignals must have already_resolved field defaulting to False."""
    from resonantforge.schemas import ResolutionSignals

    signals = ResolutionSignals(
        solution_provided=True,
        solution_type="complete",
        next_steps_present=False,
        next_steps_actionable=False,
        ownership_language_present=False,
        ownership_phrases=[],
        deflection_present=False,
        resolution_blocked=False,
    )
    assert signals.already_resolved is False


def test_resolution_signals_already_resolved_can_be_set_true():
    """ResolutionSignals.already_resolved can be set to True explicitly."""
    from resonantforge.schemas import ResolutionSignals

    signals = ResolutionSignals(
        solution_provided=True,
        solution_type="complete",
        next_steps_present=False,
        next_steps_actionable=False,
        ownership_language_present=False,
        ownership_phrases=[],
        deflection_present=False,
        resolution_blocked=False,
        already_resolved=True,
    )
    assert signals.already_resolved is True


# ── Task 3: SaaSProfile accessor methods ─────────────────────────────────────


def test_saas_profile_completion_verb_patterns_returns_list():
    """SaaSProfile.completion_verb_patterns() must return a non-empty list of strings."""
    from resonantforge.profiles.saas import SaaSProfile
    from resonantforge.profiles.lexicons import saas as saas_lex

    profile = SaaSProfile()
    result = profile.completion_verb_patterns()
    assert isinstance(result, list)
    assert len(result) > 0
    assert result == saas_lex.COMPLETION_VERB_PATTERNS


def test_saas_profile_completion_resolution_patterns_returns_list():
    """SaaSProfile.completion_resolution_patterns() must return a non-empty list of strings."""
    from resonantforge.profiles.saas import SaaSProfile
    from resonantforge.profiles.lexicons import saas as saas_lex

    profile = SaaSProfile()
    result = profile.completion_resolution_patterns()
    assert isinstance(result, list)
    assert len(result) > 0
    assert result == saas_lex.COMPLETION_RESOLUTION_PATTERNS


# ── Task 4: _already_resolved signal ────────────────────────────────────────


def test_already_resolved_true_for_ive_fixed():
    """'I've fixed the webhook config' must set already_resolved=True."""
    signals = _extract(
        agent_prose="I've fixed the webhook config. You can retry now.",
        customer_prose="I'm getting a 401 error on my webhook integration.",
        completion_verb_patterns=saas_lex.COMPLETION_VERB_PATTERNS,
        completion_resolution_patterns=saas_lex.COMPLETION_RESOLUTION_PATTERNS,
    )
    assert signals.already_resolved is True, (
        f"'I've fixed' must set already_resolved=True. Got {signals.already_resolved}"
    )


def test_already_resolved_true_for_ive_updated():
    """'I've updated your settings' must set already_resolved=True (non-fixed verb)."""
    signals = _extract(
        agent_prose="I've updated your account settings. The change is live now.",
        customer_prose="",
        completion_verb_patterns=saas_lex.COMPLETION_VERB_PATTERNS,
        completion_resolution_patterns=saas_lex.COMPLETION_RESOLUTION_PATTERNS,
    )
    assert signals.already_resolved is True, (
        f"'I've updated' must set already_resolved=True. Got {signals.already_resolved}"
    )


def test_already_resolved_false_for_ive_been_looking():
    """'I've been looking at this' must NOT set already_resolved=True."""
    signals = _extract(
        agent_prose="I've been looking at this issue for a while now.",
        customer_prose="",
        completion_verb_patterns=saas_lex.COMPLETION_VERB_PATTERNS,
        completion_resolution_patterns=saas_lex.COMPLETION_RESOLUTION_PATTERNS,
    )
    assert signals.already_resolved is False, (
        f"'I've been looking at' must NOT set already_resolved=True. Got {signals.already_resolved}"
    )


def test_already_resolved_false_for_passive_voice():
    """'The issue was fixed' does NOT match already_resolved (passive voice, known gap)."""
    signals = _extract(
        agent_prose="The issue was fixed on our end. Please retry.",
        customer_prose="",
        completion_verb_patterns=saas_lex.COMPLETION_VERB_PATTERNS,
        completion_resolution_patterns=saas_lex.COMPLETION_RESOLUTION_PATTERNS,
    )
    assert signals.already_resolved is False, (
        "Passive voice 'The issue was fixed' must NOT set already_resolved=True "
        "(known gap — first-person past-tense only)."
    )


# ── Task 4: pronoun-reference fallback in _classify_solution_type ────────────


def test_escalation_classifies_as_complete():
    """'I'm escalating this to engineering' must classify as solution_type='complete'."""
    signals = _extract(
        agent_prose="I'm escalating this to our engineering team right away.",
        customer_prose="I'm having trouble with my webhook integration.",
        completion_verb_patterns=saas_lex.COMPLETION_VERB_PATTERNS,
        completion_resolution_patterns=saas_lex.COMPLETION_RESOLUTION_PATTERNS,
    )
    assert signals.solution_type == "complete", (
        f"Escalation with pronoun proxy must classify as 'complete'. Got '{signals.solution_type}'"
    )


def test_investigative_plus_pronoun_stays_partial():
    """'I'll look into this' must stay solution_type='partial' (investigative verb, not in completion set)."""
    signals = _extract(
        agent_prose="I'll look into this for you and get back to you.",
        customer_prose="I'm having trouble with my webhook integration.",
        completion_verb_patterns=saas_lex.COMPLETION_VERB_PATTERNS,
        completion_resolution_patterns=saas_lex.COMPLETION_RESOLUTION_PATTERNS,
    )
    assert signals.solution_type == "partial", (
        f"Investigative I'll verb with pronoun must stay 'partial'. Got '{signals.solution_type}'"
    )


def test_refusal_stays_none_with_pronoun_fallback():
    """'I can't help with this' must stay solution_type='none' (no resolution pattern)."""
    signals = _extract(
        agent_prose="I can't help with this. It's outside my scope.",
        customer_prose="I'm having trouble with my integration.",
        completion_verb_patterns=saas_lex.COMPLETION_VERB_PATTERNS,
        completion_resolution_patterns=saas_lex.COMPLETION_RESOLUTION_PATTERNS,
    )
    assert signals.solution_type == "none", (
        f"Refusal with pronoun must stay 'none'. Got '{signals.solution_type}'"
    )


# ── Task 5: validate_resolution — guided target ──────────────────────────────

from resonantforge.schemas import ResolutionSignals
from resonantforge.validators.rule_engine import validate_resolution


def _make_signals(**kwargs) -> ResolutionSignals:
    """Build a ResolutionSignals with sensible defaults for testing."""
    defaults = dict(
        solution_provided=False,
        solution_type="none",
        next_steps_present=False,
        next_steps_actionable=False,
        ownership_language_present=False,
        ownership_phrases=[],
        deflection_present=False,
        resolution_blocked=False,
        already_resolved=False,
    )
    defaults.update(kwargs)
    return ResolutionSignals(**defaults)


def test_guided_passes_complete_walkthrough():
    """
    Complete solution with no ownership/temporal anchor passes guided.
    Agent gives step-by-step instructions; customer drives execution.
    """
    signals = _make_signals(
        solution_provided=True,
        solution_type="complete",
        deflection_present=False,
    )
    verdict = validate_resolution(signals=signals, target="guided")
    assert verdict.verdict.value == "pass", (
        f"Complete walkthrough must pass 'guided'. Got verdict={verdict.verdict.value}"
    )


def test_guided_fails_when_deflection_present():
    """'guided' fails when deflection_present=True, even with a complete solution."""
    signals = _make_signals(
        solution_provided=True,
        solution_type="complete",
        deflection_present=True,
    )
    verdict = validate_resolution(signals=signals, target="guided")
    assert verdict.verdict.value == "fail", (
        f"Guided with deflection must fail. Got verdict={verdict.verdict.value}"
    )


def test_guided_fails_without_complete_solution():
    """'guided' fails when solution_type is not 'complete'."""
    signals = _make_signals(
        solution_provided=True,
        solution_type="partial",
        deflection_present=False,
    )
    verdict = validate_resolution(signals=signals, target="guided")
    assert verdict.verdict.value == "fail", (
        f"Guided with partial solution must fail. Got verdict={verdict.verdict.value}"
    )


def test_guided_resolution_prefix_accepted():
    """'resolution:guided' prefix form is accepted."""
    signals = _make_signals(
        solution_provided=True,
        solution_type="complete",
        deflection_present=False,
    )
    verdict = validate_resolution(signals=signals, target="resolution:guided")
    assert verdict.verdict.value == "pass", (
        f"'resolution:guided' prefix form must pass. Got verdict={verdict.verdict.value}"
    )


# ── Task 5: validate_resolution — strong with already_resolved ───────────────


def test_strong_passes_with_already_resolved():
    """
    Fait accompli: already_resolved=True must allow strong to pass without next_steps_actionable.
    """
    signals = _make_signals(
        solution_provided=True,
        solution_type="complete",
        next_steps_actionable=False,  # no temporal anchor
        already_resolved=True,        # fait accompli
        ownership_language_present=True,
        deflection_present=False,
    )
    verdict = validate_resolution(signals=signals, target="strong")
    assert verdict.verdict.value == "pass", (
        f"Fait accompli must pass 'strong'. Got verdict={verdict.verdict.value}"
    )


def test_strong_still_requires_ownership_even_with_already_resolved():
    """already_resolved alone does not satisfy 'strong' — ownership is still required."""
    signals = _make_signals(
        solution_provided=True,
        solution_type="complete",
        next_steps_actionable=False,
        already_resolved=True,
        ownership_language_present=False,  # missing
        deflection_present=False,
    )
    verdict = validate_resolution(signals=signals, target="strong")
    assert verdict.verdict.value == "fail", (
        f"Strong without ownership must fail even with already_resolved. Got verdict={verdict.verdict.value}"
    )


def test_strong_passes_with_next_steps_actionable_regression():
    """Regression: original strong path (next_steps_actionable) still works."""
    signals = _make_signals(
        solution_provided=True,
        solution_type="complete",
        next_steps_actionable=True,   # standard path
        already_resolved=False,
        ownership_language_present=True,
        deflection_present=False,
    )
    verdict = validate_resolution(signals=signals, target="strong")
    assert verdict.verdict.value == "pass", (
        f"Standard strong path must still pass. Got verdict={verdict.verdict.value}"
    )


def test_signals_summary_includes_already_resolved():
    """signals_summary dict must include 'already_resolved' key."""
    signals = _make_signals(
        solution_provided=True,
        solution_type="complete",
        deflection_present=False,
    )
    verdict = validate_resolution(signals=signals, target="guided")
    assert "already_resolved" in verdict.signals_summary, (
        f"signals_summary must include 'already_resolved'. Got keys: {list(verdict.signals_summary.keys())}"
    )


# ── Task 5: conv_evt_00313 regression guard ───────────────────────────────────


def test_conv_evt_00313_passes_guided():
    """
    Canonical guided conversation: complete walkthrough, no ownership/temporal.
    Agent provides a 3-step onboarding walkthrough; customer executes.
    """
    agent_prose = (
        "Here's how to get started. First, go to Settings and click on Integrations. "
        "Select the CRM connector and copy the API key shown there. "
        "Paste it into your CRM's webhook configuration field and save. "
        "The connection should be active within a few minutes."
    )
    customer_prose = "I need help setting up the integration with my CRM."

    signals = extract_resolution_signals(
        agent_prose=agent_prose,
        customer_prose=customer_prose,
        resolution_patterns=saas_lex.RESOLUTION_PATTERNS,
        deflection_patterns=saas_lex.DEFLECTION_PATTERNS,
        next_steps_patterns=saas_lex.NEXT_STEPS_PATTERNS,
        temporal_anchor_patterns=saas_lex.TEMPORAL_ANCHOR_PATTERNS,
        specific_actor_patterns=saas_lex.SPECIFIC_ACTOR_PATTERNS,
        ownership_patterns=saas_lex.OWNERSHIP_PATTERNS,
        issue_keywords=saas_lex.ISSUE_KEYWORDS,
        completion_verb_patterns=saas_lex.COMPLETION_VERB_PATTERNS,
        completion_resolution_patterns=saas_lex.COMPLETION_RESOLUTION_PATTERNS,
    )

    verdict = validate_resolution(signals=signals, target="guided")
    assert verdict.verdict.value == "pass", (
        f"conv_evt_00313 walkthrough must pass 'guided'. "
        f"solution_provided={signals.solution_provided}, "
        f"solution_type={signals.solution_type}, "
        f"deflection_present={signals.deflection_present}, "
        f"verdict={verdict.verdict.value}"
    )


def test_unknown_resolution_target_returns_skip():
    """Unknown resolution target returns SKIP (regression guard)."""
    signals = _make_signals()
    verdict = validate_resolution(signals=signals, target="resolution:unknown_target")
    assert verdict.verdict.value == "skip", (
        f"Unknown target must return SKIP. Got verdict={verdict.verdict.value}"
    )
