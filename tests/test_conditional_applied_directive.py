"""
Tests for RFORGE-34: conditional_applied prose directive must instruct affirmative language.

The current directive ("correctly apply the conditional") is too vague. The LLM reads
a branching policy, applies the relevant branch, but expresses it as negation:
"we don't offer monetary refunds for annual plans." The accuracy extractor sees this as
alignment=contradicted against the base policy, failing post-gen validation every time.

The fix makes the directive explicit:
  - State what the customer IS entitled to (affirmative language)
  - Include the qualifying condition as a scope marker (for constraint_preserved=True)
  - Explicitly warn against negating framing

Covers:
  - Directive instructs affirmative language ("IS entitled to")
  - Directive instructs including the qualifying condition
  - Directive warns against negating framing
  - Directive still embeds KB chunk text
  - Directive does NOT use the word "contradict" (would cue meta-commentary)
"""
from __future__ import annotations

import random
from datetime import datetime

from resonantforge.layer1.quality_plan_injector import QualityPlanInjector
from resonantforge.schemas import (
    ConstraintType,
    KBChunk,
    SimEvent,
    SimEventType,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


REFUND_CONDITIONAL_CHUNK = KBChunk(
    chunk_id="kb_refund_conditional",
    document_id="doc_refund",
    document_path="policies/refund.md",
    chunk_text="Refunds are available within 30 days for monthly plans. Annual plans receive account credits instead.",
    constraint_type=ConstraintType.ALLOW_CONDITION,
    domains=["billing"],
)

RATE_LIMIT_CONDITIONAL_CHUNK = KBChunk(
    chunk_id="kb_rate_limits_conditional",
    document_id="doc_api",
    document_path="policies/api.md",
    chunk_text="API rate limits differ by plan: Starter plans allow 100 requests/minute; Pro plans allow 1000 requests/minute.",
    constraint_type=ConstraintType.ALLOW_CONDITION,
    domains=["features"],
)


def _make_conv_event(event_id: str, domain: str = "billing") -> SimEvent:
    return SimEvent(
        event_id=event_id,
        event_type=SimEventType.CONVERSATION_STARTED,
        account_id="acc_001",
        timestamp=datetime(2024, 1, 15, 10, 0, 0),
        day_index=0,
        month_index=0,
        payload={
            "domain": domain,
            "intent": ["billing_inquiry"],
            "agent_id": "agent_001",
            "surface_channel": "chat",
        },
    )


def _conditional_applied_plans(plans):
    """Extract supported:conditional_applied plans that have a direct accuracy spec (not rerouted control or distractor)."""
    return [
        p for p in plans
        if (
            p.rubric_targets.accuracy is not None
            and p.rubric_targets.accuracy.status == "supported"
            and p.rubric_targets.accuracy.precision == "conditional_applied"
            and p.rubric_targets.empathy is None  # accuracy spec, not control plan
            and not p.is_distractor
        )
    ]


# ── Test 1: affirmative language instruction ──────────────────────────────────


def test_conditional_applied_directive_instructs_affirmative_language():
    """
    The directive must explicitly instruct the LLM to state the positive
    entitlement (what the customer IS entitled to) rather than what they
    are not entitled to.
    """
    rng = random.Random(42)
    injector = QualityPlanInjector(rng=rng)
    events = [_make_conv_event(f"evt_{i:03d}") for i in range(80)]
    chunks = [REFUND_CONDITIONAL_CHUNK]

    plans = injector.inject(events=events, snapshots=[], kb_chunks=chunks, planted_count=50)
    ca_plans = _conditional_applied_plans(plans)
    assert ca_plans, "Expected at least one conditional_applied accuracy plan"

    for plan in ca_plans:
        assert "IS entitled" in plan.prose_generation_directives, (
            f"Directive must instruct affirmative language ('IS entitled to').\n"
            f"Got: {plan.prose_generation_directives[:400]!r}"
        )


# ── Test 2: qualifier instruction ─────────────────────────────────────────────


def test_conditional_applied_directive_instructs_include_qualifier():
    """
    The directive must instruct the LLM to include the qualifying condition
    (plan type, eligibility window, etc.) as an explicit scope marker.
    This is required for constraint_preserved=True in post-gen validation.
    """
    rng = random.Random(42)
    injector = QualityPlanInjector(rng=rng)
    events = [_make_conv_event(f"evt_{i:03d}") for i in range(80)]
    chunks = [REFUND_CONDITIONAL_CHUNK]

    plans = injector.inject(events=events, snapshots=[], kb_chunks=chunks, planted_count=50)
    ca_plans = _conditional_applied_plans(plans)
    assert ca_plans, "Expected at least one conditional_applied accuracy plan"

    for plan in ca_plans:
        assert "qualifying condition" in plan.prose_generation_directives, (
            f"Directive must instruct including the qualifying condition as a scope marker.\n"
            f"Got: {plan.prose_generation_directives[:400]!r}"
        )


# ── Test 3: negating framing warning ─────────────────────────────────────────


def test_conditional_applied_directive_warns_against_negating_framing():
    """
    The directive must explicitly warn against negating language
    (\"we don't offer\", \"you cannot\", \"not available\").
    The LLM producing negating language is the root cause of
    alignment=contradicted failures in smoke-6.
    """
    rng = random.Random(42)
    injector = QualityPlanInjector(rng=rng)
    events = [_make_conv_event(f"evt_{i:03d}") for i in range(80)]
    chunks = [REFUND_CONDITIONAL_CHUNK]

    plans = injector.inject(events=events, snapshots=[], kb_chunks=chunks, planted_count=50)
    ca_plans = _conditional_applied_plans(plans)
    assert ca_plans, "Expected at least one conditional_applied accuracy plan"

    for plan in ca_plans:
        assert "negating" in plan.prose_generation_directives.lower(), (
            f"Directive must warn against negating framing.\n"
            f"Got: {plan.prose_generation_directives[:400]!r}"
        )


# ── Test 4: chunk text still embedded ────────────────────────────────────────


def test_conditional_applied_directive_still_embeds_chunk_text():
    """The directive must still embed KB chunk text for LLM grounding."""
    rng = random.Random(42)
    injector = QualityPlanInjector(rng=rng)
    events = [_make_conv_event(f"evt_{i:03d}") for i in range(80)]
    chunks = [REFUND_CONDITIONAL_CHUNK]

    plans = injector.inject(events=events, snapshots=[], kb_chunks=chunks, planted_count=50)
    ca_plans = _conditional_applied_plans(plans)
    assert ca_plans, "Expected at least one conditional_applied accuracy plan"

    for plan in ca_plans:
        assert REFUND_CONDITIONAL_CHUNK.chunk_text in plan.prose_generation_directives, (
            f"Directive must embed KB chunk text.\n"
            f"Got: {plan.prose_generation_directives[:400]!r}"
        )


# ── Test 5: no 'contradict' keyword ──────────────────────────────────────────


def test_conditional_applied_directive_does_not_contain_contradict():
    """
    The directive must NOT contain the word 'contradict'. Exposing the
    meta-label to the LLM causes meta-commentary (acknowledging both branches),
    setting constraint_preserved=False.
    """
    rng = random.Random(42)
    injector = QualityPlanInjector(rng=rng)
    events = [_make_conv_event(f"evt_{i:03d}") for i in range(80)]
    chunks = [REFUND_CONDITIONAL_CHUNK]

    plans = injector.inject(events=events, snapshots=[], kb_chunks=chunks, planted_count=50)
    ca_plans = _conditional_applied_plans(plans)
    assert ca_plans, "Expected at least one conditional_applied accuracy plan"

    for plan in ca_plans:
        assert "contradict" not in plan.prose_generation_directives.lower(), (
            f"Directive must not contain 'contradict'.\n"
            f"Got: {plan.prose_generation_directives[:400]!r}"
        )
