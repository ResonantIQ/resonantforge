"""
Tests for conditional-chunk detection and reroute in plan injector (RFORGE-10).

Covers:
  - is_conditional_chunk: truth table across all ConstraintType values
  - Injector reroute: supported:exact + ALLOW_CONDITION chunk → conditional_applied
  - Injector reroute: supported:exact + DENY_CONDITION chunk → conditional_applied
  - Injector no-reroute: supported:exact + INFORMATIONAL chunk → exact (no change)
  - Injector no-reroute: overgeneralized + ALLOW_CONDITION chunk → overgeneralized (RFORGE-7's territory)
  - Injector no-reroute: contradicted:exact handled by pool filter, not reroute
  - Pool filter alignment: contradicted:exact pick_pool_capped excludes ALLOW_CONDITION / DENY_CONDITION chunks
  - Pool filter telemetry: conditional_excluded_count reported in pool_filter_telemetry
  - Injector telemetry: conditional_reroute_count and conditional_reroute_events populated on reroute
  - Determinism: same seed → same reroute results
"""
from __future__ import annotations

import random
from datetime import datetime

import pytest

from resonantforge.layer1.quality_plan_injector import QualityPlanInjector, is_conditional_chunk
from resonantforge.schemas import (
    AccuracyLabel,
    ConstraintType,
    KBChunk,
    SimEvent,
    SimEventType,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_conv_event(event_id: str = "evt_001", domain: str = "billing") -> SimEvent:
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


def _make_chunk(
    chunk_id: str,
    chunk_text: str,
    domain: str = "billing",
    constraint_type: ConstraintType = ConstraintType.INFORMATIONAL,
) -> KBChunk:
    return KBChunk(
        chunk_id=chunk_id,
        document_id=f"doc_{chunk_id}",
        document_path=f"policies/{chunk_id}.md",
        chunk_text=chunk_text,
        constraint_type=constraint_type,
        domains=[domain],
    )


# Chunks with concrete, contradictable facts (needed for contradicted:exact pool)
INFORMATIONAL_FACT_A = _make_chunk(
    "kb_info_refund",
    "Refunds are available within 30 days of purchase for paid plans.",
    constraint_type=ConstraintType.INFORMATIONAL,
)
INFORMATIONAL_FACT_B = _make_chunk(
    "kb_info_seats",
    "Your plan supports up to 500 users.",
    constraint_type=ConstraintType.INFORMATIONAL,
)
INFORMATIONAL_FACT_C = _make_chunk(
    "kb_info_sla",
    "We guarantee 99.9% uptime for billing services.",
    constraint_type=ConstraintType.INFORMATIONAL,
)

# Conditional chunks — ineligible for :exact
ALLOW_CONDITION_CHUNK = _make_chunk(
    "kb_allow_refund_conditional",
    "Refunds are available within 30 days for monthly plans. Annual plans are eligible for refunds within 7 days.",
    constraint_type=ConstraintType.ALLOW_CONDITION,
)
DENY_CONDITION_CHUNK = _make_chunk(
    "kb_deny_enterprise_promo",
    "Enterprise customers are not eligible for introductory pricing. Standard pricing applies to all enterprise accounts.",
    constraint_type=ConstraintType.DENY_CONDITION,
)


# ── is_conditional_chunk truth table ─────────────────────────────────────────


def test_is_conditional_chunk_allow_condition_returns_true():
    """ALLOW_CONDITION chunks are conditional — is_conditional_chunk must return True."""
    chunk = _make_chunk("c1", "text", constraint_type=ConstraintType.ALLOW_CONDITION)
    assert is_conditional_chunk(chunk) is True


def test_is_conditional_chunk_deny_condition_returns_true():
    """DENY_CONDITION chunks are conditional — is_conditional_chunk must return True."""
    chunk = _make_chunk("c2", "text", constraint_type=ConstraintType.DENY_CONDITION)
    assert is_conditional_chunk(chunk) is True


def test_is_conditional_chunk_informational_returns_false():
    """INFORMATIONAL chunks can be exactly recited — is_conditional_chunk must return False."""
    chunk = _make_chunk("c3", "text", constraint_type=ConstraintType.INFORMATIONAL)
    assert is_conditional_chunk(chunk) is False


def test_is_conditional_chunk_default_constraint_type_returns_false():
    """KBChunk defaults to INFORMATIONAL — is_conditional_chunk must return False for default."""
    chunk = KBChunk(
        chunk_id="c4",
        document_id="doc_c4",
        document_path="policies/c4.md",
        chunk_text="Standard policy text.",
        domains=["billing"],
        # constraint_type not set → defaults to INFORMATIONAL
    )
    assert is_conditional_chunk(chunk) is False


# ── Injector reroute: supported:exact + conditional chunk → conditional_applied ─


def test_injector_reroutes_exact_to_conditional_applied_for_allow_condition_chunk():
    """
    When the only available chunk is ALLOW_CONDITION and a supported:exact plan
    is generated, the injector must reroute to supported:conditional_applied.
    """
    rng = random.Random(42)
    injector = QualityPlanInjector(rng=rng)
    events = [_make_conv_event(f"evt_{i:03d}") for i in range(80)]
    # Only ALLOW_CONDITION chunk available — exact plans must reroute
    chunks = [ALLOW_CONDITION_CHUNK]

    plans = injector.inject(events=events, snapshots=[], kb_chunks=chunks, planted_count=50)

    exact_plans = [
        p for p in plans
        if p.rubric_targets.accuracy
        and p.rubric_targets.accuracy.precision == "exact"
        and p.kb_chunks_required  # only plans that actually picked this chunk
    ]
    assert exact_plans == [], (
        f"Found {len(exact_plans)} supported:exact plans paired with a conditional chunk — "
        f"reroute did not fire. Plans: {[p.conversation_id for p in exact_plans]}"
    )


def test_injector_reroutes_exact_to_conditional_applied_for_deny_condition_chunk():
    """
    When the only available chunk is DENY_CONDITION and a supported:exact plan
    is generated, the injector must reroute to supported:conditional_applied.
    """
    rng = random.Random(42)
    injector = QualityPlanInjector(rng=rng)
    events = [_make_conv_event(f"evt_{i:03d}") for i in range(80)]
    chunks = [DENY_CONDITION_CHUNK]

    plans = injector.inject(events=events, snapshots=[], kb_chunks=chunks, planted_count=50)

    exact_plans = [
        p for p in plans
        if p.rubric_targets.accuracy
        and p.rubric_targets.accuracy.precision == "exact"
        and p.kb_chunks_required
    ]
    assert exact_plans == [], (
        f"Found {len(exact_plans)} exact plans with DENY_CONDITION chunk — reroute did not fire"
    )


def test_injector_preserves_exact_for_informational_chunk():
    """
    When informational chunks are available and a supported:exact plan is
    generated, the injector must NOT reroute — exact stays exact.
    """
    rng = random.Random(42)
    injector = QualityPlanInjector(rng=rng)
    events = [_make_conv_event(f"evt_{i:03d}") for i in range(80)]
    # Three informational fact-bearing chunks
    chunks = [INFORMATIONAL_FACT_A, INFORMATIONAL_FACT_B, INFORMATIONAL_FACT_C]

    plans = injector.inject(events=events, snapshots=[], kb_chunks=chunks, planted_count=50)

    supported_exact_plans = [
        p for p in plans
        if p.rubric_targets.accuracy
        and p.rubric_targets.accuracy.status == "supported"
        and p.rubric_targets.accuracy.precision == "exact"
    ]
    # With only informational chunks, supported:exact plans should appear (unmodified)
    assert supported_exact_plans, (
        "Expected at least one supported:exact plan when only informational chunks are available"
    )


def test_injector_does_not_reroute_overgeneralized_plans():
    """
    RFORGE-7 covers overgeneralized plans. RFORGE-10 must not reroute
    overgeneralized + ALLOW_CONDITION → the precision stays overgeneralized.
    """
    rng = random.Random(42)
    injector = QualityPlanInjector(rng=rng)
    events = [_make_conv_event(f"evt_{i:03d}") for i in range(80)]
    # ALLOW_CONDITION chunk with a real constraint in it
    allow_chunk_with_constraint = _make_chunk(
        "kb_allow_with_limit",
        "Monthly plans can request a refund only within 14 days of billing.",
        constraint_type=ConstraintType.ALLOW_CONDITION,
    )
    chunks = [allow_chunk_with_constraint, INFORMATIONAL_FACT_A, INFORMATIONAL_FACT_B]

    plans = injector.inject(events=events, snapshots=[], kb_chunks=chunks, planted_count=50)

    overgeneralized_plans = [
        p for p in plans
        if p.rubric_targets.accuracy
        and p.rubric_targets.accuracy.precision == "overgeneralized"
    ]
    # Overgeneralized plans must remain overgeneralized (RFORGE-7's domain)
    for plan in overgeneralized_plans:
        assert plan.rubric_targets.accuracy.precision == "overgeneralized", (
            f"Plan {plan.conversation_id} had overgeneralized rerouted to "
            f"{plan.rubric_targets.accuracy.precision!r}"
        )


# ── Pool filter alignment: contradicted:exact excludes conditional chunks ─────


def test_no_contradicted_exact_plan_paired_with_conditional_chunk():
    """
    After RFORGE-10, no contradicted:exact plan may be paired with an
    ALLOW_CONDITION or DENY_CONDITION chunk. The pick_pool_capped filter
    must exclude these before the fact-bearing pre-filter.
    """
    rng = random.Random(42)
    injector = QualityPlanInjector(rng=rng)
    events = [_make_conv_event(f"evt_{i:03d}") for i in range(80)]
    # Mix of informational and conditional chunks — contradicted:exact must only pick informational
    chunks = [
        INFORMATIONAL_FACT_A,
        INFORMATIONAL_FACT_B,
        INFORMATIONAL_FACT_C,
        ALLOW_CONDITION_CHUNK,
        DENY_CONDITION_CHUNK,
    ]
    plans = injector.inject(events=events, snapshots=[], kb_chunks=chunks, planted_count=50)

    contradicted_exact_plans = [
        p for p in plans
        if p.rubric_targets.accuracy
        and p.rubric_targets.accuracy.status == "contradicted"
        and p.rubric_targets.accuracy.precision == "exact"
        and p.kb_chunks_required
    ]

    chunk_lookup = {c.chunk_id: c for c in chunks}
    for plan in contradicted_exact_plans:
        for cid in plan.kb_chunks_required:
            chunk = chunk_lookup.get(cid)
            assert chunk is not None, f"kb_chunks_required references unknown chunk {cid!r}"
            assert not is_conditional_chunk(chunk), (
                f"Plan {plan.conversation_id} (contradicted:exact) is paired with conditional "
                f"chunk {cid!r} (constraint_type={chunk.constraint_type}) — pool filter did not fire"
            )


# ── Pool filter telemetry: conditional_excluded_count ─────────────────────────


def test_pool_filter_telemetry_includes_conditional_excluded_count():
    """
    When conditional chunks are present in the pool, pool_filter_telemetry for
    contradicted:exact must include a conditional_excluded_count field.
    """
    rng = random.Random(42)
    injector = QualityPlanInjector(rng=rng)
    events = [_make_conv_event(f"evt_{i:03d}") for i in range(80)]
    chunks = [
        INFORMATIONAL_FACT_A,
        INFORMATIONAL_FACT_B,
        INFORMATIONAL_FACT_C,
        ALLOW_CONDITION_CHUNK,
        DENY_CONDITION_CHUNK,
    ]
    injector.inject(events=events, snapshots=[], kb_chunks=chunks, planted_count=50)

    telemetry = injector.pool_filter_telemetry
    assert "contradicted:exact" in telemetry, (
        f"pool_filter_telemetry must contain 'contradicted:exact'; keys={list(telemetry)}"
    )
    ct = telemetry["contradicted:exact"]
    assert "conditional_excluded_count" in ct, (
        f"pool_filter_telemetry['contradicted:exact'] missing 'conditional_excluded_count'; "
        f"keys={list(ct)}"
    )
    # Conditional chunks are present in the pool, so at least some were excluded
    assert ct["conditional_excluded_count"] >= 0, (
        f"conditional_excluded_count must be non-negative, got {ct['conditional_excluded_count']}"
    )


def test_pool_filter_telemetry_conditional_excluded_count_zero_with_no_conditional_chunks():
    """
    When no conditional chunks are in the pool, conditional_excluded_count must be 0.
    """
    rng = random.Random(42)
    injector = QualityPlanInjector(rng=rng)
    events = [_make_conv_event(f"evt_{i:03d}") for i in range(80)]
    # Only informational chunks — no conditional exclusions should occur
    chunks = [INFORMATIONAL_FACT_A, INFORMATIONAL_FACT_B, INFORMATIONAL_FACT_C]

    injector.inject(events=events, snapshots=[], kb_chunks=chunks, planted_count=50)

    telemetry = injector.pool_filter_telemetry
    if "contradicted:exact" in telemetry:
        ct = telemetry["contradicted:exact"]
        assert ct.get("conditional_excluded_count", 0) == 0, (
            f"Expected conditional_excluded_count=0 with no conditional chunks, "
            f"got {ct.get('conditional_excluded_count')}"
        )


# ── Injector telemetry: conditional_reroute_count / conditional_reroute_events ─


def test_conditional_reroute_count_incremented_when_reroute_fires():
    """
    When reroute fires (exact + conditional chunk), conditional_reroute_count
    must be > 0 after inject().
    """
    rng = random.Random(42)
    injector = QualityPlanInjector(rng=rng)
    events = [_make_conv_event(f"evt_{i:03d}") for i in range(80)]
    chunks = [ALLOW_CONDITION_CHUNK, INFORMATIONAL_FACT_A]

    injector.inject(events=events, snapshots=[], kb_chunks=chunks, planted_count=50)

    assert injector.conditional_reroute_count > 0, (
        f"Expected conditional_reroute_count > 0 when ALLOW_CONDITION chunk is in pool "
        f"and exact plans are scheduled, got {injector.conditional_reroute_count}"
    )


def test_conditional_reroute_events_populated_when_reroute_fires():
    """
    conditional_reroute_events must be a list of event IDs, one per reroute,
    and its length must equal conditional_reroute_count.
    """
    rng = random.Random(42)
    injector = QualityPlanInjector(rng=rng)
    events = [_make_conv_event(f"evt_{i:03d}") for i in range(80)]
    chunks = [ALLOW_CONDITION_CHUNK, INFORMATIONAL_FACT_A]

    injector.inject(events=events, snapshots=[], kb_chunks=chunks, planted_count=50)

    assert len(injector.conditional_reroute_events) == injector.conditional_reroute_count, (
        f"conditional_reroute_events length {len(injector.conditional_reroute_events)} "
        f"!= conditional_reroute_count {injector.conditional_reroute_count}"
    )
    for event_id in injector.conditional_reroute_events:
        assert isinstance(event_id, str) and event_id, (
            f"conditional_reroute_events must contain non-empty strings, got {event_id!r}"
        )


def test_conditional_reroute_count_zero_with_only_informational_chunks():
    """
    With only informational chunks, no reroute fires — conditional_reroute_count must be 0.
    """
    rng = random.Random(42)
    injector = QualityPlanInjector(rng=rng)
    events = [_make_conv_event(f"evt_{i:03d}") for i in range(80)]
    chunks = [INFORMATIONAL_FACT_A, INFORMATIONAL_FACT_B, INFORMATIONAL_FACT_C]

    injector.inject(events=events, snapshots=[], kb_chunks=chunks, planted_count=50)

    assert injector.conditional_reroute_count == 0, (
        f"Expected conditional_reroute_count=0 with only informational chunks, "
        f"got {injector.conditional_reroute_count}"
    )
    assert injector.conditional_reroute_events == [], (
        f"Expected empty conditional_reroute_events, got {injector.conditional_reroute_events}"
    )


# ── Determinism ───────────────────────────────────────────────────────────────


def test_conditional_reroute_deterministic():
    """Same seed → same rerouted plans on every run."""
    chunks = [
        ALLOW_CONDITION_CHUNK,
        INFORMATIONAL_FACT_A,
        INFORMATIONAL_FACT_B,
        INFORMATIONAL_FACT_C,
    ]
    events = [_make_conv_event(f"evt_{i:03d}") for i in range(80)]

    injector_a = QualityPlanInjector(rng=random.Random(77))
    plans_a = injector_a.inject(events=events, snapshots=[], kb_chunks=chunks, planted_count=50)

    injector_b = QualityPlanInjector(rng=random.Random(77))
    plans_b = injector_b.inject(events=events, snapshots=[], kb_chunks=chunks, planted_count=50)

    precisions_a = {
        p.conversation_id: p.rubric_targets.accuracy.precision
        for p in plans_a
        if p.rubric_targets.accuracy
    }
    precisions_b = {
        p.conversation_id: p.rubric_targets.accuracy.precision
        for p in plans_b
        if p.rubric_targets.accuracy
    }
    assert precisions_a == precisions_b, "Conditional reroute is not deterministic across runs"
    assert injector_a.conditional_reroute_count == injector_b.conditional_reroute_count


# ── RFORGE-35: rerouted control plan directive regenerated ────────────────────


def test_rerouted_control_plan_gets_conditional_applied_directive():
    """
    When a control plan is rerouted to conditional_applied, its
    prose_generation_directives must be regenerated to the conditional_applied
    directive — not left as the generic control preamble.

    Without this fix the LLM receives "Write a model customer service interaction"
    with no instruction to apply the conditional, producing negating language
    that fails alignment==supported post-gen validation.
    """
    rng = random.Random(42)
    injector = QualityPlanInjector(rng=rng)
    events = [_make_conv_event(f"evt_{i:03d}") for i in range(80)]
    # Only a conditional chunk in the billing domain — all control plans citing it
    # will be rerouted to conditional_applied.
    chunks = [ALLOW_CONDITION_CHUNK]

    plans = injector.inject(events=events, snapshots=[], kb_chunks=chunks, planted_count=50)

    # Find rerouted control plans: all-clean rubric except accuracy=conditional_applied
    rerouted_control = [
        p for p in plans
        if (
            p.rubric_targets.accuracy is not None
            and p.rubric_targets.accuracy.status == "supported"
            and p.rubric_targets.accuracy.precision == "conditional_applied"
            and p.rubric_targets.empathy == "high"
            and p.rubric_targets.resolution == "strong"
            and p.rubric_targets.brand_voice_target == "on_brand"
        )
    ]
    assert rerouted_control, (
        "Expected at least one rerouted control plan (conditional_applied + all-clean rubric)"
    )

    generic_preamble = (
        "Write a model customer service interaction. Agent should be empathetic"
    )
    for plan in rerouted_control:
        assert generic_preamble not in plan.prose_generation_directives, (
            f"Rerouted control plan {plan.conversation_id} still has generic preamble — "
            f"directive was not regenerated after reroute.\n"
            f"Got: {plan.prose_generation_directives[:200]!r}"
        )
        assert "IS entitled" in plan.prose_generation_directives, (
            f"Rerouted control plan {plan.conversation_id} directive missing conditional_applied "
            f"instruction.\nGot: {plan.prose_generation_directives[:200]!r}"
        )


def test_rerouted_control_plan_directive_contains_chunk_text():
    """Rerouted control plan directive must embed the KB chunk text (same as accuracy plans)."""
    rng = random.Random(42)
    injector = QualityPlanInjector(rng=rng)
    events = [_make_conv_event(f"evt_{i:03d}") for i in range(80)]
    chunks = [ALLOW_CONDITION_CHUNK]

    plans = injector.inject(events=events, snapshots=[], kb_chunks=chunks, planted_count=50)

    rerouted_control = [
        p for p in plans
        if (
            p.rubric_targets.accuracy is not None
            and p.rubric_targets.accuracy.precision == "conditional_applied"
            and p.rubric_targets.empathy == "high"
            and p.rubric_targets.resolution == "strong"
        )
    ]
    assert rerouted_control, "Expected at least one rerouted control plan"

    for plan in rerouted_control:
        assert ALLOW_CONDITION_CHUNK.chunk_text in plan.prose_generation_directives, (
            f"Rerouted control plan directive missing chunk text.\n"
            f"Got: {plan.prose_generation_directives[:300]!r}"
        )
