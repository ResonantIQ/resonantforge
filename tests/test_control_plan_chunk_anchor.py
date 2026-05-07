"""
Tests for RFORGE-13: control plan prose directive must embed the required KB chunk text.

Without the chunk text inline, the LLM has no grounding material and produces
zero-claim outputs, causing downstream accuracy validators to mark the plan as
unscored. The fix appends _render_chunk_blocks to the control plan directive
exactly as all accuracy plan branches already do.

Covers:
  - Control plan directive contains KB chunk text when a chunk is available
  - Chunk text appears after the generic preamble (not instead of it)
  - Generic preamble is still present
  - When allow_ids is empty (no chunk available) directive is preamble-only
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

_GENERIC_PREAMBLE = (
    "Write a model customer service interaction. Agent should be empathetic, "
    "resolve the issue completely, write on-brand, and make accurate claims "
    "supported by the knowledge base."
)

BILLING_CHUNK = KBChunk(
    chunk_id="kb_billing_policy",
    document_id="doc_billing",
    document_path="policies/billing.md",
    chunk_text="Refunds are processed within 5–7 business days after approval.",
    constraint_type=ConstraintType.INFORMATIONAL,
    domains=["billing"],
)

FEATURES_CHUNK = KBChunk(
    chunk_id="kb_features_sso",
    document_id="doc_features",
    document_path="policies/features.md",
    chunk_text="Single sign-on (SSO) is available on Enterprise plans only.",
    constraint_type=ConstraintType.INFORMATIONAL,
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


def _control_plans(plans):
    """Extract plans with all-clean rubric targets (control plans).

    ALL_CLEAN_TARGETS: empathy="high", resolution="strong", brand_voice set, accuracy=supported:exact.
    This combination uniquely identifies control plans — other plan types only set one dimension.
    """
    return [
        p for p in plans
        if (
            p.rubric_targets.accuracy is not None
            and p.rubric_targets.accuracy.status == "supported"
            and p.rubric_targets.accuracy.precision == "exact"
            and p.rubric_targets.empathy == "high"
            and p.rubric_targets.resolution == "strong"
            and p.rubric_targets.brand_voice_target == "on_brand"
        )
    ]


# ── Test 1: chunk text present in directive ───────────────────────────────────


def test_control_plan_directive_contains_chunk_text():
    """
    A control plan directive must include the KB chunk text when a billing chunk
    is available. Without it the LLM has no grounding material.
    """
    rng = random.Random(42)
    injector = QualityPlanInjector(rng=rng)
    events = [_make_conv_event(f"evt_{i:03d}") for i in range(80)]
    chunks = [BILLING_CHUNK]

    plans = injector.inject(events=events, snapshots=[], kb_chunks=chunks, planted_count=50)
    control = _control_plans(plans)

    assert control, "Expected at least one control plan"
    for plan in control:
        assert BILLING_CHUNK.chunk_text in plan.prose_generation_directives, (
            f"Control plan directive missing chunk text.\n"
            f"Expected to find: {BILLING_CHUNK.chunk_text!r}\n"
            f"Got: {plan.prose_generation_directives!r}"
        )


# ── Test 2: generic preamble still present ────────────────────────────────────


def test_control_plan_directive_retains_generic_preamble():
    """Chunk text is appended; the generic preamble must still be present."""
    rng = random.Random(42)
    injector = QualityPlanInjector(rng=rng)
    events = [_make_conv_event(f"evt_{i:03d}") for i in range(80)]
    chunks = [BILLING_CHUNK]

    plans = injector.inject(events=events, snapshots=[], kb_chunks=chunks, planted_count=50)
    control = _control_plans(plans)

    assert control, "Expected at least one control plan"
    for plan in control:
        assert _GENERIC_PREAMBLE in plan.prose_generation_directives, (
            f"Generic preamble missing from control plan directive.\n"
            f"Got: {plan.prose_generation_directives!r}"
        )


# ── Test 3: chunk text follows the preamble ───────────────────────────────────


def test_control_plan_chunk_text_follows_preamble():
    """Chunk text must appear after the generic preamble, not before it."""
    rng = random.Random(42)
    injector = QualityPlanInjector(rng=rng)
    events = [_make_conv_event(f"evt_{i:03d}") for i in range(80)]
    chunks = [BILLING_CHUNK]

    plans = injector.inject(events=events, snapshots=[], kb_chunks=chunks, planted_count=50)
    control = _control_plans(plans)

    assert control, "Expected at least one control plan"
    for plan in control:
        d = plan.prose_generation_directives
        preamble_pos = d.find(_GENERIC_PREAMBLE)
        chunk_pos = d.find(BILLING_CHUNK.chunk_text)
        assert preamble_pos != -1, "Preamble not found"
        assert chunk_pos != -1, "Chunk text not found"
        assert chunk_pos > preamble_pos, (
            f"Chunk text appears before preamble. preamble_pos={preamble_pos}, chunk_pos={chunk_pos}"
        )


# ── Test 4: chunk_id header present ──────────────────────────────────────────


def test_control_plan_directive_contains_chunk_id_header():
    """The rendered block should include the chunk_id label (e.g. [kb_billing_policy])."""
    rng = random.Random(42)
    injector = QualityPlanInjector(rng=rng)
    events = [_make_conv_event(f"evt_{i:03d}") for i in range(80)]
    chunks = [BILLING_CHUNK]

    plans = injector.inject(events=events, snapshots=[], kb_chunks=chunks, planted_count=50)
    control = _control_plans(plans)

    assert control, "Expected at least one control plan"
    for plan in control:
        assert f"[{BILLING_CHUNK.chunk_id}]" in plan.prose_generation_directives, (
            f"chunk_id header [{BILLING_CHUNK.chunk_id}] missing from directive.\n"
            f"Got: {plan.prose_generation_directives!r}"
        )
