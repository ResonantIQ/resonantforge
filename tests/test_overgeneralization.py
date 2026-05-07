"""
Tests for the closed-loop overgeneralization pipeline (RFORGE-7 + RFORGE-29).

Covers:
  - Chunk pre-filter: unconstrained chunks excluded from overgeneralized pool
  - Transformation prompt: directive embeds named constraint
  - QualityPlan carries planted_constraint
  - Validator: planted_constraint absent → overgeneralization_flag True
  - Validator: planted_constraint present → overgeneralization_flag False
  - conv_evt_00114 class: bare numeric "10 seats" → "unlimited seats" → flag
  - Determinism: same seed → same planted_constraint
"""
from __future__ import annotations

import random
from datetime import datetime

import pytest

from resonantforge.kb.constraint_extractor import extract_constraints
from resonantforge.layer1.quality_plan_injector import QualityPlanInjector
from resonantforge.schemas import (
    AccuracyLabel,
    Claim,
    ConstraintType,
    KBChunk,
    SimEvent,
    SimEventType,
)
from resonantforge.validators.extractors.accuracy import run_kb_alignment_pipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


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
    constraint_type: ConstraintType = ConstraintType.ALLOW_CONDITION,
) -> KBChunk:
    return KBChunk(
        chunk_id=chunk_id,
        document_id=f"doc_{chunk_id}",
        document_path=f"policies/{chunk_id}.md",
        chunk_text=chunk_text,
        constraint_type=constraint_type,
        domains=[domain],
    )


def _make_claim(claim_text: str) -> Claim:
    return Claim(
        claim_text=claim_text,
        claim_span=(0, len(claim_text)),
        claim_type="policy",
        normalized_subject="policy",
        normalized_predicate="states",
        normalized_object="claim",
    )


CONSTRAINED_CHUNK = _make_chunk(
    "kb_refund_policy",
    "Refunds available within 30 days for paid plans only.",
)
SEATS_CHUNK = _make_chunk(
    "kb_seats_policy",
    "The plan includes 10 seats per account.",
    domain="features",
)
UNCONSTRAINED_CHUNK = _make_chunk(
    "kb_support_info",
    "Our support team is available 24/7 to help you.",
)


# ---------------------------------------------------------------------------
# Chunk pre-filter: injector must not select unconstrained chunks for
# overgeneralized plans
# ---------------------------------------------------------------------------


def test_injector_planted_constraint_set_for_overgeneralized_plan():
    """QualityPlan.planted_constraint must be non-None for overgeneralized accuracy plans."""
    rng = random.Random(42)
    injector = QualityPlanInjector(rng=rng)
    events = [_make_conv_event(f"evt_{i:03d}") for i in range(60)]
    chunks = [
        CONSTRAINED_CHUNK,
        _make_chunk("kb_refund2", "Refunds within 14 days for enterprise customers only.", domain="billing"),
        _make_chunk("kb_billing", "Billing is processed monthly for annual subscriptions.", domain="billing"),
        UNCONSTRAINED_CHUNK,
    ]
    plans = injector.inject(events=events, snapshots=[], kb_chunks=chunks, planted_count=20)

    overgen_plans = [
        p for p in plans
        if p.rubric_targets.accuracy
        and p.rubric_targets.accuracy.precision == "overgeneralized"
    ]

    assert overgen_plans, "Expected at least one overgeneralized plan in the schedule"

    for plan in overgen_plans:
        assert plan.planted_constraint is not None, (
            f"Plan {plan.conversation_id} has precision=overgeneralized "
            f"but planted_constraint is None"
        )


def test_injector_skips_unconstrained_chunk_for_overgeneralized_plan():
    """
    When the only available chunk has no constraints, the overgeneralized plan
    must fall back (kb_chunks_required=[]) rather than planting a meaningless constraint.
    """
    rng = random.Random(42)
    injector = QualityPlanInjector(rng=rng)
    events = [_make_conv_event(f"evt_{i:03d}") for i in range(60)]
    # Only unconstrained chunks available
    chunks = [UNCONSTRAINED_CHUNK]
    plans = injector.inject(events=events, snapshots=[], kb_chunks=chunks, planted_count=20)

    overgen_plans = [
        p for p in plans
        if p.rubric_targets.accuracy
        and p.rubric_targets.accuracy.precision == "overgeneralized"
    ]

    for plan in overgen_plans:
        # Either no chunk was selected, or planted_constraint is None (fallback)
        assert not plan.kb_chunks_required or plan.planted_constraint is None, (
            f"Overgeneralized plan selected unconstrained chunk without constraint: "
            f"kb_chunks_required={plan.kb_chunks_required}, "
            f"planted_constraint={plan.planted_constraint!r}"
        )


# ---------------------------------------------------------------------------
# Transformation prompt: directive must name the planted constraint
# ---------------------------------------------------------------------------


def test_transformation_prompt_contains_planted_constraint():
    """The prose_generation_directives for an overgeneralized plan must embed the constraint."""
    rng = random.Random(42)
    injector = QualityPlanInjector(rng=rng)
    events = [_make_conv_event(f"evt_{i:03d}") for i in range(60)]
    chunks = [CONSTRAINED_CHUNK, UNCONSTRAINED_CHUNK]
    plans = injector.inject(events=events, snapshots=[], kb_chunks=chunks, planted_count=20)

    overgen_plans_with_constraint = [
        p for p in plans
        if p.rubric_targets.accuracy
        and p.rubric_targets.accuracy.precision == "overgeneralized"
        and p.planted_constraint is not None
    ]

    assert overgen_plans_with_constraint, "No overgeneralized plan with planted_constraint found"

    for plan in overgen_plans_with_constraint:
        constraint = plan.planted_constraint.lower()
        directives_lower = plan.prose_generation_directives.lower()
        assert constraint in directives_lower, (
            f"Planted constraint {constraint!r} not found in directives:\n"
            f"{plan.prose_generation_directives}"
        )


# ---------------------------------------------------------------------------
# Validator: run_kb_alignment_pipeline with planted_constraint
# ---------------------------------------------------------------------------


def test_validator_flags_when_planted_constraint_absent():
    """
    Planted constraint 'within 30 days' absent from agent claim → overgeneralization_flag=True.
    """
    claims = [_make_claim("Refunds are available for all customers.")]
    signals = run_kb_alignment_pipeline(
        claims=claims,
        kb_chunks=[CONSTRAINED_CHUNK],
        conversation_context="",
        kb_chunks_required=["kb_refund_policy"],
        synonym_map={},
        planted_constraint="within 30 days",
    )
    assert signals.overgeneralization_flag is True


def test_validator_no_flag_when_planted_constraint_present():
    """
    Planted constraint 'within 30 days' present in agent claim → no overgeneralization flag.
    """
    claims = [_make_claim("Refunds are available within 30 days of purchase.")]
    signals = run_kb_alignment_pipeline(
        claims=claims,
        kb_chunks=[CONSTRAINED_CHUNK],
        conversation_context="",
        kb_chunks_required=["kb_refund_policy"],
        synonym_map={},
        planted_constraint="within 30 days",
    )
    assert signals.overgeneralization_flag is False


def test_validator_flags_conv_evt_00114_class():
    """
    conv_evt_00114 class: chunk says '10 seats', agent says 'unlimited seats'.
    Old regex-based detector missed this; planted_constraint check must catch it.
    """
    claim = _make_claim("We offer unlimited seats for all account types.")
    signals = run_kb_alignment_pipeline(
        claims=[claim],
        kb_chunks=[SEATS_CHUNK],
        conversation_context="",
        kb_chunks_required=["kb_seats_policy"],
        synonym_map={},
        planted_constraint="10 seats",
    )
    assert signals.overgeneralization_flag is True


def test_validator_no_planted_constraint_does_not_crash():
    """
    When planted_constraint=None (organic conversation path), the pipeline
    runs without error. Organic conversations have no accuracy rubric target so
    overgeneralization_flag is not load-bearing — we only verify no exception.
    """
    claims = [_make_claim("Refunds are available for all customers.")]
    signals = run_kb_alignment_pipeline(
        claims=claims,
        kb_chunks=[CONSTRAINED_CHUNK],
        conversation_context="",
        kb_chunks_required=["kb_refund_policy"],
        synonym_map={},
        planted_constraint=None,
    )
    assert isinstance(signals.overgeneralization_flag, bool)


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_same_seed_produces_same_planted_constraint():
    """Same seed → same overgeneralized plan → same planted_constraint."""
    chunks = [CONSTRAINED_CHUNK, UNCONSTRAINED_CHUNK]
    events = [_make_conv_event(f"evt_{i:03d}") for i in range(60)]

    plans_a = QualityPlanInjector(rng=random.Random(99)).inject(
        events=events, snapshots=[], kb_chunks=chunks, planted_count=20
    )
    plans_b = QualityPlanInjector(rng=random.Random(99)).inject(
        events=events, snapshots=[], kb_chunks=chunks, planted_count=20
    )

    constraints_a = [p.planted_constraint for p in plans_a]
    constraints_b = [p.planted_constraint for p in plans_b]
    assert constraints_a == constraints_b, "Determinism broken: different seeds produced different planted_constraints"
