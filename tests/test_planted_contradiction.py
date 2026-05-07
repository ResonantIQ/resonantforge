"""
Tests for the closed-loop planted_contradiction pipeline (RFORGE-8).

Covers:
  - Chunk pre-filter: fact-free chunks excluded from contradicted:exact pool
  - Pool starvation: warning emitted when fact-bearing pool < 3
  - Injector: planted_contradiction is set and carries kb_fact + negated_form
  - Injector: falls back to kb_required=[] when no fact-bearing chunk found
  - Transformation prompt: directive names the negated_form
  - Validator: negated_form present + kb_fact absent → contradicted_flag True
  - Validator: negated_form absent → contradicted_flag False
  - Validator: both present → contradicted_flag False (meta-commentary detected)
  - Determinism: same seed → same planted_contradiction
"""
from __future__ import annotations

import logging
import random
from datetime import datetime

import pytest

from resonantforge.kb.kb_fact_extractor import extract_facts
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


def _make_claim(claim_text: str) -> Claim:
    return Claim(
        claim_text=claim_text,
        claim_span=(0, len(claim_text)),
        claim_type="policy",
        normalized_subject="policy",
        normalized_predicate="states",
        normalized_object="claim",
    )


# Fact-bearing chunks (concrete, contradictable facts)
FACT_CHUNK = _make_chunk(
    "kb_refund_policy",
    "Refunds are available within 30 days of purchase for paid plans.",
)
NUMERIC_CHUNK = _make_chunk(
    "kb_seats_policy",
    "Your plan supports up to 500 users.",
    domain="features",
)
CAPABILITY_CHUNK = _make_chunk(
    "kb_webhook",
    "The platform supports outbound webhooks for event notifications.",
    domain="features",
)
# Fact-free chunks — must never be selected for contradicted:exact
VAGUE_CHUNK = _make_chunk(
    "kb_marketing",
    "Our platform is best-in-class for enterprise customers.",
)
MARKETING_CHUNK = _make_chunk(
    "kb_transform",
    "Transform your customer experience with our powerful platform.",
)


# ── Chunk pre-filter ──────────────────────────────────────────────────────────


def test_fact_bearing_chunks_have_extractable_facts():
    """Sanity: our fixture chunks actually yield facts."""
    assert extract_facts(FACT_CHUNK.chunk_text)
    assert extract_facts(NUMERIC_CHUNK.chunk_text)
    assert extract_facts(CAPABILITY_CHUNK.chunk_text)


def test_fact_free_chunks_yield_no_facts():
    """Sanity: our vague/marketing fixture chunks yield no facts."""
    assert extract_facts(VAGUE_CHUNK.chunk_text) == []
    assert extract_facts(MARKETING_CHUNK.chunk_text) == []


# ── Injector: planted_contradiction set ───────────────────────────────────────


def test_injector_planted_contradiction_set_for_contradicted_plan():
    """QualityPlan.planted_contradiction must be non-None for contradicted:exact plans."""
    rng = random.Random(42)
    injector = QualityPlanInjector(rng=rng)
    events = [_make_conv_event(f"evt_{i:03d}") for i in range(80)]
    chunks = [
        FACT_CHUNK,
        NUMERIC_CHUNK,
        CAPABILITY_CHUNK,
        VAGUE_CHUNK,
    ]
    plans = injector.inject(events=events, snapshots=[], kb_chunks=chunks, planted_count=50)

    contradicted_plans = [
        p for p in plans
        if p.rubric_targets.accuracy
        and p.rubric_targets.accuracy.status == "contradicted"
        and p.rubric_targets.accuracy.precision == "exact"
    ]

    assert contradicted_plans, "Expected at least one contradicted:exact plan in the schedule"

    for plan in contradicted_plans:
        assert plan.planted_contradiction is not None, (
            f"Plan {plan.conversation_id} has precision=contradicted:exact "
            f"but planted_contradiction is None"
        )
        assert plan.planted_contradiction.kb_fact, "planted_contradiction.kb_fact must be non-empty"
        assert plan.planted_contradiction.negated_form, "planted_contradiction.negated_form must be non-empty"
        assert plan.planted_contradiction.fact_category in {"numeric", "categorical", "capability", "policy"}


def test_injector_skips_fact_free_chunk_for_contradicted_plan():
    """
    When only fact-free chunks are available, the contradicted:exact plan must
    fall back (kb_chunks_required=[]) rather than planting a meaningless contradiction.
    """
    rng = random.Random(42)
    injector = QualityPlanInjector(rng=rng)
    events = [_make_conv_event(f"evt_{i:03d}") for i in range(80)]
    chunks = [VAGUE_CHUNK, MARKETING_CHUNK]
    plans = injector.inject(events=events, snapshots=[], kb_chunks=chunks, planted_count=50)

    contradicted_plans = [
        p for p in plans
        if p.rubric_targets.accuracy
        and p.rubric_targets.accuracy.status == "contradicted"
        and p.rubric_targets.accuracy.precision == "exact"
    ]

    for plan in contradicted_plans:
        assert not plan.kb_chunks_required or plan.planted_contradiction is None, (
            f"Contradicted plan selected fact-free chunk: "
            f"kb_chunks_required={plan.kb_chunks_required}, "
            f"planted_contradiction={plan.planted_contradiction!r}"
        )


# ── Transformation prompt ─────────────────────────────────────────────────────


def test_transformation_prompt_contains_negated_form():
    """prose_generation_directives for a contradicted:exact plan must embed negated_form."""
    rng = random.Random(42)
    injector = QualityPlanInjector(rng=rng)
    events = [_make_conv_event(f"evt_{i:03d}") for i in range(80)]
    chunks = [FACT_CHUNK, NUMERIC_CHUNK, VAGUE_CHUNK]
    plans = injector.inject(events=events, snapshots=[], kb_chunks=chunks, planted_count=50)

    candidates = [
        p for p in plans
        if p.rubric_targets.accuracy
        and p.rubric_targets.accuracy.status == "contradicted"
        and p.rubric_targets.accuracy.precision == "exact"
        and p.planted_contradiction is not None
    ]
    assert candidates, "No contradicted:exact plan with planted_contradiction found"

    for plan in candidates:
        assert plan.planted_contradiction.negated_form in plan.prose_generation_directives, (
            f"negated_form {plan.planted_contradiction.negated_form!r} not found in directives"
        )


# ── Pool starvation ───────────────────────────────────────────────────────────


def test_pool_starvation_logged_when_fact_bearing_pool_small(caplog):
    """
    When the fact-bearing chunk pool drops below 3 after pre-filtering,
    a pool_starvation warning must be logged.
    """
    rng = random.Random(42)
    injector = QualityPlanInjector(rng=rng)
    events = [_make_conv_event(f"evt_{i:03d}") for i in range(80)]
    # Only 1 fact-bearing chunk → pool will be 1 after filtering → starvation
    chunks = [
        FACT_CHUNK,
        VAGUE_CHUNK,
        MARKETING_CHUNK,
    ]

    with caplog.at_level(logging.WARNING):
        injector.inject(events=events, snapshots=[], kb_chunks=chunks, planted_count=50)

    pool_starvation_logs = [
        r for r in caplog.records
        if "pool_starvation" in r.getMessage()
    ]
    assert pool_starvation_logs, "Expected pool_starvation warning when fact-bearing pool < 3"


# ── Determinism ───────────────────────────────────────────────────────────────


def test_planted_contradiction_deterministic():
    """Same seed → same planted_contradiction on every run."""
    chunks = [FACT_CHUNK, NUMERIC_CHUNK, CAPABILITY_CHUNK, VAGUE_CHUNK]
    events = [_make_conv_event(f"evt_{i:03d}") for i in range(80)]

    plans_a = QualityPlanInjector(rng=random.Random(99)).inject(
        events=events, snapshots=[], kb_chunks=chunks, planted_count=50
    )
    plans_b = QualityPlanInjector(rng=random.Random(99)).inject(
        events=events, snapshots=[], kb_chunks=chunks, planted_count=50
    )

    contradicted_a = {
        p.conversation_id: p.planted_contradiction
        for p in plans_a
        if p.planted_contradiction is not None
    }
    contradicted_b = {
        p.conversation_id: p.planted_contradiction
        for p in plans_b
        if p.planted_contradiction is not None
    }

    assert contradicted_a == contradicted_b, "planted_contradiction not deterministic across runs"


# ── Validator: contradicted_flag ─────────────────────────────────────────────


def test_validator_contradicted_flag_set_when_negated_form_present_kb_fact_absent():
    """
    Clean plant: agent claim contains negated_form but not kb_fact → contradicted_flag True.
    """
    planted_contradiction = {
        "kb_fact": "refunds are available within 30 days of purchase",
        "negated_form": "no refunds are issued under any circumstances",
        "fact_category": "policy",
    }
    # Agent says the negated form (the contradiction) — not the original fact
    claim_text = "No refunds are issued under any circumstances. Our policy does not allow exceptions."
    claims = [_make_claim(claim_text)]

    signals = run_kb_alignment_pipeline(
        claims=claims,
        kb_chunks=[FACT_CHUNK],
        conversation_context="",
        kb_chunks_required=["kb_refund_policy"],
        synonym_map={},
        planted_contradiction=planted_contradiction,
    )
    assert signals.contradicted_flag is True


def test_validator_contradicted_flag_false_when_negated_form_absent():
    """
    Not found: agent claim mentions neither the kb_fact nor negated_form → contradicted_flag False.
    """
    planted_contradiction = {
        "kb_fact": "refunds are available within 30 days of purchase",
        "negated_form": "no refunds are issued under any circumstances",
        "fact_category": "policy",
    }
    claim_text = "Our return and billing processes follow standard industry guidelines."
    claims = [_make_claim(claim_text)]

    signals = run_kb_alignment_pipeline(
        claims=claims,
        kb_chunks=[FACT_CHUNK],
        conversation_context="",
        kb_chunks_required=["kb_refund_policy"],
        synonym_map={},
        planted_contradiction=planted_contradiction,
    )
    assert signals.contradicted_flag is False


def test_validator_contradicted_flag_false_when_both_present():
    """
    Meta-commentary: agent claim contains both kb_fact and negated_form → contradicted_flag False.
    The LLM self-corrected or hedged; contradiction is not clean — do not reward it.
    """
    planted_contradiction = {
        "kb_fact": "refunds are available within 30 days of purchase",
        "negated_form": "no refunds are issued under any circumstances",
        "fact_category": "policy",
    }
    claim_text = (
        "While refunds are available within 30 days of purchase, "
        "no refunds are issued under any circumstances for digital goods."
    )
    claims = [_make_claim(claim_text)]

    signals = run_kb_alignment_pipeline(
        claims=claims,
        kb_chunks=[FACT_CHUNK],
        conversation_context="",
        kb_chunks_required=["kb_refund_policy"],
        synonym_map={},
        planted_contradiction=planted_contradiction,
    )
    assert signals.contradicted_flag is False


def test_validator_fallback_when_no_planted_contradiction():
    """
    When planted_contradiction is None, validator uses existing alignment path —
    contradicted_flag defaults to False (no closed-loop override).
    """
    claims = [_make_claim("Refunds are available within 30 days of purchase.")]

    signals = run_kb_alignment_pipeline(
        claims=claims,
        kb_chunks=[FACT_CHUNK],
        conversation_context="",
        kb_chunks_required=["kb_refund_policy"],
        synonym_map={},
        planted_contradiction=None,
    )
    assert signals.contradicted_flag is False


def test_directive_for_contradicted_plan_does_not_contain_contradiction_hint():
    """
    The prose_generation_directives for a contradicted:exact plan must NOT contain
    the word 'contradict' in any form. Exposing the meta-label to the LLM causes
    meta-commentary (acknowledging both the real policy and the planted claim), which
    sets kb_fact_present=True → contradicted_flag=False → validation FAIL.
    """
    rng = random.Random(42)
    injector = QualityPlanInjector(rng=rng)
    events = [_make_conv_event(f"evt_{i:03d}") for i in range(80)]
    chunks = [FACT_CHUNK, NUMERIC_CHUNK, VAGUE_CHUNK]
    plans = injector.inject(events=events, snapshots=[], kb_chunks=chunks, planted_count=50)

    candidates = [
        p for p in plans
        if p.rubric_targets.accuracy
        and p.rubric_targets.accuracy.status == "contradicted"
        and p.rubric_targets.accuracy.precision == "exact"
        and p.planted_contradiction is not None
    ]
    assert candidates, "Need at least one contradicted:exact plan with planted_contradiction"

    for plan in candidates:
        assert "contradict" not in plan.prose_generation_directives.lower(), (
            f"prose_generation_directives contains 'contradict' — this cues LLM meta-commentary.\n"
            f"Directive excerpt: {plan.prose_generation_directives[:300]!r}"
        )
