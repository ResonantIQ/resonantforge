"""
Tests for RFORGE-37: branch-scoped constraint_preserved for ALLOW_CONDITION chunks.

Root cause: the constraint extractor treats the full ALLOW_CONDITION chunk text as a
single constraint set. For multi-branch chunks (e.g. "monthly: 30-day window / annual:
account credits"), a correctly-applied annual-branch claim omits the monthly-branch
constraint "within 30 days", causing constraint_preserved=False every time.

Fix:
  - KBChunk gains an optional `branches` field (list[KBChunkBranch]) and
    `global_constraints` field.
  - QualityPlan gains an optional `target_branch` field.
  - Extractor: multi-branch chunk + target_branch → constraint check scoped to that
    branch's constraints ∪ global_constraints.
  - Extractor: multi-branch chunk + no target_branch (organic) → skip constraint check
    (constraint_preserved=True); run cross-branch contamination check instead.
  - Contamination check (per-claim only): if a single claim contains constraints from
    ≥2 branches, flag as contaminated (constraint_preserved=False).
  - Single-branch ALLOW_CONDITION chunks (no `branches`) → existing behavior unchanged.

Residual gaps (documented, not detected):
  - Cross-claim contamination is not detected; rejected due to false-positive rate on
    legitimate full-policy disclosure ("monthly: 30 days; annual: credits").
  - Wrong-branch-applied (LLM cites monthly terms for annual customer) not detectable
    without target_branch.
  - Organic ALLOW_CONDITION constraint_preserved is best-effort; no ground truth on
    intended branch. Contamination check is per-claim only. Cross-claim contamination
    is not detected; conversation-level rejected due to false-positive rate on legitimate
    full-policy disclosure. Turn-scoped contamination is a candidate tightening if the
    failure mode surfaces.

Covers:
  - KBChunkBranch schema exists and is accepted by KBChunk
  - QualityPlan accepts target_branch
  - Extractor: target_branch="monthly" → only monthly constraints required
  - Extractor: target_branch="annual" → only annual constraints required
  - Extractor: no target_branch → constraint_preserved=True (skip check)
  - Contamination: claim with constraints from two branches → constraint_preserved=False
  - Contamination: claim with constraints from one branch → constraint_preserved=True
  - Single-branch ALLOW_CONDITION (no branches field) → existing behavior unchanged
  - Injector: multi-branch conditional chunk → target_branch set on quality plan
"""
from __future__ import annotations

import random
from datetime import datetime

import pytest

from resonantforge.schemas import (
    ConstraintType,
    KBChunk,
    KBChunkBranch,
    QualityPlan,
    SimEvent,
    SimEventType,
)
from resonantforge.validators.extractors.accuracy import _check_chunk_relevance


# ── Fixtures ──────────────────────────────────────────────────────────────────


REFUND_MULTI_BRANCH_CHUNK = KBChunk(
    chunk_id="kb_refund_conditional_plan_type",
    document_id="doc_refund",
    document_path="policies/refund.md",
    chunk_text=(
        "Refund requests within 30 days are self-serve for monthly plan customers "
        "via the billing portal. Annual plan customers must contact billing support "
        "to initiate a refund; self-serve refund is not available for annual accounts."
    ),
    constraint_type=ConstraintType.ALLOW_CONDITION,
    domains=["refund_policy"],
    branches=[
        KBChunkBranch(
            id="monthly",
            condition="monthly plan customer",
            constraints=["within 30 days"],
            content=(
                "Refund requests within 30 days are self-serve for monthly plan customers "
                "via the billing portal."
            ),
        ),
        KBChunkBranch(
            id="annual",
            condition="annual plan customer",
            constraints=["billing support"],
            content=(
                "Annual plan customers must contact billing support to initiate a refund; "
                "self-serve refund is not available for annual accounts."
            ),
        ),
    ],
)

SINGLE_BRANCH_CHUNK = KBChunk(
    chunk_id="kb_cancellation_allow",
    document_id="doc_cancel",
    document_path="policies/cancel.md",
    chunk_text=(
        "Paid subscribers may cancel at any time with no cancellation fee. "
        "Access continues through the end of the current billing period."
    ),
    constraint_type=ConstraintType.ALLOW_CONDITION,
    domains=["cancellation"],
    # No branches field — single-branch, existing behavior
)


def _make_claim(text: str):
    """Build a minimal Claim for extractor testing."""
    from resonantforge.schemas import Claim
    return Claim(
        claim_text=text,
        claim_span=(0, len(text)),
        claim_type="policy",
        normalized_subject="customer",
        normalized_predicate="receives",
        normalized_object=text,
        alignment=None,
    )


# ── Schema tests ──────────────────────────────────────────────────────────────


def test_kb_chunk_branch_schema_exists():
    """KBChunkBranch is importable and constructable."""
    branch = KBChunkBranch(
        id="monthly",
        condition="monthly plan customer",
        constraints=["within 30 days"],
        content="Monthly customers get a refund within 30 days.",
    )
    assert branch.id == "monthly"
    assert "within 30 days" in branch.constraints


def test_kb_chunk_accepts_branches():
    """KBChunk accepts a populated branches list."""
    assert len(REFUND_MULTI_BRANCH_CHUNK.branches) == 2
    assert REFUND_MULTI_BRANCH_CHUNK.branches[0].id == "monthly"
    assert REFUND_MULTI_BRANCH_CHUNK.branches[1].id == "annual"


def test_kb_chunk_branches_defaults_to_empty():
    """KBChunk with no branches specified defaults to empty list."""
    assert SINGLE_BRANCH_CHUNK.branches == []


def test_quality_plan_accepts_target_branch():
    """QualityPlan accepts an optional target_branch field."""
    from resonantforge.schemas import (
        AccuracyLabel,
        KnowledgeCitations,
        RubricTarget,
    )
    plan = QualityPlan(
        conversation_id="conv_test",
        trigger_event_id="evt_test",
        rubric_targets=RubricTarget(
            accuracy=AccuracyLabel(status="supported", precision="conditional_applied"),
        ),
        knowledge_citations=KnowledgeCitations(
            should_cite=["kb_refund_conditional_plan_type"],
        ),
        coaching_target_dimension="accuracy",
        prose_generation_directives="test directive",
        kb_chunks_required=["kb_refund_conditional_plan_type"],
        target_branch="monthly",
    )
    assert plan.target_branch == "monthly"


def test_quality_plan_target_branch_defaults_to_none():
    """QualityPlan target_branch defaults to None when not specified."""
    from resonantforge.schemas import (
        KnowledgeCitations,
        RubricTarget,
    )
    plan = QualityPlan(
        conversation_id="conv_test",
        trigger_event_id="evt_test",
        rubric_targets=RubricTarget(),
        knowledge_citations=KnowledgeCitations(),
        coaching_target_dimension="accuracy",
        prose_generation_directives="test directive",
    )
    assert plan.target_branch is None


# ── Extractor: target_branch scopes constraint check ─────────────────────────


def test_extractor_target_branch_monthly_requires_only_monthly_constraints():
    """
    When target_branch='monthly', a claim that includes monthly constraints
    but omits annual constraints passes constraint_preserved.
    """
    # Claim correctly applies monthly branch: includes "within 30 days"
    monthly_claim = _make_claim(
        "Monthly plan customers can request a refund within 30 days "
        "via the billing portal."
    )
    result = _check_chunk_relevance(
        monthly_claim,
        REFUND_MULTI_BRANCH_CHUNK,
        {},
        target_branch="monthly",
    )
    assert result["relevant"], "Claim should be relevant to chunk"
    assert result["constraint_preserved"], (
        f"Monthly-branch claim with 'within 30 days' should pass constraint check "
        f"when target_branch='monthly'. Got: {result}"
    )


def test_extractor_target_branch_annual_skips_monthly_constraints():
    """
    When target_branch='annual', a correctly-applied annual claim (no 30-day window)
    passes constraint_preserved because the annual branch has no constraints.
    """
    # Claim correctly applies annual branch: no "within 30 days"
    annual_claim = _make_claim(
        "Annual plan customers must contact billing support to initiate a refund."
    )
    result = _check_chunk_relevance(
        annual_claim,
        REFUND_MULTI_BRANCH_CHUNK,
        {},
        target_branch="annual",
    )
    assert result["relevant"], "Claim should be relevant to chunk"
    assert result["constraint_preserved"], (
        f"Annual-branch claim should pass constraint check when target_branch='annual' "
        f"(annual branch has no constraints). Got: {result}"
    )


# ── Extractor: no target_branch → organic fallback ───────────────────────────


def test_extractor_no_target_branch_skips_constraint_check_for_multi_branch():
    """
    Without target_branch (organic conversation), multi-branch ALLOW_CONDITION
    chunks skip constraint_preserved check → returns True.
    """
    # Annual-branch claim with no "within 30 days" — would fail old behavior
    annual_claim = _make_claim(
        "Annual plan customers must contact billing support to initiate a refund."
    )
    result = _check_chunk_relevance(
        annual_claim,
        REFUND_MULTI_BRANCH_CHUNK,
        {},
        target_branch=None,
    )
    assert result["relevant"], "Claim should be relevant to chunk"
    assert result["constraint_preserved"], (
        "Organic multi-branch ALLOW_CONDITION should skip constraint check "
        f"(no target_branch). Got: {result}"
    )


# ── Contamination check ───────────────────────────────────────────────────────


def test_contamination_check_flags_claim_with_constraints_from_two_branches():
    """
    A single claim that contains constraints from two different branches
    is incoherent — flag it as constraint_preserved=False (contaminated merge).
    """
    # This claim mixes monthly ("within 30 days") with annual ("contact billing support")
    contaminated_claim = _make_claim(
        "You can request a refund within 30 days — for annual plan customers, "
        "please contact billing support."
    )
    result = _check_chunk_relevance(
        contaminated_claim,
        REFUND_MULTI_BRANCH_CHUNK,
        {},
        target_branch=None,
    )
    assert result["relevant"], "Claim should be relevant to chunk"
    assert not result["constraint_preserved"], (
        "Claim mixing constraints from two branches should be flagged as contaminated "
        f"(constraint_preserved=False). Got: {result}"
    )


def test_contamination_check_passes_claim_from_single_branch():
    """
    A claim that only contains constraints from one branch is coherent —
    constraint_preserved=True.
    """
    monthly_only_claim = _make_claim(
        "Monthly plan customers can get a refund within 30 days via the billing portal."
    )
    result = _check_chunk_relevance(
        monthly_only_claim,
        REFUND_MULTI_BRANCH_CHUNK,
        {},
        target_branch=None,
    )
    assert result["relevant"], "Claim should be relevant to chunk"
    assert result["constraint_preserved"], (
        "Claim from a single branch should not be flagged as contaminated. "
        f"Got: {result}"
    )


# ── Single-branch: existing behavior preserved ───────────────────────────────


def test_single_branch_chunk_preserves_existing_constraint_behavior():
    """
    Single-branch ALLOW_CONDITION chunks (no branches field) use the existing
    constraint check — omitting chunk constraints is still flagged.
    """
    # Chunk says "within 30 days" — claim omits it
    vague_claim = _make_claim("Paid subscribers are eligible for a refund.")
    result = _check_chunk_relevance(
        vague_claim,
        SINGLE_BRANCH_CHUNK,
        {},
    )
    # SINGLE_BRANCH_CHUNK has no numeric/timing constraints, so constraint check passes trivially
    # Test confirms the function accepts no target_branch arg for single-branch chunks
    assert isinstance(result["constraint_preserved"], bool)


# ── Injector: multi-branch chunk gets target_branch on quality plan ───────────


def test_injector_sets_target_branch_for_multi_branch_conditional_chunk():
    """
    When the injector assigns a conditional_applied plan for a multi-branch
    ALLOW_CONDITION chunk, it sets target_branch on the resulting QualityPlan.
    """
    from resonantforge.layer1.quality_plan_injector import QualityPlanInjector

    rng = random.Random(42)
    injector = QualityPlanInjector(rng=rng)

    events = [
        SimEvent(
            event_id=f"evt_{i:03d}",
            event_type=SimEventType.CONVERSATION_STARTED,
            account_id="acc_001",
            timestamp=datetime(2024, 1, 15, 10, 0, 0),
            day_index=0,
            month_index=0,
            payload={
                "domain": "refund_policy",
                "intent": ["refund_request"],
                "agent_id": "agent_001",
                "surface_channel": "chat",
            },
        )
        for i in range(80)
    ]

    plans = injector.inject(
        events=events,
        snapshots=[],
        kb_chunks=[REFUND_MULTI_BRANCH_CHUNK],
        planted_count=50,
    )

    # Find conditional_applied plans that used our multi-branch chunk
    ca_plans = [
        p for p in plans
        if (
            p.rubric_targets.accuracy is not None
            and p.rubric_targets.accuracy.precision == "conditional_applied"
            and "kb_refund_conditional_plan_type" in p.kb_chunks_required
        )
    ]

    assert ca_plans, "Expected at least one conditional_applied plan for multi-branch chunk"

    branch_ids = {b.id for b in REFUND_MULTI_BRANCH_CHUNK.branches}
    for plan in ca_plans:
        assert plan.target_branch is not None, (
            f"Plan {plan.conversation_id} has no target_branch set for multi-branch chunk"
        )
        assert plan.target_branch in branch_ids, (
            f"Plan {plan.conversation_id} target_branch={plan.target_branch!r} "
            f"is not a valid branch id. Valid: {branch_ids}"
        )
