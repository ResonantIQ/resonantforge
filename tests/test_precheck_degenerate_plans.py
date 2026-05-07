"""
Tests for RFORGE-36: pre-prompt guard for degenerate plans.

An overgeneralized or contradicted:exact plan with empty kb_chunks_required is
inherently invalid — the LLM has no policy content to work from and will exhaust
all 3 retries with no chance of passing. pre_prompt_validate should fail-fast on
these rather than wasting API calls.

Covers:
  - overgeneralized + empty kb_chunks_required → pre_prompt_validate returns FAIL
  - contradicted:exact + empty kb_chunks_required → pre_prompt_validate returns FAIL
  - overgeneralized + populated kb_chunks_required → pre_prompt_validate returns PASS
  - contradicted:exact + populated kb_chunks_required → pre_prompt_validate returns PASS
  - Other precisions with empty kb_chunks_required → pre_prompt_validate not affected
"""
from __future__ import annotations

from datetime import date, datetime

from resonantforge.layer1.plan_validator import PlanValidator
from resonantforge.schemas import (
    AccuracyLabel,
    DaySnapshot,
    KnowledgeCitations,
    LifecycleStage,
    QualityPlan,
    RubricTarget,
    SimEvent,
    SimEventType,
)
from resonantforge.validators.rule_engine import ValidationVerdict


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _make_event(event_id: str = "evt_001") -> SimEvent:
    return SimEvent(
        event_id=event_id,
        event_type=SimEventType.CONVERSATION_STARTED,
        account_id="acc_001",
        timestamp=datetime(2024, 1, 15, 10, 0, 0),
        day_index=0,
        month_index=0,
        payload={
            "domain": "billing",
            "intent": ["billing_inquiry"],
            "agent_id": "agent_001",
            "surface_channel": "chat",
        },
    )


def _make_snapshot(lifecycle: LifecycleStage = LifecycleStage.ACTIVE) -> DaySnapshot:
    return DaySnapshot(
        snapshot_id="snap_001",
        account_id="acc_001",
        day_index=0,
        month_index=0,
        date=date(2024, 1, 15),
        lifecycle_stage=lifecycle,
        health_state="healthy",
        health_score=0.8,
        open_tickets=0,
        recent_signals=[],
        active_agents=[],
        payment_status="current",
    )


def _make_plan(
    precision: str,
    status: str = "supported",
    kb_chunks_required: list[str] | None = None,
    event_id: str = "evt_001",
) -> QualityPlan:
    return QualityPlan(
        conversation_id=f"conv_{event_id}",
        trigger_event_id=event_id,
        rubric_targets=RubricTarget(
            accuracy=AccuracyLabel(status=status, precision=precision),
        ),
        knowledge_citations=KnowledgeCitations(),
        coaching_target_dimension="accuracy",
        prose_generation_directives="some directive",
        kb_chunks_required=kb_chunks_required or [],
    )


# ── overgeneralized + empty pool → FAIL ──────────────────────────────────────


def test_precheck_fails_overgeneralized_with_empty_kb_chunks():
    """
    An overgeneralized plan with empty kb_chunks_required is degenerate —
    the LLM has no policy content to over-generalize from.
    pre_prompt_validate must return FAIL immediately.
    """
    validator = PlanValidator()
    plan = _make_plan(precision="overgeneralized", kb_chunks_required=[])
    event = _make_event()
    snapshot = _make_snapshot()

    result = validator.pre_prompt_validate(plan, snapshot, [event])

    assert result.overall_verdict == ValidationVerdict.FAIL, (
        f"Expected FAIL for overgeneralized plan with empty kb_chunks_required, "
        f"got {result.overall_verdict}"
    )
    assert result.skip_reason, "FAIL result must include a skip_reason"


# ── contradicted:exact + empty pool → FAIL ───────────────────────────────────


def test_precheck_fails_contradicted_exact_with_empty_kb_chunks():
    """
    A contradicted:exact plan with empty kb_chunks_required and no planted
    contradiction is degenerate — alignment=contradicted is unreliable without
    a specific KB fact to contradict.
    pre_prompt_validate must return FAIL immediately.
    """
    validator = PlanValidator()
    plan = _make_plan(precision="exact", status="contradicted", kb_chunks_required=[])
    event = _make_event()
    snapshot = _make_snapshot()

    result = validator.pre_prompt_validate(plan, snapshot, [event])

    assert result.overall_verdict == ValidationVerdict.FAIL, (
        f"Expected FAIL for contradicted:exact plan with empty kb_chunks_required, "
        f"got {result.overall_verdict}"
    )
    assert result.skip_reason, "FAIL result must include a skip_reason"


# ── overgeneralized + populated pool → PASS ──────────────────────────────────


def test_precheck_passes_overgeneralized_with_populated_kb_chunks():
    """
    An overgeneralized plan with kb_chunks_required populated is valid —
    the LLM has content to work from. pre_prompt_validate must return PASS.
    """
    validator = PlanValidator()
    plan = _make_plan(
        precision="overgeneralized",
        kb_chunks_required=["kb_refund_policy"],
    )
    event = _make_event()
    snapshot = _make_snapshot()

    result = validator.pre_prompt_validate(plan, snapshot, [event])

    assert result.overall_verdict == ValidationVerdict.PASS, (
        f"Expected PASS for overgeneralized plan with kb_chunks_required populated, "
        f"got {result.overall_verdict} (skip_reason={result.skip_reason!r})"
    )


# ── contradicted:exact + populated pool → PASS ───────────────────────────────


def test_precheck_passes_contradicted_exact_with_populated_kb_chunks():
    """
    A contradicted:exact plan with kb_chunks_required populated is valid.
    pre_prompt_validate must return PASS.
    """
    validator = PlanValidator()
    plan = _make_plan(
        precision="exact",
        status="contradicted",
        kb_chunks_required=["kb_refund_policy"],
    )
    event = _make_event()
    snapshot = _make_snapshot()

    result = validator.pre_prompt_validate(plan, snapshot, [event])

    assert result.overall_verdict == ValidationVerdict.PASS, (
        f"Expected PASS for contradicted:exact plan with kb_chunks_required populated, "
        f"got {result.overall_verdict} (skip_reason={result.skip_reason!r})"
    )


# ── supported:exact + empty pool → PASS (not affected) ───────────────────────


def test_precheck_does_not_affect_exact_supported_with_empty_kb_chunks():
    """
    The guard is only for overgeneralized and contradicted:exact. A
    supported:exact plan with empty kb_chunks_required passes the new check
    (other validators may still flag it, but not this guard).
    """
    validator = PlanValidator()
    plan = _make_plan(precision="exact", status="supported", kb_chunks_required=[])
    event = _make_event()
    snapshot = _make_snapshot()

    result = validator.pre_prompt_validate(plan, snapshot, [event])

    assert result.overall_verdict == ValidationVerdict.PASS, (
        f"supported:exact with empty kb_chunks should not be blocked by degenerate-plan guard, "
        f"got {result.overall_verdict}"
    )
