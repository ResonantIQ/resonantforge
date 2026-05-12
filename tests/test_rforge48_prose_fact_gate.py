"""
RFORGE-48 — Negative-target conversations must not count against prose_fact_rate gate.

TDD step: RED — tests fail until SkipRateTracker and pipeline.py are updated.
"""
from __future__ import annotations

import pytest

from resonantforge.layer1.plan_validator import SkipRateTracker
from resonantforge.schemas import AccuracyLabel, QualityPlan, RubricTarget, KnowledgeCitations


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _positive_plan() -> QualityPlan:
    return QualityPlan(
        conversation_id="conv-pos",
        trigger_event_id="evt-pos",
        rubric_targets=RubricTarget(
            empathy="high",
            resolution="strong",
            brand_voice_target="on_brand",
            accuracy=AccuracyLabel(status="supported", precision="exact"),
        ),
        knowledge_citations=KnowledgeCitations(),
        prose_generation_directives="Positive target.",
    )


def _negative_empathy_plan() -> QualityPlan:
    return QualityPlan(
        conversation_id="conv-neg-empathy",
        trigger_event_id="evt-neg",
        rubric_targets=RubricTarget(
            empathy="low",
            resolution="strong",
            brand_voice_target="on_brand",
            accuracy=AccuracyLabel(status="supported", precision="exact"),
        ),
        knowledge_citations=KnowledgeCitations(),
        prose_generation_directives="Negative empathy target.",
    )


def _negative_accuracy_plan() -> QualityPlan:
    return QualityPlan(
        conversation_id="conv-neg-accuracy",
        trigger_event_id="evt-neg-acc",
        rubric_targets=RubricTarget(
            empathy="high",
            resolution="strong",
            brand_voice_target="on_brand",
            accuracy=AccuracyLabel(status="contradicted", precision="exact"),
        ),
        knowledge_citations=KnowledgeCitations(),
        prose_generation_directives="Negative accuracy target.",
    )


def _negative_resolution_plan() -> QualityPlan:
    return QualityPlan(
        conversation_id="conv-neg-resolution",
        trigger_event_id="evt-neg-res",
        rubric_targets=RubricTarget(
            empathy="high",
            resolution="weak",
            brand_voice_target="on_brand",
            accuracy=AccuracyLabel(status="supported", precision="exact"),
        ),
        knowledge_citations=KnowledgeCitations(),
        prose_generation_directives="Negative resolution target.",
    )


def _negative_brand_voice_plan() -> QualityPlan:
    return QualityPlan(
        conversation_id="conv-neg-bv",
        trigger_event_id="evt-neg-bv",
        rubric_targets=RubricTarget(
            empathy="high",
            resolution="strong",
            brand_voice_target="off_brand",
            accuracy=AccuracyLabel(status="supported", precision="exact"),
        ),
        knowledge_citations=KnowledgeCitations(),
        prose_generation_directives="Negative brand voice target.",
    )


# ─── SkipRateTracker structure ────────────────────────────────────────────────

def test_skip_rate_tracker_has_negative_prose_fact_counters() -> None:
    """SkipRateTracker must expose negative_prose_fact_attempts and negative_prose_fact_failures."""
    tracker = SkipRateTracker()
    assert hasattr(tracker, "negative_prose_fact_attempts"), (
        "SkipRateTracker must have negative_prose_fact_attempts field"
    )
    assert hasattr(tracker, "negative_prose_fact_failures"), (
        "SkipRateTracker must have negative_prose_fact_failures field"
    )
    assert tracker.negative_prose_fact_attempts == 0
    assert tracker.negative_prose_fact_failures == 0


def test_negative_prose_fact_rate_property() -> None:
    """negative_prose_fact_rate property returns ratio of negative failures to attempts."""
    tracker = SkipRateTracker(
        negative_prose_fact_attempts=10,
        negative_prose_fact_failures=4,
    )
    assert tracker.negative_prose_fact_rate == pytest.approx(0.4)


def test_negative_prose_fact_rate_zero_when_no_attempts() -> None:
    """negative_prose_fact_rate returns 0.0 when negative_prose_fact_attempts == 0."""
    tracker = SkipRateTracker()
    assert tracker.negative_prose_fact_rate == 0.0


# ─── Gate isolation ───────────────────────────────────────────────────────────

def test_check_gates_ignores_negative_counters() -> None:
    """check_gates() must not fire on negative_prose_fact counters, even at 100% failure rate."""
    tracker = SkipRateTracker(
        negative_prose_fact_attempts=100,
        negative_prose_fact_failures=100,  # 100% negative failure rate — gate must NOT fire
        prose_fact_attempts=5,
        prose_fact_failures=0,             # positive rate = 0% — gate healthy
    )
    violations = tracker.check_gates()
    prose_violations = [v for v in violations if v.gate_name == "prose_fact_rate"]
    assert len(prose_violations) == 0, (
        f"check_gates() must not fire prose_fact_rate gate based on negative counters; "
        f"got violations: {prose_violations}"
    )


def test_check_gates_fires_on_positive_counters_only() -> None:
    """check_gates() fires prose_fact_rate gate based on positive counters exceeding threshold."""
    tracker = SkipRateTracker(
        prose_fact_attempts=10,
        prose_fact_failures=2,             # 20% positive failure rate — gate MUST fire
        negative_prose_fact_attempts=100,
        negative_prose_fact_failures=0,    # zero negative failures — irrelevant
    )
    violations = tracker.check_gates()
    prose_violations = [v for v in violations if v.gate_name == "prose_fact_rate"]
    assert len(prose_violations) == 1, (
        f"check_gates() must fire prose_fact_rate gate when positive rate=0.20 (>0.10); "
        f"got {len(prose_violations)} violations"
    )


# ─── _quality_plan_is_negative helper ────────────────────────────────────────

def test_quality_plan_is_negative_returns_false_for_positive_plan() -> None:
    """_quality_plan_is_negative returns False when all rubric targets are positive."""
    from resonantforge.pipeline import _quality_plan_is_negative
    assert _quality_plan_is_negative(_positive_plan()) is False


def test_quality_plan_is_negative_returns_false_for_none() -> None:
    """_quality_plan_is_negative returns False when quality_plan is None (organic conv)."""
    from resonantforge.pipeline import _quality_plan_is_negative
    assert _quality_plan_is_negative(None) is False


def test_quality_plan_is_negative_empathy_low() -> None:
    """_quality_plan_is_negative returns True when empathy target is 'low'."""
    from resonantforge.pipeline import _quality_plan_is_negative
    assert _quality_plan_is_negative(_negative_empathy_plan()) is True


def test_quality_plan_is_negative_accuracy_contradicted() -> None:
    """_quality_plan_is_negative returns True when accuracy status is 'contradicted'."""
    from resonantforge.pipeline import _quality_plan_is_negative
    assert _quality_plan_is_negative(_negative_accuracy_plan()) is True


def test_quality_plan_is_negative_resolution_weak() -> None:
    """_quality_plan_is_negative returns True when resolution target is 'weak'."""
    from resonantforge.pipeline import _quality_plan_is_negative
    assert _quality_plan_is_negative(_negative_resolution_plan()) is True


def test_quality_plan_is_negative_brand_voice_off_brand() -> None:
    """_quality_plan_is_negative returns True when brand_voice_target is 'off_brand'."""
    from resonantforge.pipeline import _quality_plan_is_negative
    assert _quality_plan_is_negative(_negative_brand_voice_plan()) is True


def test_quality_plan_is_negative_accuracy_precision_overgeneralized() -> None:
    """_quality_plan_is_negative returns True when accuracy precision is 'overgeneralized'."""
    from resonantforge.pipeline import _quality_plan_is_negative
    plan = QualityPlan(
        conversation_id="conv-neg-prec",
        trigger_event_id="evt-neg-prec",
        rubric_targets=RubricTarget(
            empathy="high",
            resolution="strong",
            brand_voice_target="on_brand",
            accuracy=AccuracyLabel(status="supported", precision="overgeneralized"),
        ),
        knowledge_citations=KnowledgeCitations(),
        prose_generation_directives="Negative precision target.",
    )
    assert _quality_plan_is_negative(plan) is True
