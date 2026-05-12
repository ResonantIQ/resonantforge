"""
RFORGE-47 — Tests for _derive_expected_failures() and the updated _write_label_template().

TDD step: RED — all tests should fail until implementation lands.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from resonantforge.schemas import AccuracyLabel, QualityPlan, RubricTarget, KnowledgeCitations
from resonantforge.replay.extractor import _derive_expected_failures, _write_label_template


# ─── Fixtures ────────────────────────────────────────────────────────────────

def _make_plan(
    *,
    empathy: str | None = "high",
    resolution: str | None = "strong",
    brand_voice_target: str | None = "on_brand",
    accuracy_status: str = "supported",
    accuracy_precision: str = "exact",
) -> QualityPlan:
    return QualityPlan(
        conversation_id="conv-test",
        trigger_event_id="evt-1",
        rubric_targets=RubricTarget(
            empathy=empathy,
            resolution=resolution,
            brand_voice_target=brand_voice_target,
            accuracy=AccuracyLabel(status=accuracy_status, precision=accuracy_precision),
        ),
        knowledge_citations=KnowledgeCitations(),
        prose_generation_directives="Test directive.",
    )


# ─── _derive_expected_failures ────────────────────────────────────────────────

class TestDeriveExpectedFailures:
    """Unit tests for the pure function that maps rubric targets → expected_failures dict."""

    def test_all_clean_returns_all_false_and_pass(self):
        plan = _make_plan()
        ef, outcome = _derive_expected_failures(plan)
        assert ef["empathy"] is False
        assert ef["resolution"] is False
        assert ef["brand_voice"] is False
        assert ef["accuracy"] is False
        assert outcome == "pass"

    def test_low_empathy_marks_empathy_true(self):
        plan = _make_plan(empathy="low")
        ef, outcome = _derive_expected_failures(plan)
        assert ef["empathy"] is True
        assert outcome == "fail"

    def test_high_empathy_marks_empathy_false(self):
        plan = _make_plan(empathy="high")
        ef, outcome = _derive_expected_failures(plan)
        assert ef["empathy"] is False

    def test_weak_resolution_marks_resolution_true(self):
        plan = _make_plan(resolution="weak")
        ef, outcome = _derive_expected_failures(plan)
        assert ef["resolution"] is True
        assert outcome == "fail"

    def test_guided_resolution_marks_resolution_false(self):
        plan = _make_plan(resolution="guided")
        ef, outcome = _derive_expected_failures(plan)
        assert ef["resolution"] is False

    def test_strong_resolution_marks_resolution_false(self):
        plan = _make_plan(resolution="strong")
        ef, outcome = _derive_expected_failures(plan)
        assert ef["resolution"] is False

    def test_off_brand_marks_brand_voice_true(self):
        plan = _make_plan(brand_voice_target="off_brand")
        ef, outcome = _derive_expected_failures(plan)
        assert ef["brand_voice"] is True
        assert outcome == "fail"

    def test_on_brand_marks_brand_voice_false(self):
        plan = _make_plan(brand_voice_target="on_brand")
        ef, outcome = _derive_expected_failures(plan)
        assert ef["brand_voice"] is False

    def test_contradicted_marks_accuracy_true(self):
        plan = _make_plan(accuracy_status="contradicted", accuracy_precision="exact")
        ef, outcome = _derive_expected_failures(plan)
        assert ef["accuracy"] is True
        assert outcome == "fail"

    def test_overgeneralized_marks_accuracy_true(self):
        plan = _make_plan(accuracy_status="supported", accuracy_precision="overgeneralized")
        ef, outcome = _derive_expected_failures(plan)
        assert ef["accuracy"] is True
        assert outcome == "fail"

    def test_missing_constraint_marks_accuracy_true(self):
        plan = _make_plan(accuracy_status="supported", accuracy_precision="missing_constraint")
        ef, outcome = _derive_expected_failures(plan)
        assert ef["accuracy"] is True
        assert outcome == "fail"

    def test_condition_missed_marks_accuracy_true(self):
        plan = _make_plan(accuracy_status="supported", accuracy_precision="condition_missed")
        ef, outcome = _derive_expected_failures(plan)
        assert ef["accuracy"] is True
        assert outcome == "fail"

    def test_exact_supported_marks_accuracy_false(self):
        plan = _make_plan(accuracy_status="supported", accuracy_precision="exact")
        ef, outcome = _derive_expected_failures(plan)
        assert ef["accuracy"] is False

    def test_conditional_applied_marks_accuracy_false(self):
        plan = _make_plan(accuracy_status="supported", accuracy_precision="conditional_applied")
        ef, outcome = _derive_expected_failures(plan)
        assert ef["accuracy"] is False

    def test_none_quality_plan_returns_all_false_uncertain(self):
        ef, outcome = _derive_expected_failures(None)
        assert ef["empathy"] is False
        assert ef["resolution"] is False
        assert ef["brand_voice"] is False
        assert ef["accuracy"] is False
        assert outcome == "uncertain"

    def test_claim_extraction_and_layer1_always_false(self):
        """claim_extraction and layer1_signal are not derived from rubric_targets."""
        plan = _make_plan(empathy="low")
        ef, _ = _derive_expected_failures(plan)
        assert ef["claim_extraction"] is False
        assert ef["layer1_signal"] is False

    def test_multiple_failures_all_captured(self):
        plan = _make_plan(
            empathy="low",
            resolution="weak",
            brand_voice_target="off_brand",
            accuracy_status="contradicted",
        )
        ef, outcome = _derive_expected_failures(plan)
        assert ef["empathy"] is True
        assert ef["resolution"] is True
        assert ef["brand_voice"] is True
        assert ef["accuracy"] is True
        assert outcome == "fail"


# ─── _write_label_template with quality_plan ─────────────────────────────────

class TestWriteLabelTemplateWithQualityPlan:
    """Integration tests: _write_label_template must persist derived labels to disk."""

    def test_write_label_with_failing_plan_persists_failures(self):
        plan = _make_plan(empathy="low", resolution="weak")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            _write_label_template(out, "conv-1", quality_plan=plan)
            data = json.loads((out / "labels.json").read_text())
        assert data["expected_failures"]["empathy"] is True
        assert data["expected_failures"]["resolution"] is True
        assert data["expected_outcome"] == "fail"

    def test_write_label_with_clean_plan_has_all_false(self):
        plan = _make_plan()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            _write_label_template(out, "conv-2", quality_plan=plan)
            data = json.loads((out / "labels.json").read_text())
        assert all(v is False for v in data["expected_failures"].values())
        assert data["expected_outcome"] == "pass"

    def test_write_label_without_plan_stays_uncertain(self):
        """Passing quality_plan=None preserves the old uncertain/all-false behavior."""
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            _write_label_template(out, "conv-3", quality_plan=None)
            data = json.loads((out / "labels.json").read_text())
        assert data["expected_outcome"] == "uncertain"
        assert all(v is False for v in data["expected_failures"].values())

    def test_existing_label_not_overwritten(self):
        plan = _make_plan(empathy="low")
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            # Pre-write a human-authored label
            existing = {"expected_outcome": "pass", "expected_failures": {"empathy": False}}
            (out / "labels.json").write_text(json.dumps(existing))
            _write_label_template(out, "conv-4", quality_plan=plan)
            data = json.loads((out / "labels.json").read_text())
        # Must still show original human-authored content
        assert data["expected_outcome"] == "pass"
        assert data["expected_failures"]["empathy"] is False
