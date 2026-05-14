"""
Tests for distractor planting (RFORGE-73).

Covers:
  - QualityPlan.is_distractor field exists
  - AccuracyAnswerDetail.is_distractor_trap field exists
  - "distractor" in VALID_TAGS
  - _build_schedule produces distractor slots within budget
  - _derive_answer_key propagates is_distractor_trap
  - _write_label_template tags "distractor" for distractor plans
  - write_single_envelope produces answer_key with is_distractor_trap
  - format_summary includes distractor FP rate when distractors present
"""
from __future__ import annotations

import json
from pathlib import Path
import pytest

from resonantforge.layer1.quality_plan_injector import QualityPlanInjector
from resonantforge.replay.diff import format_summary
from resonantforge.replay.extractor import _derive_answer_key, _write_label_template, write_single_envelope
from resonantforge.replay.schemas import (
    VALID_TAGS,
    AccuracyAnswerDetail,
    AgreementResult,
    AnswerKey,
    BrandVoiceConfig,
    EnvelopeLexicons,
    EnvelopeMetadata,
    ReplayLabels,
    ReplayResult,
)
from resonantforge.schemas import (
    AccuracyLabel,
    KnowledgeCitations,
    PlantedContradiction,
    QualityPlan,
    RubricTarget,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_distractor_plan(conv_id: str = "conv_distractor_001") -> QualityPlan:
    return QualityPlan(
        conversation_id=conv_id,
        trigger_event_id=f"evt_{conv_id}",
        rubric_targets=RubricTarget(
            accuracy=AccuracyLabel(status="supported", precision="exact")
        ),
        knowledge_citations=KnowledgeCitations(should_cite=["kb_chunk_1"]),
        prose_generation_directives="State the KB fact correctly using your own words.",
        coaching_target_dimension="accuracy",
        kb_chunks_required=["kb_chunk_1"],
        is_distractor=True,
    )


def _make_normal_plan(conv_id: str = "conv_normal_001") -> QualityPlan:
    return QualityPlan(
        conversation_id=conv_id,
        trigger_event_id=f"evt_{conv_id}",
        rubric_targets=RubricTarget(
            accuracy=AccuracyLabel(status="contradicted", precision="exact")
        ),
        knowledge_citations=KnowledgeCitations(should_cite=["kb_chunk_1"]),
        prose_generation_directives="Contradict the KB.",
        coaching_target_dimension="accuracy",
        kb_chunks_required=["kb_chunk_1"],
        is_distractor=False,
    )


def _make_labels(conv_id: str, tags: list[str], expected_failures: dict | None = None) -> ReplayLabels:
    return ReplayLabels(
        schema_version=1,
        conv_id=conv_id,
        label_version=1,
        labeled_at="2026-05-14",
        labeled_by="tj",
        expected_outcome="pass",
        expected_failures={
            "accuracy": False,
            "empathy": False,
            "resolution": False,
            "brand_voice": False,
            "claim_extraction": False,
            "layer1_signal": False,
            **(expected_failures or {}),
        },
        confidence="high",
        tags=tags,
        rationale="Distractor test fixture.",
    )


def _make_replay_result(
    conv_id: str,
    *,
    is_distractor: bool,
    accuracy_fp: bool = False,
    outcome: str = "pass",
) -> ReplayResult:
    tags = ["distractor"] if is_distractor else ["planted"]
    labels = _make_labels(conv_id, tags=tags)
    false_positive_rules = ["accuracy"] if accuracy_fp else []
    return ReplayResult(
        conv_id=conv_id,
        dimension_verdicts=[],
        claim_extraction_ok=True,
        overall_outcome=outcome,
        agreement=AgreementResult(
            outcome_match=not accuracy_fp,
            rule_match=not accuracy_fp,
            false_positive_rules=false_positive_rules,
            missed_rules=[],
        ),
        envelope_metadata=EnvelopeMetadata(conv_id=conv_id),
        labels=labels,
    )


def _minimal_envelope_args(tmp_path: Path, conv_id: str, *, quality_plan: QualityPlan) -> dict:
    return dict(
        output_dir=tmp_path / conv_id,
        conv_id=conv_id,
        agent_prose="The refund policy is within five business days.",
        customer_prose="Can I get a refund?",
        quality_plan=quality_plan,
        kb_chunks=[],
        lexicons=EnvelopeLexicons(),
        brand_voice=BrandVoiceConfig(
            variant_id="bv_test",
            feature_profiles={
                "bv_test": {
                    "sentence_length_range": [1, 200],
                    "question_count_range": [0, 20],
                    "hedging_range": [0, 20],
                    "directive_range": [0, 20],
                    "aligned_term_field": "warm_terms_count",
                    "aligned_terms_min": 0,
                    "formality_range": [0.0, 1.0],
                }
            },
        ),
        extracted_claims=[],
        pipeline_version="0.0.0",
        kb_version="sha256:test",
        source_corpus="test",
    )


# ---------------------------------------------------------------------------
# Schema fields
# ---------------------------------------------------------------------------

class TestSchemaFields:
    def test_quality_plan_has_is_distractor_field(self) -> None:
        plan = _make_distractor_plan()
        assert plan.is_distractor is True

    def test_quality_plan_is_distractor_defaults_false(self) -> None:
        plan = _make_normal_plan()
        assert plan.is_distractor is False

    def test_accuracy_answer_detail_has_is_distractor_trap(self) -> None:
        detail = AccuracyAnswerDetail(
            status="supported",
            precision="exact",
            is_distractor_trap=True,
        )
        assert detail.is_distractor_trap is True

    def test_accuracy_answer_detail_is_distractor_trap_defaults_false(self) -> None:
        detail = AccuracyAnswerDetail(status="supported", precision="exact")
        assert detail.is_distractor_trap is False

    def test_distractor_in_valid_tags(self) -> None:
        assert "distractor" in VALID_TAGS


# ---------------------------------------------------------------------------
# _build_schedule: distractor slots
# ---------------------------------------------------------------------------

class TestBuildScheduleDistractors:
    def _injector(self) -> QualityPlanInjector:
        import random
        return QualityPlanInjector(rng=random.Random(42))

    def test_schedule_contains_distractor_slots(self) -> None:
        inj = self._injector()
        schedule = inj._build_schedule(50, distractor_count=2)
        distractor_slots = [s for s in schedule if s["type"] == "distractor"]
        assert len(distractor_slots) == 2

    def test_schedule_total_length_unchanged(self) -> None:
        inj = self._injector()
        schedule_without = inj._build_schedule(50, distractor_count=0)
        schedule_with = inj._build_schedule(50, distractor_count=2)
        assert len(schedule_with) == len(schedule_without)

    def test_schedule_distractor_count_zero_produces_no_distractor_slots(self) -> None:
        inj = self._injector()
        schedule = inj._build_schedule(50, distractor_count=0)
        assert not any(s["type"] == "distractor" for s in schedule)

    def test_schedule_distractor_capped_by_accuracy_budget(self) -> None:
        """When accuracy budget is smaller than distractor_count, clamp."""
        inj = self._injector()
        # planted_count=6: 2 controls, 4 remaining, per_dim=1, leftover=0
        # accuracy budget=1; requesting 2 distractors → clamp to 1
        schedule = inj._build_schedule(6, distractor_count=2)
        distractor_slots = [s for s in schedule if s["type"] == "distractor"]
        assert len(distractor_slots) == 1
        assert len(schedule) == 6

    def test_default_distractor_count_is_2(self) -> None:
        inj = self._injector()
        schedule = inj._build_schedule(50)
        distractor_slots = [s for s in schedule if s["type"] == "distractor"]
        assert len(distractor_slots) == 2


# ---------------------------------------------------------------------------
# _derive_answer_key: is_distractor_trap
# ---------------------------------------------------------------------------

class TestDeriveAnswerKeyDistractor:
    def test_distractor_plan_sets_is_distractor_trap(self) -> None:
        plan = _make_distractor_plan()
        key = _derive_answer_key("conv_distractor_001", plan)
        assert key.accuracy_detail is not None
        assert key.accuracy_detail.is_distractor_trap is True

    def test_non_distractor_plan_is_distractor_trap_false(self) -> None:
        plan = _make_normal_plan()
        key = _derive_answer_key("conv_normal_001", plan)
        assert key.accuracy_detail is not None
        assert key.accuracy_detail.is_distractor_trap is False

    def test_distractor_plan_expected_failures_accuracy_false(self) -> None:
        plan = _make_distractor_plan()
        key = _derive_answer_key("conv_distractor_001", plan)
        assert key.expected_failures["accuracy"] is False
        assert key.expected_outcome == "pass"

    def test_distractor_plan_severity_none(self) -> None:
        """Distractor is a pass case — severity must be None."""
        plan = _make_distractor_plan()
        key = _derive_answer_key("conv_distractor_001", plan)
        assert key.severity is None


# ---------------------------------------------------------------------------
# _write_label_template: distractor tag
# ---------------------------------------------------------------------------

class TestWriteLabelTemplateDistractor:
    def test_distractor_plan_tagged_in_labels(self, tmp_path: Path) -> None:
        plan = _make_distractor_plan()
        _write_label_template(tmp_path, "conv_distractor_001", quality_plan=plan)
        raw = json.loads((tmp_path / "labels.json").read_text(encoding="utf-8"))
        assert "distractor" in raw["tags"]

    def test_non_distractor_plan_not_tagged(self, tmp_path: Path) -> None:
        plan = _make_normal_plan()
        _write_label_template(tmp_path, "conv_normal_001", quality_plan=plan)
        raw = json.loads((tmp_path / "labels.json").read_text(encoding="utf-8"))
        assert "distractor" not in raw["tags"]


# ---------------------------------------------------------------------------
# write_single_envelope: answer_key has is_distractor_trap
# ---------------------------------------------------------------------------

class TestWriteSingleEnvelopeDistractor:
    def test_distractor_answer_key_has_is_distractor_trap(self, tmp_path: Path) -> None:
        plan = _make_distractor_plan()
        write_single_envelope(**_minimal_envelope_args(tmp_path, "conv_distractor_001", quality_plan=plan))
        raw = json.loads(
            (tmp_path / "conv_distractor_001" / "answer_key.json").read_text(encoding="utf-8")
        )
        key = AnswerKey.model_validate(raw)
        assert key.accuracy_detail is not None
        assert key.accuracy_detail.is_distractor_trap is True

    def test_normal_answer_key_has_is_distractor_trap_false(self, tmp_path: Path) -> None:
        plan = _make_normal_plan()
        write_single_envelope(**_minimal_envelope_args(tmp_path, "conv_normal_001", quality_plan=plan))
        raw = json.loads(
            (tmp_path / "conv_normal_001" / "answer_key.json").read_text(encoding="utf-8")
        )
        key = AnswerKey.model_validate(raw)
        assert key.accuracy_detail is not None
        assert key.accuracy_detail.is_distractor_trap is False


# ---------------------------------------------------------------------------
# format_summary: distractor FP rate
# ---------------------------------------------------------------------------

class TestFormatSummaryDistractor:
    def test_distractor_fp_rate_shown_when_distractors_present(self) -> None:
        results = [
            _make_replay_result("conv_distractor_001", is_distractor=True, accuracy_fp=True),
            _make_replay_result("conv_distractor_002", is_distractor=True, accuracy_fp=False),
            _make_replay_result("conv_planted_001", is_distractor=False, accuracy_fp=False),
        ]
        summary = format_summary(results)
        assert "Distractor FP rate" in summary
        assert "1/2" in summary

    def test_no_distractor_section_when_no_distractors(self) -> None:
        results = [
            _make_replay_result("conv_planted_001", is_distractor=False, accuracy_fp=False),
        ]
        summary = format_summary(results)
        assert "Distractor FP rate" not in summary

    def test_zero_distractor_fp_shown_correctly(self) -> None:
        results = [
            _make_replay_result("conv_distractor_001", is_distractor=True, accuracy_fp=False),
            _make_replay_result("conv_distractor_002", is_distractor=True, accuracy_fp=False),
        ]
        summary = format_summary(results)
        assert "Distractor FP rate" in summary
        assert "0/2" in summary
