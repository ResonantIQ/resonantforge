"""
Tests for resonantforge.replay.agreement.compute_agreement.
"""
from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from resonantforge.replay.agreement import compute_agreement
from resonantforge.replay.schemas import ReplayLabels
from resonantforge.schemas import DimensionVerdict, ValidationVerdict


def _make_verdict(dimension: str, verdict: str) -> DimensionVerdict:
    return DimensionVerdict(
        dimension=dimension,
        verdict=ValidationVerdict(verdict),
        target="test",
        signals_summary={},
    )


def _make_labels(expected_outcome: str, expected_failures: dict) -> ReplayLabels:
    return ReplayLabels(
        schema_version=1,
        conv_id="test",
        label_version=1,
        labeled_at="2026-05-05",
        labeled_by="tj",
        expected_outcome=expected_outcome,
        expected_failures={
            "accuracy": False,
            "empathy": False,
            "resolution": False,
            "brand_voice": False,
            "claim_extraction": False,
            "layer1_signal": False,
            **expected_failures,
        },
        confidence="high",
        tags=[],
        rationale="test rationale",
        revised_from=None,
        revision_notes=None,
    )


class TestComputeAgreement:
    def test_perfect_pass_agreement(self) -> None:
        verdicts = [_make_verdict("empathy", "pass"), _make_verdict("resolution", "pass")]
        labels = _make_labels("pass", {})
        result = compute_agreement(verdicts, True, "pass", labels)
        assert result.outcome_match is True
        assert result.rule_match is True
        assert result.false_positive_rules == []
        assert result.missed_rules == []

    def test_perfect_fail_agreement(self) -> None:
        verdicts = [_make_verdict("empathy", "fail")]
        labels = _make_labels("fail", {"empathy": True})
        result = compute_agreement(verdicts, True, "fail", labels)
        assert result.outcome_match is True
        assert result.rule_match is True

    def test_outcome_mismatch(self) -> None:
        verdicts = [_make_verdict("empathy", "pass")]
        labels = _make_labels("fail", {"empathy": True})
        result = compute_agreement(verdicts, True, "pass", labels)
        assert result.outcome_match is False

    def test_uncertain_outcome_match_is_none(self) -> None:
        verdicts = [_make_verdict("empathy", "pass")]
        labels = _make_labels("uncertain", {})
        result = compute_agreement(verdicts, True, "pass", labels)
        assert result.outcome_match is None

    def test_false_positive_detected(self) -> None:
        # Validator fails empathy but label says empathy should pass
        verdicts = [_make_verdict("empathy", "fail")]
        labels = _make_labels("pass", {})  # label says all pass
        result = compute_agreement(verdicts, True, "fail", labels)
        assert "empathy" in result.false_positive_rules
        assert result.rule_match is False

    def test_missed_rule_detected(self) -> None:
        # Label says resolution should fail but validator passes it
        verdicts = [_make_verdict("resolution", "pass")]
        labels = _make_labels("fail", {"resolution": True})
        result = compute_agreement(verdicts, True, "pass", labels)
        assert "resolution" in result.missed_rules
        assert result.rule_match is False

    def test_skip_verdicts_not_counted_as_fail(self) -> None:
        # Skip verdicts are neutral — don't add to validator_failed
        verdicts = [_make_verdict("accuracy", "skip")]
        labels = _make_labels("pass", {})
        result = compute_agreement(verdicts, True, "pass", labels)
        assert result.rule_match is True
        assert result.false_positive_rules == []

    def test_claim_extraction_failure_adds_rule(self) -> None:
        verdicts = []
        labels = _make_labels("fail", {"claim_extraction": True})
        result = compute_agreement(verdicts, False, "fail", labels)  # claim_extraction_ok=False
        assert result.rule_match is True
        assert "claim_extraction" not in result.false_positive_rules

    def test_false_positive_and_missed_rules_are_sorted(self) -> None:
        # Multiple diverging rules — output should be sorted for determinism
        verdicts = [
            _make_verdict("resolution", "fail"),
            _make_verdict("brand_voice", "fail"),
        ]
        labels = _make_labels("fail", {"empathy": True})
        result = compute_agreement(verdicts, True, "fail", labels)
        assert result.false_positive_rules == sorted(result.false_positive_rules)
        assert result.missed_rules == sorted(result.missed_rules)
