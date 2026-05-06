"""
Tests for resonantforge.replay.diff: format_conv_line and format_summary.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from resonantforge.replay.diff import format_conv_line, format_summary
from resonantforge.replay.schemas import AgreementResult, ReplayLabels, ReplayResult, EnvelopeMetadata
from tests.replay_fixtures.builder import build_fixtures, _labels


def _make_labels_obj(**kwargs) -> ReplayLabels:
    base = _labels(
        "test_conv",
        expected_outcome=kwargs.get("expected_outcome", "pass"),
        expected_failures=kwargs.get("expected_failures", {}),
        confidence=kwargs.get("confidence", "high"),
        tags=kwargs.get("tags", []),
        rationale=kwargs.get("rationale", "test"),
    )
    return ReplayLabels.model_validate(base)


def _make_result(
    conv_id: str = "conv_test",
    overall_outcome: str = "pass",
    outcome_match: bool | None = True,
    rule_match: bool = True,
    false_positive_rules: list | None = None,
    missed_rules: list | None = None,
) -> ReplayResult:
    agreement = AgreementResult(
        outcome_match=outcome_match,
        rule_match=rule_match,
        false_positive_rules=false_positive_rules or [],
        missed_rules=missed_rules or [],
    )
    return ReplayResult(
        conv_id=conv_id,
        dimension_verdicts=[],
        claim_extraction_ok=True,
        overall_outcome=overall_outcome,  # type: ignore[arg-type]
        agreement=agreement,
        envelope_metadata=EnvelopeMetadata(conv_id=conv_id),
        labels=_make_labels_obj(),
    )


class TestFormatConvLine:
    def test_pass_conv_shows_pass(self) -> None:
        result = _make_result("conv_abc", overall_outcome="pass")
        line = format_conv_line(result)
        assert "[PASS]" in line
        assert "conv_abc" in line

    def test_fail_conv_shows_fail(self) -> None:
        result = _make_result("conv_xyz", overall_outcome="fail")
        line = format_conv_line(result)
        assert "[FAIL]" in line

    def test_outcome_match_shown(self) -> None:
        result = _make_result(outcome_match=True)
        line = format_conv_line(result)
        assert "outcome_match=Y" in line

    def test_outcome_mismatch_shown(self) -> None:
        result = _make_result(outcome_match=False)
        line = format_conv_line(result)
        assert "outcome_match=N" in line

    def test_uncertain_shown(self) -> None:
        result = _make_result(outcome_match=None)
        line = format_conv_line(result)
        assert "outcome_match=uncertain" in line

    def test_false_positive_rules_shown(self) -> None:
        result = _make_result(false_positive_rules=["empathy", "resolution"])
        line = format_conv_line(result)
        assert "empathy" in line
        assert "resolution" in line

    def test_no_divergence_shown_as_dash(self) -> None:
        result = _make_result()
        line = format_conv_line(result)
        assert "fp=-" in line
        assert "miss=-" in line


class TestFormatSummary:
    def test_empty_results(self) -> None:
        assert format_summary([]) == "No results."

    def test_all_pass_summary(self) -> None:
        results = [_make_result(f"conv_{i}") for i in range(3)]
        summary = format_summary(results)
        assert "3 conversation(s)" in summary
        assert "3/3" in summary

    def test_uncertain_excluded_from_rates(self) -> None:
        results = [
            _make_result("c1", outcome_match=True),
            _make_result("c2", outcome_match=None),  # uncertain
        ]
        summary = format_summary(results)
        # Only 1 certain conv, not 2
        assert "1/1" in summary

    def test_diverging_rules_listed(self) -> None:
        results = [
            _make_result("c1", false_positive_rules=["empathy"]),
            _make_result("c2", missed_rules=["resolution"]),
        ]
        summary = format_summary(results)
        assert "empathy" in summary
        assert "resolution" in summary

    def test_perfect_summary_message(self) -> None:
        results = [_make_result()]
        summary = format_summary(results)
        assert "All rules matched labels exactly." in summary
