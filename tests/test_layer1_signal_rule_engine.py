"""
Tests for the layer1_signal rule engine validator (RFORGE-21).

Covers:
  - DECLINING health + distress_signals_present=True → PASS
  - DECLINING health + distress_signals_present=False → FAIL
  - CRITICAL health + distress_signals_present=True → PASS
  - CRITICAL health + distress_signals_present=False → FAIL
  - HEALTHY health → SKIP (regardless of distress)
  - CHURNED health → SKIP
"""
from __future__ import annotations

import pytest

from resonantforge.schemas import Layer1SignalSignals, ValidationVerdict


def _signals(
    health_state: str,
    *,
    signal_expected: bool,
    distress_present: bool,
    distress_terms: list[str] | None = None,
    max_signal_strength: float = 0.6,
) -> Layer1SignalSignals:
    return Layer1SignalSignals(
        health_state=health_state,
        signal_expected=signal_expected,
        distress_signals_present=distress_present,
        distress_terms=distress_terms or (["cancel"] if distress_present else []),
        max_signal_strength=max_signal_strength,
    )


def test_declining_distress_present_pass() -> None:
    """DECLINING + distress_signals_present → PASS."""
    from resonantforge.validators.rule_engine import validate_layer1_signal

    sig = _signals("declining", signal_expected=True, distress_present=True)
    verdict = validate_layer1_signal(sig)
    assert verdict.verdict == ValidationVerdict.PASS
    assert verdict.dimension == "layer1_signal"


def test_declining_no_distress_fail() -> None:
    """DECLINING + no distress_signals → FAIL (sentiment_drop not reflected)."""
    from resonantforge.validators.rule_engine import validate_layer1_signal

    sig = _signals("declining", signal_expected=True, distress_present=False)
    verdict = validate_layer1_signal(sig)
    assert verdict.verdict == ValidationVerdict.FAIL


def test_critical_distress_present_pass() -> None:
    """CRITICAL + distress_signals_present → PASS."""
    from resonantforge.validators.rule_engine import validate_layer1_signal

    sig = _signals("critical", signal_expected=True, distress_present=True)
    verdict = validate_layer1_signal(sig)
    assert verdict.verdict == ValidationVerdict.PASS


def test_critical_no_distress_fail() -> None:
    """CRITICAL + no distress_signals → FAIL."""
    from resonantforge.validators.rule_engine import validate_layer1_signal

    sig = _signals("critical", signal_expected=True, distress_present=False)
    verdict = validate_layer1_signal(sig)
    assert verdict.verdict == ValidationVerdict.FAIL


def test_healthy_always_skip() -> None:
    """HEALTHY health → SKIP regardless of distress_signals_present."""
    from resonantforge.validators.rule_engine import validate_layer1_signal

    sig = _signals("healthy", signal_expected=False, distress_present=False)
    verdict = validate_layer1_signal(sig)
    assert verdict.verdict == ValidationVerdict.SKIP


def test_healthy_with_distress_still_skip() -> None:
    """HEALTHY with distress terms in prose → SKIP (false positive protection)."""
    from resonantforge.validators.rule_engine import validate_layer1_signal

    sig = _signals("healthy", signal_expected=False, distress_present=True)
    verdict = validate_layer1_signal(sig)
    assert verdict.verdict == ValidationVerdict.SKIP


def test_churned_always_skip() -> None:
    """CHURNED health → SKIP (terminal state)."""
    from resonantforge.validators.rule_engine import validate_layer1_signal

    sig = _signals("churned", signal_expected=False, distress_present=False)
    verdict = validate_layer1_signal(sig)
    assert verdict.verdict == ValidationVerdict.SKIP


def test_verdict_signals_summary_has_required_keys() -> None:
    """signals_summary in the DimensionVerdict includes health_state and distress_present."""
    from resonantforge.validators.rule_engine import validate_layer1_signal

    sig = _signals("declining", signal_expected=True, distress_present=True)
    verdict = validate_layer1_signal(sig)
    assert "health_state" in verdict.signals_summary
    assert "distress_signals_present" in verdict.signals_summary
    assert "signal_expected" in verdict.signals_summary
