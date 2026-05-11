"""
Tests for the layer1_signal extractor (RFORGE-21).

The extractor checks whether customer-turn prose reflects the planted Layer 1
health state (sentiment_drop signal type, SaaS profile, v1).

Covers:
  - DECLINING health + distress signals in customer prose → distress_signals_present=True
  - DECLINING health + no distress signals → distress_signals_present=False
  - CRITICAL health + strong distress → distress_signals_present=True, signal_expected=True
  - HEALTHY health → signal_expected=False
  - CHURNED health → signal_expected=False (edge state — treat like HEALTHY)
  - PlantedHealthContext with churn_signals populates max_signal_strength
  - Empty customer prose with DECLINING health → distress_signals_present=False
"""
from __future__ import annotations

import pytest

from resonantforge.replay.schemas import PlantedHealthContext


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _ctx(health_state: str, *, strength: float = 0.6) -> PlantedHealthContext:
    """Build a minimal PlantedHealthContext for testing."""
    signals = []
    if health_state in ("declining", "critical"):
        signals = [{"signal_type": "sentiment_drop", "strength": strength}]
    return PlantedHealthContext(health_state=health_state, churn_signals=signals)


def _extract(customer_prose: str, health_state: str = "declining", *, strength: float = 0.6):
    from resonantforge.validators.extractors.layer1_signal import extract_layer1_signal_signals
    return extract_layer1_signal_signals(
        customer_prose=customer_prose,
        planted_health_context=_ctx(health_state, strength=strength),
    )


# ---------------------------------------------------------------------------
# signal_expected flag
# ---------------------------------------------------------------------------


def test_signal_expected_declining() -> None:
    """DECLINING health → signal_expected=True."""
    signals = _extract("Hello, I need some help.", "declining")
    assert signals.signal_expected is True


def test_signal_expected_critical() -> None:
    """CRITICAL health → signal_expected=True."""
    signals = _extract("Hello, I need some help.", "critical")
    assert signals.signal_expected is True


def test_signal_not_expected_healthy() -> None:
    """HEALTHY health → signal_expected=False."""
    signals = _extract("Hi, quick question about billing.", "healthy")
    assert signals.signal_expected is False


def test_signal_not_expected_churned() -> None:
    """CHURNED health → signal_expected=False (terminal state, no distress to detect)."""
    signals = _extract("I would like information about my account.", "churned")
    assert signals.signal_expected is False


# ---------------------------------------------------------------------------
# distress_signals_present detection
# ---------------------------------------------------------------------------


def test_declining_with_cancel_mention() -> None:
    """'cancel' in customer prose → distress_signals_present=True for DECLINING."""
    signals = _extract(
        "I'm thinking about canceling my subscription if this isn't fixed soon.",
        "declining",
    )
    assert signals.distress_signals_present is True
    assert len(signals.distress_terms) > 0


def test_declining_with_frustrated() -> None:
    """'frustrated' in customer prose → distress_signals_present=True."""
    signals = _extract(
        "I'm really frustrated with the constant outages.",
        "declining",
    )
    assert signals.distress_signals_present is True


def test_declining_with_unacceptable() -> None:
    """'unacceptable' in customer prose → distress_signals_present=True."""
    signals = _extract(
        "This is completely unacceptable. We're paying a lot for this service.",
        "declining",
    )
    assert signals.distress_signals_present is True


def test_declining_no_distress_in_polite_query() -> None:
    """Polite customer query with no distress terms → distress_signals_present=False."""
    signals = _extract(
        "Hello, I have a question about how to set up the API integration.",
        "declining",
    )
    assert signals.distress_signals_present is False


def test_declining_empty_customer_prose() -> None:
    """Empty customer prose → distress_signals_present=False (nothing to detect)."""
    signals = _extract("", "declining")
    assert signals.distress_signals_present is False


def test_critical_with_multiple_distress_terms() -> None:
    """CRITICAL health + multiple distress terms → all matched terms returned."""
    signals = _extract(
        "I'm incredibly frustrated and this is unacceptable. I want to cancel.",
        "critical",
    )
    assert signals.distress_signals_present is True
    assert len(signals.distress_terms) >= 2


# ---------------------------------------------------------------------------
# max_signal_strength
# ---------------------------------------------------------------------------


def test_max_signal_strength_from_churn_signal() -> None:
    """max_signal_strength reflects the planted churn signal strength."""
    signals = _extract("I want to cancel.", "declining", strength=0.85)
    assert abs(signals.max_signal_strength - 0.85) < 1e-6


def test_max_signal_strength_zero_for_healthy() -> None:
    """HEALTHY health has no churn signals → max_signal_strength=0.0."""
    signals = _extract("Quick billing question.", "healthy")
    assert signals.max_signal_strength == 0.0


# ---------------------------------------------------------------------------
# health_state passthrough
# ---------------------------------------------------------------------------


def test_health_state_preserved_in_signals() -> None:
    """The planted health_state is preserved verbatim in the signal struct."""
    signals = _extract("Some customer text.", "declining")
    assert signals.health_state == "declining"

    signals2 = _extract("Some customer text.", "critical")
    assert signals2.health_state == "critical"
