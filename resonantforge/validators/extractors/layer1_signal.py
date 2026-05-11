# resonantforge/validators/extractors/layer1_signal.py
"""
Layer 1 signal extractor — deterministic (string/regex, no LLM).

Checks whether customer-turn prose reflects the planted Layer 1 health state
(RFORGE-21). v1 scope: SaaS profile, sentiment_drop signal type only.

The rule:
  - DECLINING or CRITICAL health → expect distress signals in customer prose
  - HEALTHY or CHURNED health → signal not expected (validator returns SKIP)
"""
from __future__ import annotations

import re

from resonantforge.replay.schemas import PlantedHealthContext
from resonantforge.schemas import Layer1SignalSignals

# Health states where distress signals are expected in customer prose.
_SIGNAL_EXPECTED_STATES = frozenset({"declining", "critical"})

# Distress terms that reflect a sentiment_drop churn signal in SaaS conversations.
# Substring match (case-insensitive) against customer prose.
_DISTRESS_TERMS: list[str] = [
    "cancel",
    "canceling",
    "cancellation",
    "frustrated",
    "frustration",
    "unacceptable",
    "disappointed",
    "terrible",
    "useless",
    "ridiculous",
    "fed up",
    "unhappy",
    "angry",
    "awful",
    "horrible",
    "waste of money",
    "wasted",
    "switching to",
    "switch away",
    "moving on",
    "leaving",
    "escalate",
    "speak to your manager",
    "talk to your manager",
    "this is a joke",
    "not working",
    "still broken",
    "disgusted",
    "outrageous",
]


def extract_layer1_signal_signals(
    customer_prose: str,
    planted_health_context: PlantedHealthContext,
) -> Layer1SignalSignals:
    """
    Extract layer1_signal dimension signals from customer-turn prose.

    Args:
        customer_prose: Customer turns only (speaker prefix stripped).
        planted_health_context: Frozen Layer 1 health state and churn signals.

    Returns:
        Layer1SignalSignals with all fields populated.
    """
    health_state = planted_health_context.health_state.lower()
    signal_expected = health_state in _SIGNAL_EXPECTED_STATES

    # Derive max_signal_strength from planted churn signals.
    strengths = [
        float(s.get("strength", 0.0))
        for s in planted_health_context.churn_signals
        if s.get("signal_type") == "sentiment_drop"
    ]
    max_signal_strength = max(strengths) if strengths else 0.0

    # Detect distress terms in customer prose (case-insensitive substring match).
    prose_lower = customer_prose.lower()
    matched: list[str] = [term for term in _DISTRESS_TERMS if term in prose_lower]

    return Layer1SignalSignals(
        health_state=health_state,
        signal_expected=signal_expected,
        distress_signals_present=bool(matched),
        distress_terms=matched,
        max_signal_strength=max_signal_strength,
    )
