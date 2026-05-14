"""
Tests that the prose generator system prompt includes the credential guardrail
introduced in response to GitHub issue #13.

The guardrail must be present in the system prompt regardless of quality plan
or account context, since it constrains agent behavior universally.
"""
from __future__ import annotations

from datetime import datetime

import pytest

from resonantforge.pipeline import _build_prompt
from resonantforge.schemas import SimEvent, SimEventType


_GUARDRAIL = "The agent must never ask the customer to share passwords, API keys, tokens"


def _make_conv_event() -> SimEvent:
    return SimEvent(
        event_id="evt_test",
        event_type=SimEventType.CONVERSATION_STARTED,
        account_id="acc_test",
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


def test_credential_guardrail_present_without_quality_plan() -> None:
    """Guardrail appears in system prompt when no quality plan is active."""
    system_prompt, _ = _build_prompt(
        account_id="acc_test",
        month_index=0,
        conv_event=_make_conv_event(),
        month_summary={},
        quality_plan=None,
        profile_name="saas",
    )
    assert _GUARDRAIL in system_prompt


def test_credential_guardrail_present_with_quality_plan() -> None:
    """Guardrail appears in system prompt even when a quality plan is injected."""
    from unittest.mock import MagicMock

    plan = MagicMock()
    plan.prose_generation_directives = "Agent should be terse and procedural."

    system_prompt, _ = _build_prompt(
        account_id="acc_test",
        month_index=0,
        conv_event=_make_conv_event(),
        month_summary={"avg_health_score": 0.8, "dominant_health_state": "healthy", "payment_failures": 0},
        quality_plan=plan,
        profile_name="saas",
    )
    assert _GUARDRAIL in system_prompt
