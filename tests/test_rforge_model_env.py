"""
Tests for RFORGE_MODEL environment variable support (RFORGE-70).

Verifies that _call_anthropic respects the RFORGE_MODEL env var at call time,
falling back to the hardcoded default when the variable is not set.  The
Anthropic client is patched so no real API calls are made.
"""
from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from resonantforge.pipeline import _call_anthropic


def _make_mock_client(response_text: str = "mock response") -> MagicMock:
    """Return a mock Anthropic client whose messages.create returns a plausible response."""
    block = SimpleNamespace(text=response_text)
    response = SimpleNamespace(content=[block])
    client = MagicMock()
    client.messages.create.return_value = response
    return client


def test_default_model_when_env_not_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """When RFORGE_MODEL is absent, the model passed to the API is the hardcoded default."""
    monkeypatch.delenv("RFORGE_MODEL", raising=False)
    client = _make_mock_client()

    text, cc, cr = _call_anthropic(client, "system", "user")

    assert text == "mock response"
    _, kwargs = client.messages.create.call_args
    assert kwargs["model"] == "claude-haiku-4-5-20251001"


def test_override_model_via_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    """When RFORGE_MODEL is set, the model passed to the API matches the env var value."""
    monkeypatch.setenv("RFORGE_MODEL", "claude-haiku-4-6")
    client = _make_mock_client()

    text, cc, cr = _call_anthropic(client, "system", "user")

    assert text == "mock response"
    _, kwargs = client.messages.create.call_args
    assert kwargs["model"] == "claude-haiku-4-6"
