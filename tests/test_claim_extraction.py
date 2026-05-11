"""
Tests for extract_claims_llm retry and fail-loud behaviour (fix/extract-claims-fail-loud).

Three assertions:
  1. All 3 attempts return malformed JSON → ClaimExtractionError raised
  2. First attempt malformed, second valid → retry succeeds, claims returned
  3. Validator integration: extraction failure produces 'claim_extraction' FAIL verdict,
     not a passing 'accuracy' verdict
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from resonantforge.validators.extractors.accuracy import (
    ClaimExtractionError,
    extract_claims_llm,
)
from resonantforge.validators.rule_engine import validate_all_dimensions
from resonantforge.schemas import (
    AccuracyLabel,
    DimensionVerdict,
    ValidationVerdict,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_anthropic_client(*text_responses: str, stop_reason: str = "end_turn") -> MagicMock:
    """Return a mock Anthropic client whose messages.create() yields successive responses."""
    client = MagicMock()
    side_effects = []
    for text in text_responses:
        msg = MagicMock()
        msg.content = [SimpleNamespace(text=text)]
        msg.stop_reason = stop_reason
        side_effects.append(msg)
    client.messages.create.side_effect = side_effects
    return client


VALID_CLAIMS_JSON = json.dumps([
    {
        "claim_text": "We offer a 30-day refund policy.",
        "claim_span": [0, 36],
        "claim_type": "policy",
        "subject": "we",
        "predicate": "offer",
        "object": "30-day refund policy",
    }
])

AGENT_PROSE = "We offer a 30-day refund policy."


# ---------------------------------------------------------------------------
# Assertion 1: all 3 attempts malformed → ClaimExtractionError
# ---------------------------------------------------------------------------

def test_all_attempts_fail_raises_claim_extraction_error():
    """Three malformed JSON responses must raise ClaimExtractionError, never return []."""
    client = _make_anthropic_client(
        "Sorry, here are the claims: ...",   # attempt 1 — invalid JSON
        "Here you go: {broken json",          # attempt 2 — still invalid
        "{broken",                            # attempt 3 — syntax error
    )

    with pytest.raises(ClaimExtractionError) as exc_info:
        extract_claims_llm(AGENT_PROSE, anthropic_client=client, max_attempts=3)

    err = exc_info.value
    assert err.attempts == 3
    assert len(err.raw_excerpt) <= 200


def test_claim_extraction_error_message_includes_excerpt():
    client = _make_anthropic_client("not json", "still not json", "nope")
    with pytest.raises(ClaimExtractionError) as exc_info:
        extract_claims_llm(AGENT_PROSE, anthropic_client=client, max_attempts=3)
    assert "3 attempt" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Assertion 2: first attempt malformed, second valid → retry succeeds
# ---------------------------------------------------------------------------

def test_retry_succeeds_on_second_attempt():
    """A single parse failure should trigger a retry and succeed on valid JSON."""
    client = _make_anthropic_client(
        "Here are the claims as requested!",  # attempt 1 — not JSON
        VALID_CLAIMS_JSON,                    # attempt 2 — valid
    )

    claims = extract_claims_llm(AGENT_PROSE, anthropic_client=client, max_attempts=3)

    assert len(claims) == 1
    assert claims[0].claim_text == "We offer a 30-day refund policy."
    # Only two LLM calls should have been made
    assert client.messages.create.call_count == 2


def test_retry_prompt_has_strict_suffix_on_second_attempt():
    """Second attempt must append the stricter JSON-only suffix to the prompt."""
    client = _make_anthropic_client(
        "not json",         # triggers retry
        VALID_CLAIMS_JSON,  # succeeds
    )

    extract_claims_llm(AGENT_PROSE, anthropic_client=client, max_attempts=3)

    first_call_content = client.messages.create.call_args_list[0][1]["messages"][0]["content"]
    second_call_content = client.messages.create.call_args_list[1][1]["messages"][0]["content"]

    assert "IMPORTANT" not in first_call_content
    assert "IMPORTANT" in second_call_content


# ---------------------------------------------------------------------------
# Assertion 3: validator integration — claim_extraction failure is distinct from accuracy
# ---------------------------------------------------------------------------

def _make_rubric_targets(accuracy_status: str = "supported", accuracy_precision: str = "exact"):
    """Minimal RubricTarget-like object with accuracy set."""
    accuracy = AccuracyLabel(status=accuracy_status, precision=accuracy_precision)
    rubric = MagicMock()
    rubric.accuracy = accuracy
    rubric.empathy = None
    rubric.resolution = None
    rubric.brand_voice_target = None
    rubric.brand_voice_against = None
    return rubric


def test_claim_extraction_failure_produces_claim_extraction_verdict_not_accuracy():
    """
    When ClaimExtractionError is raised, the pipeline should record a 'claim_extraction'
    FAIL verdict. The 'accuracy' dimension should not produce a PASS verdict on an empty
    claim set (the old silent-failure path).

    This test simulates what pipeline.py does on ClaimExtractionError:
    - accuracy_signals = None
    - a DimensionVerdict(dimension='claim_extraction', verdict=FAIL) is appended
    - validate_all_dimensions is called with accuracy_signals=None (skips accuracy rule)
    """
    rubric_targets = _make_rubric_targets()

    # Mimic pipeline: extraction failed → accuracy_signals is None, inject verdict
    accuracy_signals = None
    verdicts = validate_all_dimensions(
        empathy_signals=None,
        resolution_signals=None,
        brand_voice_signals=None,
        accuracy_signals=accuracy_signals,
        rubric_targets=rubric_targets,
        brand_voice_variant_id="bv_warm_exploratory",
        feature_profiles={},
    )

    # Manually append the claim_extraction verdict as pipeline.py does
    verdicts.append(DimensionVerdict(
        dimension="claim_extraction",
        verdict=ValidationVerdict.FAIL,
        target="claim_extraction",
        signals_summary={"conv_id": "test-conv-001", "attempts": 3, "raw_excerpt": "bad json"},
    ))

    dimensions = {v.dimension for v in verdicts}
    assert "claim_extraction" in dimensions, "claim_extraction rule failure must be present"
    assert "accuracy" not in dimensions, (
        "accuracy verdict must not appear when claim extraction failed — "
        "previously the silent [] return would produce a passing accuracy verdict"
    )

    claim_verdict = next(v for v in verdicts if v.dimension == "claim_extraction")
    assert claim_verdict.verdict == ValidationVerdict.FAIL


def test_successful_extraction_does_not_add_claim_extraction_verdict():
    """Normal path: extraction succeeds, no claim_extraction verdict is added."""
    client = _make_anthropic_client(VALID_CLAIMS_JSON)
    claims = extract_claims_llm(AGENT_PROSE, anthropic_client=client, max_attempts=3)

    assert len(claims) == 1
    # No ClaimExtractionError was raised — in the pipeline, claim_extraction_error stays None
    # and no extra verdict is appended. This test verifies the happy path is unaffected.


# ---------------------------------------------------------------------------
# RFORGE-27: structured log fields must appear in the log output
# ---------------------------------------------------------------------------

def test_json_parse_failure_log_contains_structured_fields(caplog):
    """
    Key fields (stop_reason, response_length) must appear in the warning log text.
    Previously these were passed via extra={} which Python's default logger silently drops.
    """
    import logging

    client = _make_anthropic_client(
        "not json",   # attempt 1
        "not json",   # attempt 2
        "not json",   # attempt 3
        stop_reason="max_tokens",
    )

    with caplog.at_level(logging.WARNING, logger="resonantforge.validators.extractors.accuracy"):
        with pytest.raises(ClaimExtractionError):
            extract_claims_llm(AGENT_PROSE, anthropic_client=client, max_attempts=3)

    warning_texts = [r.message for r in caplog.records if "JSON parse failed" in r.message]
    assert warning_texts, "expected at least one 'JSON parse failed' warning"

    first = warning_texts[0]
    assert "stop_reason" in first, f"stop_reason missing from log: {first!r}"
    assert "response_length" in first, f"response_length missing from log: {first!r}"
    assert "max_tokens" in first, f"stop_reason value missing from log: {first!r}"
