"""
Tests for the layer1_signal envelope and label schema additions (RFORGE-21).

Covers:
  - PlantedHealthContext validates correctly
  - ReplayEnvelope accepts optional planted_health_context
  - Existing envelopes (no planted_health_context) still validate (backward compat)
  - labels.json REQUIRED_FAILURE_KEYS includes layer1_signal
  - Label template written by _write_label_template includes layer1_signal key
  - ReplayLabels with missing layer1_signal in expected_failures raises ValueError
"""
from __future__ import annotations

import json
import pytest
from pathlib import Path


# ---------------------------------------------------------------------------
# PlantedHealthContext
# ---------------------------------------------------------------------------


def test_planted_health_context_validates() -> None:
    """PlantedHealthContext accepts valid health states and churn signals."""
    from resonantforge.replay.schemas import PlantedHealthContext

    ctx = PlantedHealthContext(
        health_state="declining",
        churn_signals=[{"signal_type": "sentiment_drop", "strength": 0.7}],
    )
    assert ctx.health_state == "declining"
    assert len(ctx.churn_signals) == 1


def test_planted_health_context_empty_signals() -> None:
    """PlantedHealthContext with empty churn_signals is valid (HEALTHY accounts)."""
    from resonantforge.replay.schemas import PlantedHealthContext

    ctx = PlantedHealthContext(health_state="healthy", churn_signals=[])
    assert ctx.health_state == "healthy"
    assert ctx.churn_signals == []


# ---------------------------------------------------------------------------
# ReplayEnvelope — planted_health_context optional
# ---------------------------------------------------------------------------


def _minimal_envelope_dict(*, with_health_context: bool = False) -> dict:
    """Build a minimal valid ReplayEnvelope dict for schema validation tests."""
    base = {
        "schema_version": 1,
        "conv_id": "conv_test_001",
        "agent_prose": "Agent response.",
        "customer_prose": "Customer question.",
        "quality_plan": {
            "conversation_id": "conv_test_001",
            "trigger_event_id": "evt_001",
            "rubric_targets": {},
            "knowledge_citations": {},
            "prose_generation_directives": "test directive",
            "kb_chunks_required": [],
        },
        "kb_chunks": [],
        "lexicons": {},
        "brand_voice": {"variant_id": "bv_friendly", "feature_profiles": {}},
        "validator_inputs": {
            "accuracy": {
                "extracted_claims": [],
                "extraction_meta": {
                    "model": "claude-haiku-4-5-20251001",
                    "prompt_version": "sha256:abc",
                    "extracted_at": "2026-01-01T00:00:00+00:00",
                    "claims_hash": "sha256:def",
                },
            }
        },
        "metadata": {
            "conv_id": "conv_test_001",
            "source_corpus": "",
            "pipeline_version": "0.2.0",
            "kb_version": "v1",
        },
    }
    if with_health_context:
        base["planted_health_context"] = {
            "health_state": "declining",
            "churn_signals": [{"signal_type": "sentiment_drop", "strength": 0.6}],
        }
    return base


def test_envelope_accepts_planted_health_context() -> None:
    """ReplayEnvelope validates with planted_health_context present."""
    from resonantforge.replay.schemas import ReplayEnvelope

    env = ReplayEnvelope.model_validate(_minimal_envelope_dict(with_health_context=True))
    assert env.planted_health_context is not None
    assert env.planted_health_context.health_state == "declining"


def test_envelope_backward_compat_no_health_context() -> None:
    """Existing envelopes without planted_health_context validate fine (optional field)."""
    from resonantforge.replay.schemas import ReplayEnvelope

    env = ReplayEnvelope.model_validate(_minimal_envelope_dict(with_health_context=False))
    assert env.planted_health_context is None


# ---------------------------------------------------------------------------
# REQUIRED_FAILURE_KEYS includes layer1_signal
# ---------------------------------------------------------------------------


def test_required_failure_keys_includes_layer1_signal() -> None:
    """REQUIRED_FAILURE_KEYS must include layer1_signal for agreement to work."""
    from resonantforge.replay.schemas import REQUIRED_FAILURE_KEYS

    assert "layer1_signal" in REQUIRED_FAILURE_KEYS


def test_labels_missing_layer1_signal_fails_validation() -> None:
    """ReplayLabels with expected_failures missing layer1_signal must raise ValueError."""
    from pydantic import ValidationError
    from resonantforge.replay.schemas import ReplayLabels

    with pytest.raises(ValidationError, match="layer1_signal"):
        ReplayLabels.model_validate({
            "schema_version": 1,
            "conv_id": "conv_test_001",
            "label_version": 1,
            "labeled_at": "2026-01-01T00:00:00+00:00",
            "labeled_by": "tj",
            "expected_outcome": "pass",
            "expected_failures": {
                "accuracy": False,
                "empathy": False,
                "resolution": False,
                "brand_voice": False,
                "claim_extraction": False,
                # layer1_signal intentionally missing
            },
            "confidence": "high",
            "rationale": "all good",
        })


# ---------------------------------------------------------------------------
# Label template includes layer1_signal
# ---------------------------------------------------------------------------


def test_label_template_includes_layer1_signal(tmp_path: Path) -> None:
    """_write_label_template must include layer1_signal in expected_failures."""
    from resonantforge.replay.extractor import _write_label_template

    _write_label_template(tmp_path, "conv_test_001")
    label_data = json.loads((tmp_path / "labels.json").read_text())
    assert "layer1_signal" in label_data["expected_failures"]
