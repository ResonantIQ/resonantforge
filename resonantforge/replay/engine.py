"""
Replay engine: run validators against frozen envelopes at zero LLM cost.

Public API:
  load_envelope(path)       → ReplayEnvelope
  load_labels(path)         → ReplayLabels
  replay_one(envelope, labels) → ReplayResult

Determinism contract (enforced when RFORGE_REPLAY_MODE=1):
  1. No LLM calls — extracted_claims is injected directly into run_kb_alignment_pipeline,
     bypassing extract_claims_llm entirely. extract_claims_llm is patched to raise
     ReplayModeError if accidentally called.
  2. No datetime.now() / time.time() in the validator code path.
  3. Stable sort — synonym map iteration uses sorted() (longest-first); KB alignment
     iterates candidate_chunks in envelope load order.
  4. No RNG — none of the 4 extractors use randomness.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from resonantforge.replay.agreement import compute_agreement
from resonantforge.replay.schemas import (
    AgreementResult,
    ReplayEnvelope,
    ReplayLabels,
    ReplayResult,
)
from resonantforge.schemas import (
    Claim,
    DimensionVerdict,
    KBChunk,
    ValidationVerdict,
)
from resonantforge.validators.extractors.accuracy import run_kb_alignment_pipeline
from resonantforge.validators.extractors.brand_voice import extract_brand_voice_signals
from resonantforge.validators.extractors.empathy import extract_empathy_signals
from resonantforge.validators.extractors.resolution import extract_resolution_signals
from resonantforge.validators.rule_engine import validate_all_dimensions


# ---------------------------------------------------------------------------
# Sentinel errors
# ---------------------------------------------------------------------------


class EnvelopeSchemaError(ValueError):
    """Raised when an envelope.json fails schema validation."""


class LabelSchemaError(ValueError):
    """Raised when a labels.json fails schema validation."""


class ReplayModeError(RuntimeError):
    """Raised when a prohibited call (LLM, datetime.now) occurs during replay."""


# ---------------------------------------------------------------------------
# REPLAY_MODE determinism guard
# ---------------------------------------------------------------------------

_REPLAY_MODE = os.environ.get("RFORGE_REPLAY_MODE", "0") == "1"


def _install_replay_guard() -> None:
    """
    Patch extract_claims_llm to raise ReplayModeError if called during replay.

    This makes the "no LLM calls" invariant fail-loud rather than silently
    returning stale or random results. Called once at module import if
    RFORGE_REPLAY_MODE=1.
    """
    import resonantforge.validators.extractors.accuracy as _accuracy_mod

    def _guarded_extract_claims_llm(*args: Any, **kwargs: Any) -> None:
        raise ReplayModeError(
            "extract_claims_llm called during replay — this violates the "
            "determinism contract. The replay engine must inject frozen "
            "extracted_claims directly into run_kb_alignment_pipeline."
        )

    _accuracy_mod.extract_claims_llm = _guarded_extract_claims_llm  # type: ignore[attr-defined]


if _REPLAY_MODE:
    _install_replay_guard()


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def load_envelope(path: Path) -> ReplayEnvelope:
    """
    Load and validate an envelope.json file.

    Raises:
        EnvelopeSchemaError: if the file is missing, not valid JSON, or fails
            the ReplayEnvelope Pydantic schema (including schema_version check
            and conv_id cross-validation with metadata).
    """
    if not path.exists():
        raise EnvelopeSchemaError(f"envelope not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise EnvelopeSchemaError(f"envelope is not valid JSON: {path}\n{exc}") from exc
    try:
        return ReplayEnvelope.model_validate(raw)
    except ValidationError as exc:
        raise EnvelopeSchemaError(f"envelope schema error in {path}:\n{exc}") from exc


def load_labels(path: Path, *, conv_id: str) -> ReplayLabels:
    """
    Load and validate a labels.json file, cross-checking conv_id against the envelope.

    Args:
        path: path to labels.json
        conv_id: the conv_id from the sibling envelope (must match)

    Raises:
        LabelSchemaError: if the file is missing, not valid JSON, fails the
            ReplayLabels Pydantic schema, or conv_id does not match the envelope.
    """
    if not path.exists():
        raise LabelSchemaError(f"labels not found: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LabelSchemaError(f"labels is not valid JSON: {path}\n{exc}") from exc
    try:
        labels = ReplayLabels.model_validate(raw)
    except ValidationError as exc:
        raise LabelSchemaError(f"label schema error in {path}:\n{exc}") from exc

    if labels.conv_id != conv_id:
        raise LabelSchemaError(
            f"LabelSchemaError: labels.conv_id={labels.conv_id!r} does not match "
            f"envelope conv_id={conv_id!r} in {path}"
        )
    return labels


# ---------------------------------------------------------------------------
# Core replay engine
# ---------------------------------------------------------------------------


def replay_one(envelope: ReplayEnvelope, labels: ReplayLabels) -> ReplayResult:
    """
    Run all validators against a frozen envelope and compute agreement with labels.

    This is the free inner loop of validator iteration. No LLM calls are made:
      - Empathy, resolution, brand_voice: deterministic extractors, frozen lexicons
      - Accuracy: run_kb_alignment_pipeline with frozen extracted_claims (no LLM)

    The overall_outcome is FAIL when any dimension verdict is FAIL; PASS otherwise
    (SKIP verdicts are neutral — they neither pass nor fail the conversation).

    Args:
        envelope: validated ReplayEnvelope from load_envelope()
        labels: validated ReplayLabels from load_labels()

    Returns:
        ReplayResult with dimension verdicts, overall outcome, and agreement layers
    """
    lex = envelope.lexicons
    plan = envelope.quality_plan
    rubric = plan.rubric_targets

    # 1. Empathy — deterministic extractor
    empathy_signals = extract_empathy_signals(
        agent_prose=envelope.agent_prose,
        acknowledgment_phrases=lex.acknowledgment_phrases,
        emotion_lexicon=lex.emotion_lexicon,
        apology_lexicon=lex.apology_lexicon,
        action_verb_lexicon=lex.action_verb_lexicon,
    )

    # 2. Resolution — deterministic extractor
    resolution_signals = extract_resolution_signals(
        agent_prose=envelope.agent_prose,
        customer_prose=envelope.customer_prose,
        resolution_patterns=lex.resolution_patterns,
        deflection_patterns=lex.deflection_patterns,
        next_steps_patterns=lex.next_steps_patterns,
        temporal_anchor_patterns=lex.temporal_anchor_patterns,
        specific_actor_patterns=lex.specific_actor_patterns,
        ownership_patterns=lex.ownership_patterns,
        issue_keywords=lex.issue_keywords,
    )

    # 3. Brand voice — deterministic extractor
    brand_voice_signals = extract_brand_voice_signals(
        agent_prose=envelope.agent_prose,
        hedging_lexicon=lex.hedging_lexicon,
        directive_lexicon=lex.directive_lexicon,
        warm_terms=lex.warm_terms,
        clinical_terms=lex.clinical_terms,
        contraction_patterns=lex.contraction_patterns,
    )

    # 4. Accuracy — inject frozen extracted_claims, bypass extract_claims_llm entirely.
    # planted_constraint and planted_contradiction are forwarded from the quality_plan
    # so the closed-loop detection paths fire correctly during replay.
    frozen_claims: list[Claim] = envelope.validator_inputs.accuracy.extracted_claims
    accuracy_signals = run_kb_alignment_pipeline(
        claims=frozen_claims,
        kb_chunks=envelope.kb_chunks,
        conversation_context=envelope.customer_prose,
        kb_chunks_required=plan.kb_chunks_required,
        synonym_map=lex.synonym_map,
        planted_constraint=plan.planted_constraint,
        planted_contradiction=(
            plan.planted_contradiction.model_dump() if plan.planted_contradiction else None
        ),
    )

    # 5. Rule engine — validate all dimensions with targets from frozen quality plan
    bv_variant = envelope.brand_voice.variant_id
    bv_profiles = envelope.brand_voice.feature_profiles

    dimension_verdicts: list[DimensionVerdict] = validate_all_dimensions(
        empathy_signals=empathy_signals if rubric.empathy else None,
        resolution_signals=resolution_signals if rubric.resolution else None,
        brand_voice_signals=brand_voice_signals if rubric.brand_voice_target else None,
        accuracy_signals=accuracy_signals if rubric.accuracy else None,
        rubric_targets=rubric,
        brand_voice_variant_id=bv_variant,
        feature_profiles=bv_profiles,
    )

    # 6. Overall outcome: FAIL if any dimension failed; PASS otherwise
    overall_outcome: str = "pass"
    for verdict in dimension_verdicts:
        if verdict.verdict == ValidationVerdict.FAIL:
            overall_outcome = "fail"
            break

    # 7. Three-layer agreement
    agreement = compute_agreement(
        dimension_verdicts=dimension_verdicts,
        claim_extraction_ok=True,  # always True in replay (no LLM extraction)
        overall_outcome=overall_outcome,
        labels=labels,
    )

    return ReplayResult(
        conv_id=envelope.conv_id,
        dimension_verdicts=dimension_verdicts,
        claim_extraction_ok=True,
        overall_outcome=overall_outcome,  # type: ignore[arg-type]
        agreement=agreement,
        envelope_metadata=envelope.metadata,
        labels=labels,
    )
