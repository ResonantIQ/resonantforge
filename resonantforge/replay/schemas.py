"""
Pydantic models for replay corpus artifacts: envelopes, labels, and replay results.

These schemas enforce the v1 contracts defined in:
  harness/docs/replay-envelope-schema.md
  harness/docs/replay-label-schema.md
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from resonantforge.schemas import Claim, KBChunk, QualityPlan


# ---------------------------------------------------------------------------
# Envelope sub-models
# ---------------------------------------------------------------------------


class EnvelopeLexicons(BaseModel):
    """All lexicons frozen at extraction time, keyed by extractor consumer."""

    # Empathy extractor
    acknowledgment_phrases: list[str] = Field(default_factory=list)
    emotion_lexicon: list[str] = Field(default_factory=list)
    apology_lexicon: list[str] = Field(default_factory=list)
    action_verb_lexicon: list[str] = Field(default_factory=list)

    # Resolution extractor
    resolution_patterns: list[str] = Field(default_factory=list)
    deflection_patterns: list[str] = Field(default_factory=list)
    next_steps_patterns: list[str] = Field(default_factory=list)
    temporal_anchor_patterns: list[str] = Field(default_factory=list)
    specific_actor_patterns: list[str] = Field(default_factory=list)
    ownership_patterns: list[str] = Field(default_factory=list)
    issue_keywords: list[str] = Field(default_factory=list)

    # Brand voice extractor
    hedging_lexicon: list[str] = Field(default_factory=list)
    directive_lexicon: list[str] = Field(default_factory=list)
    warm_terms: list[str] = Field(default_factory=list)
    clinical_terms: list[str] = Field(default_factory=list)
    contraction_patterns: list[str] = Field(default_factory=list)

    # Accuracy extractor (Steps 2–8)
    synonym_map: dict[str, str] = Field(default_factory=dict)


class BrandVoiceConfig(BaseModel):
    """Frozen brand voice variant config."""

    variant_id: str
    feature_profiles: dict[str, Any]


class ExtractionMeta(BaseModel):
    """Provenance for the extract_claims_llm call that produced extracted_claims."""

    model: str
    prompt_version: str  # SHA-256 of CLAIM_EXTRACTION_PROMPT at extraction time
    extracted_at: str    # ISO-8601 UTC timestamp
    claims_hash: str     # SHA-256 of sorted canonical JSON of the extracted claims


class AccuracyValidatorInputs(BaseModel):
    """Pre-extracted claims — the sole LLM-derived artifact in the envelope."""

    extracted_claims: list[Claim] = Field(default_factory=list)
    extraction_meta: ExtractionMeta


class ValidatorInputs(BaseModel):
    """Frozen validator inputs section of the envelope."""

    accuracy: AccuracyValidatorInputs = Field(default_factory=AccuracyValidatorInputs)


class EnvelopeMetadata(BaseModel):
    """Provenance metadata for a replay envelope."""

    conv_id: str
    source_corpus: str = ""
    pipeline_version: str = ""
    kb_version: str = ""
    generator_version: str = ""
    extraction_timestamp: str = ""


# ---------------------------------------------------------------------------
# Top-level envelope
# ---------------------------------------------------------------------------


class ReplayEnvelope(BaseModel):
    """
    Self-contained snapshot of every input a validator needs to reproduce its
    verdict without calling any LLM or external service.

    Schema version 1. See harness/docs/replay-envelope-schema.md.
    """

    schema_version: int
    conv_id: str
    agent_prose: str
    customer_prose: str
    quality_plan: QualityPlan
    kb_chunks: list[KBChunk] = Field(default_factory=list)
    lexicons: EnvelopeLexicons
    brand_voice: BrandVoiceConfig
    validator_inputs: ValidatorInputs = Field(default_factory=ValidatorInputs)
    metadata: EnvelopeMetadata

    @field_validator("schema_version")
    @classmethod
    def _require_v1(cls, v: int) -> int:
        if v != 1:
            raise ValueError(f"EnvelopeSchemaError: expected schema_version=1, got {v}")
        return v

    @model_validator(mode="after")
    def _conv_id_in_metadata(self) -> "ReplayEnvelope":
        if self.metadata.conv_id and self.metadata.conv_id != self.conv_id:
            raise ValueError(
                f"EnvelopeSchemaError: metadata.conv_id={self.metadata.conv_id!r} "
                f"does not match top-level conv_id={self.conv_id!r}"
            )
        return self


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------

VALID_TAGS = frozenset({
    "high_confidence_fail",
    "high_confidence_pass",
    "edge_case",
    "borderline",
    "claim_extraction_stress",
    "multi_chunk",
    "deny_condition",
    "overgeneralization",
    "planted",
    "organic",
    "skipped_during_generation",
})

REQUIRED_FAILURE_KEYS = frozenset({"accuracy", "empathy", "resolution", "brand_voice", "claim_extraction"})


class ReplayLabels(BaseModel):
    """
    Human-authored ground truth for one conversation envelope.

    Schema version 1. See harness/docs/replay-label-schema.md.
    """

    schema_version: int
    conv_id: str
    label_version: int
    labeled_at: str
    labeled_by: str
    expected_outcome: Literal["pass", "fail", "uncertain"]
    expected_failures: dict[str, bool]
    confidence: Literal["high", "medium", "low"]
    tags: list[str] = Field(default_factory=list)
    rationale: str
    revised_from: Optional[int] = None
    revision_notes: Optional[str] = None

    @field_validator("schema_version")
    @classmethod
    def _require_v1(cls, v: int) -> int:
        if v != 1:
            raise ValueError(f"LabelSchemaError: expected schema_version=1, got {v}")
        return v

    @field_validator("label_version")
    @classmethod
    def _positive_label_version(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"LabelSchemaError: label_version must be a positive integer, got {v}")
        return v

    @field_validator("rationale")
    @classmethod
    def _rationale_not_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("LabelSchemaError: rationale must not be empty")
        return v

    @field_validator("tags")
    @classmethod
    def _tags_closed_enum(cls, v: list[str]) -> list[str]:
        unknown = [t for t in v if t not in VALID_TAGS]
        if unknown:
            raise ValueError(
                f"LabelSchemaError: unknown tag(s) {unknown!r}. "
                f"Valid tags: {sorted(VALID_TAGS)}"
            )
        return v

    @field_validator("expected_failures")
    @classmethod
    def _expected_failures_complete(cls, v: dict[str, bool]) -> dict[str, bool]:
        missing = REQUIRED_FAILURE_KEYS - set(v.keys())
        if missing:
            raise ValueError(
                f"LabelSchemaError: expected_failures is missing required key(s): {sorted(missing)}"
            )
        return v

    @model_validator(mode="after")
    def _revision_audit_trail(self) -> "ReplayLabels":
        if self.revised_from is not None and self.revision_notes is None:
            raise ValueError(
                "LabelSchemaError: revised_from is set but revision_notes is null — "
                "revision audit trail is incomplete"
            )
        return self


# ---------------------------------------------------------------------------
# Replay result types
# ---------------------------------------------------------------------------


class AgreementResult(BaseModel):
    """Three-layer agreement between validator output and human labels."""

    outcome_match: Optional[bool]  # None when expected_outcome is "uncertain"
    rule_match: bool
    false_positive_rules: list[str]  # rules that fired but label says should pass
    missed_rules: list[str]          # rules label says should fail that didn't fire


class ReplayResult(BaseModel):
    """Full output of running the replay engine against one (envelope, labels) pair."""

    conv_id: str
    dimension_verdicts: list[Any]  # list[DimensionVerdict] — imported lazily to avoid cycles
    claim_extraction_ok: bool  # always True in replay (no LLM call for extraction)
    overall_outcome: Literal["pass", "fail"]
    agreement: AgreementResult
    envelope_metadata: EnvelopeMetadata
    labels: ReplayLabels
