"""
ResonantForge corpus generator — all Pydantic v2 schemas.

This module is the single source of truth for every data shape produced or
consumed by the harness: simulation events, conversation records, quality
plans, KB chunks, signal extraction results, validation verdicts, agent
profiles, correction records, and the top-level manifest.
"""

from __future__ import annotations

from datetime import date, datetime
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Section 1 — Core event / simulation schemas
# ---------------------------------------------------------------------------


class SimEventType(str, Enum):
    """Engagement-agnostic vocabulary of simulation events."""

    ACCOUNT_CREATED = "account_created"
    CONVERSATION_STARTED = "conversation_started"
    CONVERSATION_ENDED = "conversation_ended"
    PAYMENT_RECEIVED = "payment_received"
    PAYMENT_FAILED = "payment_failed"
    FEATURE_USAGE = "feature_usage"
    SUPPORT_TICKET_OPENED = "support_ticket_opened"
    SUPPORT_TICKET_RESOLVED = "support_ticket_resolved"
    CHURN_SIGNAL_DETECTED = "churn_signal_detected"
    ESCALATION_DETECTED = "escalation_detected"
    RENEWAL_APPROACHING = "renewal_approaching"
    RENEWAL_COMPLETED = "renewal_completed"
    RENEWAL_LAPSED = "renewal_lapsed"
    COACHING_NOTE_ISSUED = "coaching_note_issued"
    SCORE_CORRECTION = "score_correction"


class LifecycleStage(str, Enum):
    """Customer account lifecycle stage."""

    ONBOARDING = "onboarding"
    ACTIVE = "active"
    AT_RISK = "at_risk"
    CHURNED = "churned"


class HealthState(str, Enum):
    """Coarse account health classification."""

    HEALTHY = "healthy"
    DECLINING = "declining"
    CRITICAL = "critical"
    CHURNED = "churned"


class SimEvent(BaseModel):
    """
    A single discrete event emitted by the simulation engine.

    Events are the lowest-level unit of simulation output; they drive
    conversation generation, snapshot calculation, and downstream analytics.
    """

    event_id: str
    event_type: SimEventType
    account_id: str
    agent_id: Optional[str] = None
    conversation_id: Optional[str] = None
    timestamp: datetime
    day_index: int  # 0-based day within simulation
    month_index: int  # 0-based month
    payload: dict[str, Any] = Field(default_factory=dict)


class DaySnapshot(BaseModel):
    """
    A point-in-time rollup of account state captured once per simulated day.

    Snapshots are pre-computed aggregates that downstream layers (analytics,
    intelligence scoring) can read without replaying the full event stream.
    """

    snapshot_id: str
    account_id: str
    day_index: int
    month_index: int
    date: date
    lifecycle_stage: LifecycleStage
    health_state: HealthState
    health_score: float  # 0.0–1.0
    open_tickets: int
    recent_signals: list[str]  # signal type names
    active_agents: list[str]  # agent IDs active this day
    payment_status: Literal["current", "overdue", "failed"]
    renewal_days_remaining: Optional[int] = None

    @field_validator("health_score")
    @classmethod
    def health_score_range(cls, v: float) -> float:
        """Ensure health score is within the 0.0–1.0 range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"health_score must be 0.0–1.0, got {v}")
        return v


# ---------------------------------------------------------------------------
# Section 2 — Conversation and quality plan schemas
# ---------------------------------------------------------------------------


class ConversationStartedPayload(BaseModel):
    """
    Typed payload for CONVERSATION_STARTED simulation events.

    ``domain`` (single-valued) drives KB chunk selection at injection time.
    ``intent`` (multi-valued) is topic metadata about what the customer is
    trying to accomplish and is available for downstream analytics.
    """

    surface_channel: str
    customer_name: str
    agent_id: str
    domain: str  # e.g. "billing", "api", "refunds"
    intent: list[str] = Field(default_factory=list)  # e.g. ["how_to", "feature_request"]


class ConversationRecord(BaseModel):
    """
    A generated customer-support conversation with metadata.

    The ``prose`` field contains the full verbatim conversation text as
    produced by the prose generator. ``is_planted_quality`` flags conversations
    that were deliberately crafted to exercise a specific rubric target.
    ``planted_constraint`` carries the specific constraint phrase planted for
    overgeneralization events so the validator can do a deterministic presence
    check instead of regex guessing.
    """

    conversation_id: str
    account_id: str
    agent_id: str
    surface_channel: str  # e.g. "intercom", "hubspot", "zendesk", "zoom", "email"
    started_at: datetime
    ended_at: datetime
    turn_count: int
    prose: str  # full generated conversation text
    trigger_event_id: str
    is_planted_quality: bool = False
    tone_variant: Optional[str] = None  # brand voice variant id if contaminated; None if dominant
    planted_constraint: Optional[str] = None  # normalized constraint phrase for overgeneralization events


class AccuracyLabel(BaseModel):
    """
    Two-axis label describing how accurately an agent's claims map to KB truth.

    ``status`` captures the binary alignment verdict; ``precision`` captures
    the specific failure mode when the agent was wrong or underspecified.
    """

    status: Literal["supported", "contradicted", "insufficient_information"]
    precision: Literal[
        "exact",
        "overgeneralized",
        "missing_constraint",
        "conditional_applied",
        "condition_missed",
    ]


class RubricTarget(BaseModel):
    """
    The intended rubric outcome for a planted-quality conversation.

    Every field is optional so that a plan can target a single dimension
    without specifying the others.
    """

    empathy: Optional[Literal["low", "high"]] = None
    resolution: Optional[Literal["weak", "strong"]] = None
    brand_voice_against: Optional[str] = None  # brand voice variant ID
    brand_voice_target: Optional[Literal["on_brand", "off_brand"]] = None
    accuracy: Optional[AccuracyLabel] = None


class KnowledgeCitations(BaseModel):
    """
    Ground-truth KB citation constraints for a quality-plan conversation.

    ``should_cite`` lists chunk IDs the agent ought to reference; ``must_not_cite``
    lists chunk IDs (or the wildcard ``["*"]``) the agent must not use.
    """

    should_cite: list[str] = Field(default_factory=list)  # chunk IDs
    must_not_cite: list[str] = Field(default_factory=list)  # chunk IDs or ["*"]


class QualityPlan(BaseModel):
    """
    Section 11.1 — directive driving prose generation for a planted conversation.

    A QualityPlan is authored by the corpus planner and consumed by the prose
    generator. It specifies rubric targets, KB constraints, and the plain-English
    directives the generator must follow when producing conversation prose.
    ``planted_constraint`` is set for overgeneralization plans: the specific
    constraint phrase extracted from the KB chunk that the generator is instructed
    to drop, enabling deterministic ground-truth validation.
    """

    conversation_id: str
    trigger_event_id: str
    rubric_targets: RubricTarget
    knowledge_citations: KnowledgeCitations
    coaching_target_dimension: Optional[str] = None
    prose_generation_directives: str  # plain-English for the prose generator
    cat11_gate: Optional[str] = None  # which Cat 11 gate this planted case tests
    multi_chunk_required: bool = False
    kb_chunks_required: list[str] = Field(default_factory=list)  # ground truth chunks
    planted_constraint: Optional[str] = None  # normalized constraint phrase for overgeneralization events


# ---------------------------------------------------------------------------
# Section 3 — KB chunk schema
# ---------------------------------------------------------------------------


class ConstraintType(str, Enum):
    """Policy constraint semantics for a KB chunk."""

    ALLOW_CONDITION = "allow_condition"
    DENY_CONDITION = "deny_condition"
    INFORMATIONAL = "informational"


class KBChunk(BaseModel):
    """
    A single retrievable unit of the knowledge base.

    Chunk IDs are deterministic (e.g. ``kb_chunk_refund_policy_v3``) so that
    quality plans and validation results can reference them by stable key across
    corpus regeneration runs with the same seed.

    ``domains`` drives KB selection: the injector filters candidates to chunks
    where the conversation's domain appears in this list. Empty means the chunk
    is not yet tagged and falls through to legacy selection logic.

    ``adversarial`` marks chunks that are intentional traps (overgeneralisations,
    stale policies, keyword-match bait). They default to ``must_not_cite`` and
    can only enter ``should_cite`` when the caller explicitly opts in.
    """

    chunk_id: str  # deterministic: kb_chunk_refund_policy_v3 style
    document_id: str
    document_path: str  # e.g. policies/refund_policy.md
    chunk_text: str
    constraint_type: ConstraintType = ConstraintType.INFORMATIONAL
    effective_date: Optional[date] = None
    superseded_by: Optional[str] = None  # chunk_id of newer version
    counterintuitive: bool = False
    cat11_gate: Optional[str] = None  # which gate this chunk supports
    tone_variant: Optional[str] = None  # brand voice variant id if contaminated; None if dominant
    domains: list[str] = Field(default_factory=list)  # topic domains this chunk covers
    # intent_tags narrow chunk eligibility to events whose intent list intersects this set.
    # Used by the plan injector's satisfiability pre-check: a chunk with
    # intent_tags=["webhook_configuration"] will only be selected for events that carry
    # "webhook_configuration" in their intent field.  Empty list means "match any intent
    # within domain" — backward compatible with all pre-RFORGE-11 chunks.
    intent_tags: list[str] = Field(default_factory=list)
    adversarial: bool = False  # opt-in required to include in should_cite
    sanity_probe: bool = False  # rotational probe per separability hypothesis schedule (PR4+)
    claims: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Section 4 — Signal schemas (per-dimension, per Section 5.1)
# ---------------------------------------------------------------------------


class EmpathySignals(BaseModel):
    """
    Extracted lexical and structural signals used to score empathy.

    ``fake_empathy_flag`` is a derived field: True when acknowledgment language
    is present but the agent fails to follow through with concrete action.
    """

    acknowledgment_present: bool
    acknowledgment_phrases: list[str]
    emotional_language_present: bool
    emotional_terms: list[str]
    apology_present: bool
    apology_terms: list[str]
    customer_emotion_referenced: bool
    response_length_tokens: int
    follow_through_present: bool
    fake_empathy_flag: bool  # derived: acknowledgment_present AND NOT follow_through_present


class ResolutionSignals(BaseModel):
    """
    Extracted signals indicating whether the agent resolved the customer's issue.

    ``deflection_present`` is True when the agent redirects without addressing
    the root issue; ``resolution_blocked`` when external factors prevented resolution.
    """

    solution_provided: bool
    solution_type: Literal["none", "partial", "complete"]
    next_steps_present: bool
    next_steps_actionable: bool
    ownership_language_present: bool
    ownership_phrases: list[str]
    deflection_present: bool
    resolution_blocked: bool


class BrandVoiceSignals(BaseModel):
    """
    Stylometric features extracted from agent prose for brand-voice scoring.

    ``formality_score`` is a normalized 0.0–1.0 value (0 = very informal,
    1 = very formal). ``vocabulary_match`` is a free-form counter dict
    (e.g. ``{"warm_terms_count": 3, "clinical_terms_count": 1}``).
    """

    avg_sentence_length: float
    sentence_count: int
    question_count: int
    hedging_terms_count: int
    hedging_terms: list[str]
    directive_terms_count: int
    directive_terms: list[str]
    formality_score: float  # 0.0–1.0
    exclamation_count: int
    vocabulary_match: dict[str, int]  # {"warm_terms_count": ..., "clinical_terms_count": ...}

    @field_validator("formality_score")
    @classmethod
    def formality_score_range(cls, v: float) -> float:
        """Ensure formality score is within the 0.0–1.0 range."""
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"formality_score must be 0.0–1.0, got {v}")
        return v


class Claim(BaseModel):
    """
    A single factual claim extracted from agent prose, normalized for matching.

    ``claim_span`` is a character-level [start, end] index into the conversation
    prose, enabling precise attribution when surfacing evidence to human reviewers.
    """

    claim_text: str
    claim_span: tuple[int, int]  # [start_char, end_char]
    claim_type: Literal["policy", "procedural", "factual"]
    normalized_subject: str
    normalized_predicate: str
    normalized_object: str
    alignment: Optional[Literal["supported", "contradicted", "partial", "not_found"]] = None


class AccuracySignals(BaseModel):
    """
    Claim-level accuracy signals linking agent prose to KB ground truth.

    ``blocking_constraint_violated`` is True when the agent applied a DENY_CONDITION
    chunk incorrectly (i.e. gave the customer a benefit they were explicitly excluded from).
    ``multi_chunk_satisfied`` is only meaningful when ``multi_chunk_required`` is True.
    """

    claims: list[Claim]
    kb_chunks_used: list[str]  # chunk IDs retrieved by harness
    kb_chunks_required: list[str]  # planted ground truth
    alignment: Literal["supported", "contradicted", "partial", "not_found"]
    constraint_preserved: bool
    overgeneralization_flag: bool
    blocking_constraint_violated: bool
    multi_chunk_required: bool
    multi_chunk_satisfied: bool


class ConversationSignals(BaseModel):
    """
    Full set of extracted signals for a single conversation, keyed by dimension.

    The ``meta`` dict captures extraction provenance (e.g. whether an LLM was
    used, extraction confidence, freeform notes).
    """

    conversation_id: str
    empathy: EmpathySignals
    resolution: ResolutionSignals
    brand_voice: BrandVoiceSignals
    accuracy: AccuracySignals
    meta: dict[str, Any] = Field(default_factory=dict)  # llm_extraction_used, extraction_confidence, notes


# ---------------------------------------------------------------------------
# Section 5 — Validation result and disagreement record
# ---------------------------------------------------------------------------


class ValidationVerdict(str, Enum):
    """Coarse three-way validation outcome."""

    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"


class GateSeverity(str, Enum):
    """Severity of a quality gate violation."""

    WARNING = "warning"
    ERROR = "error"


class GateViolation(BaseModel):
    """
    Structured record of a single quality gate violation from SkipRateTracker.check_gates().

    WARNING violations are logged but do not abort the run.
    ERROR violations cause the pipeline to raise RuntimeError after writing the manifest.
    """

    gate_name: str
    severity: GateSeverity
    actual_value: float
    threshold: float
    message: str


class DimensionVerdict(BaseModel):
    """
    Verdict for a single scoring dimension within a validation run.

    ``signals_summary`` is a dimension-specific dict of the key signals that
    drove the verdict, kept as a freeform dict for extensibility.
    """

    dimension: str
    verdict: ValidationVerdict
    target: str
    signals_summary: dict[str, Any]


class ValidationResult(BaseModel):
    """
    Aggregated validation outcome for a single conversation.

    ``overall_verdict`` is FAIL if any dimension verdict is FAIL; SKIP if
    the conversation was excluded from validation for a documented reason;
    PASS otherwise. ``retry_count`` tracks how many prose-regeneration retries
    were needed before the conversation passed (or exhausted retries).
    """

    conversation_id: str
    overall_verdict: ValidationVerdict
    dimension_verdicts: list[DimensionVerdict]
    skip_reason: Optional[str] = None
    retry_count: int = 0


class SkippedConversationRecord(BaseModel):
    """
    Records full context for a conversation skipped after exhausting prose retries.

    Written to skipped_conversations.jsonl so rejected conversations can be
    debugged without re-running the pipeline.  All fields are present for
    POST-GEN SKIP paths; pre-prompt and API-error skips may have None prose
    and empty verdicts.
    """

    conversation_id: str
    account_id: str
    event_id: Optional[str]
    quality_plan_summary: Optional[dict[str, Any]]
    final_retry_count: int
    final_verdicts: list[dict[str, Any]]
    agent_prose_snippet: Optional[str]
    final_prose: Optional[str] = None
    kb_chunks_required: Optional[list[str]]
    timestamp: str


class DisagreementRecord(BaseModel):
    """
    Records a disagreement between the rule-based validator and the soft judge.

    Used to surface borderline cases for human calibration and to tune
    validator thresholds over time.
    """

    conversation_id: str
    validator_verdict: str  # e.g. "empathy:low"
    soft_judge_perception: str  # e.g. "empathy:medium"
    disagreement: bool
    disagreement_class: str  # e.g. "ambiguous_phrasing", "borderline_case"
    notes: Optional[str] = None


# ---------------------------------------------------------------------------
# Section 6 — Agent profile schemas (Section 11.2)
# ---------------------------------------------------------------------------


class HistoryType(str, Enum):
    """Characterizes an agent's performance trajectory over time."""

    IMPROVING = "improving"
    RECURRING_WEAKNESS = "recurring_weakness"
    NEW = "new"
    MIXED = "mixed"


class SkillProfile(BaseModel):
    """
    Relative skill percentiles for a synthetic agent vs. their peer cohort.

    Percentiles are integers in [0, 100]. They drive score distribution
    parameters during conversation generation — higher percentile agents
    produce higher-quality prose on average.
    """

    empathy_percentile: int  # 0–100 vs peer cohort
    resolution_percentile: int
    brand_voice_percentile: int
    accuracy_percentile: int

    @field_validator(
        "empathy_percentile",
        "resolution_percentile",
        "brand_voice_percentile",
        "accuracy_percentile",
    )
    @classmethod
    def percentile_range(cls, v: int) -> int:
        """Ensure all percentiles are in [0, 100]."""
        if not 0 <= v <= 100:
            raise ValueError(f"Percentile must be 0–100, got {v}")
        return v


class AgentProfile(BaseModel):
    """
    Section 11.2 — synthetic agent fixture driving conversation quality.

    Each agent is deterministically associated with a tenant, a skill profile,
    and a history type. The history type controls how scores evolve over the
    simulation timeline (e.g. IMPROVING agents trend upward month-over-month).

    ``name`` is a human-readable display name for the synthetic agent.
    ``linked_conversation_id`` pins this agent to one conversation from the
    organic corpus so that downstream layers can retrieve a representative
    sample without a full table scan.
    """

    agent_id: str
    tenant_id: str
    history_type: HistoryType
    tenure_days: int
    skill_profile: SkillProfile
    name: str = ""
    linked_conversation_id: Optional[str] = None


class CoachingEvent(BaseModel):
    """
    A coaching note issued to an agent, anchored to a specific conversation.

    ``triggering_conversation_id`` provides the causal anchor — the system
    must be able to trace every coaching note back to the conversation that
    prompted it. ``status`` reflects the agent's response to the note.
    """

    coaching_id: str
    agent_id: str
    issued_at: datetime
    criterion_targeted: str
    triggering_conversation_id: str  # causal anchor
    note_text: str
    status: Literal["acknowledged", "disputed", "edited", "ignored"]


class TrajectoryRow(BaseModel):
    """
    A single row in an agent's score trajectory table.

    ``coaching_phase`` partitions rows relative to coaching interventions so
    that the intelligence layer can measure pre/post deltas. When
    ``coaching_phase`` is ``post_coaching``, ``post_coaching_of`` references
    the coaching_id that preceded this score.
    """

    trajectory_id: str
    conversation_id: str
    agent_id: str
    ai_score: int  # 0–100
    human_score: Optional[int] = None  # set when correction applied
    scored_at: datetime
    coaching_phase: Literal["pre_coaching", "post_coaching", "uncoached"]
    post_coaching_of: Optional[str] = None  # coaching_id reference
    tone_variant: Optional[str] = None  # brand voice variant id if contaminated; None if dominant

    @field_validator("ai_score")
    @classmethod
    def ai_score_range(cls, v: int) -> int:
        """Ensure AI score is in [0, 100]."""
        if not 0 <= v <= 100:
            raise ValueError(f"ai_score must be 0–100, got {v}")
        return v

    @field_validator("human_score")
    @classmethod
    def human_score_range(cls, v: Optional[int]) -> Optional[int]:
        """Ensure human score, when present, is in [0, 100]."""
        if v is not None and not 0 <= v <= 100:
            raise ValueError(f"human_score must be 0–100, got {v}")
        return v


class DisputeRecord(BaseModel):
    """
    An agent's formal dispute of an AI score on a specific criterion.

    Disputes are a distinct signal from corrections: corrections come from QA
    managers, disputes come from agents. The intelligence layer can detect
    patterns of dispute (e.g. an agent who disputes empathy scores consistently).
    """

    dispute_id: str
    agent_id: str
    conversation_id: str
    criterion: str
    disputed_score: int
    proposed_score: int
    rationale: str
    timestamp: datetime


# ---------------------------------------------------------------------------
# Section 7 — Correction record schema (Section 11.3)
# ---------------------------------------------------------------------------


class NoiseClass(str, Enum):
    """Type of synthetic noise injected into a correction record."""

    CLEAN = "clean"
    SCORE_JITTER = "score_jitter"
    TERSE_RATIONALE = "terse_rationale"
    CONTRADICTORY = "contradictory"
    BURSTY_TIMESTAMP = "bursty_timestamp"
    CROSS_TENANT = "cross_tenant"
    NON_EXPLAINING_RATIONALE = "non_explaining_rationale"


class PatternClass(str, Enum):
    """The signal pattern a correction record is designed to exercise."""

    SYSTEMATIC_UPWARD = "systematic_upward"
    COUNTEREXAMPLE = "counterexample"
    CROSS_CRITERION_NOISE = "cross_criterion_noise"
    CROSS_TENANT_BOUNDARY = "cross_tenant_boundary"


class CorrectionRecord(BaseModel):
    """
    Section 11.3 — a human score correction planted in the corpus.

    ``ai_score`` is the oracle score plus optional jitter (controlled by
    ``noise_class``). ``human_score`` is the planted correction value.
    ``pattern_class`` determines which intelligence pattern this record is
    designed to surface.
    """

    correction_id: str
    tenant_id: str
    agent_id: str
    conversation_id: str
    criterion: str
    ai_score: int  # oracle score + jitter
    human_score: int  # planted correction value
    rationale: str
    pattern_class: PatternClass
    timestamp: datetime
    noise_class: NoiseClass = NoiseClass.CLEAN


# ---------------------------------------------------------------------------
# Section 8 — Manifest schema (Section 11.4)
# ---------------------------------------------------------------------------


class Manifest(BaseModel):
    """
    Section 11.4 — top-level corpus manifest written at generation completion.

    The manifest is the authoritative record of what was generated, how, and
    with what seed. All hashes are SHA-256 hex digests of the corresponding
    NDJSON output files, enabling deterministic verification across environments.
    """

    generator_version: str = "0.2.0"
    profile_name: str
    profile_version: str
    seed: int
    accounts: int
    months: int
    generated_at: datetime
    # Counts
    event_count: int
    snapshot_count: int
    conversation_count: int
    skipped_conversation_count: int
    planted_quality_count: int
    knowledge_base_doc_count: int
    knowledge_base_chunk_count: int
    agent_count: int
    corrections_count: int
    # Hashes (SHA-256 hex of corresponding NDJSON output files)
    events_hash: str
    snapshots_hash: str
    conversations_hash: str
    planted_quality_hash: str
    kb_chunks_hash: str
    tenant_config_hash: str
    agent_fixtures_hash: str
    corrections_hash: str
    # Skip / disagreement rates
    prose_fact_violation_rate: float
    validator_rule_failure_rate: float
    disagreement_rate: float
    # Prompt cache telemetry (Anthropic ephemeral caching)
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    cache_hit_rate: float = 0.0
    cache_estimated_savings_usd: float = 0.0
    # Gate abort fields — populated when a hard gate fires and the run is aborted
    gate_aborted: bool = False
    gate_violations: list[dict[str, Any]] = Field(default_factory=list)
    # KB domain telemetry — populated after Phase 5 when domain-tagged chunks are used
    kb_version: str = ""  # sha256 of sorted chunk_ids + chunk_text
    kb_chunk_count: int = 0  # mirrors knowledge_base_chunk_count; kept for API symmetry
    domain_distribution_observed: dict[str, int] = Field(default_factory=dict)
    chunk_selection_frequency: dict[str, int] = Field(default_factory=dict)
    # PR3b: coverage backfill telemetry — populated when a CoverageBackfill is wired in
    backfill_activations: list[dict[str, Any]] = Field(default_factory=list)
    cells_requiring_backfill: list[str] = Field(default_factory=list)
    cells_satisfied_by_normal: list[str] = Field(default_factory=list)
    deficit_at_run_end: dict[str, int] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# Section 9 — Brand voice and tenant config
# ---------------------------------------------------------------------------


class BrandVoiceVariant(BaseModel):
    """
    A named brand voice specification paired with calibrated feature ranges.

    ``content`` is a ~150-200 word prose description of the brand voice, written
    in the second person (e.g. "You are warm, conversational, …"). It is injected
    directly into the prose generator prompt. ``feature_profile`` holds calibrated
    numeric ranges used by the brand-voice validator to distinguish on-brand from
    off-brand responses for this particular variant.
    """

    id: str  # e.g. "bv_warm_exploratory"
    label: str
    content: str  # ~150-200 word brand voice spec
    feature_profile: Optional[dict[str, Any]] = None  # calibrated ranges for validator


class CoachingStyleOverlay(BaseModel):
    """
    A coaching tone/style overlay paired with a specific brand voice variant.

    Overlays modify how coaching notes are framed (e.g. "direct and corrective"
    vs "socratic and reflective") without altering the underlying rubric targets.
    Each overlay is permanently coupled to one brand voice variant.
    """

    id: str
    label: str
    content: str
    brand_voice_variant_id: str  # paired with


class TenantConfig(BaseModel):
    """
    Full tenant configuration driving corpus generation for one tenant.

    ``knowledge_base_subset`` is either ``"all"`` or a named subset key defined
    in the KB layer. ``rubric_weights`` must sum to 1.0 across the four core
    dimensions (empathy, resolution, brand_voice, accuracy). The default weights
    are the platform defaults and can be overridden per tenant.
    """

    tenant_id: str
    brand_voice_variants: list[BrandVoiceVariant]
    coaching_style_overlays: list[CoachingStyleOverlay] = Field(default_factory=list)
    knowledge_base_subset: str = "all"
    rubric_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "empathy": 0.25,
            "resolution": 0.30,
            "brand_voice": 0.20,
            "accuracy": 0.25,
        }
    )
