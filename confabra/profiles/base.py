"""Abstract Profile interface for Layer 2 implementations."""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from confabra.schemas import BrandVoiceVariant, CoachingStyleOverlay


@dataclass
class LifecycleStageConfig:
    """Detailed configuration for a single lifecycle stage."""

    name: str
    display_name: str
    health_range: tuple[float, float]  # (min, max) health score for this stage
    typical_event_rate: float  # conversations per day


@dataclass
class SignalDescriptor:
    """Describes a named signal type and how it is detected."""

    name: str
    description: str
    detection_mechanism: str  # "llm" | "embedding" | "rule"


@dataclass
class SurfaceChannel:
    """A conversation surface channel with its associated prose style hint."""

    name: str
    display_name: str
    prose_style: str  # brief description of the channel's writing style


@dataclass
class PersonaArchetype:
    """A synthetic agent persona archetype used to seed agent fixtures."""

    name: str
    role: str  # e.g. "senior_support_agent", "junior_cs_rep"
    empathy_baseline: float  # 0.0-1.0
    resolution_baseline: float
    formality: float  # 0.0-1.0


@dataclass
class AgentHistoryTemplate:
    """
    Template describing an agent's performance trajectory type.

    Used by the agent fixture generator to seed realistic score histories
    with causally-anchored coaching events.
    """

    history_type: str  # "improving" | "recurring_weakness" | "new" | "mixed"
    score_trajectory_shape: str  # "upward" | "flat" | "none" | "mixed"
    coaching_event_count: int  # typical number of prior coaching events
    description: str


@dataclass
class CorrectionPattern:
    """
    A single planted correction pattern for the corrections log artifact.

    Pattern classes map to Cat L1 signal-detection categories:
    - systematic_upward / counterexample: the two directions of systematic bias
    - cross_criterion_noise: corrections on unrelated criteria (should not cause spurious learning)
    - cross_tenant_boundary: corrections that must not bleed across tenant isolation
    """

    criterion: str
    direction: str  # "upward" | "downward"
    pattern_class: str  # "systematic_upward" | "counterexample" | "cross_criterion_noise" | "cross_tenant_boundary"
    rationale_template: str  # template string for generating rationale text


@dataclass
class ValueMetric:
    """Primary value metric for accounts in this profile."""

    name: str  # e.g. "monthly_arr"
    unit: str  # e.g. "cents"


@dataclass
class ReferenceSurface:
    """The primary surface channel used as the canonical conversation surface."""

    primary: str  # primary surface channel name


class Profile(ABC):
    """
    Abstract base class for Layer 2 industry profiles.

    Each profile defines the industry-specific vocabulary, surface channels,
    persona archetypes, KB content, brand voice variants, and correction
    patterns that drive corpus generation for a particular vertical.

    SaaS and PS are the two concrete implementations shipped in v1.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this profile, e.g. 'saas' or 'ps'."""
        ...

    @property
    @abstractmethod
    def version(self) -> str:
        """Semantic version of this profile definition, e.g. '1.1.0'."""
        ...

    @abstractmethod
    def lifecycle_stages(self) -> list[str]:
        """Return list of lifecycle stage names (used by state machine)."""

    @abstractmethod
    def lifecycle_stage_configs(self) -> list[LifecycleStageConfig]:
        """Return detailed lifecycle stage configurations including health ranges."""

    @abstractmethod
    def signal_vocabulary(self) -> dict[str, SignalDescriptor]:
        """Return the signal type descriptors for this profile's domain."""

    @abstractmethod
    def surface_channels(self) -> list[SurfaceChannel]:
        """Return the available surface channels and their prose style hints."""

    @abstractmethod
    def persona_pool(self) -> list[PersonaArchetype]:
        """Return the agent persona archetypes for this profile."""

    @abstractmethod
    def perturbation_density(self) -> dict[str, float]:
        """Return perturbation class densities (fraction of conversations to perturb)."""

    @abstractmethod
    def value_metric(self) -> ValueMetric:
        """Return the primary value metric used for accounts in this profile."""

    @abstractmethod
    def reference_surface(self) -> ReferenceSurface:
        """Return the primary reference surface channel for this profile."""

    @abstractmethod
    def prose_style_directives(self) -> dict[str, str]:
        """Return a channel → prose style directive mapping for the prose generator."""

    @abstractmethod
    def brand_voice_variants(self) -> list[BrandVoiceVariant]:
        """Return the brand voice variants available to tenants using this profile."""

    def coaching_style_overlays(self) -> list[CoachingStyleOverlay]:
        """
        Return coaching style overlays.

        Returns an empty list at the scaffolding stage; P1-7 (coaching overlay
        generation) populates this in a later task.
        """
        return []

    @abstractmethod
    def knowledge_base_content(self) -> "list[KBChunk]":
        """Return the KB chunks that seed the knowledge base for this profile."""

    @abstractmethod
    def agent_history_templates(self) -> dict[str, AgentHistoryTemplate]:
        """Return a mapping of history_type → AgentHistoryTemplate."""

    @abstractmethod
    def correction_pattern_set(self) -> list[CorrectionPattern]:
        """Return the planted correction patterns for this profile."""

    def brand_voice_feature_profiles(self) -> dict:
        """
        Return calibrated feature profiles for brand voice variants.

        Used by the brand-voice validator to distinguish on-brand from off-brand
        responses.  Defaults to an empty dict; profiles with multiple brand voice
        variants should override this.
        """
        return {}

    def planted_quality_count(self) -> int:
        """
        Number of planted quality conversations for this profile.

        SaaS = 50 (full Cat coverage), PS = 15 (vocabulary-overfit smoke).
        """
        return 50

    def default_accounts(self) -> int:
        """Default number of synthetic accounts for this profile."""
        return 25

    def default_months(self) -> int:
        """Default simulation duration in months for this profile."""
        return 6
