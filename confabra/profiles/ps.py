"""Professional Services Layer 2 profile — secondary, minimal smoke."""
from __future__ import annotations
from confabra.profiles.base import (
    Profile, LifecycleStageConfig, SignalDescriptor, SurfaceChannel,
    PersonaArchetype, AgentHistoryTemplate, CorrectionPattern, ValueMetric, ReferenceSurface
)
from confabra.schemas import BrandVoiceVariant, CoachingStyleOverlay, KBChunk
from confabra.profiles.lexicons import ps as ps_lex
from confabra.kb.ps_content import get_ps_kb_chunks


class PSProfile(Profile):
    """
    Professional Services profile — 5 accounts × 3 months, vocabulary-overfit smoke.

    Industry: B2B professional services delivery (consulting, implementation, advisory).
    Reference surface: client call notes.
    Scope: smaller corpus than SaaS; designed to verify that the vocabulary-overfit
    detection fires correctly when PS-domain terms dominate the embedding space.
    """

    @property
    def name(self) -> str:
        return "ps"

    @property
    def version(self) -> str:
        return "1.1.0"

    def lifecycle_stages(self) -> list[str]:
        """Return the four PS engagement lifecycle stages."""
        return ["discovery", "scoping", "delivery", "closure"]

    def lifecycle_stage_configs(self) -> list[LifecycleStageConfig]:
        """Return detailed lifecycle stage configs for PS engagements."""
        return [
            LifecycleStageConfig("discovery", "Discovery", (0.7, 1.0), 0.5),
            LifecycleStageConfig("scoping", "Scoping", (0.6, 1.0), 0.6),
            LifecycleStageConfig("delivery", "Delivery", (0.4, 1.0), 0.4),
            LifecycleStageConfig("closure", "Closure", (0.5, 1.0), 0.2),
        ]

    def signal_vocabulary(self) -> dict[str, SignalDescriptor]:
        """Return the six PS-specific signal descriptors."""
        return {
            "scope_creep": SignalDescriptor("scope_creep", "Customer requesting work outside agreed scope", "llm"),
            "delivery_risk": SignalDescriptor("delivery_risk", "Timeline at risk due to blockers or delays", "rule"),
            "stakeholder_escalation": SignalDescriptor("stakeholder_escalation", "Issue escalated to client executive", "embedding"),
            "milestone_approved": SignalDescriptor("milestone_approved", "Client approved a deliverable or milestone", "rule"),
            "change_request": SignalDescriptor("change_request", "Formal change request submitted", "rule"),
            "satisfaction_risk": SignalDescriptor("satisfaction_risk", "Client expressed dissatisfaction with delivery", "llm"),
        }

    def surface_channels(self) -> list[SurfaceChannel]:
        """Return the six PS surface channels."""
        return [
            SurfaceChannel("client_call", "Client Call Notes", "Meeting notes, action items, formal but conversational"),
            SurfaceChannel("deliverable_review", "Deliverable Review Email", "Formal email with structured feedback"),
            SurfaceChannel("status_update", "Weekly Status Update", "Structured report: RAG status, accomplishments, risks, next week"),
            SurfaceChannel("email", "Client Email Thread", "Professional email, formal sign-off, clear subject line"),
            SurfaceChannel("slack", "Slack DM", "Informal, quick updates, may reference documents"),
            SurfaceChannel("document_comment", "Document Comment", "Inline review comment on a deliverable"),
        ]

    def persona_pool(self) -> list[PersonaArchetype]:
        """Return the four PS consultant persona archetypes."""
        return [
            PersonaArchetype("engagement_manager", "Engagement Manager", 0.7, 0.8, 0.8),
            PersonaArchetype("delivery_lead", "Delivery Lead", 0.6, 0.9, 0.8),
            PersonaArchetype("consultant", "Senior Consultant", 0.7, 0.8, 0.75),
            PersonaArchetype("analyst", "Business Analyst", 0.75, 0.7, 0.7),
        ]

    def perturbation_density(self) -> dict[str, float]:
        """
        Return per-class perturbation densities for PS conversations.

        Lower densities than SaaS because the PS corpus is smaller (5 accounts × 3 months)
        and the primary goal is vocabulary-overfit detection, not broad noise coverage.
        """
        return {
            "adversarial_input": 0.03,
            "sarcasm": 0.02,
            "negation": 0.03,
            "noise": 0.01,
            "cross_channel": 0.05,
            "ambiguous_signal": 0.08,
        }

    def value_metric(self) -> ValueMetric:
        """Return the PS primary value metric: engagement value in cents."""
        return ValueMetric("engagement_value", "cents")

    def reference_surface(self) -> ReferenceSurface:
        """Return client call notes as the reference surface for PS conversations."""
        return ReferenceSurface("client_call")

    def prose_style_directives(self) -> dict[str, str]:
        """Return per-channel prose generation directives for the prose generator."""
        return {
            "client_call": "Write structured client call notes: attendees, agenda items discussed, decisions made, action items with owners and due dates.",
            "deliverable_review": "Write a formal deliverable review email: reference specific sections, provide constructive feedback with clear acceptance criteria.",
            "status_update": "Write a weekly project status update: RAG status (Red/Amber/Green), accomplishments this week, risks and mitigations, planned activities next week.",
            "email": "Write a formal professional services email with clear subject, structured body, professional sign-off.",
            "slack": "Write a brief Slack message for quick project update or question. Informal but professional.",
            "document_comment": "Write an inline review comment on a consulting deliverable. Specific, actionable, references document section.",
        }

    def brand_voice_variants(self) -> list[BrandVoiceVariant]:
        """
        Return the single PS brand voice variant.

        PS uses a single baseline variant — the vocabulary-overfit smoke test does
        not require multiple brand voice conditions to be effective.
        """
        return [
            BrandVoiceVariant(
                id="bv_baseline",
                label="Brand-voice-agnostic baseline",
                content=(
                    "Write in a professional, collaborative consulting voice. Be precise and structured. "
                    "Acknowledge client concerns with appropriate gravitas. Provide clear recommendations "
                    "backed by analysis. Use formal register appropriate to professional services. "
                    "No specific brand voice constraints beyond professionalism."
                ),
                feature_profile=ps_lex.BRAND_VOICE_FEATURE_PROFILES.get("bv_baseline"),
            ),
        ]

    def knowledge_base_content(self) -> list[KBChunk]:
        """Return the PS KB chunks from the ps_content fixture module."""
        return get_ps_kb_chunks()

    def agent_history_templates(self) -> dict[str, AgentHistoryTemplate]:
        """Return the four PS consultant history templates (same types as SaaS, smaller counts)."""
        return {
            "improving": AgentHistoryTemplate("improving", "upward", 2, "Consultant improving on engagement quality metrics after coaching."),
            "recurring_weakness": AgentHistoryTemplate("recurring_weakness", "flat", 3, "Consultant repeatedly coached on same delivery gap without improvement."),
            "new": AgentHistoryTemplate("new", "none", 0, "New consultant (<14 days). Sparse history."),
            "mixed": AgentHistoryTemplate("mixed", "mixed", 2, "Mixed improvement pattern across criteria."),
        }

    def correction_pattern_set(self) -> list[CorrectionPattern]:
        """
        Return the 5-pattern PS correction set.

        Minimal set covering all four required pattern classes (systematic_upward,
        counterexample, cross_criterion_noise, cross_tenant_boundary) at the
        smallest viable size for smoke testing.
        """
        return [
            CorrectionPattern("empathy", "upward", "systematic_upward", "consultant acknowledged scope issue but PS context requires more formal empathy expression"),
            CorrectionPattern("resolution", "upward", "systematic_upward", "resolution scored low but client email confirmed acceptance; resolution was complete"),
            CorrectionPattern("accuracy", "downward", "counterexample", "consultant cited SOW but claim overgeneralized the scope commitment"),
            CorrectionPattern("brand_voice", "upward", "cross_criterion_noise", "formal register appropriate for executive stakeholder; not overly clinical"),
            CorrectionPattern("empathy", "upward", "cross_tenant_boundary", "tenant B PS correction: empathy upward on technically correct but cold responses"),
        ]

    def get_lexicons(self):
        """Return the ps lexicon module."""
        return ps_lex

    def brand_voice_feature_profiles(self) -> dict:
        """Return calibrated feature profiles from the ps lexicon module."""
        return ps_lex.BRAND_VOICE_FEATURE_PROFILES

    def planted_quality_count(self) -> int:
        """PS plants 15 quality conversations for vocabulary-overfit smoke testing."""
        return 15

    def default_accounts(self) -> int:
        return 5

    def default_months(self) -> int:
        return 3
