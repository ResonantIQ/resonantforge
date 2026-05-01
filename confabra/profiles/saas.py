"""SaaS (Software-as-a-Service) Layer 2 profile — primary, full coverage."""
from __future__ import annotations
from confabra.profiles.base import (
    Profile, LifecycleStageConfig, SignalDescriptor, SurfaceChannel,
    PersonaArchetype, AgentHistoryTemplate, CorrectionPattern, ValueMetric, ReferenceSurface
)
from confabra.schemas import BrandVoiceVariant, CoachingStyleOverlay, KBChunk
from confabra.profiles.lexicons import saas as saas_lex
from confabra.kb.saas_content import get_saas_kb_chunks

# Canonical plan-tier names for this profile. Detectors, validators, and the
# invariant checker all import from here — never hardcode tier names elsewhere.
CANONICAL_TIER_NAMES: frozenset[str] = frozenset({"Starter", "Growth", "Enterprise"})


class SaaSProfile(Profile):
    """
    SaaS profile — 25 accounts × 6 months, full Cat coverage.

    Industry: B2B SaaS customer support.
    Reference surface: Intercom conversation.
    Primary AI surfaces: scorer, coaching notes, health signals, customer timeline.

    This is the primary Layer 2 profile.  It defines the widest signal vocabulary,
    the most surface channels, and the full 16-pattern correction set that exercises
    all four Cat L1 signal-detection categories (systematic_upward, counterexample,
    cross_criterion_noise, cross_tenant_boundary).
    """

    @property
    def name(self) -> str:
        return "saas"

    @property
    def version(self) -> str:
        return "1.1.0"

    def lifecycle_stages(self) -> list[str]:
        """Return the four SaaS account lifecycle stages."""
        return ["onboarding", "active", "at_risk", "churned"]

    def lifecycle_stage_configs(self) -> list[LifecycleStageConfig]:
        """Return detailed lifecycle stage configs with health ranges and event rates."""
        return [
            LifecycleStageConfig("onboarding", "Onboarding", (0.6, 1.0), 0.8),
            LifecycleStageConfig("active", "Active", (0.5, 1.0), 0.4),
            LifecycleStageConfig("at_risk", "At Risk", (0.2, 0.6), 0.6),
            LifecycleStageConfig("churned", "Churned", (0.0, 0.3), 0.1),
        ]

    def signal_vocabulary(self) -> dict[str, SignalDescriptor]:
        """Return the eight SaaS-specific signal descriptors."""
        return {
            "churn_intent": SignalDescriptor(
                "churn_intent", "Customer expresses intent to cancel or not renew", "embedding"
            ),
            "escalation_detected": SignalDescriptor(
                "escalation_detected", "Conversation escalated to higher support tier", "embedding"
            ),
            "payment_risk": SignalDescriptor(
                "payment_risk", "Payment failure or credit card issue detected", "rule"
            ),
            "renewal_risk": SignalDescriptor(
                "renewal_risk", "Renewal approaching with declining health indicators", "rule"
            ),
            "product_adoption_low": SignalDescriptor(
                "product_adoption_low", "Feature usage below expected baseline for plan tier", "rule"
            ),
            "positive_outcome": SignalDescriptor(
                "positive_outcome", "Successful resolution, upsell, or feature adoption", "llm"
            ),
            "feature_request": SignalDescriptor(
                "feature_request", "Customer requests a feature not currently available", "llm"
            ),
            "bug_report": SignalDescriptor(
                "bug_report", "Customer reports a reproducible product defect", "llm"
            ),
        }

    def surface_channels(self) -> list[SurfaceChannel]:
        """Return the six surface channels available in SaaS conversations."""
        return [
            SurfaceChannel("intercom", "Intercom Conversation", "Conversational chat, informal but professional, typically 5-15 turns"),
            SurfaceChannel("hubspot", "HubSpot Deal Note", "CRM notes, terse and factual, 1-3 paragraphs, past-tense summary"),
            SurfaceChannel("zendesk", "Zendesk Ticket", "Formal ticket format, numbered steps for technical issues, email-like"),
            SurfaceChannel("zoom", "Zoom Transcript", "Spoken-word transcript, includes filler words, interruptions, informal register"),
            SurfaceChannel("email", "Email Thread", "Professional email format, clear subject, structured body with action items"),
            SurfaceChannel("in_app", "In-App Message", "Short messages, often single-turn, mobile-first phrasing"),
        ]

    def persona_pool(self) -> list[PersonaArchetype]:
        """Return the six SaaS agent persona archetypes."""
        return [
            PersonaArchetype("senior_support", "Senior Support Engineer", 0.8, 0.9, 0.7),
            PersonaArchetype("junior_cs", "Junior Customer Success Rep", 0.9, 0.6, 0.5),
            PersonaArchetype("technical_specialist", "Technical Integration Specialist", 0.5, 0.9, 0.8),
            PersonaArchetype("account_manager", "Account Manager", 0.7, 0.7, 0.7),
            PersonaArchetype("onboarding_specialist", "Onboarding Specialist", 0.85, 0.75, 0.6),
            PersonaArchetype("billing_support", "Billing Support", 0.6, 0.85, 0.7),
        ]

    def perturbation_density(self) -> dict[str, float]:
        """
        Return per-class perturbation densities for SaaS conversations.

        Fractions are calibrated so the corpus contains realistic noise without
        overwhelming the signal: adversarial inputs (5%), sarcasm (3%),
        negation (4%), typos/transcription noise (2%), cross-channel (8%),
        ambiguous signals (10%).
        """
        return {
            "adversarial_input": 0.05,
            "sarcasm": 0.03,
            "negation": 0.04,
            "noise": 0.02,
            "cross_channel": 0.08,
            "ambiguous_signal": 0.10,
        }

    def value_metric(self) -> ValueMetric:
        """Return the SaaS primary value metric: monthly ARR in cents."""
        return ValueMetric("monthly_arr", "cents")

    def reference_surface(self) -> ReferenceSurface:
        """Return Intercom as the reference surface for SaaS conversations."""
        return ReferenceSurface("intercom")

    def prose_style_directives(self) -> dict[str, str]:
        """Return per-channel prose generation directives for the prose generator."""
        return {
            "intercom": (
                "Write a realistic Intercom customer support conversation. "
                "Use a conversational, professional tone. Include customer greetings and sign-offs. "
                "Format as: Customer: [message] Agent: [message]. 5-15 turns typical."
            ),
            "hubspot": (
                "Write a HubSpot CRM note summarizing a customer interaction. "
                "Past tense, terse, factual. Include: date, issue discussed, resolution status, next actions. "
                "1-3 short paragraphs. No greeting or sign-off."
            ),
            "zendesk": (
                "Write a Zendesk support ticket thread. "
                "Customer description in first message, then agent response with numbered troubleshooting steps. "
                "Formal email-like format with ticket reference numbers."
            ),
            "zoom": (
                "Write a Zoom call transcript for a customer success call. "
                "Include natural speech patterns, occasional filler words. "
                "Format as: [AGENT NAME]: [speech] [CUSTOMER NAME]: [speech]"
            ),
            "email": (
                "Write a professional email thread about a customer issue or account review. "
                "Clear subject line, structured body, formal sign-off. 2-4 emails in thread."
            ),
            "in_app": (
                "Write a short in-app message exchange (2-4 messages). "
                "Mobile-first, very brief, single-purpose. May use simple markdown."
            ),
        }

    def brand_voice_variants(self) -> list[BrandVoiceVariant]:
        """
        Return the three SaaS brand voice variants.

        - bv_warm_exploratory: warm, curious, collaborative — exercises high empathy + question_count signal
        - bv_direct_clinical: precise, efficient, formal — exercises low empathy + directive_terms signal
        - bv_baseline: neutral professional — used as the control condition
        """
        return [
            BrandVoiceVariant(
                id="bv_warm_exploratory",
                label="Warm and exploratory",
                content=(
                    "Our voice is warm, curious, and human. We treat every customer as a real person "
                    "with a real problem, not a ticket to close. Our agents explore issues collaboratively — "
                    "they ask questions to understand the full picture before jumping to solutions. "
                    "We use natural language, contractions, and first-person plural to convey partnership "
                    "('let's figure this out together'). We're genuinely enthusiastic about helping and "
                    "that comes through without being performative. We acknowledge feelings before facts. "
                    "We avoid jargon unless the customer uses it first. Our sign-offs are warm and personal. "
                    "Length: we don't rush — a slightly longer message that truly helps is better than a "
                    "terse message that leaves questions open. Questions to the customer are encouraged "
                    "to ensure we've understood correctly before acting."
                ),
                feature_profile=saas_lex.BRAND_VOICE_FEATURE_PROFILES.get("bv_warm_exploratory"),
            ),
            BrandVoiceVariant(
                id="bv_direct_clinical",
                label="Direct and clinical",
                content=(
                    "Our voice is precise, efficient, and action-oriented. Every message moves the "
                    "conversation forward. We state facts clearly, provide numbered steps when applicable, "
                    "and avoid filler language. We use formal register — no contractions, no colloquialisms. "
                    "We do not speculate or hedge: if we don't know, we say so and escalate. "
                    "We address the technical issue first, then the process around it. "
                    "We use active voice and imperative constructions for instructions. "
                    "We do not use exclamation marks or emotionally-loaded language. "
                    "Length: messages are as short as the situation requires — we eliminate any word "
                    "that doesn't add information. We do not ask clarifying questions unless truly necessary; "
                    "we act on the most reasonable interpretation of the customer's request."
                ),
                feature_profile=saas_lex.BRAND_VOICE_FEATURE_PROFILES.get("bv_direct_clinical"),
            ),
            BrandVoiceVariant(
                id="bv_baseline",
                label="Brand-voice-agnostic baseline",
                content=(
                    "Write in a professional, helpful customer service voice. Be clear and accurate. "
                    "Acknowledge the customer's situation appropriately. Provide complete, actionable responses. "
                    "Match the formality level to the channel (chat is more casual than email). "
                    "No specific brand voice constraints — optimize for clarity and helpfulness."
                ),
                feature_profile=saas_lex.BRAND_VOICE_FEATURE_PROFILES.get("bv_baseline"),
            ),
        ]

    def knowledge_base_content(self) -> list[KBChunk]:
        """Return the SaaS KB chunks from the saas_content fixture module."""
        return get_saas_kb_chunks()

    def agent_history_templates(self) -> dict[str, AgentHistoryTemplate]:
        """
        Return the four SaaS agent history templates.

        Types: improving (upward trajectory after coaching), recurring_weakness
        (flat despite repeated coaching), new (no history), mixed (one improving
        + one chronic weak + one uncoached criterion).
        """
        return {
            "improving": AgentHistoryTemplate(
                "improving",
                "upward",
                coaching_event_count=3,
                description="Agent showing measurable improvement after coaching. Score trajectory is upward on coached dimension.",
            ),
            "recurring_weakness": AgentHistoryTemplate(
                "recurring_weakness",
                "flat",
                coaching_event_count=4,
                description="Agent coached 3+ times on same criterion with no improvement. Chronic performance gap.",
            ),
            "new": AgentHistoryTemplate(
                "new",
                "none",
                coaching_event_count=0,
                description="New agent (<14 days tenure). Sparse history, no coaching events, no causal markers.",
            ),
            "mixed": AgentHistoryTemplate(
                "mixed",
                "mixed",
                coaching_event_count=3,
                description="One criterion improving (with causal anchors), another in recurring-weakness pattern, third uncoached.",
            ),
        }

    def correction_pattern_set(self) -> list[CorrectionPattern]:
        """
        Return the 16-pattern SaaS correction set.

        Distribution by class:
        - 5 systematic_upward: the primary signal the intelligence layer must detect
        - 3 counterexample: opposing direction corrections to prevent false positives
        - 5 cross_criterion_noise: corrections on unrelated criteria (should not trigger spurious learning)
        - 3 cross_tenant_boundary: planted for isolation testing in Cat L1
        """
        return [
            # Systematic upward patterns (5)
            CorrectionPattern("empathy", "upward", "systematic_upward", "agent was terse but customer ended satisfied; empathy reads low but outcome was positive"),
            CorrectionPattern("empathy", "upward", "systematic_upward", "agent did not use acknowledgment phrases but body of response showed genuine care"),
            CorrectionPattern("resolution", "upward", "systematic_upward", "agent resolved the issue without explicit ownership language; resolution was complete"),
            CorrectionPattern("brand_voice", "upward", "systematic_upward", "brand voice scoring too strict on formality; this agent's casual register was appropriate for the channel"),
            CorrectionPattern("accuracy", "upward", "systematic_upward", "agent's claim was supported by KB even though exact phrasing differed from chunk text"),
            # Counterexamples (3 — opposing systematic direction)
            CorrectionPattern("empathy", "downward", "counterexample", "agent used acknowledgment phrases but did not follow through; empathy was performative"),
            CorrectionPattern("resolution", "downward", "counterexample", "agent said would resolve but did not provide actionable steps; resolution incomplete"),
            CorrectionPattern("accuracy", "downward", "counterexample", "agent cited policy correctly but overgeneralized the constraint; precision was insufficient"),
            # Cross-criterion noise (5 — corrections on unrelated criteria, should not cause spurious learning)
            CorrectionPattern("empathy", "upward", "cross_criterion_noise", "customer used informal language so agent matched register; not a brand voice issue"),
            CorrectionPattern("brand_voice", "downward", "cross_criterion_noise", "long response appropriate given complexity; not a resolution weakness"),
            CorrectionPattern("resolution", "upward", "cross_criterion_noise", "deflection language present but customer preferred self-service; was appropriate"),
            CorrectionPattern("accuracy", "upward", "cross_criterion_noise", "outdated policy cited but information given was still factually correct at time of interaction"),
            CorrectionPattern("empathy", "downward", "cross_criterion_noise", "customer was angry; agent stayed professional rather than emotional; appropriate"),
            # Cross-tenant boundary (3 — for isolation testing in Cat L1)
            CorrectionPattern("empathy", "upward", "cross_tenant_boundary", "tenant B systematic correction: empathy upward on efficient agents"),
            CorrectionPattern("resolution", "upward", "cross_tenant_boundary", "tenant B systematic correction: resolution upward when ticket closed same day"),
            CorrectionPattern("accuracy", "downward", "cross_tenant_boundary", "tenant B systematic correction: accuracy downward when KB not explicitly cited"),
        ]

    def brand_voice_feature_profiles(self) -> dict:
        """Return calibrated feature profiles from the saas lexicon module."""
        return saas_lex.BRAND_VOICE_FEATURE_PROFILES

    # ------------------------------------------------------------------
    # Lexicon accessors — delegate to saas lexicon module constants
    # ------------------------------------------------------------------

    def acknowledgment_phrases(self) -> list[str]:
        return saas_lex.ACKNOWLEDGMENT_PHRASES

    def emotion_lexicon(self) -> list[str]:
        return saas_lex.EMOTION_LEXICON

    def apology_lexicon(self) -> list[str]:
        return saas_lex.APOLOGY_LEXICON

    def action_verb_lexicon(self) -> list[str]:
        return saas_lex.ACTION_VERB_LEXICON

    def hedging_lexicon(self) -> list[str]:
        return saas_lex.HEDGING_LEXICON

    def directive_lexicon(self) -> list[str]:
        return saas_lex.DIRECTIVE_LEXICON

    def warm_terms(self) -> list[str]:
        return saas_lex.WARM_TERMS

    def clinical_terms(self) -> list[str]:
        return saas_lex.CLINICAL_TERMS

    def contraction_patterns(self) -> list[str]:
        return saas_lex.CONTRACTION_PATTERNS

    def resolution_patterns(self) -> list[str]:
        return saas_lex.RESOLUTION_PATTERNS

    def deflection_patterns(self) -> list[str]:
        return saas_lex.DEFLECTION_PATTERNS

    def next_steps_patterns(self) -> list[str]:
        return saas_lex.NEXT_STEPS_PATTERNS

    def temporal_anchor_patterns(self) -> list[str]:
        return saas_lex.TEMPORAL_ANCHOR_PATTERNS

    def specific_actor_patterns(self) -> list[str]:
        return saas_lex.SPECIFIC_ACTOR_PATTERNS

    def ownership_patterns(self) -> list[str]:
        return saas_lex.OWNERSHIP_PATTERNS

    def issue_keywords(self) -> list[str]:
        return saas_lex.ISSUE_KEYWORDS

    def synonym_map(self) -> dict[str, str]:
        return saas_lex.SYNONYM_MAP

    def planted_quality_count(self) -> int:
        """SaaS plants 50 quality conversations to exercise full Cat coverage."""
        return 50

    def default_accounts(self) -> int:
        return 25

    def default_months(self) -> int:
        return 6
