"""Quality plan injector — plants rubric-dimension targets into the organic event log."""

import random
from confabra.schemas import (
    QualityPlan, RubricTarget, AccuracyLabel, KnowledgeCitations,
    SimEvent, SimEventType, DaySnapshot, LifecycleStage
)
from confabra.layer1.plan_validator import PlanValidator

# Accuracy label distribution
ACCURACY_DISTRIBUTION = [
    (3, AccuracyLabel(status="supported", precision="exact")),
    (3, AccuracyLabel(status="supported", precision="overgeneralized")),
    (3, AccuracyLabel(status="supported", precision="conditional_applied")),
    (3, AccuracyLabel(status="contradicted", precision="exact")),
]

# All-clean control target
ALL_CLEAN_TARGETS = RubricTarget(
    empathy="high",
    resolution="strong",
    brand_voice_against="bv_baseline",
    brand_voice_target="on_brand",
    accuracy=AccuracyLabel(status="supported", precision="exact"),
)


def _pick_conversation_events(
    events: list[SimEvent],
    rng: random.Random,
    exclude_ids: set[str],
    prefer_health_states: list[str] | None = None,
    snapshots_by_account: dict[str, list[DaySnapshot]] | None = None,
) -> SimEvent | None:
    """
    Pick a CONVERSATION_STARTED event not already assigned to a quality plan.

    Filters the full event list to CONVERSATION_STARTED events that haven't yet
    been claimed by a previous plan, then returns a random choice. Returns None
    when no eligible events remain.
    """
    candidates = [
        e for e in events
        if e.event_type == SimEventType.CONVERSATION_STARTED
        and e.event_id not in exclude_ids
    ]
    if not candidates:
        return None
    return rng.choice(candidates)


def _prose_directive_for_empathy(target: str) -> str:
    """
    Return a plain-English prose generation directive for the empathy dimension.

    ``low`` targets produce terse, transactional exchanges; ``high`` targets
    require explicit acknowledgment of customer emotion followed by concrete action.
    """
    if target == "low":
        return (
            "Agent should be terse and procedural. Focus on steps, not feelings. "
            "Do not acknowledge customer frustration. Avoid emotional language, apologies, or empathy phrases. "
            "Keep responses short and transactional."
        )
    return (
        "Agent should explicitly acknowledge the customer's frustration or difficulty. "
        "Use empathy phrases ('I understand', 'I can see why that's frustrating'). "
        "Follow acknowledgments with clear action steps. Avoid being terse."
    )


def _prose_directive_for_resolution(target: str) -> str:
    """
    Return a plain-English prose generation directive for the resolution dimension.

    ``weak`` targets use deflection and vagueness; ``strong`` targets require
    ownership language and actionable next steps with temporal anchors.
    """
    if target == "weak":
        return (
            "Agent should be vague about resolution. Use deflection phrases like 'check the documentation' "
            "or 'our team will look into it' without committing to specific next steps. "
            "Do not provide a complete solution or temporal anchors."
        )
    return (
        "Agent should provide a complete, specific resolution. Include ownership language ('I will', 'let me'). "
        "Provide actionable next steps with temporal anchors ('by tomorrow', 'within 24 hours'). "
        "Reference the customer's specific issue directly."
    )


def _prose_directive_for_brand_voice(target: str, variant: str) -> str:
    """
    Return a plain-English prose generation directive for the brand voice dimension.

    ``off_brand`` deliberately mismatches the brand voice spec; ``on_brand``
    requires strict adherence to sentence length, question frequency, and vocabulary.
    """
    if target == "off_brand":
        return (
            "Agent should write in a tone that deliberately mismatches the brand voice spec. "
            "If the brand voice is warm-exploratory, use clinical direct language. "
            "If the brand voice is direct-clinical, use warm colloquial language."
        )
    return (
        "Agent should write strictly within the brand voice spec provided. "
        "Match sentence length, question frequency, hedging style, and vocabulary to the spec."
    )


def _prose_directive_for_accuracy(label: AccuracyLabel, kb_chunk_ids: list[str]) -> str:
    """
    Return a plain-English prose generation directive for the accuracy dimension.

    The directive varies by accuracy label: contradicted claims must directly contradict
    the KB; overgeneralized claims omit a key constraint; conditional_applied claims
    correctly apply a policy condition to the customer's context; exact claims mirror
    the KB faithfully.
    """
    if label.status == "contradicted":
        return (
            "Agent should make a claim about refund/policy that directly contradicts the KB. "
            "The claim should be specific and verifiable. "
            f"The relevant KB chunks are: {kb_chunk_ids}."
        )
    if label.precision == "overgeneralized":
        return (
            "Agent should make a claim that is technically supported by the KB but misses an important constraint. "
            "For example, if the policy says 'refunds within 30 days for paid plans only', the agent says 'refunds are available'. "
            f"The relevant KB chunks are: {kb_chunk_ids}."
        )
    if label.precision == "conditional_applied":
        return (
            "Agent should correctly apply the conditional from the KB to the customer's context. "
            "The customer's situation triggers a condition in the policy; the agent must recognize and apply it. "
            f"The relevant KB chunks are: {kb_chunk_ids}."
        )
    # supported, exact
    return (
        "Agent should make an accurate claim that exactly matches the KB policy, including all constraints. "
        f"The relevant KB chunks are: {kb_chunk_ids}."
    )


class QualityPlanInjector:
    """
    Picks conversations from the organic event log and assigns quality plans.

    The injector selects CONVERSATION_STARTED events from the Layer 1 event log
    and assigns rubric-dimension targets + prose generation directives to each.
    Assignments are deterministic given the seed-controlled ``rng``. The injector
    pre-validates each plan against the account snapshot via PlanValidator and
    skips conversations that fail the pre-prompt gate.
    """

    def __init__(self, rng: random.Random, profile_name: str = "saas"):
        """
        Initialise the injector with a seeded RNG and a profile name.

        Args:
            rng:          A seeded ``random.Random`` instance — all shuffles and
                          picks go through this so corpus generation is deterministic.
            profile_name: Profile label (``"saas"`` or ``"ps"``); controls which
                          planted count is expected from the caller.
        """
        self.rng = rng
        self.profile_name = profile_name
        self.validator = PlanValidator()

    def inject(
        self,
        events: list[SimEvent],
        snapshots: list[DaySnapshot],
        planted_count: int = 50,
        kb_chunk_ids: list[str] | None = None,
    ) -> list[QualityPlan]:
        """
        Pick conversations from the event log and generate quality plans.

        Shuffles eligible CONVERSATION_STARTED events (seed-deterministic), then
        assigns one plan spec per event until the ``planted_count`` target is met
        or events are exhausted. Plans that fail the PlanValidator pre-prompt gate
        are silently skipped (the validator failure is a data-consistency signal;
        not all accounts will have a matching snapshot on the conversation day).

        Args:
            events:        Full organic event log from the state machine.
            snapshots:     Full snapshot log from the snapshot emitter.
            planted_count: Target number of planted conversations (50 for SaaS, 15 for PS).
            kb_chunk_ids:  Available KB chunk IDs; defaults to four placeholder IDs that
                           will be replaced by real KB fixture IDs in Phase 2.

        Returns:
            List of QualityPlan records, one per accepted planted conversation.
        """
        # Default KB chunk IDs if not provided (real values come from KB generator in Phase 2)
        if kb_chunk_ids is None:
            kb_chunk_ids = [
                "kb_chunk_refund_policy_v3",
                "kb_chunk_sla_terms_v1",
                "kb_chunk_onboarding_v2",
                "kb_chunk_refund_deny_usage_v1",  # deny_condition chunk
            ]

        # Build snapshot lookup: account_id → snapshots
        snaps_by_account: dict[str, list[DaySnapshot]] = {}
        for snap in snapshots:
            snaps_by_account.setdefault(snap.account_id, []).append(snap)

        # Build conversation events lookup
        conv_events = [e for e in events if e.event_type == SimEventType.CONVERSATION_STARTED]

        if len(conv_events) < planted_count:
            # Fewer conversations than needed — plant what we have
            planted_count = len(conv_events)

        # Shuffle conversation events (seed-deterministic)
        available = list(conv_events)
        self.rng.shuffle(available)

        plans: list[QualityPlan] = []
        assigned_ids: set[str] = set()

        # Build the plan schedule
        schedule = self._build_schedule(planted_count)

        for i, plan_spec in enumerate(schedule):
            if i >= len(available):
                break

            conv_event = available[i]
            assigned_ids.add(conv_event.event_id)

            # Find the account snapshot for this event's day
            account_snaps = snaps_by_account.get(conv_event.account_id, [])
            account_snap = next(
                (s for s in account_snaps if s.day_index == conv_event.day_index),
                account_snaps[-1] if account_snaps else None
            )

            plan = self._build_plan(conv_event, account_snap, plan_spec, kb_chunk_ids)

            # Pre-validate (soft check — we log but don't hard-abort in injector)
            if account_snap:
                account_events = [e for e in events if e.account_id == conv_event.account_id]
                result = self.validator.pre_prompt_validate(plan, account_snap, account_events)
                if result.overall_verdict.value == "fail":
                    # Skip this plan and try next available conversation
                    assigned_ids.discard(conv_event.event_id)
                    continue

            plans.append(plan)

        return plans

    def _build_schedule(self, count: int) -> list[dict]:
        """
        Build an ordered list of plan specifications for each planted conversation.

        Produces exactly ``count`` spec dicts. Starts with 2 all-clean controls,
        then distributes the remainder across 4 rubric dimensions. Accuracy slots
        are filled from ACCURACY_DISTRIBUTION in order (3 each of 4 label types).

        Args:
            count: Total number of plan specs to generate.

        Returns:
            List of spec dicts, each with a ``"type"`` key and dimension-specific fields.
        """
        schedule = []

        # 2 all-clean controls
        for _ in range(min(2, count)):
            schedule.append({"type": "control"})

        remaining = count - len(schedule)

        # Proportional distribution across dimensions
        per_dim = remaining // 4
        leftover = remaining - per_dim * 4

        for i in range(per_dim):
            schedule.append({"type": "empathy", "target": "low" if i % 2 == 0 else "high"})
        for i in range(per_dim):
            schedule.append({"type": "resolution", "target": "weak" if i % 2 == 0 else "strong"})
        for i in range(per_dim):
            schedule.append({"type": "brand_voice", "target": "off_brand" if i % 2 == 0 else "on_brand"})

        # Accuracy distribution — expand each (count_each, label) pair into individual specs.
        # Variable renamed from ``count`` to ``count_each`` to avoid shadowing the parameter.
        accuracy_items = []
        for count_each, label in ACCURACY_DISTRIBUTION:
            for _ in range(count_each):
                accuracy_items.append({"type": "accuracy", "label": label})
        # Take as many as fit in per_dim slots plus any leftover
        for item in accuracy_items[:per_dim + leftover]:
            schedule.append(item)

        return schedule

    def _build_plan(
        self,
        conv_event: SimEvent,
        account_snap: DaySnapshot | None,
        spec: dict,
        kb_chunk_ids: list[str],
    ) -> QualityPlan:
        """
        Build a single QualityPlan from a plan specification dict.

        Derives rubric targets, KB citation constraints, Cat 11 gate labels,
        and prose generation directives from the spec type. The ``conversation_id``
        is deterministic: it embeds the trigger event ID so plans are traceable
        back to specific simulation events without a separate index.

        Args:
            conv_event:   The CONVERSATION_STARTED event this plan targets.
            account_snap: The DaySnapshot for the account on the conversation day,
                          or None if no snapshot exists for that day.
            spec:         Plan specification dict produced by ``_build_schedule``.
            kb_chunk_ids: Available KB chunk IDs for citation constraints.

        Returns:
            A fully-populated QualityPlan ready for pre-prompt validation.
        """
        # Pick KB chunks for this plan
        allow_chunks = [c for c in kb_chunk_ids if "deny" not in c][:2]
        deny_chunks = [c for c in kb_chunk_ids if "deny" in c][:1]

        # Initialise all fields with safe defaults; branches override as needed.
        rubric: RubricTarget
        citations: KnowledgeCitations
        directives: str
        cat11_gate: str | None = None
        multi_chunk: bool = False
        kb_required: list[str] = []
        coaching_dim: str

        if spec["type"] == "control":
            rubric = ALL_CLEAN_TARGETS
            citations = KnowledgeCitations(should_cite=allow_chunks[:1], must_not_cite=[])
            directives = (
                "Write a model customer service interaction. Agent should be empathetic, resolve the issue completely, "
                "write on-brand, and make accurate claims supported by the knowledge base."
            )
            cat11_gate = None
            multi_chunk = False
            kb_required = allow_chunks[:1]
            coaching_dim = "empathy"

        elif spec["type"] == "empathy":
            rubric = RubricTarget(empathy=spec["target"])
            citations = KnowledgeCitations()
            directives = _prose_directive_for_empathy(spec["target"])
            cat11_gate = None
            multi_chunk = False
            kb_required = []
            coaching_dim = "empathy"

        elif spec["type"] == "resolution":
            rubric = RubricTarget(resolution=spec["target"])
            citations = KnowledgeCitations()
            directives = _prose_directive_for_resolution(spec["target"])
            cat11_gate = None
            multi_chunk = False
            kb_required = []
            coaching_dim = "resolution"

        elif spec["type"] == "brand_voice":
            rubric = RubricTarget(
                brand_voice_against="bv_baseline",
                brand_voice_target=spec["target"],
            )
            citations = KnowledgeCitations()
            directives = _prose_directive_for_brand_voice(spec["target"], "bv_baseline")
            cat11_gate = None
            multi_chunk = False
            kb_required = []
            coaching_dim = "brand_voice"

        else:  # accuracy
            label: AccuracyLabel = spec["label"]
            if label.status == "insufficient_information":
                citations = KnowledgeCitations(should_cite=[], must_not_cite=["*"])
                kb_required = []
                multi_chunk = False
            elif label.precision == "conditional_applied":
                # Gate 1: needs allow + deny chunk to test conditional application
                citations = KnowledgeCitations(should_cite=allow_chunks + deny_chunks, must_not_cite=[])
                kb_required = allow_chunks + deny_chunks
                multi_chunk = True
            else:
                citations = KnowledgeCitations(should_cite=allow_chunks[:1], must_not_cite=deny_chunks)
                kb_required = allow_chunks[:1]
                multi_chunk = False
            rubric = RubricTarget(accuracy=label)
            directives = _prose_directive_for_accuracy(label, kb_chunk_ids[:3])
            cat11_gate = (
                "gate_1" if label.precision == "conditional_applied"
                else "gate_3" if label.precision == "overgeneralized"
                else None
            )
            coaching_dim = "accuracy"

        return QualityPlan(
            conversation_id=f"conv_planted_{conv_event.event_id}",
            trigger_event_id=conv_event.event_id,
            rubric_targets=rubric,
            knowledge_citations=citations,
            coaching_target_dimension=coaching_dim,
            prose_generation_directives=directives,
            cat11_gate=cat11_gate,
            multi_chunk_required=multi_chunk,
            kb_chunks_required=kb_required,
        )
