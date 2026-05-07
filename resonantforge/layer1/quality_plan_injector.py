"""Quality plan injector — plants rubric-dimension targets into the organic event log."""

import logging
import math
import random

from resonantforge.kb.constraint_extractor import extract_constraints
from resonantforge.kb.kb_fact_extractor import extract_facts
from resonantforge.kb.negation import negate_fact
from resonantforge.schemas import (
    ConstraintType, KBChunk, PlantedContradiction, QualityPlan, RubricTarget, AccuracyLabel,
    KnowledgeCitations, SimEvent, SimEventType, DaySnapshot, LifecycleStage,
)
from resonantforge.layer1.plan_validator import PlanValidator

_logger = logging.getLogger(__name__)

# Event types eligible for QualityPlan injection.  Raise ValueError on others.
_ELIGIBLE_EVENT_TYPES = {SimEventType.CONVERSATION_STARTED}

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


def _render_chunk_blocks(header: str, chunks: list[KBChunk]) -> str:
    """Render a labeled section of KB chunk content for inline prompt injection."""
    if not chunks:
        return ""
    blocks = "\n---\n".join(f"[{c.chunk_id}]\n{c.chunk_text}" for c in chunks)
    return f"\n\n{header}\n---\n{blocks}\n---"


def _prose_directive_for_accuracy(
    label: AccuracyLabel,
    should_cite: list[KBChunk],
    must_not_cite: list[KBChunk],
    planted_constraint: str | None = None,
    planted_contradiction: PlantedContradiction | None = None,
) -> str:
    """
    Return a plain-English prose generation directive for the accuracy dimension.

    The directive varies by accuracy label: contradicted claims must directly contradict
    the KB; overgeneralized claims omit a key constraint; conditional_applied claims
    correctly apply a policy condition to the customer's context; exact claims mirror
    the KB faithfully.

    For overgeneralized plans, ``planted_constraint`` names the specific constraint to
    drop. This produces a controlled transformation (reliable) rather than an open-ended
    generation directive (unreliable).

    Chunk text is embedded inline so the LLM has the actual policy content to
    ground its claims in — passing only chunk IDs produced zero-claim outputs.
    """
    if label.status == "contradicted":
        if planted_contradiction is not None:
            base = (
                f"Rewrite the agent's response to state the following as confirmed company policy:\n"
                f'"{planted_contradiction.negated_form}"\n\n'
                f"Present this as an accurate statement of company policy. "
                f"Do not mention any alternative policies, time periods, or exceptions."
            )
        else:
            base = (
                "Agent should make a claim that directly contradicts the KB. "
                "The claim should be specific and verifiable."
            )
        base += _render_chunk_blocks("KB policy content (for context):", should_cite)
        return base
    if label.precision == "overgeneralized":
        if planted_constraint:
            base = (
                f"Rewrite the agent's response using the KB policy content below, but REMOVE "
                f"or OMIT this specific constraint: \"{planted_constraint}\"\n\n"
                f"State the policy as if it applies universally — without that limit, condition, "
                f"or restriction. Do not add new claims. Keep the rest of the response unchanged."
            )
        else:
            base = (
                "Agent should make a claim that is technically supported by the KB but "
                "removes or omits any limiting, conditional, or qualifying language "
                "(phrases like 'only', numeric limits, plan restrictions, date windows). "
                "State the policy as if it applies universally without exceptions."
            )
        base += _render_chunk_blocks("Relevant KB policy content:", should_cite)
        base += _render_chunk_blocks("Do not cite or mirror the following stale/adversarial content:", must_not_cite)
        return base
    if label.precision == "conditional_applied":
        base = (
            "Agent should correctly apply the conditional from the KB to the customer's context. "
            "The customer's situation triggers a condition in the policy; the agent must recognize and apply it."
        )
        base += _render_chunk_blocks("Relevant KB policy content:", should_cite)
        return base
    # supported, exact
    base = "Agent should make an accurate claim that exactly matches the KB policy, including all constraints."
    base += _render_chunk_blocks("Relevant KB policy content:", should_cite)
    base += _render_chunk_blocks("Do not cite or mirror the following stale/adversarial content:", must_not_cite)
    return base


def is_conditional_chunk(chunk: KBChunk) -> bool:
    """
    Return True when the chunk has conditional structure ineligible for :exact precision.

    ALLOW_CONDITION and DENY_CONDITION chunks have branching or qualifying clauses
    (e.g. "monthly plans get X, annual plans get Y"). An agent answering correctly
    applies the relevant branch, but cannot recite the full conditional verbatim —
    so :exact is unsatisfiable. Use :conditional_applied instead (RFORGE-10).
    """
    return chunk.constraint_type in {ConstraintType.ALLOW_CONDITION, ConstraintType.DENY_CONDITION}


def _chunk_satisfies_intent(chunk: KBChunk, event_intent: list[str]) -> bool:
    """
    Return True when the chunk's topic is plausibly covered by the event's intent.

    Empty intent_tags matches any intent (backward compatible with pre-RFORGE-11
    chunks).  A non-empty list requires at least one tag to intersect the event's
    intent list — if there is no intersection the chunk is topically misaligned and
    would generate a misleading training signal.

    This function is the core of the satisfiability pre-check added in RFORGE-11.
    It is deliberately simple: the caller (QualityPlanInjector._build_plan) handles
    retry logic and the fallback to kb_required=[].
    """
    if not chunk.intent_tags:
        return True
    return any(tag in event_intent for tag in chunk.intent_tags)


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
        # Pool-starvation telemetry — accumulated during inject(), read by pipeline.
        self.pool_starvation_count: int = 0
        self.pool_starvation_events: list[dict] = []
        self._pool_telemetry_observations: dict[str, list[dict]] = {}
        self.pool_filter_telemetry: dict = {}
        # Conditional-reroute telemetry (RFORGE-10) — :exact swapped to :conditional_applied.
        self.conditional_reroute_count: int = 0
        self.conditional_reroute_events: list[str] = []

    def inject(
        self,
        events: list[SimEvent],
        snapshots: list[DaySnapshot],
        planted_count: int = 50,
        kb_chunks: list[KBChunk] | None = None,
        include_adversarial_in_should_cite: bool = False,
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
            kb_chunks:     Available KBChunk objects from the KB generator.  When any
                           chunk carries a non-empty ``domains`` list the injector uses
                           domain-based filtering; otherwise it falls back to legacy
                           non-adversarial / deny-chunk heuristics.
            include_adversarial_in_should_cite:
                           When True, adversarial chunks may appear in ``should_cite``
                           in addition to ``must_not_cite``.  Default False.

        Returns:
            List of QualityPlan records, one per accepted planted conversation.
        """
        chunks: list[KBChunk] = kb_chunks if kb_chunks is not None else []

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

            plan = self._build_plan(
                conv_event,
                account_snap,
                plan_spec,
                chunks,
                include_adversarial_in_should_cite=include_adversarial_in_should_cite,
            )

            # Pre-validate (soft check — we log but don't hard-abort in injector)
            if account_snap:
                account_events = [e for e in events if e.account_id == conv_event.account_id]
                result = self.validator.pre_prompt_validate(plan, account_snap, account_events)
                if result.overall_verdict.value == "fail":
                    # Skip this plan and try next available conversation
                    assigned_ids.discard(conv_event.event_id)
                    continue

            plans.append(plan)

        # Compute aggregate pool_filter_telemetry from per-plan observations.
        self.pool_filter_telemetry = {}
        for label_key, observations in self._pool_telemetry_observations.items():
            if not observations:
                continue
            initial_sizes = [o["initial_pool_size"] for o in observations]
            fact_sizes = [o["fact_bearing_pool_size"] for o in observations]
            conditional_excluded_counts = [o.get("conditional_excluded", 0) for o in observations]
            self.pool_filter_telemetry[label_key] = {
                "initial_pool_avg": sum(initial_sizes) / len(initial_sizes),
                "after_contradiction_filter_avg": sum(fact_sizes) / len(fact_sizes),
                "final_pool_avg": sum(fact_sizes) / len(fact_sizes),
                "min_final_pool": min(fact_sizes),
                "max_final_pool": max(fact_sizes),
                "conditional_excluded_count": sum(conditional_excluded_counts),
            }

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
        kb_chunks: list[KBChunk],
        *,
        include_adversarial_in_should_cite: bool = False,
    ) -> QualityPlan:
        """
        Build a single QualityPlan from a plan specification dict.

        Derives rubric targets, KB citation constraints, Cat 11 gate labels,
        and prose generation directives from the spec type. The ``conversation_id``
        is deterministic: it embeds the trigger event ID so plans are traceable
        back to specific simulation events without a separate index.

        When the KB is domain-tagged (any chunk has ``domains`` set), chunks are
        filtered to ``event.domain`` candidates before allow/deny splitting.
        Otherwise falls back to the legacy ``"deny" in chunk_id`` heuristic so
        existing tests pass against untagged KB content.

        Args:
            conv_event:   The CONVERSATION_STARTED event this plan targets.
            account_snap: The DaySnapshot for the account on the conversation day,
                          or None if no snapshot exists for that day.
            spec:         Plan specification dict produced by ``_build_schedule``.
            kb_chunks:    Available KBChunk objects.
            include_adversarial_in_should_cite:
                          When True adversarial chunks may land in should_cite.

        Returns:
            A fully-populated QualityPlan ready for pre-prompt validation.

        Raises:
            ValueError: if the event type is not in the eligible allowlist.
            ValueError: if domain is missing from a domain-aware event.
            ValueError: if no KB candidates exist for the event's domain (domain-aware KB only).
        """
        # --- Allowlist check ---
        if conv_event.event_type not in _ELIGIBLE_EVENT_TYPES:
            raise ValueError(
                f"Event type '{conv_event.event_type}' is not eligible for QualityPlan "
                f"injection.  Allowed types: {_ELIGIBLE_EVENT_TYPES}"
            )

        # --- Domain-aware KB selection ---
        domain_aware = any(c.domains for c in kb_chunks) if kb_chunks else False

        if domain_aware:
            domain = conv_event.payload.get("domain", "")
            if not domain:
                raise ValueError(
                    f"Event {conv_event.event_id}: 'domain' missing from payload "
                    f"but KB is domain-tagged — injection cannot proceed."
                )
            domain_candidates = [c for c in kb_chunks if domain in c.domains]
            if not domain_candidates:
                # Legacy state machine emits domain strings ('api', 'billing', 'refunds')
                # that predate the 13-domain vocabulary. Fall back to the full KB pool
                # until PR3 updates the state machine to emit vocabulary-aligned strings.
                domain_candidates = list(kb_chunks)

            # Split into allow pool (non-adversarial) and deny pool.
            # DENY_CONDITION chunks are always deny-pool regardless of adversarial flag.
            deny_pool = [
                c for c in domain_candidates
                if c.adversarial or c.constraint_type == ConstraintType.DENY_CONDITION
            ]
            allow_pool = [c for c in domain_candidates if c not in deny_pool]

            # Within-topic normalization: cap the pick pool so chunk-count-rich domains
            # don't dominate selection diversity across the corpus.
            if include_adversarial_in_should_cite:
                # When opted in, adversarial chunks compete for should_cite slots too.
                # Cap applies to ALL domain candidates.
                pick_pool = sorted(domain_candidates, key=lambda c: c.chunk_id)
                cap = max(3, math.ceil(math.sqrt(len(pick_pool))))
                pick_pool_capped = pick_pool[:cap]
                deny_pool_for_must_not: list[KBChunk] = []  # no must_not_cite when opted in
            else:
                # Default: only non-adversarial chunks for should_cite;
                # adversarial/deny chunks go to must_not_cite.
                pick_pool = sorted(allow_pool, key=lambda c: c.chunk_id)
                cap = max(3, math.ceil(math.sqrt(len(pick_pool)))) if pick_pool else 3
                pick_pool_capped = pick_pool[:cap]
                deny_pool_for_must_not = sorted(deny_pool, key=lambda c: c.chunk_id)

            def _pick_from(pool: list[KBChunk]) -> KBChunk | None:
                """Seed-based deterministic pick from a sorted candidate pool."""
                if not pool:
                    return None
                idx = self.rng.randint(0, len(pool) - 1)
                return pool[idx]

            picked = _pick_from(pick_pool_capped)

            # --- Satisfiability pre-check (RFORGE-11) ---
            # Verify that the picked allow-pool chunk's topic is plausibly covered by the
            # event's intent.  If not, retry with up to N=2 additional RNG picks from the
            # same capped pool, consuming additional RNG values from the seeded generator.
            #
            # Fallback policy: if all retries fail, drop kb_required to [].
            # Wrong-chunk plans generate misleading training signal.  Skip rate may increase
            # temporarily; data quality goes up.
            #
            # Seed note: the seed is preserved across runs but the result may differ from
            # pre-fix runs if retries are triggered.  This is expected and intentional —
            # the pre-fix runs produced topically misaligned plans; the post-fix runs do not.
            event_intent: list[str] = conv_event.payload.get("intent", [])
            _MAX_SATISFIABILITY_RETRIES = 2

            if picked is not None and not _chunk_satisfies_intent(picked, event_intent):
                retried: KBChunk | None = None
                for _ in range(_MAX_SATISFIABILITY_RETRIES):
                    candidate = _pick_from(pick_pool_capped)
                    if candidate is not None and _chunk_satisfies_intent(candidate, event_intent):
                        retried = candidate
                        break
                # If all retries failed, fall back to no required chunk rather than ship a
                # mismatched plan that the accuracy validator will correctly flag as not_found.
                picked = retried  # None if all retries exhausted

            picked_deny_chunk = _pick_from(deny_pool_for_must_not)

            allow_ids = [picked.chunk_id] if picked is not None else []
            deny_ids: list[str] = [picked_deny_chunk.chunk_id] if picked_deny_chunk is not None else []

            # All domain candidates as chunk_id list for prose directives.
            all_candidate_ids = [c.chunk_id for c in sorted(domain_candidates, key=lambda c: c.chunk_id)]
        else:
            # --- Legacy fallback for untagged KB (e.g. real saas_content.py) ---
            allow_ids = [c.chunk_id for c in kb_chunks if "deny" not in c.chunk_id and not c.adversarial][:2]
            deny_ids = [c.chunk_id for c in kb_chunks if "deny" in c.chunk_id or c.adversarial][:1]
            all_candidate_ids = allow_ids + deny_ids

        # Initialise all fields with safe defaults; branches override as needed.
        rubric: RubricTarget
        citations: KnowledgeCitations
        directives: str
        cat11_gate: str | None = None
        multi_chunk: bool = False
        kb_required: list[str] = []
        coaching_dim: str
        planted_constraint: str | None = None
        planted_contradiction: PlantedContradiction | None = None

        if spec["type"] == "control":
            rubric = ALL_CLEAN_TARGETS
            citations = KnowledgeCitations(should_cite=allow_ids[:1], must_not_cite=[])
            _ctrl_preamble = (
                "Write a model customer service interaction. Agent should be empathetic, resolve the issue completely, "
                "write on-brand, and make accurate claims supported by the knowledge base."
            )
            _ctrl_chunk_map = {c.chunk_id: c for c in kb_chunks}
            _ctrl_chunks = [_ctrl_chunk_map[x] for x in allow_ids[:1] if x in _ctrl_chunk_map]
            directives = _ctrl_preamble + _render_chunk_blocks("Relevant KB policy content:", _ctrl_chunks)
            cat11_gate = None
            multi_chunk = False
            kb_required = allow_ids[:1]
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
            planted_constraint: str | None = None
            if label.status == "insufficient_information":
                citations = KnowledgeCitations(should_cite=[], must_not_cite=["*"])
                kb_required = []
                multi_chunk = False
            elif label.precision == "conditional_applied":
                # Gate 1: needs allow + deny chunk to test conditional application
                combined = allow_ids + deny_ids
                citations = KnowledgeCitations(should_cite=combined, must_not_cite=[])
                kb_required = combined
                multi_chunk = True
            elif label.precision == "overgeneralized":
                # Pre-filter: only chunks with extractable constraints qualify.
                # If the picked chunk has no constraints, retry up to 2 times;
                # fall back to kb_required=[] rather than ship a constraint-free plan.
                chunks_by_id = {c.chunk_id: c for c in kb_chunks}
                allow_chunk = chunks_by_id.get(allow_ids[0]) if allow_ids else None

                if allow_chunk is not None:
                    constraints = extract_constraints(allow_chunk.chunk_text)
                    if not constraints:
                        # Retry from pick_pool_capped for up to 2 additional candidates
                        if domain_aware:
                            for _ in range(2):
                                candidate = _pick_from(pick_pool_capped)
                                if candidate is not None:
                                    candidate_constraints = extract_constraints(candidate.chunk_text)
                                    if candidate_constraints:
                                        allow_chunk = candidate
                                        constraints = candidate_constraints
                                        allow_ids = [candidate.chunk_id]
                                        break
                            else:
                                # All retries exhausted — fall back
                                allow_chunk = None
                                constraints = []
                        else:
                            allow_chunk = None
                            constraints = []

                if allow_chunk is not None and constraints:
                    # Deterministic pick: use seeded RNG index into sorted constraint list
                    sorted_constraints = sorted(constraints, key=lambda c: c.normalized)
                    idx = self.rng.randint(0, len(sorted_constraints) - 1)
                    planted_constraint = sorted_constraints[idx].normalized
                    citations = KnowledgeCitations(should_cite=[allow_chunk.chunk_id], must_not_cite=deny_ids)
                    kb_required = [allow_chunk.chunk_id]
                else:
                    citations = KnowledgeCitations(should_cite=[], must_not_cite=deny_ids)
                    kb_required = []
                multi_chunk = False
            elif label.status == "contradicted" and label.precision == "exact":
                # Pre-filter: only chunks with extractable, contradictable facts qualify.
                # RFORGE-10: also exclude conditional chunks — negating a branch of a conditional
                # produces unsatisfiable plans. The reroute path for conditional+exact only covers
                # supported plans; contradicted:exact keeps its own dedicated pool filter here.
                non_conditional_pool = [c for c in pick_pool_capped if not is_conditional_chunk(c)]
                conditional_excluded = len(pick_pool_capped) - len(non_conditional_pool)
                fact_bearing_pool = [c for c in non_conditional_pool if extract_facts(c.chunk_text)]
                initial_pool_size = len(pick_pool_capped)
                fact_pool_size = len(fact_bearing_pool)

                # Pool instrumentation: warn when diversity is too low for reliable seeding.
                if fact_pool_size < 3:
                    _logger.warning(
                        "pool_starvation: fact-bearing pool=%d < 3 for contradicted:exact event_id=%s",
                        fact_pool_size,
                        conv_event.event_id,
                    )
                    self.pool_starvation_count += 1
                    self.pool_starvation_events.append({
                        "event_id": conv_event.event_id,
                        "plan_type": "contradicted:exact",
                        "filter_stage": "fact_bearing",
                        "pool_size_at_failure": fact_pool_size,
                    })

                # Record per-plan observation for aggregate telemetry.
                label_key = "contradicted:exact"
                if label_key not in self._pool_telemetry_observations:
                    self._pool_telemetry_observations[label_key] = []
                self._pool_telemetry_observations[label_key].append({
                    "initial_pool_size": initial_pool_size,
                    "fact_bearing_pool_size": fact_pool_size,
                    "conditional_excluded": conditional_excluded,
                })

                allow_chunk_c: KBChunk | None = _pick_from(fact_bearing_pool)

                if allow_chunk_c is not None:
                    facts = extract_facts(allow_chunk_c.chunk_text)
                    sorted_facts = sorted(facts, key=lambda f: f.normalized)
                    fact_idx = self.rng.randint(0, len(sorted_facts) - 1)
                    picked_fact = sorted_facts[fact_idx]
                    negated = negate_fact(picked_fact)
                    planted_contradiction = PlantedContradiction(
                        kb_fact=picked_fact.normalized,
                        negated_form=negated,
                        fact_category=picked_fact.fact_category,
                    )
                    citations = KnowledgeCitations(
                        should_cite=[allow_chunk_c.chunk_id], must_not_cite=deny_ids
                    )
                    kb_required = [allow_chunk_c.chunk_id]
                else:
                    # No fact-bearing chunk available — fall back rather than ship a groundless plan.
                    citations = KnowledgeCitations(should_cite=[], must_not_cite=deny_ids)
                    kb_required = []
                multi_chunk = False
            else:
                citations = KnowledgeCitations(should_cite=allow_ids[:1], must_not_cite=deny_ids)
                kb_required = allow_ids[:1]
                multi_chunk = False
            rubric = RubricTarget(accuracy=label)
            chunks_by_id = {c.chunk_id: c for c in kb_chunks}
            should_cite_chunks = [chunks_by_id[cid] for cid in citations.should_cite if cid in chunks_by_id]
            must_not_cite_chunks = [chunks_by_id[cid] for cid in citations.must_not_cite if cid in chunks_by_id]
            directives = _prose_directive_for_accuracy(
                label, should_cite_chunks, must_not_cite_chunks,
                planted_constraint=planted_constraint,
                planted_contradiction=planted_contradiction,
            )
            cat11_gate = (
                "gate_1" if label.precision == "conditional_applied"
                else "gate_3" if label.precision == "overgeneralized"
                else None
            )
            coaching_dim = "accuracy"

        # RFORGE-10: reroute :exact → :conditional_applied for any plan paired with a
        # conditional chunk. Applies to all spec types including control plans (which
        # hardcode accuracy=supported:exact via ALL_CLEAN_TARGETS).
        if (
            rubric.accuracy is not None
            and rubric.accuracy.precision == "exact"
            and kb_required
        ):
            _rc = next((c for c in kb_chunks if c.chunk_id == kb_required[0]), None)
            if _rc is not None and is_conditional_chunk(_rc):
                _new_acc = AccuracyLabel(status=rubric.accuracy.status, precision="conditional_applied")
                rubric = rubric.model_copy(update={"accuracy": _new_acc})
                if spec["type"] == "accuracy":
                    # Re-generate directives to reflect the rerouted precision.
                    _cbi = {c.chunk_id: c for c in kb_chunks}
                    _sc = [_cbi[x] for x in citations.should_cite if x in _cbi]
                    _mnc = [_cbi[x] for x in citations.must_not_cite if x in _cbi]
                    directives = _prose_directive_for_accuracy(
                        _new_acc, _sc, _mnc,
                        planted_constraint=planted_constraint,
                        planted_contradiction=planted_contradiction,
                    )
                self.conditional_reroute_count += 1
                self.conditional_reroute_events.append(conv_event.event_id)

        return QualityPlan(
            conversation_id=f"conv_{conv_event.event_id}",
            trigger_event_id=conv_event.event_id,
            rubric_targets=rubric,
            knowledge_citations=citations,
            coaching_target_dimension=coaching_dim,
            prose_generation_directives=directives,
            cat11_gate=cat11_gate,
            multi_chunk_required=multi_chunk,
            kb_chunks_required=kb_required,
            planted_constraint=planted_constraint,
            planted_contradiction=planted_contradiction,
        )
