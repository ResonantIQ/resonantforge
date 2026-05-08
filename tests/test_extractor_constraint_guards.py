"""
Tests for RFORGE-39 and RFORGE-40: extractor constraint-check guards.

RFORGE-39: Skip constraint check for INFORMATIONAL chunks.
  INFORMATIONAL chunks provide context, not enforceable policy conditions.
  _extract_constraints_from_text can find incidental numeric phrases (e.g. "up to 30"
  from a queuing timeout) that never appear in realistic claims, causing false
  constraint_preserved=False on every correctly-applied response.

RFORGE-40: Don't flag overgeneralization when claim best alignment is "contradicted".
  In ALLOW+DENY multi-chunk plans, the agent correctly applies the DENY scenario
  (e.g. annual plan, no refund). The ALLOW chunk's constraint ("within 30 days")
  is absent from the claim — correctly. But with the global overgeneralization
  accumulator, the ALLOW chunk fires first and sets overall_overgeneralization=True.
  The guard at line 592 prevents the DENY chunk from adding more, but can't undo
  what the ALLOW chunk already set.

  Fix: track overgeneralization per claim. If a claim's best alignment is
  "contradicted", discard any per-claim overgeneralization — the agent correctly
  applied the DENY scenario.
"""
from __future__ import annotations

from resonantforge.schemas import Claim, ConstraintType, KBChunk
from resonantforge.validators.extractors.accuracy import (
    _check_chunk_relevance,
    run_kb_alignment_pipeline,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

_API_RATE_LIMITS_INFORMATIONAL = KBChunk(
    chunk_id="kb_api_rate_limits",
    document_id="doc_api",
    document_path="policies/api.md",
    chunk_text=(
        "API rate limits differ by plan: Starter plans receive 1,000 requests per minute; "
        "Enterprise plans receive 10,000 requests per minute. Requests that exceed the limit "
        "are queued for up to 30 seconds before returning a 429 Too Many Requests error. "
        "Burst headroom is not available on Starter."
    ),
    constraint_type=ConstraintType.INFORMATIONAL,
    domains=["api_and_webhooks"],
)

_REFUND_ALLOW_CHUNK = KBChunk(
    chunk_id="kb_refund_allow",
    document_id="doc_refund",
    document_path="policies/refund.md",
    chunk_text=(
        "Refunds are available within 30 days of purchase for all paid plan customers. "
        "Eligible customers receive a full refund to the original payment method within "
        "5-7 business days of an approved request."
    ),
    constraint_type=ConstraintType.ALLOW_CONDITION,
    domains=["refund_policy"],
)

_REFUND_DENY_CHUNK = KBChunk(
    chunk_id="kb_refund_deny_annual",
    document_id="doc_refund",
    document_path="policies/refund.md",
    chunk_text=(
        "Refunds are available on monthly plans only. Annual plan customers are NOT eligible "
        "for monetary refunds. Annual plan customers who wish to discontinue should contact "
        "their account manager to discuss plan credit options applicable to future purchases."
    ),
    constraint_type=ConstraintType.DENY_CONDITION,
    domains=["refund_policy"],
)


def _make_claim(text: str, subject: str = "customer", obj: str | None = None) -> Claim:
    obj = obj or text
    return Claim(
        claim_text=text,
        claim_span=(0, len(text)),
        claim_type="policy",
        normalized_subject=subject,
        normalized_predicate="receives",
        normalized_object=obj,
        alignment=None,
    )


# ── RFORGE-39: INFORMATIONAL chunks skip constraint check ────────────────────


def test_informational_chunk_constraint_preserved_true_for_partial_claim():
    """
    INFORMATIONAL chunk: claim covers the topic but omits the queuing timeout detail.
    constraint_preserved must be True — INFORMATIONAL chunks don't enforce constraints.
    """
    claim = _make_claim(
        "Starter plan customers receive 1,000 requests per minute via the API.",
        subject="starter plan customer api",
        obj="requests minute api",
    )
    result = _check_chunk_relevance(claim, _API_RATE_LIMITS_INFORMATIONAL, {})
    assert result["relevant"], "Claim should be relevant to the API rate limits chunk"
    assert result["constraint_preserved"], (
        "INFORMATIONAL chunk must not trigger constraint check — "
        f"claim correctly states rate limit without queuing detail. Got: {result}"
    )


def test_informational_chunk_no_overgeneralization_in_pipeline():
    """
    End-to-end: INFORMATIONAL chunk with numeric content — no overgeneralization flag.
    """
    claim = _make_claim(
        "Enterprise plan customers get 10,000 API requests per minute.",
        subject="enterprise plan customer api",
        obj="requests minute api",
    )
    signals = run_kb_alignment_pipeline(
        claims=[claim],
        kb_chunks=[_API_RATE_LIMITS_INFORMATIONAL],
        conversation_context="",
        kb_chunks_required=["kb_api_rate_limits"],
        synonym_map={},
        planted_constraint=None,
    )
    assert signals.overgeneralization_flag is False, (
        "INFORMATIONAL chunk triggered overgeneralization_flag — "
        f"constraint check must be skipped for INFORMATIONAL chunks. signals={signals}"
    )
    assert signals.constraint_preserved is True, (
        f"constraint_preserved=False for INFORMATIONAL chunk. signals={signals}"
    )


def test_allow_condition_chunk_still_checks_constraints():
    """
    Regression guard: ALLOW_CONDITION chunks still enforce constraint check.
    Claim drops 'within 30 days' → overgeneralization_flag=True.
    """
    claim = _make_claim(
        "Paid plan customers are eligible for a full refund.",
        subject="customer refund",
        obj="refund",
    )
    signals = run_kb_alignment_pipeline(
        claims=[claim],
        kb_chunks=[_REFUND_ALLOW_CHUNK],
        conversation_context="",
        kb_chunks_required=["kb_refund_allow"],
        synonym_map={},
        planted_constraint=None,
    )
    assert signals.overgeneralization_flag is True, (
        "ALLOW_CONDITION chunk should still flag overgeneralization when "
        f"'within 30 days' is missing from the claim. signals={signals}"
    )


# ── RFORGE-40: contradicted claim not flagged as overgeneralized ─────────────


def test_policy_claim_preserves_constraint_clears_procedural_claim_violation():
    """
    Multiple claims: one procedural (no constraints), one policy (has 'within 30 days').
    With per-chunk any-wins: the policy claim's preservation clears the procedural
    violation → constraint_preserved=True (RFORGE-40).
    """
    procedural = _make_claim(
        "I can help you get that refund sorted out right now.",
        subject="customer refund",
        obj="refund",
    )
    policy = _make_claim(
        "Monthly plan customers are eligible for a refund within 30 days of purchase.",
        subject="customer refund monthly",
        obj="refund days monthly",
    )
    signals = run_kb_alignment_pipeline(
        claims=[procedural, policy],
        kb_chunks=[_REFUND_ALLOW_CHUNK],
        conversation_context="",
        kb_chunks_required=["kb_refund_allow"],
        synonym_map={},
        planted_constraint=None,
    )
    assert signals.overgeneralization_flag is False, (
        "Policy claim 'within 30 days' should satisfy the ALLOW chunk constraint "
        "for the whole response via any-wins — procedural claim violation should not "
        f"override the policy claim's preservation. signals={signals}"
    )
    assert signals.constraint_preserved is True, (
        f"constraint_preserved=False despite policy claim preserving 'within 30 days'. signals={signals}"
    )


def test_single_claim_without_constraint_still_flagged():
    """
    Regression guard: when NO claim preserves the constraint,
    overgeneralization still fires. Any-wins requires at least one preserver.
    """
    vague = _make_claim(
        "Refunds are available for all paid plan customers.",
        subject="customer refund",
        obj="refund paid",
    )
    signals = run_kb_alignment_pipeline(
        claims=[vague],
        kb_chunks=[_REFUND_ALLOW_CHUNK],
        conversation_context="",
        kb_chunks_required=["kb_refund_allow"],
        synonym_map={},
        planted_constraint=None,
    )
    assert signals.overgeneralization_flag is True, (
        "No claim preserves 'within 30 days' — overgeneralization must still fire. "
        f"signals={signals}"
    )


def test_contradicted_claim_not_counted_in_overgen_tracking():
    """
    Regression guard: claims contradicted by a DENY chunk are excluded from
    per-chunk any-wins tracking. The DENY chunk contradiction does not
    drive overgeneralization_flag.
    """
    contradicted = _make_claim(
        "All customers can request a refund at any time.",
        subject="customers refunds",
        obj="refunds",
    )
    signals = run_kb_alignment_pipeline(
        claims=[contradicted],
        kb_chunks=[_REFUND_DENY_CHUNK],
        conversation_context="",
        kb_chunks_required=["kb_refund_deny_annual"],
        synonym_map={},
        planted_constraint=None,
    )
    assert signals.alignment == "contradicted", (
        f"Expected contradicted alignment. Got: {signals.alignment}"
    )
    assert signals.overgeneralization_flag is False, (
        "DENY chunk contradiction must not set overgeneralization_flag. "
        f"signals={signals}"
    )


# ── RFORGE-40b: injector does not include DENY chunks in conditional_applied kb_required


def test_injector_conditional_applied_excludes_deny_from_kb_required():
    """
    conditional_applied plans must not include DENY_CONDITION chunks in
    kb_chunks_required. DENY chunks cause false 'contradicted' alignment in the
    extractor when the agent correctly applies the ALLOW branch (RFORGE-40).
    """
    import random
    from datetime import datetime
    from resonantforge.layer1.quality_plan_injector import QualityPlanInjector
    from resonantforge.schemas import (
        ConstraintType, KBChunk, SimEvent, SimEventType,
    )

    deny_chunk = KBChunk(
        chunk_id="kb_annual_deny",
        document_id="doc_refund",
        document_path="policies/refund.md",
        chunk_text=(
            "Refunds are available on monthly plans only. Annual plan customers are NOT "
            "eligible for monetary refunds."
        ),
        constraint_type=ConstraintType.DENY_CONDITION,
        domains=["refund_policy"],
    )

    rng = random.Random(42)
    injector = QualityPlanInjector(rng=rng)

    events = [
        SimEvent(
            event_id=f"evt_{i:03d}",
            event_type=SimEventType.CONVERSATION_STARTED,
            account_id="acc_001",
            timestamp=datetime(2024, 1, 15, 10, 0, 0),
            day_index=0,
            month_index=0,
            payload={
                "domain": "refund_policy",
                "intent": ["refund_request"],
                "agent_id": "agent_001",
                "surface_channel": "chat",
            },
        )
        for i in range(80)
    ]

    plans = injector.inject(
        events=events,
        snapshots=[],
        kb_chunks=[_REFUND_ALLOW_CHUNK, deny_chunk],
        planted_count=50,
    )

    ca_plans = [
        p for p in plans
        if (
            p.rubric_targets.accuracy is not None
            and p.rubric_targets.accuracy.precision == "conditional_applied"
        )
    ]

    assert ca_plans, "Expected at least one conditional_applied plan"

    for plan in ca_plans:
        deny_in_required = any(
            "deny" in cid.lower() or cid == "kb_annual_deny"
            for cid in plan.kb_chunks_required
        )
        assert not deny_in_required, (
            f"Plan {plan.conversation_id} has DENY chunk in kb_chunks_required: "
            f"{plan.kb_chunks_required}. DENY chunks must not be in kb_required for "
            f"conditional_applied plans."
        )
