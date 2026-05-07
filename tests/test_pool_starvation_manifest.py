"""
Tests for pool-starvation manifest telemetry (RFORGE-8).

Covers:
  - pool_starvation_count = 0 when no starvation occurs (≥ 3 fact-bearing chunks)
  - pool_starvation_count and pool_starvation_events populated correctly when starvation fires
  - pool_filter_telemetry structure and invariants (averages, min/max)
"""
from __future__ import annotations

import random
from datetime import datetime

from resonantforge.layer1.quality_plan_injector import QualityPlanInjector
from resonantforge.schemas import (
    ConstraintType,
    KBChunk,
    SimEvent,
    SimEventType,
)


# ── Fixtures (shared with test_planted_contradiction.py) ──────────────────────


def _make_conv_event(event_id: str, domain: str = "billing") -> SimEvent:
    return SimEvent(
        event_id=event_id,
        event_type=SimEventType.CONVERSATION_STARTED,
        account_id="acc_001",
        timestamp=datetime(2024, 1, 15, 10, 0, 0),
        day_index=0,
        month_index=0,
        payload={
            "domain": domain,
            "intent": ["billing_inquiry"],
            "agent_id": "agent_001",
            "surface_channel": "chat",
        },
    )


def _make_chunk(chunk_id: str, chunk_text: str, domain: str = "billing") -> KBChunk:
    return KBChunk(
        chunk_id=chunk_id,
        document_id=f"doc_{chunk_id}",
        document_path=f"policies/{chunk_id}.md",
        chunk_text=chunk_text,
        constraint_type=ConstraintType.ALLOW_CONDITION,
        domains=[domain],
    )


# Fact-bearing chunks
FACT_CHUNK = _make_chunk(
    "kb_refund_policy",
    "Refunds are available within 30 days of purchase for paid plans.",
)
NUMERIC_CHUNK = _make_chunk(
    "kb_seats_policy",
    "Your plan supports up to 500 users.",
    domain="features",
)
CAPABILITY_CHUNK = _make_chunk(
    "kb_webhook",
    "The platform supports outbound webhooks for event notifications.",
    domain="features",
)
# Fact-free chunks
VAGUE_CHUNK = _make_chunk(
    "kb_marketing",
    "Our platform is best-in-class for enterprise customers.",
)
MARKETING_CHUNK = _make_chunk(
    "kb_transform",
    "Transform your customer experience with our powerful platform.",
)


# Three fact-bearing chunks all in "billing" domain — ensures the per-event
# domain-filtered pool stays ≥ 3 so starvation never fires.
BILLING_FACT_A = _make_chunk(
    "kb_billing_refund",
    "Refunds are available within 30 days of purchase for paid plans.",
    domain="billing",
)
BILLING_FACT_B = _make_chunk(
    "kb_billing_seats",
    "Your plan supports up to 500 billing users.",
    domain="billing",
)
BILLING_FACT_C = _make_chunk(
    "kb_billing_sla",
    "We guarantee 99.9% uptime for billing services.",
    domain="billing",
)


# ── Test 1: No starvation when pool has ≥ 3 fact-bearing chunks ──────────────


def test_pool_starvation_count_zero_when_no_starvation():
    """
    With 3 fact-bearing chunks all in the same domain as the events,
    no pool-starvation event should be recorded and pool_starvation_count must be 0.
    """
    rng = random.Random(42)
    injector = QualityPlanInjector(rng=rng)
    events = [_make_conv_event(f"evt_{i:03d}") for i in range(80)]
    # All 3 fact-bearing chunks are in the "billing" domain; events are also billing.
    chunks = [BILLING_FACT_A, BILLING_FACT_B, BILLING_FACT_C, VAGUE_CHUNK]

    injector.inject(events=events, snapshots=[], kb_chunks=chunks, planted_count=50)

    assert injector.pool_starvation_count == 0, (
        f"Expected pool_starvation_count=0, got {injector.pool_starvation_count}"
    )
    assert injector.pool_starvation_events == [], (
        f"Expected empty pool_starvation_events, got {injector.pool_starvation_events}"
    )


# ── Test 2: Starvation count and events populated correctly ──────────────────


def test_pool_starvation_count_and_events_when_starvation_fires():
    """
    With only 1 fact-bearing chunk, every contradicted:exact plan triggers
    pool_starvation. The count and events list must agree, and each event
    record must carry the required diagnostic fields.
    """
    rng = random.Random(42)
    injector = QualityPlanInjector(rng=rng)
    events = [_make_conv_event(f"evt_{i:03d}") for i in range(80)]
    # Only 1 fact-bearing chunk → pool = 1 < 3 → starvation for every contradicted:exact plan
    chunks = [FACT_CHUNK, VAGUE_CHUNK, MARKETING_CHUNK]

    injector.inject(events=events, snapshots=[], kb_chunks=chunks, planted_count=50)

    assert injector.pool_starvation_count >= 1, (
        "Expected at least one pool_starvation incident with a single fact-bearing chunk"
    )
    assert len(injector.pool_starvation_events) == injector.pool_starvation_count, (
        "pool_starvation_events length must equal pool_starvation_count"
    )

    required_keys = {"event_id", "plan_type", "filter_stage", "pool_size_at_failure"}
    for record in injector.pool_starvation_events:
        assert required_keys <= set(record.keys()), (
            f"pool_starvation_events record missing keys: "
            f"{required_keys - set(record.keys())!r}, got {record}"
        )
        assert record["plan_type"] == "contradicted:exact", (
            f"Expected plan_type='contradicted:exact', got {record['plan_type']!r}"
        )
        assert record["filter_stage"] == "fact_bearing", (
            f"Expected filter_stage='fact_bearing', got {record['filter_stage']!r}"
        )
        assert record["pool_size_at_failure"] < 3, (
            f"pool_size_at_failure={record['pool_size_at_failure']} must be < 3"
        )


# ── Test 3: pool_filter_telemetry structure and invariants ───────────────────


def test_pool_filter_telemetry_structure_and_invariants():
    """
    pool_filter_telemetry must be keyed by plan type. For contradicted:exact,
    all required sub-fields must be present and the numeric invariants must hold:
    after_contradiction_filter_avg <= initial_pool_avg, min <= max.
    """
    rng = random.Random(42)
    injector = QualityPlanInjector(rng=rng)
    events = [_make_conv_event(f"evt_{i:03d}") for i in range(80)]
    chunks = [FACT_CHUNK, VAGUE_CHUNK, MARKETING_CHUNK]

    injector.inject(events=events, snapshots=[], kb_chunks=chunks, planted_count=50)

    telemetry = injector.pool_filter_telemetry
    assert isinstance(telemetry, dict), (
        f"pool_filter_telemetry must be a dict, got {type(telemetry)}"
    )
    assert "contradicted:exact" in telemetry, (
        f"pool_filter_telemetry must contain 'contradicted:exact' key; keys={list(telemetry)}"
    )

    ct = telemetry["contradicted:exact"]
    required_subkeys = {
        "initial_pool_avg",
        "after_contradiction_filter_avg",
        "final_pool_avg",
        "min_final_pool",
        "max_final_pool",
    }
    assert required_subkeys <= set(ct.keys()), (
        f"pool_filter_telemetry['contradicted:exact'] missing keys: "
        f"{required_subkeys - set(ct.keys())!r}"
    )

    # Fact-bearing filter can only shrink the pool
    assert ct["after_contradiction_filter_avg"] <= ct["initial_pool_avg"], (
        f"after_contradiction_filter_avg={ct['after_contradiction_filter_avg']} "
        f"> initial_pool_avg={ct['initial_pool_avg']} — filter grew the pool"
    )
    assert ct["min_final_pool"] <= ct["max_final_pool"], (
        f"min_final_pool={ct['min_final_pool']} > max_final_pool={ct['max_final_pool']}"
    )
    assert ct["min_final_pool"] >= 0, (
        f"min_final_pool={ct['min_final_pool']} must be non-negative"
    )
    assert ct["final_pool_avg"] >= 0.0, (
        f"final_pool_avg={ct['final_pool_avg']} must be non-negative"
    )
