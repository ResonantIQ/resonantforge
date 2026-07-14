"""
Tests for first-class customer-contact modeling and the two relationship churn
signals it makes testable (`relationship_champion_at_risk`,
`relationship_single_threaded`).

Coverage:
- The pure detector mirrors match the engine's Rule 4 / Rule 5 semantics.
- The ContactPlanner is deterministic (contacts, labels, snapshot rollups).
- Every planted positive genuinely fires; every decoy and null genuinely does
  not, and decoys are flagged as such.
- Labels are internally consistent with their own evidence and with the
  as-of-now snapshot rollup.
- Contact attribution is complete and well-formed.
- The relationship validator agrees with the planted ground truth (independent
  recompute from raw attribution).
"""
from __future__ import annotations

from collections import Counter, defaultdict

import pytest

from resonantforge.layer1.state_machine import StateMachine
from resonantforge.layer1.contact_planner import (
    champion_at_risk_fires,
    single_threaded_fires,
)
from resonantforge.layer1.sim_events import validate_event_payload
from resonantforge.profiles.saas import SaaSProfile
from resonantforge.schemas import (
    SimEventType,
    ContactRole,
    ContactRollup,
    RelationshipDetector,
    ConversationRecord,
)
from resonantforge.validators.relationship import validate_account_relationships


SEED = 42
N_ACCOUNTS = 40
N_MONTHS = 6


@pytest.fixture(scope="module")
def sim():
    """A single simulated corpus, reused across tests in this module."""
    sm = StateMachine(seed=SEED, num_accounts=N_ACCOUNTS, num_months=N_MONTHS, profile=SaaSProfile())
    events, snapshots = sm.simulate()
    return sm, events, snapshots


def _champion(days_since, freq, seen=True):
    return ContactRollup(
        contact_id="c", name="X", role=ContactRole.DECISION_MAKER, is_champion=True,
        last_seen_day_index=(100 if seen else None),
        days_since_seen=(days_since if seen else None),
        engagement_frequency=(freq if seen else None),
    )


# ---------------------------------------------------------------------------
# Pure detector mirrors
# ---------------------------------------------------------------------------


def test_champion_mirror_no_champions_never_fires():
    assert champion_at_risk_fires([]) is False


def test_champion_mirror_silence_fires():
    assert champion_at_risk_fires([_champion(30, 5.0)]) is True   # exactly 30d
    assert champion_at_risk_fires([_champion(45, 5.0)]) is True
    assert champion_at_risk_fires([_champion(29, 5.0)]) is False  # <30 and freq ok


def test_champion_mirror_low_frequency_fires():
    assert champion_at_risk_fires([_champion(5, 0.5)]) is True    # active but <1/mo
    assert champion_at_risk_fires([_champion(5, 1.0)]) is False   # exactly 1/mo is fine
    assert champion_at_risk_fires([_champion(5, 2.0)]) is False


def test_champion_mirror_never_seen_is_skipped():
    # A champion with no recency is skipped (engine guard) — does not fire.
    assert champion_at_risk_fires([_champion(0, 0.0, seen=False)]) is False


def test_single_threaded_mirror_semantics():
    # Fires: >2 all-time, exactly 1 recent, >=3 prior.
    assert single_threaded_fires(4, 1, 3) is True
    # Guard: <=2 all-time suppresses.
    assert single_threaded_fires(2, 1, 3) is False
    # Must be exactly 1 recent.
    assert single_threaded_fires(4, 0, 3) is False
    assert single_threaded_fires(4, 2, 3) is False
    # Prior must be >=3.
    assert single_threaded_fires(4, 1, 2) is False


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_planner_determinism():
    def run():
        sm = StateMachine(seed=SEED, num_accounts=N_ACCOUNTS, num_months=N_MONTHS, profile=SaaSProfile())
        sm.simulate()
        return sm

    a, b = run(), run()
    assert [c.model_dump_json() for c in a.contacts] == [c.model_dump_json() for c in b.contacts]
    assert [l.model_dump_json() for l in a.relationship_labels] == [
        l.model_dump_json() for l in b.relationship_labels
    ]
    assert [s.model_dump_json() for s in a.snapshots] == [s.model_dump_json() for s in b.snapshots]


# ---------------------------------------------------------------------------
# Planted positives fire, decoys / nulls do not
# ---------------------------------------------------------------------------


def _labels_by(sm):
    out = defaultdict(dict)  # account_id -> {detector: label}
    for lab in sm.relationship_labels:
        out[lab.account_id][lab.detector] = lab
    return out


def test_positives_fire(sim):
    sm, _, _ = sim
    by = _labels_by(sm)
    seen_scenarios = Counter()
    for account_id, labs in by.items():
        champ = labs[RelationshipDetector.CHAMPION_AT_RISK]
        single = labs[RelationshipDetector.SINGLE_THREADED]
        scenario = champ.scenario
        seen_scenarios[scenario] += 1
        if scenario == "single_threaded_positive":
            assert single.expected_fire is True
            assert single.is_decoy is False
        elif scenario in ("champion_silence_positive", "champion_lowfreq_positive"):
            assert champ.expected_fire is True
            assert champ.is_decoy is False
    # The larger corpus must actually contain each positive scenario.
    assert seen_scenarios["single_threaded_positive"] >= 1
    assert seen_scenarios["champion_silence_positive"] >= 1
    assert seen_scenarios["champion_lowfreq_positive"] >= 1


def test_decoys_do_not_fire_but_are_flagged(sim):
    sm, _, _ = sim
    by = _labels_by(sm)
    champ_decoys = single_decoys = 0
    for labs in by.values():
        champ = labs[RelationshipDetector.CHAMPION_AT_RISK]
        single = labs[RelationshipDetector.SINGLE_THREADED]
        if champ.scenario == "champion_active_decoy":
            # Near-miss: account HAS a champion but it's active → must not fire.
            assert champ.expected_fire is False
            assert champ.is_decoy is True
            assert champ.evidence["has_champion"] is True
            champ_decoys += 1
        if single.scenario in ("single_threaded_decoy_narrowed", "single_threaded_decoy_guard"):
            assert single.expected_fire is False
            assert single.is_decoy is True
            single_decoys += 1
    assert champ_decoys >= 1
    assert single_decoys >= 1


def test_nulls_do_not_fire(sim):
    sm, _, _ = sim
    by = _labels_by(sm)
    for labs in by.values():
        for lab in labs.values():
            if lab.scenario in ("null_multithreaded", "sparse_single"):
                assert lab.expected_fire is False
                assert lab.is_decoy is False


# ---------------------------------------------------------------------------
# Label / evidence / rollup consistency
# ---------------------------------------------------------------------------


def test_labels_consistent_with_their_evidence(sim):
    sm, _, _ = sim
    for lab in sm.relationship_labels:
        ev = lab.evidence
        if lab.detector == RelationshipDetector.SINGLE_THREADED:
            got = single_threaded_fires(
                ev["total_contacts_all_time"],
                ev["distinct_active_contacts_60d"],
                ev["distinct_active_contacts_prior_180d"],
            )
        else:
            got = champion_at_risk_fires([ContactRollup(**r) for r in ev["champions"]])
        assert got == lab.expected_fire, (lab.account_id, lab.detector, lab.scenario)


def test_final_snapshot_rollup_matches_single_thread_label(sim):
    sm, events, snapshots = sim
    final_snap = {}
    for s in snapshots:
        cur = final_snap.get(s.account_id)
        if cur is None or s.day_index > cur.day_index:
            final_snap[s.account_id] = s
    single_labels = {
        lab.account_id: lab
        for lab in sm.relationship_labels
        if lab.detector == RelationshipDetector.SINGLE_THREADED
    }
    for account_id, lab in single_labels.items():
        snap = final_snap[account_id]
        assert snap.day_index == lab.as_of_day_index
        assert snap.distinct_active_contacts_60d == lab.evidence["distinct_active_contacts_60d"]
        assert snap.distinct_active_contacts_prior_180d == lab.evidence["distinct_active_contacts_prior_180d"]
        assert snap.total_contacts_all_time == lab.evidence["total_contacts_all_time"]


# ---------------------------------------------------------------------------
# Contact attribution integrity
# ---------------------------------------------------------------------------


def test_every_contact_engaged_and_referenced(sim):
    sm, events, _ = sim
    contact_ids = {c.contact_id for c in sm.contacts}
    # Every emitted contact engaged at least once.
    for c in sm.contacts:
        assert c.engagement_count >= 1
        assert c.first_seen_day_index >= 0
        assert c.last_seen_day_index >= c.first_seen_day_index
    # Every attributed conversation event references an emitted contact.
    attributed = 0
    for e in events:
        if e.event_type == SimEventType.CONVERSATION_STARTED:
            cid = e.payload.get("contact_id")
            assert cid is not None, f"conversation {e.event_id} was not attributed"
            assert cid in contact_ids
            assert e.payload.get("customer_name")  # contact name set
            assert e.payload.get("contact_role")
            attributed += 1
    assert attributed > 0


def test_total_contacts_all_time_equals_contact_rows(sim):
    sm, _, snapshots = sim
    contacts_per_account = Counter(c.account_id for c in sm.contacts)
    final_snap = {}
    for s in snapshots:
        cur = final_snap.get(s.account_id)
        if cur is None or s.day_index > cur.day_index:
            final_snap[s.account_id] = s
    for account_id, n_contacts in contacts_per_account.items():
        # Every emitted contact engages, so the final rollup's total equals the
        # number of contact rows (matches the engine's contact-row count).
        assert final_snap[account_id].total_contacts_all_time == n_contacts


# ---------------------------------------------------------------------------
# Relationship validator agrees with ground truth
# ---------------------------------------------------------------------------


def test_validator_agrees_with_planted_labels(sim):
    sm, events, snapshots = sim
    # Build ConversationRecords + day map from CONVERSATION_STARTED events.
    day_by_conv: dict[str, int] = {}
    convs_by_account: dict[str, list[ConversationRecord]] = defaultdict(list)
    for e in events:
        if e.event_type != SimEventType.CONVERSATION_STARTED:
            continue
        conv_id = f"conv_{e.event_id}"
        day_by_conv[conv_id] = e.day_index
        convs_by_account[e.account_id].append(
            ConversationRecord(
                conversation_id=conv_id,
                account_id=e.account_id,
                agent_id=e.payload.get("agent_id", "a"),
                surface_channel="intercom",
                started_at=e.timestamp,
                ended_at=e.timestamp,
                turn_count=1,
                prose="",
                trigger_event_id=e.event_id,
                contact_id=e.payload.get("contact_id"),
                contact_name=e.payload.get("customer_name"),
                contact_role=e.payload.get("contact_role"),
            )
        )
    contacts_by_account: dict[str, list] = defaultdict(list)
    for c in sm.contacts:
        contacts_by_account[c.account_id].append(c)
    final_day = {}
    for s in snapshots:
        final_day[s.account_id] = max(final_day.get(s.account_id, -1), s.day_index)

    checked = 0
    for account_id in contacts_by_account:
        verdicts = validate_account_relationships(
            account_id=account_id,
            contacts=contacts_by_account[account_id],
            conversations=convs_by_account[account_id],
            day_by_conv=day_by_conv,
            labels=sm.relationship_labels,
            as_of_day_index=final_day[account_id],
        )
        for v in verdicts:
            assert v.verdict.value == "pass", (
                account_id, v.dimension, v.signals_summary
            )
            checked += 1
    assert checked > 0


# ---------------------------------------------------------------------------
# sim_events optional fields
# ---------------------------------------------------------------------------


def test_contact_fields_not_required_by_validate_payload():
    # A CONVERSATION_STARTED payload without contact fields is still valid
    # (they are enrichment fields added post-emission).
    missing = validate_event_payload(
        SimEventType.CONVERSATION_STARTED,
        {"surface_channel": "intercom", "agent_id": "a", "customer_name": "n", "domain": "d"},
    )
    assert missing == []
