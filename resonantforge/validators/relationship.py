# resonantforge/validators/relationship.py
"""
Relationship-signal validator — account-level, deterministic (no LLM).

This is the relationship counterpart to the per-conversation dimension
validators. It follows the same "feature extractor + rule engine" split, but the
unit of validation is an *account × detector* rather than a conversation:

- **Feature extractor** (:func:`extract_relationship_features`) independently
  recomputes the windowed contact-engagement features (distinct active contacts
  in the trailing 60d / prior 60–240d windows, total contacts, per-champion
  recency and frequency) straight from the attributed ``ConversationRecord``
  stream — *not* from the pre-rolled ``DaySnapshot`` fields. Recomputing from raw
  attribution is the point: it cross-checks that the snapshot pre-roll and the
  planted labels agree with a second, independent implementation.

- **Rule engine** (:func:`validate_account_relationships`) applies the shared
  detector oracle (``champion_at_risk_fires`` / ``single_threaded_fires`` — the
  same thresholds the engine and the label generator use) and compares the
  extracted outcome against the planted ``RelationshipLabel.expected_fire``. A
  mismatch is a FAIL: the corpus's own ground truth disagrees with an
  independent recompute, which is exactly what the disagreement ledger exists to
  surface.

Only the feature extraction is re-implemented here; the thresholds stay in
``contact_planner`` so they cannot drift between planting and validation.
"""
from __future__ import annotations

from resonantforge.schemas import (
    Contact,
    ContactRollup,
    ConversationRecord,
    RelationshipLabel,
    RelationshipDetector,
    DimensionVerdict,
    ValidationVerdict,
)
from resonantforge.layer1.contact_planner import (
    champion_at_risk_fires,
    single_threaded_fires,
    RECENT_WINDOW_DAYS,
    PRIOR_WINDOW_START_DAYS,
    PRIOR_WINDOW_END_DAYS,
    FREQUENCY_WINDOW_DAYS,
    FREQUENCY_WINDOW_MONTHS,
)


class RelationshipFeatures:
    """Extracted account-level relationship features as-of a day."""

    def __init__(
        self,
        account_id: str,
        as_of_day_index: int,
        distinct_active_contacts_60d: int,
        distinct_active_contacts_prior_180d: int,
        total_contacts_all_time: int,
        champion_rollups: list[ContactRollup],
    ):
        self.account_id = account_id
        self.as_of_day_index = as_of_day_index
        self.distinct_active_contacts_60d = distinct_active_contacts_60d
        self.distinct_active_contacts_prior_180d = distinct_active_contacts_prior_180d
        self.total_contacts_all_time = total_contacts_all_time
        self.champion_rollups = champion_rollups

    def as_summary(self) -> dict:
        return {
            "distinct_active_contacts_60d": self.distinct_active_contacts_60d,
            "distinct_active_contacts_prior_180d": self.distinct_active_contacts_prior_180d,
            "total_contacts_all_time": self.total_contacts_all_time,
            "champions": [r.model_dump() for r in self.champion_rollups],
        }

    # Convenience: apply the shared oracle to the extracted features.
    def champion_fires(self) -> bool:
        return champion_at_risk_fires(self.champion_rollups)

    def single_threaded_fires(self) -> bool:
        return single_threaded_fires(
            self.total_contacts_all_time,
            self.distinct_active_contacts_60d,
            self.distinct_active_contacts_prior_180d,
        )


def _contact_days(
    conversations: list[ConversationRecord],
    day_by_conv: dict[str, int],
) -> dict[str, list[int]]:
    """Map contact_id → sorted list of conversation day-indices it participated in."""
    days: dict[str, list[int]] = {}
    for c in conversations:
        if not c.contact_id:
            continue
        day = day_by_conv.get(c.conversation_id)
        if day is None:
            continue
        days.setdefault(c.contact_id, []).append(day)
    for v in days.values():
        v.sort()
    return days


def extract_relationship_features(
    account_id: str,
    contacts: list[Contact],
    conversations: list[ConversationRecord],
    day_by_conv: dict[str, int],
    as_of_day_index: int,
) -> RelationshipFeatures:
    """
    Recompute the account's relationship features from raw attribution.

    Args:
        account_id:       Account to extract for.
        contacts:         All Contact rows for this account.
        conversations:    All ConversationRecords for this account (must carry
                          ``contact_id`` attribution).
        day_by_conv:      Map conversation_id → simulation day-index (from the
                          trigger event; conversations don't store day-index
                          directly).
        as_of_day_index:  The day the windows end on (the account's "now").
    """
    contact_days = _contact_days(conversations, day_by_conv)
    champion_ids = [c.contact_id for c in contacts if c.is_champion]
    role_by_id = {c.contact_id: c.role for c in contacts}
    name_by_id = {c.contact_id: c.name for c in contacts}
    D = as_of_day_index

    def _distinct(lo_days_ago: int, hi_days_ago: int) -> int:
        lo_day = D - hi_days_ago
        hi_day = D - lo_days_ago
        return sum(
            1
            for days in contact_days.values()
            if any(lo_day <= d <= hi_day for d in days if d <= D)
        )

    distinct_60d = _distinct(0, RECENT_WINDOW_DAYS)
    distinct_prior = _distinct(PRIOR_WINDOW_START_DAYS + 1, PRIOR_WINDOW_END_DAYS)
    total_all_time = sum(1 for days in contact_days.values() if any(d <= D for d in days))

    rollups: list[ContactRollup] = []
    for cid in champion_ids:
        seen = [d for d in contact_days.get(cid, []) if d <= D]
        if not seen:
            rollups.append(
                ContactRollup(
                    contact_id=cid, name=name_by_id.get(cid, ""),
                    role=role_by_id[cid], is_champion=True,
                    last_seen_day_index=None, days_since_seen=None,
                    engagement_frequency=None,
                )
            )
            continue
        last_seen = max(seen)
        touches = sum(1 for d in seen if D - d <= FREQUENCY_WINDOW_DAYS)
        rollups.append(
            ContactRollup(
                contact_id=cid, name=name_by_id.get(cid, ""),
                role=role_by_id[cid], is_champion=True,
                last_seen_day_index=last_seen, days_since_seen=D - last_seen,
                engagement_frequency=round(touches / FREQUENCY_WINDOW_MONTHS, 4),
            )
        )

    return RelationshipFeatures(
        account_id=account_id,
        as_of_day_index=D,
        distinct_active_contacts_60d=distinct_60d,
        distinct_active_contacts_prior_180d=distinct_prior,
        total_contacts_all_time=total_all_time,
        champion_rollups=rollups,
    )


def _validate_one(
    features: RelationshipFeatures,
    detector: RelationshipDetector,
    label: RelationshipLabel | None,
) -> DimensionVerdict:
    if detector == RelationshipDetector.CHAMPION_AT_RISK:
        computed = features.champion_fires()
    else:
        computed = features.single_threaded_fires()

    dimension = detector.value
    if label is None:
        return DimensionVerdict(
            dimension=dimension,
            verdict=ValidationVerdict.SKIP,
            target="unlabeled",
            signals_summary=features.as_summary(),
        )

    passed = computed == label.expected_fire
    return DimensionVerdict(
        dimension=dimension,
        verdict=ValidationVerdict.PASS if passed else ValidationVerdict.FAIL,
        target=f"expected_fire={label.expected_fire}",
        signals_summary={
            **features.as_summary(),
            "computed_fire": computed,
            "expected_fire": label.expected_fire,
            "scenario": label.scenario,
            "is_decoy": label.is_decoy,
        },
    )


def validate_account_relationships(
    account_id: str,
    contacts: list[Contact],
    conversations: list[ConversationRecord],
    day_by_conv: dict[str, int],
    labels: list[RelationshipLabel],
    as_of_day_index: int,
) -> list[DimensionVerdict]:
    """
    Validate both relationship detectors for one account.

    Returns two DimensionVerdicts (champion-at-risk, single-threaded). A FAIL
    means the independently-extracted features disagree with the planted label —
    a corpus-integrity signal for the disagreement ledger.
    """
    features = extract_relationship_features(
        account_id, contacts, conversations, day_by_conv, as_of_day_index
    )
    label_by_detector = {lab.detector: lab for lab in labels if lab.account_id == account_id}
    return [
        _validate_one(features, RelationshipDetector.CHAMPION_AT_RISK,
                      label_by_detector.get(RelationshipDetector.CHAMPION_AT_RISK)),
        _validate_one(features, RelationshipDetector.SINGLE_THREADED,
                      label_by_detector.get(RelationshipDetector.SINGLE_THREADED)),
    ]
