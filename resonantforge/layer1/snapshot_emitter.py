"""Snapshot enrichment and query utilities."""

from resonantforge.schemas import DaySnapshot, SimEvent, LifecycleStage, HealthState


class SnapshotEmitter:
    """
    Wraps the snapshot log with query and enrichment utilities.

    Accepts the raw outputs of StateMachine.simulate() and builds an index
    keyed by account_id for O(accounts) lookup instead of O(snapshots) scans.
    All mutation of the underlying lists happens in the state machine; this
    class is read-only.
    """

    def __init__(self, snapshots: list[DaySnapshot], events: list[SimEvent]):
        """
        Initialise the emitter with the simulation outputs.

        Args:
            snapshots: All DaySnapshot records produced by StateMachine.simulate().
            events:    All SimEvent records produced by StateMachine.simulate().
        """
        self._snapshots = snapshots
        self._events = events
        # Build per-account index at construction time; avoids repeated scans.
        self._by_account: dict[str, list[DaySnapshot]] = {}
        for snap in snapshots:
            self._by_account.setdefault(snap.account_id, []).append(snap)

    # ------------------------------------------------------------------
    # Query utilities
    # ------------------------------------------------------------------

    def get_account_snapshots(self, account_id: str) -> list[DaySnapshot]:
        """
        Get all snapshots for an account, ordered by day_index ascending.

        Returns an empty list if the account has no snapshots (e.g. churned
        on day 0 before any snapshot was written — unlikely but safe).
        """
        return sorted(
            self._by_account.get(account_id, []), key=lambda s: s.day_index
        )

    def get_snapshot_at(self, account_id: str, day_index: int) -> DaySnapshot | None:
        """
        Get the snapshot for an account on a specific day.

        Returns None if no snapshot exists for that account/day combination.
        """
        snaps = self.get_account_snapshots(account_id)
        for s in snaps:
            if s.day_index == day_index:
                return s
        return None

    def account_health_at(self, account_id: str, day_index: int) -> HealthState | None:
        """
        Return the HealthState for an account on a given day, or None if not found.

        Convenience wrapper around get_snapshot_at for callers that only need
        the health dimension without loading the full snapshot.
        """
        snap = self.get_snapshot_at(account_id, day_index)
        return snap.health_state if snap else None

    def account_stage_at(
        self, account_id: str, day_index: int
    ) -> LifecycleStage | None:
        """
        Return the LifecycleStage for an account on a given day, or None if not found.

        Convenience wrapper around get_snapshot_at for callers that only need
        the lifecycle dimension without loading the full snapshot.
        """
        snap = self.get_snapshot_at(account_id, day_index)
        return snap.lifecycle_stage if snap else None

    def events_for_account_day(
        self, account_id: str, day_index: int
    ) -> list[SimEvent]:
        """
        Get all events for an account on a given day.

        Iterates the full event list — this is acceptable because events are
        used infrequently compared to snapshot queries and the corpus size is
        bounded (< 100k events for typical runs).
        """
        return [
            e
            for e in self._events
            if e.account_id == account_id and e.day_index == day_index
        ]

    # ------------------------------------------------------------------
    # Summary / aggregation helpers
    # ------------------------------------------------------------------

    def month_summary(self, account_id: str, month_index: int) -> dict:
        """
        Summary statistics for an account's month (used in prompt context building).

        Returns a dict with:
        - avg_health_score: mean health score across all days in the month
        - dominant_health_state: the HealthState that appeared most often
        - lifecycle_stage: the stage recorded on the last day of the month
        - event_count: total events emitted for this account in the month
        - conversation_count: number of conversation_started events
        - payment_failures: number of payment_failed events
        - open_tickets_end_of_month: open ticket count on the final snapshot

        Returns an empty dict if the account has no snapshots for the month.
        """
        snaps = [
            s
            for s in self.get_account_snapshots(account_id)
            if s.month_index == month_index
        ]
        if not snaps:
            return {}

        events = [
            e
            for e in self._events
            if e.account_id == account_id and e.month_index == month_index
        ]

        return {
            "avg_health_score": round(
                sum(s.health_score for s in snaps) / len(snaps), 3
            ),
            "dominant_health_state": max(
                set(s.health_state for s in snaps),
                key=lambda h: sum(1 for s in snaps if s.health_state == h),
            ),
            "lifecycle_stage": snaps[-1].lifecycle_stage,
            "event_count": len(events),
            "conversation_count": sum(
                1 for e in events if e.event_type.value == "conversation_started"
            ),
            "payment_failures": sum(
                1 for e in events if e.event_type.value == "payment_failed"
            ),
            "open_tickets_end_of_month": snaps[-1].open_tickets,
        }

    def all_account_ids(self) -> list[str]:
        """Return all account IDs that have at least one snapshot, in insertion order."""
        return list(self._by_account.keys())
