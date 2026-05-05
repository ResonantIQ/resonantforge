"""Actor-local clocks for consistent per-actor timestamps."""

from datetime import datetime, timedelta
from dataclasses import dataclass, field


@dataclass
class ActorClock:
    """
    Deterministic actor-local clock that advances by adding minutes.

    Each actor (account, agent) owns its own clock so that timestamps within
    a single actor's event stream are always monotonically increasing, while
    different actors' clocks advance independently. The clock never goes
    backwards — ``tick_to`` is a no-op when the target is in the past.
    """

    actor_id: str
    current_time: datetime

    def advance(self, minutes: int) -> datetime:
        """
        Advance clock by the given number of minutes and return the new time.

        Always moves forward; negative values are accepted by ``timedelta``
        but callers should treat them as a logic error.
        """
        self.current_time += timedelta(minutes=minutes)
        return self.current_time

    def tick_to(self, target: datetime) -> None:
        """
        Advance to target time. No-op if the clock is already at or past target.

        Use this to synchronise an actor's clock to an external event time (e.g.
        a globally-scheduled payment date) without risking backwards movement.
        """
        if target > self.current_time:
            self.current_time = target


class ClockRegistry:
    """
    Registry of actor clocks, all initialised to the same base time.

    Clocks are created lazily on first access and are keyed by actor ID.
    Every clock starts at ``base_time``, ensuring that actors who never
    receive events still have a valid, deterministic starting point.
    """

    def __init__(self, base_time: datetime):
        """Initialise the registry with the simulation start time."""
        self._base_time = base_time
        self._clocks: dict[str, ActorClock] = {}

    def get(self, actor_id: str) -> ActorClock:
        """
        Return the ActorClock for actor_id, creating it at base_time if new.

        Idempotent: repeated calls for the same actor_id return the same object.
        """
        if actor_id not in self._clocks:
            self._clocks[actor_id] = ActorClock(
                actor_id=actor_id, current_time=self._base_time
            )
        return self._clocks[actor_id]

    def all_actors(self) -> list[str]:
        """Return all actor IDs that have been registered so far."""
        return list(self._clocks.keys())
