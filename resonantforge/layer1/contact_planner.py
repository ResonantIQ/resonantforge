"""
Customer-contact modeling: generate first-class contacts, attribute
conversations to them, plant the two relationship churn conditions the
Resonant IQ engine detects, and emit ground-truth labels.

Why this is a post-simulation pass
-----------------------------------
The two engine detectors this feeds — ``relationship_champion_at_risk``
(Rule 4) and ``relationship_single_threaded`` (Rule 5) — evaluate an account's
contact-engagement graph as-of "now" (the account's final day). Planting them
therefore requires knowing each account's *realized* lifespan, which is only
known after the day-by-day simulation has run (accounts churn early at emergent,
health-driven times). So contacts are planned in a single pass over the finished
event/snapshot stream rather than woven into the day loop.

Determinism
-----------
The planner draws from a seed-derived ``random.Random`` (salted so it never
perturbs the state machine's own event RNG stream). Given the same seed and the
same simulation output, it produces byte-identical contacts, attribution,
snapshot rollups, and labels.

Ground truth by construction
----------------------------
Every account is assigned one exclusive *relationship scenario* (positive,
decoy, or null), but scenarios are only assigned to accounts whose realized
conversation distribution can actually express them — so a planted positive
genuinely fires and a planted decoy genuinely does not. Labels are then computed
from the *actual* resulting rollup via :func:`champion_at_risk_fires` /
:func:`single_threaded_fires`, which are line-for-line mirrors of the engine's
detector logic. The label is thus, by construction, exactly what a correct
detector must output.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

from resonantforge.schemas import (
    SimEvent,
    DaySnapshot,
    SimEventType,
    Contact,
    ContactRole,
    ContactRollup,
    RelationshipLabel,
    RelationshipDetector,
)

# ---------------------------------------------------------------------------
# Thresholds — mirror the engine (churn-detectors.ts + cross-stream-helpers.ts).
# Kept here as the single source of truth for both attribution targeting and the
# label-generating mirror functions, so the two can never drift apart.
# ---------------------------------------------------------------------------

CHAMPION_SILENCE_DAYS = 30          # Rule 4: days-since-seen ≥ this → fires
RECENT_WINDOW_DAYS = 60             # Rule 5: distinct contacts in trailing 60d
PRIOR_WINDOW_START_DAYS = 60        # Rule 5: prior window is [60, 240) days ago
PRIOR_WINDOW_END_DAYS = 240
SINGLE_THREAD_MIN_ALLTIME = 3       # Rule 5 guard: totalContactsAllTime > 2
# Per-contact engagement frequency window: 3 average months (3 × 30.44d), then
# the touch count is divided by 3 to yield touches/month (RIQAPP-186).
FREQUENCY_WINDOW_DAYS = 91
FREQUENCY_WINDOW_MONTHS = 3.0

# Salt mixed into the seed so contact planning is deterministic but independent
# of the state machine's event RNG (planning must not shift the event stream).
_CONTACT_RNG_SALT = 0x00C0FFEE

# Attribution safety margins — keep planted signal values clear of the engine's
# exact window boundaries so an off-by-one day at ingestion can't flip a label.
_CHAMPION_SILENT_BY = 40            # silent-champion convs are ≥40d old (≥30 with margin)
_CHAMPION_ACTIVE_WITHIN = 15        # active-champion decoy: latest champion conv ≤15d old
_CHAMPION_LOWFREQ_WITHIN = 29       # low-freq champion: latest champion conv ≤29d (still <30)


# ---------------------------------------------------------------------------
# Pure detector mirrors — the ground-truth oracle for both label generation and
# the relationship validator. These MUST stay faithful to the TypeScript engine.
# ---------------------------------------------------------------------------


def champion_at_risk_fires(champion_rollups: list[ContactRollup]) -> bool:
    """
    Mirror of ``detectChampionSilence`` (churn-detectors.ts Rule 4).

    Fires when the account has ≥1 champion contact and, for a champion with a
    known last-seen, either it hasn't been seen in ≥30 days OR its engagement
    frequency is below 1/month. Champions never seen (last_seen is None) are
    skipped, exactly as the engine skips contacts with no live recency.
    """
    if not champion_rollups:
        return False
    for c in champion_rollups:
        if c.last_seen_day_index is None or c.days_since_seen is None:
            continue
        silent = c.days_since_seen >= CHAMPION_SILENCE_DAYS
        engagement_low = c.engagement_frequency is not None and c.engagement_frequency < 1
        if silent or engagement_low:
            return True
    return False


def single_threaded_fires(
    total_contacts_all_time: int,
    distinct_active_contacts_60d: int,
    distinct_active_contacts_prior_180d: int,
) -> bool:
    """
    Mirror of ``detectSingleThreadRisk`` (churn-detectors.ts Rule 5).

    Suppressed unless the account ever had >2 contacts. Fires only when exactly
    one contact was active in the last 60 days but ≥3 distinct contacts were
    active in the prior 60–240-day window.
    """
    if total_contacts_all_time <= 2:
        return False
    if distinct_active_contacts_60d != 1:
        return False
    if distinct_active_contacts_prior_180d < SINGLE_THREAD_MIN_ALLTIME:
        return False
    return True


# ---------------------------------------------------------------------------
# Rollup computation — reused for every DaySnapshot AND for the final-day label.
# ---------------------------------------------------------------------------


@dataclass
class _ContactState:
    """Mutable per-contact bookkeeping while attributing conversations."""

    contact: Contact
    conv_days: list[int] = field(default_factory=list)  # sorted ascending


def _distinct_in_window(
    states: dict[str, _ContactState], day: int, lo_days_ago: int, hi_days_ago: int
) -> int:
    """
    Count distinct contacts with ≥1 conversation whose age (in days before
    ``day``) falls in [lo_days_ago, hi_days_ago]. Ages are inclusive on both
    ends; callers pass window bounds that already avoid the exact engine
    boundaries for planted cases.
    """
    lo_day = day - hi_days_ago
    hi_day = day - lo_days_ago
    return sum(
        1
        for st in states.values()
        if any(lo_day <= d <= hi_day for d in st.conv_days if d <= day)
    )


def compute_rollup_at(
    states: dict[str, _ContactState], champion_ids: list[str], day: int
) -> tuple[int, int, int, list[ContactRollup]]:
    """
    Compute the four contact-rollup snapshot fields as-of ``day``.

    Returns ``(distinct_60d, distinct_prior_180d, total_all_time,
    champion_rollups)``. Trailing windows end on ``day``; ``total_all_time`` is
    the count of contacts that have engaged on or before ``day``.
    """
    distinct_60d = _distinct_in_window(states, day, 0, RECENT_WINDOW_DAYS)
    # Prior window: 60–240 days ago → ages in [61, 240] to sit strictly before
    # the recent window (which owns age 0..60).
    distinct_prior = _distinct_in_window(
        states, day, PRIOR_WINDOW_START_DAYS + 1, PRIOR_WINDOW_END_DAYS
    )
    total_all_time = sum(
        1 for st in states.values() if any(d <= day for d in st.conv_days)
    )

    rollups: list[ContactRollup] = []
    for cid in champion_ids:
        st = states[cid]
        seen = [d for d in st.conv_days if d <= day]
        if not seen:
            rollups.append(
                ContactRollup(
                    contact_id=st.contact.contact_id,
                    name=st.contact.name,
                    role=st.contact.role,
                    is_champion=True,
                    last_seen_day_index=None,
                    days_since_seen=None,
                    engagement_frequency=None,
                )
            )
            continue
        last_seen = max(seen)
        touches_in_window = sum(1 for d in seen if day - d <= FREQUENCY_WINDOW_DAYS)
        freq = touches_in_window / FREQUENCY_WINDOW_MONTHS
        rollups.append(
            ContactRollup(
                contact_id=st.contact.contact_id,
                name=st.contact.name,
                role=st.contact.role,
                is_champion=True,
                last_seen_day_index=last_seen,
                days_since_seen=day - last_seen,
                engagement_frequency=round(freq, 4),
            )
        )
    return distinct_60d, distinct_prior, total_all_time, rollups


# ---------------------------------------------------------------------------
# Deterministic name generation
# ---------------------------------------------------------------------------

_FIRST_NAMES = [
    "Avery", "Jordan", "Riley", "Morgan", "Casey", "Taylor", "Quinn", "Rowan",
    "Sydney", "Devon", "Harper", "Reese", "Emerson", "Finley", "Marlowe",
    "Priya", "Diego", "Mei", "Omar", "Nadia", "Kenji", "Zara", "Lucas", "Ingrid",
]
_LAST_NAMES = [
    "Chen", "Patel", "Nguyen", "Okafor", "Rossi", "Kowalski", "Silva", "Haddad",
    "Larsson", "Mbeki", "Fischer", "Romano", "Delgado", "Ivanov", "Yamamoto",
    "Ahmed", "Novak", "Costa", "Reyes", "Bauer", "Kim", "Dubois", "Singh",
]

_ROLE_CYCLE = [
    ContactRole.DECISION_MAKER,
    ContactRole.DAY_TO_DAY_USER,
    ContactRole.TECHNICAL_ADMIN,
    ContactRole.ECONOMIC_BUYER,
    ContactRole.EXECUTIVE_SPONSOR,
]


# ---------------------------------------------------------------------------
# Scenario catalogue
# ---------------------------------------------------------------------------

# (scenario_id, target_fraction, is_decoy_for)
# is_decoy_for names the detector a decoy superficially resembles; None for
# positives and plain nulls.
_SCENARIO_PLAN: list[tuple[str, float]] = [
    ("single_threaded_positive", 0.16),
    ("champion_silence_positive", 0.14),
    ("champion_lowfreq_positive", 0.08),
    ("single_threaded_decoy_narrowed", 0.10),
    ("single_threaded_decoy_guard", 0.07),
    ("champion_active_decoy", 0.11),
    # remainder → null_multithreaded / sparse_single
]

_DECOY_SCENARIOS = {
    "single_threaded_decoy_narrowed",
    "single_threaded_decoy_guard",
    "champion_active_decoy",
}


@dataclass
class _AccountConvStats:
    """Per-account conversation distribution used for scenario eligibility."""

    account_id: str
    final_day: int
    conv_events: list[SimEvent]  # CONVERSATION_STARTED, sorted by (day_index, event_id)
    n_recent: int   # convs with age ≤ 60d
    n_prior: int    # convs with age in [61, 240]
    earliest_day: int
    latest_day: int
    n_last91: int   # convs with age ≤ 91d


class ContactPlanner:
    """
    Plans contacts, attributes conversations, enriches snapshots, and emits
    ground-truth relationship labels over a finished simulation.
    """

    def __init__(
        self,
        seed: int,
        accounts: list,  # list[AccountSkeleton] — duck-typed to avoid import cycle
        events: list[SimEvent],
        snapshots: list[DaySnapshot],
    ):
        self.rng = random.Random(seed ^ _CONTACT_RNG_SALT)
        self.seed = seed
        self._account_ids = [a.account_id for a in accounts]
        self.events = events
        self.snapshots = snapshots
        self.contacts: list[Contact] = []
        self.labels: list[RelationshipLabel] = []
        # Filled during plan(); keyed by account_id.
        self._states_by_account: dict[str, dict[str, _ContactState]] = {}
        self._champions_by_account: dict[str, list[str]] = {}

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def plan(self) -> None:
        """Run the full contact-planning pass, mutating events + snapshots."""
        stats = self._scan_accounts()
        scenarios = self._assign_scenarios(stats)
        for account_id in self._account_ids:
            if account_id not in stats:
                continue  # account with no snapshots (churned day 0) — nothing to plan
            self._plan_account(stats[account_id], scenarios[account_id])
        self._enrich_snapshots()
        self._emit_labels(stats, scenarios)

    # ------------------------------------------------------------------
    # Scan
    # ------------------------------------------------------------------

    def _scan_accounts(self) -> dict[str, _AccountConvStats]:
        final_day: dict[str, int] = {}
        for s in self.snapshots:
            d = final_day.get(s.account_id, -1)
            if s.day_index > d:
                final_day[s.account_id] = s.day_index

        convs_by_account: dict[str, list[SimEvent]] = {}
        for e in self.events:
            if e.event_type == SimEventType.CONVERSATION_STARTED:
                convs_by_account.setdefault(e.account_id, []).append(e)

        stats: dict[str, _AccountConvStats] = {}
        for account_id, D in final_day.items():
            convs = sorted(
                convs_by_account.get(account_id, []),
                key=lambda e: (e.day_index, e.event_id),
            )
            if not convs:
                # No conversations — account still gets an (empty) contact set and
                # trivially-negative labels; represent with a stats stub.
                stats[account_id] = _AccountConvStats(
                    account_id=account_id, final_day=D, conv_events=[],
                    n_recent=0, n_prior=0, earliest_day=D, latest_day=D, n_last91=0,
                )
                continue
            days = [e.day_index for e in convs]
            n_recent = sum(1 for d in days if D - d <= RECENT_WINDOW_DAYS)
            n_prior = sum(1 for d in days if PRIOR_WINDOW_START_DAYS + 1 <= D - d <= PRIOR_WINDOW_END_DAYS)
            n_last91 = sum(1 for d in days if D - d <= FREQUENCY_WINDOW_DAYS)
            stats[account_id] = _AccountConvStats(
                account_id=account_id,
                final_day=D,
                conv_events=convs,
                n_recent=n_recent,
                n_prior=n_prior,
                earliest_day=min(days),
                latest_day=max(days),
                n_last91=n_last91,
            )
        return stats

    # ------------------------------------------------------------------
    # Scenario assignment (deterministic, eligibility-gated)
    # ------------------------------------------------------------------

    def _eligible(self, scenario: str, st: _AccountConvStats) -> bool:
        D = st.final_day
        if scenario == "single_threaded_positive":
            return st.n_prior >= 3 and st.n_recent >= 1
        if scenario == "champion_silence_positive":
            # Need at least one conv old enough to attribute to a silent champion.
            return st.conv_events != [] and st.earliest_day <= D - _CHAMPION_SILENT_BY
        if scenario == "champion_lowfreq_positive":
            # A recent conv for the champion (<30d) plus ≥1 other recent conv so
            # the account stays multi-threaded and single-thread doesn't fire.
            has_recent_for_champion = any(
                D - e.day_index <= _CHAMPION_LOWFREQ_WITHIN for e in st.conv_events
            )
            return has_recent_for_champion and st.n_recent >= 2
        if scenario == "single_threaded_decoy_narrowed":
            return st.n_prior >= 3 and st.n_recent >= 2
        if scenario == "single_threaded_decoy_guard":
            return st.n_recent >= 1 and st.n_prior >= 1
        if scenario == "champion_active_decoy":
            # Champion must be safely clear of BOTH fire branches: seen recently
            # (≤15d) AND frequency comfortably > 1/mo. Needs ≥5 touches in the
            # trailing 91d so the champion can hold ≥4 (freq ≈1.33) while a second
            # contact keeps a recent touch (distinct_60d ≥ 2, no single-thread).
            has_active = any(D - e.day_index <= _CHAMPION_ACTIVE_WITHIN for e in st.conv_events)
            return st.n_last91 >= 5 and has_active
        return False

    def _assign_scenarios(self, stats: dict[str, _AccountConvStats]) -> dict[str, str]:
        n = len(self._account_ids)
        assigned: dict[str, str] = {}
        # Assign in a fixed scenario priority order; within each, take eligible
        # unassigned accounts in account_id order up to the scenario's quota.
        ordered_ids = sorted(stats.keys())
        for scenario, frac in _SCENARIO_PLAN:
            quota = round(frac * n)
            if quota <= 0:
                continue
            taken = 0
            for account_id in ordered_ids:
                if taken >= quota:
                    break
                if account_id in assigned:
                    continue
                if self._eligible(scenario, stats[account_id]):
                    assigned[account_id] = scenario
                    taken += 1
        # Remainder → null control. Multi-contact-capable accounts become
        # null_multithreaded; the rest are sparse_single.
        for account_id in ordered_ids:
            if account_id in assigned:
                continue
            st = stats[account_id]
            if st.n_prior >= 2 and st.n_recent >= 2:
                assigned[account_id] = "null_multithreaded"
            else:
                assigned[account_id] = "sparse_single"
        return assigned

    # ------------------------------------------------------------------
    # Per-account planning: build contacts + attribute conversations
    # ------------------------------------------------------------------

    def _make_contact(
        self, account_id: str, index: int, is_champion: bool, scenario: str
    ) -> Contact:
        first = _FIRST_NAMES[self.rng.randrange(len(_FIRST_NAMES))]
        last = _LAST_NAMES[self.rng.randrange(len(_LAST_NAMES))]
        role = ContactRole.DECISION_MAKER if is_champion else _ROLE_CYCLE[index % len(_ROLE_CYCLE)]
        return Contact(
            contact_id=f"contact_{account_id}_{index:02d}",
            account_id=account_id,
            name=f"{first} {last}",
            role=role,
            is_champion=is_champion,
            relationship_scenario=scenario,
        )

    def _plan_account(self, st: _AccountConvStats, scenario: str) -> None:
        account_id = st.account_id
        D = st.final_day
        convs = st.conv_events

        # Decide contact roster size + champion designation per scenario, then
        # produce an attribution mapping event_id -> contact_index.
        if scenario == "single_threaded_positive":
            contacts = [self._make_contact(account_id, i, False, scenario) for i in range(4)]
            attribution = self._attr_single_threaded_positive(convs, D)
        elif scenario == "champion_silence_positive":
            contacts = [
                self._make_contact(account_id, 0, True, scenario),
                self._make_contact(account_id, 1, False, scenario),
                self._make_contact(account_id, 2, False, scenario),
            ]
            attribution = self._attr_champion_silence_positive(convs, D)
        elif scenario == "champion_lowfreq_positive":
            contacts = [
                self._make_contact(account_id, 0, True, scenario),
                self._make_contact(account_id, 1, False, scenario),
                self._make_contact(account_id, 2, False, scenario),
            ]
            attribution = self._attr_champion_lowfreq_positive(convs, D)
        elif scenario == "single_threaded_decoy_narrowed":
            contacts = [self._make_contact(account_id, i, False, scenario) for i in range(4)]
            attribution = self._attr_single_threaded_decoy_narrowed(convs, D)
        elif scenario == "single_threaded_decoy_guard":
            contacts = [self._make_contact(account_id, i, False, scenario) for i in range(2)]
            attribution = self._attr_single_threaded_decoy_guard(convs, D)
        elif scenario == "champion_active_decoy":
            contacts = [
                self._make_contact(account_id, 0, True, scenario),
                self._make_contact(account_id, 1, False, scenario),
            ]
            attribution = self._attr_champion_active_decoy(convs, D)
        elif scenario == "null_multithreaded":
            contacts = [self._make_contact(account_id, i, False, scenario) for i in range(3)]
            attribution = self._attr_round_robin(convs, n_contacts=3)
        else:  # sparse_single
            contacts = [self._make_contact(account_id, 0, False, scenario)]
            attribution = {e.event_id: 0 for e in convs}

        # Build contact state, drop contacts that ended up with zero convs so the
        # emitted roster equals the set of contacts that actually engaged (keeps
        # total_contacts_all_time == len(contacts), matching the engine's row
        # count where every ingested contact has engagement).
        used_indices = set(attribution.values())
        states: dict[str, _ContactState] = {}
        index_to_contact: dict[int, Contact] = {}
        for idx, c in enumerate(contacts):
            if idx in used_indices:
                states[c.contact_id] = _ContactState(contact=c)
                index_to_contact[idx] = c

        for e in convs:
            idx = attribution[e.event_id]
            contact = index_to_contact[idx]
            states[contact.contact_id].conv_days.append(e.day_index)
            self._attribute_event(e, contact)

        for st_c in states.values():
            st_c.conv_days.sort()
            c = st_c.contact
            c.engagement_count = len(st_c.conv_days)
            c.first_seen_day_index = st_c.conv_days[0] if st_c.conv_days else -1
            c.last_seen_day_index = st_c.conv_days[-1] if st_c.conv_days else -1

        self.contacts.extend(st_c.contact for st_c in states.values())
        self._states_by_account[account_id] = states
        self._champions_by_account[account_id] = [
            cid for cid, s in states.items() if s.contact.is_champion
        ]

    def _attribute_event(self, e: SimEvent, contact: Contact) -> None:
        """Stamp a conversation event's payload with its customer contact."""
        e.payload["customer_name"] = contact.name
        e.payload["contact_id"] = contact.contact_id
        e.payload["contact_role"] = contact.role.value
        e.payload["is_champion_contact"] = contact.is_champion

    # ------------------------------------------------------------------
    # Attribution strategies (return event_id -> contact_index)
    # ------------------------------------------------------------------

    @staticmethod
    def _age(e: SimEvent, D: int) -> int:
        return D - e.day_index

    def _attr_round_robin(self, convs: list[SimEvent], n_contacts: int) -> dict[str, int]:
        return {e.event_id: i % n_contacts for i, e in enumerate(convs)}

    def _attr_single_threaded_positive(self, convs: list[SimEvent], D: int) -> dict[str, int]:
        """
        Recent-window convs → the single surviving thread (contact 0); prior
        convs → round-robin across contacts 1..3 so ≥3 distinct appear in the
        60–240d window. Result: distinct_60d == 1, distinct_prior ≥ 3.
        """
        out: dict[str, int] = {}
        prior_rr = 0
        prior_contacts = [1, 2, 3]
        for e in convs:
            if self._age(e, D) <= RECENT_WINDOW_DAYS:
                out[e.event_id] = 0
            else:
                out[e.event_id] = prior_contacts[prior_rr % len(prior_contacts)]
                prior_rr += 1
        return out

    def _attr_champion_silence_positive(self, convs: list[SimEvent], D: int) -> dict[str, int]:
        """
        Champion (contact 0) gets only old convs (age ≥ 40d), and only those;
        everything else is split across non-champion contacts 1..2 so the account
        stays alive and multi-threaded. Champion last-seen ≥40d ago → fires.
        """
        out: dict[str, int] = {}
        other_rr = 0
        champion_assigned = False
        for e in convs:
            age = self._age(e, D)
            if age >= _CHAMPION_SILENT_BY:
                # Give the champion its engagement, but keep some old convs on
                # others too so the prior window isn't champion-only. Assign the
                # first eligible old conv to the champion, alternate the rest.
                if not champion_assigned:
                    out[e.event_id] = 0
                    champion_assigned = True
                else:
                    out[e.event_id] = 1 + (other_rr % 2)
                    other_rr += 1
            else:
                out[e.event_id] = 1 + (other_rr % 2)
                other_rr += 1
        # Safety: if no conv was old enough for the champion (shouldn't happen —
        # eligibility guarantees earliest ≤ D-40), fall back to giving contact 0
        # the earliest conv.
        if not champion_assigned and convs:
            out[convs[0].event_id] = 0
        return out

    def _attr_champion_lowfreq_positive(self, convs: list[SimEvent], D: int) -> dict[str, int]:
        """
        Champion (contact 0) gets exactly ONE recent conv (age ≤ 29d) and nothing
        else, so its frequency over the trailing 3 months is 1/3 < 1 while its
        last-seen is <30d (not silent). Fires via the frequency branch only.
        Everything else → non-champion contacts 1..2.
        """
        out: dict[str, int] = {}
        # Pick the champion's single conv: the most recent conv within the
        # low-freq window (largest day_index with age ≤ 29d).
        champion_event_id = None
        for e in convs:  # convs sorted ascending by day → last match is most recent
            if self._age(e, D) <= _CHAMPION_LOWFREQ_WITHIN:
                champion_event_id = e.event_id
        other_rr = 0
        for e in convs:
            if e.event_id == champion_event_id:
                out[e.event_id] = 0
            else:
                out[e.event_id] = 1 + (other_rr % 2)
                other_rr += 1
        return out

    def _attr_single_threaded_decoy_narrowed(self, convs: list[SimEvent], D: int) -> dict[str, int]:
        """
        Near-miss: narrows to TWO active threads (not one). Recent convs split
        across contacts 0 and 1; prior convs round-robin across 1..3 so ≥3
        distinct appear in the prior window. distinct_60d == 2 → must NOT fire.
        """
        out: dict[str, int] = {}
        recent_rr = 0
        prior_rr = 0
        prior_contacts = [1, 2, 3]
        for e in convs:
            if self._age(e, D) <= RECENT_WINDOW_DAYS:
                out[e.event_id] = recent_rr % 2  # contacts 0 and 1
                recent_rr += 1
            else:
                out[e.event_id] = prior_contacts[prior_rr % len(prior_contacts)]
                prior_rr += 1
        return out

    def _attr_single_threaded_decoy_guard(self, convs: list[SimEvent], D: int) -> dict[str, int]:
        """
        Near-miss: an account that IS effectively single-threaded (1 recent
        contact, was 2) but only ever had TWO contacts — the engine's guard
        (totalContactsAllTime > 2) suppresses it. distinct_60d == 1,
        distinct_prior == 2, total == 2 → must NOT fire.
        """
        out: dict[str, int] = {}
        prior_rr = 0
        for e in convs:
            if self._age(e, D) <= RECENT_WINDOW_DAYS:
                out[e.event_id] = 0
            else:
                out[e.event_id] = prior_rr % 2  # contacts 0 and 1 → 2 distinct prior
                prior_rr += 1
        return out

    def _attr_champion_active_decoy(self, convs: list[SimEvent], D: int) -> dict[str, int]:
        """
        Near-miss: the account HAS a champion (so Rule 4 has champions to check),
        but the champion is actively engaged — seen ≤15d ago with ≥3 touches over
        the trailing 3 months (frequency ≥ 1). Must NOT fire.
        """
        out: dict[str, int] = {}
        # Champion (contact 0) holds the NEWEST touch (so days_since is small) plus
        # 3 more of the trailing-91d touches → ≥4 touches, freq ≈1.33/mo (clear of
        # the <1 boundary). The 2nd-newest touch goes to contact 1 so the account
        # keeps ≥2 distinct recent contacts and single-thread cannot fire.
        recent91 = [e.event_id for e in convs if self._age(e, D) <= FREQUENCY_WINDOW_DAYS]
        if len(recent91) >= 5:
            champion_set = {recent91[-1]} | set(recent91[-5:-2])  # newest + 3 older-within-91d
        else:  # eligibility guarantees ≥5, but stay safe
            champion_set = set(recent91[-1:])
        for e in convs:
            out[e.event_id] = 0 if e.event_id in champion_set else 1
        return out

    # ------------------------------------------------------------------
    # Snapshot enrichment
    # ------------------------------------------------------------------

    def _enrich_snapshots(self) -> None:
        for s in self.snapshots:
            states = self._states_by_account.get(s.account_id)
            if not states:
                continue
            champions = self._champions_by_account.get(s.account_id, [])
            d60, dprior, total, rollups = compute_rollup_at(states, champions, s.day_index)
            s.distinct_active_contacts_60d = d60
            s.distinct_active_contacts_prior_180d = dprior
            s.total_contacts_all_time = total
            s.champion_rollups = rollups

    # ------------------------------------------------------------------
    # Ground-truth label emission
    # ------------------------------------------------------------------

    def _emit_labels(
        self, stats: dict[str, _AccountConvStats], scenarios: dict[str, str]
    ) -> None:
        for account_id in sorted(stats.keys()):
            D = stats[account_id].final_day
            scenario = scenarios[account_id]
            states = self._states_by_account.get(account_id, {})
            champions = self._champions_by_account.get(account_id, [])
            d60, dprior, total, rollups = compute_rollup_at(states, champions, D)

            champ_fire = champion_at_risk_fires(rollups)
            single_fire = single_threaded_fires(total, d60, dprior)

            champ_evidence = {
                "has_champion": bool(rollups),
                "champions": [r.model_dump() for r in rollups],
            }
            single_evidence = {
                "total_contacts_all_time": total,
                "distinct_active_contacts_60d": d60,
                "distinct_active_contacts_prior_180d": dprior,
            }

            self.labels.append(
                RelationshipLabel(
                    account_id=account_id,
                    detector=RelationshipDetector.CHAMPION_AT_RISK,
                    expected_fire=champ_fire,
                    is_decoy=(scenario == "champion_active_decoy" and not champ_fire),
                    scenario=scenario,
                    as_of_day_index=D,
                    rationale=self._champion_rationale(scenario, rollups, champ_fire),
                    evidence=champ_evidence,
                )
            )
            self.labels.append(
                RelationshipLabel(
                    account_id=account_id,
                    detector=RelationshipDetector.SINGLE_THREADED,
                    expected_fire=single_fire,
                    is_decoy=(
                        scenario
                        in ("single_threaded_decoy_narrowed", "single_threaded_decoy_guard")
                        and not single_fire
                    ),
                    scenario=scenario,
                    as_of_day_index=D,
                    rationale=self._single_rationale(scenario, total, d60, dprior, single_fire),
                    evidence=single_evidence,
                )
            )

    @staticmethod
    def _champion_rationale(scenario: str, rollups: list, fires: bool) -> str:
        if not rollups:
            return "Account has no champion contact; Rule 4 cannot fire."
        r = rollups[0]
        if fires:
            if r.days_since_seen is not None and r.days_since_seen >= CHAMPION_SILENCE_DAYS:
                return f"Champion last seen {r.days_since_seen}d ago (≥30) — silence fires Rule 4."
            return f"Champion engagement {r.engagement_frequency}/mo (<1) — low-frequency fires Rule 4."
        if scenario == "champion_active_decoy":
            return (
                f"Decoy: champion active (seen {r.days_since_seen}d ago, "
                f"{r.engagement_frequency}/mo) — must NOT fire."
            )
        return "Champion present and engaged; Rule 4 does not fire."

    @staticmethod
    def _single_rationale(scenario: str, total: int, d60: int, dprior: int, fires: bool) -> str:
        base = f"total={total}, active_60d={d60}, prior_60_240d={dprior}"
        if fires:
            return f"Single-threaded: {base} (1 recent, ≥3 prior, >2 all-time) — fires Rule 5."
        if scenario == "single_threaded_decoy_narrowed":
            return f"Decoy: narrowed to {d60} active threads, not 1 ({base}) — must NOT fire."
        if scenario == "single_threaded_decoy_guard":
            return f"Decoy: only {total} contacts all-time — guard suppresses ({base})."
        return f"Not single-threaded ({base}); Rule 5 does not fire."
