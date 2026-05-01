"""Deterministic account lifecycle state machine."""

import random
from datetime import datetime, date, timedelta
from dataclasses import dataclass, field
from typing import Protocol

from confabra.schemas import (
    SimEvent,
    DaySnapshot,
    SimEventType,
    LifecycleStage,
    HealthState,
)
from confabra.layer1.sim_events import validate_event_payload
from confabra.layer1.clocks import ClockRegistry

# ---------------------------------------------------------------------------
# Industry fallback list (kept for backward-compat with duck-typed FakeProfile
# in existing tests; real Profile objects use profile.name instead).
# ---------------------------------------------------------------------------

FALLBACK_INDUSTRIES = [
    "saas_tech",
    "ecommerce",
    "fintech",
    "healthcare_adjacent",
    "media",
]


# ---------------------------------------------------------------------------
# Profile protocol — duck-typed so both FakeProfile (tests) and real Profile
# subclasses (Task 14) satisfy the interface without forcing inheritance.
# ---------------------------------------------------------------------------


class Profile(Protocol):
    """
    Minimal protocol that Layer 1 requires from a Layer 2 industry profile.

    Real Profile subclasses (SaaSProfile, PSProfile) satisfy this protocol
    via inheritance.  FakeProfile in tests satisfies it via duck typing.

    ``name`` is used by ``generate_accounts`` to label the industry column
    when a real Profile is provided; ``FALLBACK_INDUSTRIES`` is used only when
    the profile's name is not available (legacy duck-typed callers).
    """

    @property
    def name(self) -> str:
        """Short identifier for this profile, e.g. 'saas'."""
        ...

    def lifecycle_stages(self) -> list[str]:
        """Return the list of lifecycle stage labels supported by this profile."""
        ...


# ---------------------------------------------------------------------------
# Account skeleton
# ---------------------------------------------------------------------------


@dataclass
class AccountSkeleton:
    """
    Immutable per-account attributes decided at account-generation time.

    These fields seed the day-by-day simulation loop but are never mutated
    during the simulation.  Health state and lifecycle stage evolve in local
    variables within ``simulate()``; the skeleton just records where each
    account starts.
    """

    account_id: str
    plan_tier: str  # "starter" | "growth" | "enterprise"
    industry: str  # from FALLBACK_INDUSTRIES (or profile in Task 14)
    start_date: date  # first day of simulation for this account
    initial_health: HealthState
    lifecycle_stage: LifecycleStage
    active_agent_ids: list[str]  # agents assigned to this account
    renewal_period_months: int  # 1 or 12
    monthly_arr_cents: int


# ---------------------------------------------------------------------------
# State machine
# ---------------------------------------------------------------------------


class StateMachine:
    """
    Deterministic account lifecycle state machine.

    Driven entirely by ``random.Random(seed)`` — never the module-level
    ``random`` functions.  Given the same seed, ``simulate()`` always produces
    an identical event stream and snapshot list.

    Usage::

        sm = StateMachine(seed=42, num_accounts=10, num_months=3, profile=p)
        events, snapshots = sm.simulate()
    """

    def __init__(
        self,
        seed: int,
        num_accounts: int,
        num_months: int,
        profile: "Profile",
    ):
        """
        Initialise the state machine.

        Args:
            seed: PRNG seed — all randomness derives from this value.
            num_accounts: Number of synthetic accounts to generate and simulate.
            num_months: Number of simulated months (each treated as 30 days).
            profile: Layer 2 profile object (duck-typed; see ``Profile`` protocol).
        """
        self.rng = random.Random(seed)
        self.seed = seed
        self.num_accounts = num_accounts
        self.num_months = num_months
        self.profile = profile
        self.base_date = date(2025, 7, 1)  # simulation starts 2025-07-01
        self.events: list[SimEvent] = []
        self.snapshots: list[DaySnapshot] = []
        self._event_counter = 0
        self._snapshot_counter = 0

        # Weighted-random domain selection. Pull weights from the profile when
        # available; fall back to a minimal list for duck-typed test profiles
        # that predate the domain_weights() method (PR1/PR2 FakeProfile).
        _raw_weights: dict[str, int] = getattr(profile, "domain_weights", lambda: {})()
        if _raw_weights:
            self._domain_names: list[str] = list(_raw_weights.keys())
            self._domain_w: list[int] = list(_raw_weights.values())
        else:
            self._domain_names = ["billing_and_invoicing", "technical_issue", "how_to_usage"]
            self._domain_w = [1, 1, 1]  # uniform over legacy fallback set

        self._domain_intents: dict[str, list[str]] = getattr(
            profile, "domain_intents", lambda: {}
        )()

    # ------------------------------------------------------------------
    # ID helpers
    # ------------------------------------------------------------------

    def _next_event_id(self) -> str:
        """Increment the event counter and return a zero-padded event ID."""
        self._event_counter += 1
        return f"evt_{self._event_counter:05d}"

    def _next_snapshot_id(self) -> str:
        """Increment the snapshot counter and return a zero-padded snapshot ID."""
        self._snapshot_counter += 1
        return f"snap_{self._snapshot_counter:05d}"

    def _peek_next_event_id(self) -> str:
        """
        Return the event ID that the *next* call to ``_next_event_id`` will produce.

        Used when a caller needs to reserve a conversation ID before any events
        for that conversation have been emitted (e.g. building the conv_id that
        both CONVERSATION_STARTED and CONVERSATION_ENDED will share).
        """
        return f"evt_{self._event_counter + 1:05d}"

    # ------------------------------------------------------------------
    # Account generation
    # ------------------------------------------------------------------

    def generate_accounts(self) -> list[AccountSkeleton]:
        """
        Generate ``num_accounts`` AccountSkeletons with stratified attributes.

        Stratification rule (by fractional position in the account list):
        - First 60 % → HEALTHY / ACTIVE
        - Next 30 %  → DECLINING / AT_RISK
        - Last 10 %  → CRITICAL / AT_RISK

        Plan tiers cycle: starter → growth → enterprise.
        Monthly ARR is plan-tier-scaled from three base prices.
        Each account gets 1–3 randomly assigned agent IDs (pool of 12).
        """
        accounts: list[AccountSkeleton] = []
        plan_tiers = ["starter", "growth", "enterprise"]
        base_prices = [9900, 49900, 199900]
        plan_multipliers = {"starter": 1, "growth": 3, "enterprise": 10}

        for i in range(self.num_accounts):
            plan = plan_tiers[i % len(plan_tiers)]

            # Stratified health based on fractional position
            frac = i / max(self.num_accounts, 1)
            if frac < 0.6:
                init_health = HealthState.HEALTHY
                stage = LifecycleStage.ACTIVE
            elif frac < 0.9:
                init_health = HealthState.DECLINING
                stage = LifecycleStage.AT_RISK
            else:
                init_health = HealthState.CRITICAL
                stage = LifecycleStage.AT_RISK

            # 1–3 agents per account, drawn from a pool of 12
            num_agents = self.rng.randint(1, 3)
            agent_ids = [
                f"agent_{self.rng.randint(1, 12):03d}" for _ in range(num_agents)
            ]

            # Use the real profile's name when available; fall back to the
            # legacy FALLBACK_INDUSTRIES list for duck-typed FakeProfile callers
            # that do not expose a ``name`` property.
            profile_name = getattr(self.profile, "name", None)
            industry = profile_name if profile_name else self.rng.choice(FALLBACK_INDUSTRIES)
            base_price = self.rng.choice(base_prices)
            monthly_arr = base_price * plan_multipliers[plan]

            accounts.append(
                AccountSkeleton(
                    account_id=f"acct_{i + 1:03d}",
                    plan_tier=plan,
                    industry=industry,
                    start_date=self.base_date,
                    initial_health=init_health,
                    lifecycle_stage=stage,
                    active_agent_ids=agent_ids,
                    renewal_period_months=12 if plan == "enterprise" else 1,
                    monthly_arr_cents=monthly_arr,
                )
            )

        return accounts

    # ------------------------------------------------------------------
    # Simulation loop
    # ------------------------------------------------------------------

    def simulate(self) -> tuple[list[SimEvent], list[DaySnapshot]]:
        """
        Run the full simulation for all accounts across all months.

        For each account the loop:
        1. Emits ACCOUNT_CREATED.
        2. Iterates day-by-day, emitting probabilistic events (conversations,
           payments, tickets, churn signals, renewal notices).
        3. Updates health score via the deterministic formula and transitions
           health state according to the lifecycle rules.
        4. Appends a DaySnapshot after each day's events.
        5. Breaks early when an account reaches CHURNED.

        Returns:
            A tuple of (events, snapshots) in emission order.
        """
        accounts = self.generate_accounts()

        for account in accounts:
            self._simulate_account(account)

        return self.events, self.snapshots

    def _simulate_account(self, account: AccountSkeleton) -> None:
        """
        Drive the day-by-day lifecycle loop for a single account.

        Extracted from ``simulate()`` for readability; mutates ``self.events``
        and ``self.snapshots`` in place.
        """
        # Emit account creation event
        self._emit_event(
            event_type=SimEventType.ACCOUNT_CREATED,
            account_id=account.account_id,
            agent_id=None,
            conversation_id=None,
            event_date=account.start_date,
            day_index=0,
            month_index=0,
            payload={"plan_tier": account.plan_tier, "industry": account.industry},
        )

        # Mutable simulation state
        current_health = account.initial_health
        current_stage = account.lifecycle_stage
        health_score = (
            0.8
            if current_health == HealthState.HEALTHY
            else 0.5
            if current_health == HealthState.DECLINING
            else 0.2
        )
        open_tickets = 0
        payment_failures_recent = 0
        days_in_current_state = 0
        churn_signal_days = 0
        total_days = self.num_months * 30

        for month_idx in range(self.num_months):
            for day_idx_in_month in range(30):  # simplified: each month = 30 days
                global_day = month_idx * 30 + day_idx_in_month
                current_date = account.start_date + timedelta(days=global_day)

                # Determine which agents are active today (0 to all of them)
                k = self.rng.randint(0, len(account.active_agent_ids))
                active_agents_today = (
                    self.rng.sample(account.active_agent_ids, k=k) if k > 0 else []
                )

                # ---- Conversation events ----
                if self.rng.random() < 0.3 and active_agents_today:
                    agent = self.rng.choice(active_agents_today)
                    # Reserve the conv_id before emitting any events for it
                    conv_id = f"conv_{self._event_counter + 1:05d}"
                    _domain = self.rng.choices(self._domain_names, weights=self._domain_w, k=1)[0]
                    _intent_pool = self._domain_intents.get(_domain, [])
                    _intent = (
                        [self.rng.choice(_intent_pool)] if _intent_pool else []
                    )
                    self._emit_event(
                        event_type=SimEventType.CONVERSATION_STARTED,
                        account_id=account.account_id,
                        agent_id=agent,
                        conversation_id=conv_id,
                        event_date=current_date,
                        day_index=global_day,
                        month_index=month_idx,
                        payload={
                            "surface_channel": "intercom",
                            "agent_id": agent,
                            "customer_name": f"Customer_{account.account_id}",
                            "domain": _domain,
                            "intent": _intent,
                        },
                    )
                    duration = self.rng.randint(5, 45)
                    turns = self.rng.randint(3, 15)
                    resolution = self.rng.choice(["resolved", "escalated", "pending"])
                    self._emit_event(
                        event_type=SimEventType.CONVERSATION_ENDED,
                        account_id=account.account_id,
                        agent_id=agent,
                        conversation_id=conv_id,
                        event_date=current_date,
                        day_index=global_day,
                        month_index=month_idx,
                        payload={
                            "duration_minutes": duration,
                            "turn_count": turns,
                            "resolution_status": resolution,
                        },
                    )

                # ---- Payment events (once per month, on day 0 of each month) ----
                if day_idx_in_month == 0:
                    if (
                        current_health != HealthState.CRITICAL
                        and self.rng.random() < 0.9
                    ):
                        self._emit_event(
                            event_type=SimEventType.PAYMENT_RECEIVED,
                            account_id=account.account_id,
                            agent_id=None,
                            conversation_id=None,
                            event_date=current_date,
                            day_index=global_day,
                            month_index=month_idx,
                            payload={
                                "amount_cents": account.monthly_arr_cents,
                                "plan_tier": account.plan_tier,
                            },
                        )
                        payment_failures_recent = max(0, payment_failures_recent - 1)
                    else:
                        self._emit_event(
                            event_type=SimEventType.PAYMENT_FAILED,
                            account_id=account.account_id,
                            agent_id=None,
                            conversation_id=None,
                            event_date=current_date,
                            day_index=global_day,
                            month_index=month_idx,
                            payload={
                                "amount_cents": account.monthly_arr_cents,
                                "failure_reason": "card_declined",
                            },
                        )
                        payment_failures_recent += 1

                # ---- Support ticket events ----
                ticket_prob = (
                    0.05
                    if current_health == HealthState.HEALTHY
                    else 0.15
                    if current_health == HealthState.DECLINING
                    else 0.25
                )
                if self.rng.random() < ticket_prob:
                    ticket_id = f"ticket_{global_day}_{account.account_id}"
                    open_tickets += 1
                    self._emit_event(
                        event_type=SimEventType.SUPPORT_TICKET_OPENED,
                        account_id=account.account_id,
                        agent_id=None,
                        conversation_id=None,
                        event_date=current_date,
                        day_index=global_day,
                        month_index=month_idx,
                        payload={
                            "ticket_id": ticket_id,
                            "category": self.rng.choice(
                                ["billing", "technical", "feature_request"]
                            ),
                            "priority": self.rng.choice(["low", "medium", "high"]),
                        },
                    )
                    # 70 % chance the ticket resolves same day
                    if self.rng.random() < 0.7:
                        open_tickets = max(0, open_tickets - 1)
                        self._emit_event(
                            event_type=SimEventType.SUPPORT_TICKET_RESOLVED,
                            account_id=account.account_id,
                            agent_id=None,
                            conversation_id=None,
                            event_date=current_date,
                            day_index=global_day,
                            month_index=month_idx,
                            payload={
                                "ticket_id": ticket_id,
                                "resolution_minutes": self.rng.randint(15, 480),
                            },
                        )

                # ---- Churn signals for at-risk / critical accounts ----
                if current_health in (
                    HealthState.DECLINING,
                    HealthState.CRITICAL,
                ) and self.rng.random() < 0.2:
                    strength = self.rng.uniform(0.4, 1.0)
                    self._emit_event(
                        event_type=SimEventType.CHURN_SIGNAL_DETECTED,
                        account_id=account.account_id,
                        agent_id=None,
                        conversation_id=None,
                        event_date=current_date,
                        day_index=global_day,
                        month_index=month_idx,
                        payload={"signal_type": "sentiment_drop", "strength": strength},
                    )
                    if strength > 0.7:
                        churn_signal_days += 1

                # ---- Renewal approaching notice (mid-final-month) ----
                if month_idx == self.num_months - 1 and day_idx_in_month == 14:
                    days_remaining = total_days - global_day
                    self._emit_event(
                        event_type=SimEventType.RENEWAL_APPROACHING,
                        account_id=account.account_id,
                        agent_id=None,
                        conversation_id=None,
                        event_date=current_date,
                        day_index=global_day,
                        month_index=month_idx,
                        payload={
                            "days_remaining": days_remaining,
                            "plan_tier": account.plan_tier,
                        },
                    )

                # ---- Health score update (deterministic formula + PRNG noise) ----
                payment_ok = (
                    1.0
                    if payment_failures_recent == 0
                    else max(0.0, 1.0 - payment_failures_recent * 0.3)
                )
                low_ticket_rate = (
                    1.0
                    if open_tickets == 0
                    else max(0.0, 1.0 - open_tickets * 0.2)
                )
                engagement = (
                    self.rng.uniform(0.5, 1.0)
                    if current_health == HealthState.HEALTHY
                    else self.rng.uniform(0.1, 0.6)
                )
                health_score = round(
                    payment_ok * 0.4 + low_ticket_rate * 0.3 + engagement * 0.3, 3
                )
                # Clamp to [0.0, 1.0] to satisfy the Pydantic validator
                health_score = max(0.0, min(1.0, health_score))

                # ---- State transitions ----
                days_in_current_state += 1

                if current_health == HealthState.HEALTHY:
                    if (
                        payment_failures_recent >= 2
                        or open_tickets >= 3
                        or churn_signal_days > 0
                    ):
                        current_health = HealthState.DECLINING
                        current_stage = LifecycleStage.AT_RISK
                        days_in_current_state = 0

                elif current_health == HealthState.DECLINING:
                    if days_in_current_state >= 14 and health_score < 0.3:
                        current_health = HealthState.CRITICAL
                        days_in_current_state = 0
                    elif (
                        payment_failures_recent == 0
                        and churn_signal_days == 0
                        and days_in_current_state >= 7
                    ):
                        # Recovery: payment healthy and no churn signals for a week
                        current_health = HealthState.HEALTHY
                        current_stage = LifecycleStage.ACTIVE
                        days_in_current_state = 0

                elif current_health == HealthState.CRITICAL:
                    if days_in_current_state >= 7:
                        current_health = HealthState.CHURNED
                        current_stage = LifecycleStage.CHURNED
                        self._emit_event(
                            event_type=SimEventType.RENEWAL_LAPSED,
                            account_id=account.account_id,
                            agent_id=None,
                            conversation_id=None,
                            event_date=current_date,
                            day_index=global_day,
                            month_index=month_idx,
                            payload={
                                "plan_tier": account.plan_tier,
                                "lapse_reason": "critical_unresolved",
                            },
                        )
                        days_in_current_state = 0

                # ---- Day snapshot ----
                renewal_days_remaining: int | None = None
                if month_idx == self.num_months - 1:
                    renewal_days_remaining = total_days - global_day

                self.snapshots.append(
                    DaySnapshot(
                        snapshot_id=self._next_snapshot_id(),
                        account_id=account.account_id,
                        day_index=global_day,
                        month_index=month_idx,
                        date=current_date,
                        lifecycle_stage=current_stage,
                        health_state=current_health,
                        health_score=health_score,
                        open_tickets=open_tickets,
                        recent_signals=(
                            ["churn_risk"] if churn_signal_days > 0 else []
                        ),
                        active_agents=active_agents_today,
                        payment_status=(
                            "current" if payment_failures_recent == 0 else "failed"
                        ),
                        renewal_days_remaining=renewal_days_remaining,
                    )
                )

                # Stop simulating once churned
                if current_health == HealthState.CHURNED:
                    break

            if current_health == HealthState.CHURNED:
                break

    # ------------------------------------------------------------------
    # Event factory
    # ------------------------------------------------------------------

    def _emit_event(
        self,
        event_type: SimEventType,
        account_id: str,
        agent_id: str | None,
        conversation_id: str | None,
        event_date: date,
        day_index: int,
        month_index: int,
        payload: dict,
    ) -> SimEvent:
        """
        Create a SimEvent, append it to ``self.events``, and return it.

        Timestamp is set to a random hour (08:00–18:59) on ``event_date``,
        drawn from the shared PRNG so the result is fully deterministic.
        ``day_index`` and ``month_index`` are passed in explicitly by the
        simulation loop to avoid any ambiguity about which day is "current".
        """
        timestamp = datetime.combine(event_date, datetime.min.time()).replace(
            hour=self.rng.randint(8, 18),
            minute=self.rng.randint(0, 59),
            second=0,
            microsecond=0,
        )
        event = SimEvent(
            event_id=self._next_event_id(),
            event_type=event_type,
            account_id=account_id,
            agent_id=agent_id,
            conversation_id=conversation_id,
            timestamp=timestamp,
            day_index=day_index,
            month_index=month_index,
            payload=payload,
        )
        self.events.append(event)
        return event
