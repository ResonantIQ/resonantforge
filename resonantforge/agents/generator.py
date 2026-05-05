"""
Agent fixture generator — produces per-agent directories under corpus/<profile>/agents/.

Each agent directory contains four deterministic files:

- ``profile.json``           — AgentProfile JSON (identity, skill baseline, history type)
- ``coaching_history.jsonl`` — CoachingEvent records with causal anchors
- ``score_trajectory.jsonl`` — TrajectoryRow records partitioned into coaching phases
- ``dispute_history.jsonl``  — DisputeRecord records (agent-initiated score disputes)

The causal anchor invariant is the core constraint: every ``CoachingEvent`` must
reference a ``TrajectoryRow`` whose ``scored_at`` timestamp precedes the coaching
event's ``issued_at``.  The generator enforces this by constructing the timeline
chronologically and back-filling ``triggering_conversation_id`` from rows that
already exist at coaching-event insertion time.

Public API
----------
- :func:`generate_agents` — build and write all agent fixtures for one profile
"""

from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from resonantforge.profiles.base import Profile
from resonantforge.utils.atomic_write import atomic_write_text, atomic_write_jsonl
from resonantforge.schemas import (
    AgentProfile,
    CoachingEvent,
    DisputeRecord,
    HistoryType,
    SkillProfile,
    TrajectoryRow,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Epoch from which all agent timestamps are offset.
_BASE_TS = datetime(2025, 11, 1, 9, 0, 0, tzinfo=timezone.utc)

# Agent count per profile name.
_AGENT_COUNTS: dict[str, int] = {
    "saas": 12,
    "ps": 4,
}

# History types in canonical order (3 agents each for SaaS, 1 each for PS).
_HISTORY_TYPES = [
    HistoryType.IMPROVING,
    HistoryType.RECURRING_WEAKNESS,
    HistoryType.NEW,
    HistoryType.MIXED,
]

# Plausible synthetic agent first+last names — 12 names for SaaS coverage.
_NAMES = [
    "Alex Rivera",
    "Jordan Kim",
    "Morgan Patel",
    "Casey Chen",
    "Taylor Okonkwo",
    "Drew Nakamura",
    "Avery Singh",
    "Quinn Alvarez",
    "Sage Lindqvist",
    "River Osei",
    "Blake Ferreira",
    "Rowan Johansson",
]

# Four criteria targeted by coaching and trajectory generation.
_CRITERIA = ["empathy", "resolution", "brand_voice", "accuracy"]


# ---------------------------------------------------------------------------
# Internal helpers — ID generation
# ---------------------------------------------------------------------------


def _agent_id(index: int) -> str:
    """Return zero-padded agent ID like ``agent_001``."""
    return f"agent_{index:03d}"


def _traj_id(global_counter: int) -> str:
    """Return zero-padded trajectory ID like ``traj_007``."""
    return f"traj_{global_counter:03d}"


def _coach_id(global_counter: int) -> str:
    """Return zero-padded coaching ID like ``coach_001``."""
    return f"coach_{global_counter:03d}"


def _disp_id(global_counter: int) -> str:
    """Return zero-padded dispute ID like ``disp_001``."""
    return f"disp_{global_counter:03d}"


def _conv_id(index: int) -> str:
    """Return zero-padded synthetic conversation ID like ``conv_00142``."""
    return f"conv_{index:05d}"


# ---------------------------------------------------------------------------
# Internal helpers — conversation ID pool
# ---------------------------------------------------------------------------


def _build_conv_pool(
    organic_conversation_ids: list[str],
    rng: random.Random,
    size: int,
) -> list[str]:
    """
    Return a shuffled pool of *size* conversation IDs.

    When ``organic_conversation_ids`` is non-empty the pool is drawn from it
    (cycling if needed).  When empty — e.g. during standalone testing — the
    pool is populated with synthetic IDs seeded from *rng*.

    .. note::
        **Minimum safe size for organic IDs.**  When ``organic_conversation_ids``
        is provided but contains fewer unique IDs than there are distinct slots
        across all trajectory rows, the cycling logic will repeat IDs — multiple
        trajectory rows will reference the same conversation.  This is harmless
        for schema validity but produces unrealistic fixtures where different
        scoring events appear to reference the same conversation.  Callers
        should provide at least 50 organic conversation IDs to avoid duplicate
        references in trajectory fixtures.
    """
    if organic_conversation_ids:
        pool: list[str] = []
        # Cycle through the organic list until we have enough.
        for i in range(size):
            pool.append(organic_conversation_ids[i % len(organic_conversation_ids)])
        rng.shuffle(pool)
        return pool
    # Standalone mode: generate synthetic IDs.
    start = rng.randint(1, 50_000)
    return [_conv_id(start + i) for i in range(size)]


# ---------------------------------------------------------------------------
# Internal helpers — score generation per history type
# ---------------------------------------------------------------------------


def _jitter(rng: random.Random, base: int, spread: int = 4) -> int:
    """Return ``base`` ± ``spread``, clamped to [0, 100]."""
    return min(100, max(0, base + rng.randint(-spread, spread)))


def _ts(day_offset: float, hour_offset: int = 0) -> datetime:
    """
    Return an absolute timestamp at *day_offset* days and *hour_offset* hours
    after ``_BASE_TS``.
    """
    return _BASE_TS + timedelta(days=day_offset, hours=hour_offset)


# ---------------------------------------------------------------------------
# Coaching note text templates (brief but plausible)
# ---------------------------------------------------------------------------

_NOTE_TEXTS: dict[str, list[str]] = {
    "empathy": [
        (
            "Agent showed improvement in empathy after this conversation but needs to maintain "
            "consistency. Focus on acknowledging customer emotions before presenting solutions."
        ),
        (
            "Follow-up coaching: customer felt unheard in this exchange. Practice active listening "
            "cues and explicitly validate the customer's frustration before troubleshooting."
        ),
        (
            "Third empathy coaching: agent continues to skip emotional acknowledgment. "
            "Review the warm-voice brand guidelines and apply them from the opening message."
        ),
    ],
    "resolution": [
        (
            "Resolution was incomplete — agent provided partial steps but did not confirm "
            "whether the customer's issue was fully addressed. Always close the loop explicitly."
        ),
        (
            "Second resolution coaching: agent again deflected without ownership language. "
            "Use 'I'll take care of this' framing and provide a concrete next step with a timeline."
        ),
        (
            "Third resolution coaching: recurring pattern of incomplete closures. Escalation "
            "protocol should be triggered when the agent cannot resolve within two exchanges."
        ),
    ],
    "accuracy": [
        (
            "Agent cited the correct policy area but overgeneralized the constraint. "
            "Always check the specific plan tier before quoting entitlements."
        ),
    ],
    "brand_voice": [
        (
            "Response tone was too clinical for an Intercom channel. The warm-exploratory brand "
            "voice requires contractions and collaborative language ('let's figure this out')."
        ),
    ],
}


def _note_text(criterion: str, index: int, rng: random.Random) -> str:
    """Return a coaching note text for *criterion* at coaching sequence *index*."""
    texts = _NOTE_TEXTS.get(criterion, [_NOTE_TEXTS["resolution"][0]])
    return texts[min(index, len(texts) - 1)]


# ---------------------------------------------------------------------------
# Internal helpers — causal checks
# ---------------------------------------------------------------------------


def _pick_triggering_conv(
    all_rows: list[dict[str, Any]],
    rng: random.Random,
    before_ts: datetime | None = None,
) -> str:
    """
    Pick a conversation ID from trajectory rows that precede *before_ts*.

    When *before_ts* is supplied only rows with ``scored_at`` strictly before
    that timestamp are eligible, enforcing the causal-anchor invariant.
    Selects from the last 5 eligible rows (recency bias) to keep the causal
    link plausible — a manager coaches on a recently-observed conversation, not
    one from weeks ago.

    Args:
        all_rows:  Full list of trajectory-row dicts accumulated so far.
        rng:       Seeded RNG for deterministic selection.
        before_ts: Upper bound on ``scored_at``; rows at or after this
                   timestamp are excluded.  Pass ``None`` to skip filtering.
    """
    if before_ts is not None:
        before_iso = before_ts.isoformat()
        eligible = [r for r in all_rows if r["scored_at"] < before_iso]
    else:
        eligible = all_rows
    if not eligible:
        raise ValueError(
            f"_pick_triggering_conv: no trajectory rows precede {before_ts}; "
            "causal-anchor invariant cannot be satisfied."
        )
    recent = eligible[-5:] if len(eligible) >= 5 else eligible
    return rng.choice(recent)["conversation_id"]


# ---------------------------------------------------------------------------
# Archetype builders
# ---------------------------------------------------------------------------


def _build_improving(
    agent_id: str,
    conv_pool: list[str],
    rng: random.Random,
    traj_counter: list[int],
    coach_counter: list[int],
    disp_counter: list[int],
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Build trajectory, coaching, and dispute records for the *improving* archetype.

    Timeline (30 days):
    - Days 0–9:   pre-coaching rows, targeted criterion ~55-65
    - Day 10:     coaching event 1 (criterion: empathy, status: acknowledged)
    - Days 11–19: post-coaching rows after coach_001, targeted criterion ~68-78
    - Day 20:     coaching event 2 (criterion: resolution, status: acknowledged)
    - Days 21–30: post-coaching rows after coach_002, targeted criterion ~78-88

    Returns ``(trajectory_rows, coaching_events, dispute_records)`` as plain dicts
    ready for JSONL serialisation.
    """
    traj_rows: list[dict] = []
    coaching: list[dict] = []
    disputes: list[dict] = []

    row_count = rng.randint(20, 30)
    # Distribute rows across three phases roughly evenly.
    phase1_n = row_count // 3
    phase2_n = row_count // 3
    phase3_n = row_count - phase1_n - phase2_n

    conv_idx = 0

    # --- Phase 1: pre-coaching ---
    for i in range(phase1_n):
        day = rng.uniform(0, 9)
        row = TrajectoryRow(
            trajectory_id=_traj_id(traj_counter[0]),
            conversation_id=conv_pool[conv_idx % len(conv_pool)],
            agent_id=agent_id,
            ai_score=_jitter(rng, 60),
            human_score=None,
            scored_at=_ts(day, rng.randint(8, 17)),
            coaching_phase="pre_coaching",
            post_coaching_of=None,
            tone_variant=None,
        )
        traj_rows.append(row.model_dump(mode="json"))
        traj_counter[0] += 1
        conv_idx += 1

    # --- Coaching event 1 ---
    c1_id = _coach_id(coach_counter[0])
    coach_counter[0] += 1
    c1_issued_at = _ts(10, 9)
    trigger1 = _pick_triggering_conv(traj_rows, rng, before_ts=c1_issued_at)
    c1 = CoachingEvent(
        coaching_id=c1_id,
        agent_id=agent_id,
        issued_at=c1_issued_at,
        criterion_targeted="empathy",
        triggering_conversation_id=trigger1,
        note_text=_note_text("empathy", 0, rng),
        status="acknowledged",
    )
    coaching.append(c1.model_dump(mode="json"))

    # --- Phase 2: post-coaching after c1 ---
    for i in range(phase2_n):
        day = rng.uniform(11, 19)
        row = TrajectoryRow(
            trajectory_id=_traj_id(traj_counter[0]),
            conversation_id=conv_pool[conv_idx % len(conv_pool)],
            agent_id=agent_id,
            ai_score=_jitter(rng, 73),
            human_score=None,
            scored_at=_ts(day, rng.randint(8, 17)),
            coaching_phase="post_coaching",
            post_coaching_of=c1_id,
            tone_variant=None,
        )
        traj_rows.append(row.model_dump(mode="json"))
        traj_counter[0] += 1
        conv_idx += 1

    # --- Coaching event 2 ---
    c2_id = _coach_id(coach_counter[0])
    coach_counter[0] += 1
    c2_issued_at = _ts(20, 9)
    trigger2 = _pick_triggering_conv(traj_rows, rng, before_ts=c2_issued_at)
    c2 = CoachingEvent(
        coaching_id=c2_id,
        agent_id=agent_id,
        issued_at=c2_issued_at,
        criterion_targeted="resolution",
        triggering_conversation_id=trigger2,
        note_text=_note_text("resolution", 0, rng),
        status="acknowledged",
    )
    coaching.append(c2.model_dump(mode="json"))

    # --- Phase 3: post-coaching after c2 ---
    for i in range(phase3_n):
        day = rng.uniform(21, 30)
        row = TrajectoryRow(
            trajectory_id=_traj_id(traj_counter[0]),
            conversation_id=conv_pool[conv_idx % len(conv_pool)],
            agent_id=agent_id,
            ai_score=_jitter(rng, 83),
            human_score=None,
            scored_at=_ts(day, rng.randint(8, 17)),
            coaching_phase="post_coaching",
            post_coaching_of=c2_id,
            tone_variant=None,
        )
        traj_rows.append(row.model_dump(mode="json"))
        traj_counter[0] += 1
        conv_idx += 1

    # --- 0–1 disputes ---
    n_disputes = rng.randint(0, 1)
    for _ in range(n_disputes):
        disputed_row = rng.choice(traj_rows)
        d = DisputeRecord(
            dispute_id=_disp_id(disp_counter[0]),
            agent_id=agent_id,
            conversation_id=disputed_row["conversation_id"],
            criterion=rng.choice(_CRITERIA),
            disputed_score=disputed_row["ai_score"],
            proposed_score=min(100, disputed_row["ai_score"] + rng.randint(5, 15)),
            rationale="Score was too conservative given the positive outcome of the conversation.",
            timestamp=_ts(rng.uniform(5, 29), 14),
        )
        disputes.append(d.model_dump(mode="json"))
        disp_counter[0] += 1

    return traj_rows, coaching, disputes


def _build_recurring_weakness(
    agent_id: str,
    conv_pool: list[str],
    rng: random.Random,
    traj_counter: list[int],
    coach_counter: list[int],
    disp_counter: list[int],
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Build records for the *recurring_weakness* archetype.

    Timeline (45 days):
    - 3 coaching events on the same criterion ("resolution"), spaced ~15 days apart.
    - Scores remain flat (~55-62) despite repeated coaching.
    - Coaching statuses: acknowledged → disputed → ignored.
    - 2–3 disputes.
    """
    traj_rows: list[dict] = []
    coaching: list[dict] = []
    disputes: list[dict] = []

    row_count = rng.randint(20, 30)
    # Four phases: pre c1, post c1, post c2, post c3.
    phase_n = row_count // 4
    remainder = row_count - phase_n * 4

    conv_idx = 0

    coach_days = [15.0, 30.0, 42.0]
    phase_ranges = [(0, 14), (16, 29), (31, 41), (43, 45)]
    coach_statuses = ["acknowledged", "disputed", "ignored"]
    coach_ids: list[str] = []

    for phase_idx, (day_start, day_end) in enumerate(phase_ranges):
        n_rows = phase_n + (1 if phase_idx < remainder else 0)

        if phase_idx == 0:
            phase_label = "pre_coaching"
            post_of = None
        else:
            phase_label = "post_coaching"
            post_of = coach_ids[phase_idx - 1]

        for _ in range(n_rows):
            day = rng.uniform(day_start, day_end)
            row = TrajectoryRow(
                trajectory_id=_traj_id(traj_counter[0]),
                conversation_id=conv_pool[conv_idx % len(conv_pool)],
                agent_id=agent_id,
                ai_score=_jitter(rng, 57),  # flat — no improvement
                human_score=None,
                scored_at=_ts(day, rng.randint(8, 17)),
                coaching_phase=phase_label,
                post_coaching_of=post_of,
                tone_variant=None,
            )
            traj_rows.append(row.model_dump(mode="json"))
            traj_counter[0] += 1
            conv_idx += 1

        # Issue coaching event after each of the first 3 phases.
        if phase_idx < 3:
            c_id = _coach_id(coach_counter[0])
            coach_counter[0] += 1
            coach_ids.append(c_id)
            c_issued_at = _ts(coach_days[phase_idx], 9)
            # Causal anchor: only rows scored before this coaching event are eligible.
            trigger = _pick_triggering_conv(traj_rows, rng, before_ts=c_issued_at)
            c = CoachingEvent(
                coaching_id=c_id,
                agent_id=agent_id,
                issued_at=c_issued_at,
                criterion_targeted="resolution",
                triggering_conversation_id=trigger,
                note_text=_note_text("resolution", phase_idx, rng),
                status=coach_statuses[phase_idx],
            )
            coaching.append(c.model_dump(mode="json"))

    # --- 2–3 disputes ---
    n_disputes = rng.randint(2, 3)
    for _ in range(n_disputes):
        disputed_row = rng.choice(traj_rows)
        d = DisputeRecord(
            dispute_id=_disp_id(disp_counter[0]),
            agent_id=agent_id,
            conversation_id=disputed_row["conversation_id"],
            criterion="resolution",
            disputed_score=disputed_row["ai_score"],
            proposed_score=min(100, disputed_row["ai_score"] + rng.randint(8, 20)),
            rationale=(
                "Score was too generous given incomplete resolution; "
                "the follow-up ticket shows the issue recurred within 48 hours."
            ),
            timestamp=_ts(rng.uniform(10, 44), 14),
        )
        disputes.append(d.model_dump(mode="json"))
        disp_counter[0] += 1

    return traj_rows, coaching, disputes


def _build_new_no_history(
    agent_id: str,
    conv_pool: list[str],
    rng: random.Random,
    traj_counter: list[int],
    coach_counter: list[int],
    disp_counter: list[int],
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Build records for the *new_no_history* archetype.

    Timeline: only the most recent 14 days; 5–10 rows; no coaching events;
    no disputes.  All rows are ``coaching_phase="uncoached"``.
    """
    traj_rows: list[dict] = []

    row_count = rng.randint(5, 10)
    # Anchor to the "most recent 14 days" by using day offsets close to the end
    # of the 45-day window — but since the agent is new, we just use days 0–13.
    conv_idx = 0
    for _ in range(row_count):
        day = rng.uniform(0, 13)
        row = TrajectoryRow(
            trajectory_id=_traj_id(traj_counter[0]),
            conversation_id=conv_pool[conv_idx % len(conv_pool)],
            agent_id=agent_id,
            ai_score=_jitter(rng, 68, spread=8),
            human_score=None,
            scored_at=_ts(day, rng.randint(8, 17)),
            coaching_phase="uncoached",
            post_coaching_of=None,
            tone_variant=None,
        )
        traj_rows.append(row.model_dump(mode="json"))
        traj_counter[0] += 1
        conv_idx += 1

    return traj_rows, [], []


def _build_mixed_history(
    agent_id: str,
    conv_pool: list[str],
    rng: random.Random,
    traj_counter: list[int],
    coach_counter: list[int],
    disp_counter: list[int],
) -> tuple[list[dict], list[dict], list[dict]]:
    """
    Build records for the *mixed_history* archetype.

    Three criterion sub-patterns over 45 days:

    - Criterion A ("empathy"):   improving — 2 coaching events, upward trajectory.
    - Criterion B ("resolution"): recurring_weakness — 3 coaching events, flat.
    - Criterion C ("accuracy"):   uncoached — rows exist but no coaching events.

    Each criterion contributes approximately one-third of total rows
    (target 25–35 total).  Rows are interleaved chronologically in the output
    JSONL but grouped by criterion here for clarity.

    Disputes: 1–2.
    """
    traj_rows: list[dict] = []
    coaching: list[dict] = []
    disputes: list[dict] = []

    total_target = rng.randint(25, 35)
    per_criterion = total_target // 3
    extra = total_target - per_criterion * 3

    conv_idx = 0

    # ---- Criterion A: empathy (improving) ----
    a_rows = per_criterion + (1 if extra > 0 else 0)
    extra = max(0, extra - 1)

    # Phase 1 pre-coaching (days 0–14)
    a_phase1 = a_rows // 3
    a_phase2 = a_rows // 3
    a_phase3 = a_rows - a_phase1 - a_phase2

    c_a1_id = _coach_id(coach_counter[0])
    coach_counter[0] += 1
    c_a2_id = _coach_id(coach_counter[0])
    coach_counter[0] += 1

    for i in range(a_phase1):
        day = rng.uniform(0, 13)
        row = TrajectoryRow(
            trajectory_id=_traj_id(traj_counter[0]),
            conversation_id=conv_pool[conv_idx % len(conv_pool)],
            agent_id=agent_id,
            ai_score=_jitter(rng, 60),
            human_score=None,
            scored_at=_ts(day, rng.randint(8, 17)),
            coaching_phase="pre_coaching",
            post_coaching_of=None,
            tone_variant=None,
        )
        traj_rows.append(row.model_dump(mode="json"))
        traj_counter[0] += 1
        conv_idx += 1

    # Coaching A1 at day 14
    ca1_issued_at = _ts(14, 9)
    trigger_a1 = _pick_triggering_conv(traj_rows, rng, before_ts=ca1_issued_at)
    ca1 = CoachingEvent(
        coaching_id=c_a1_id,
        agent_id=agent_id,
        issued_at=ca1_issued_at,
        criterion_targeted="empathy",
        triggering_conversation_id=trigger_a1,
        note_text=_note_text("empathy", 0, rng),
        status="acknowledged",
    )
    coaching.append(ca1.model_dump(mode="json"))

    for i in range(a_phase2):
        day = rng.uniform(15, 27)
        row = TrajectoryRow(
            trajectory_id=_traj_id(traj_counter[0]),
            conversation_id=conv_pool[conv_idx % len(conv_pool)],
            agent_id=agent_id,
            ai_score=_jitter(rng, 73),
            human_score=None,
            scored_at=_ts(day, rng.randint(8, 17)),
            coaching_phase="post_coaching",
            post_coaching_of=c_a1_id,
            tone_variant=None,
        )
        traj_rows.append(row.model_dump(mode="json"))
        traj_counter[0] += 1
        conv_idx += 1

    # Coaching A2 at day 28
    ca2_issued_at = _ts(28, 9)
    trigger_a2 = _pick_triggering_conv(traj_rows, rng, before_ts=ca2_issued_at)
    ca2 = CoachingEvent(
        coaching_id=c_a2_id,
        agent_id=agent_id,
        issued_at=ca2_issued_at,
        criterion_targeted="empathy",
        triggering_conversation_id=trigger_a2,
        note_text=_note_text("empathy", 1, rng),
        status="disputed",
    )
    coaching.append(ca2.model_dump(mode="json"))

    for i in range(a_phase3):
        day = rng.uniform(29, 44)
        row = TrajectoryRow(
            trajectory_id=_traj_id(traj_counter[0]),
            conversation_id=conv_pool[conv_idx % len(conv_pool)],
            agent_id=agent_id,
            ai_score=_jitter(rng, 83),
            human_score=None,
            scored_at=_ts(day, rng.randint(8, 17)),
            coaching_phase="post_coaching",
            post_coaching_of=c_a2_id,
            tone_variant=None,
        )
        traj_rows.append(row.model_dump(mode="json"))
        traj_counter[0] += 1
        conv_idx += 1

    # ---- Criterion B: resolution (recurring_weakness) ----
    b_rows = per_criterion + (1 if extra > 0 else 0)
    extra = max(0, extra - 1)

    b_coach_days = [10.0, 22.0, 37.0]
    b_phase_ranges = [(0, 9), (11, 21), (23, 36), (38, 45)]
    b_statuses = ["acknowledged", "disputed", "ignored"]
    b_phase_n = b_rows // 4
    b_remainder = b_rows - b_phase_n * 4
    b_coach_ids: list[str] = []

    for phase_idx, (day_start, day_end) in enumerate(b_phase_ranges):
        n = b_phase_n + (1 if phase_idx < b_remainder else 0)
        if phase_idx == 0:
            p_label = "pre_coaching"
            post_of = None
        else:
            p_label = "post_coaching"
            post_of = b_coach_ids[phase_idx - 1]

        for _ in range(n):
            day = rng.uniform(day_start, day_end)
            row = TrajectoryRow(
                trajectory_id=_traj_id(traj_counter[0]),
                conversation_id=conv_pool[conv_idx % len(conv_pool)],
                agent_id=agent_id,
                ai_score=_jitter(rng, 57),
                human_score=None,
                scored_at=_ts(day, rng.randint(8, 17)),
                coaching_phase=p_label,
                post_coaching_of=post_of,
                tone_variant=None,
            )
            traj_rows.append(row.model_dump(mode="json"))
            traj_counter[0] += 1
            conv_idx += 1

        if phase_idx < 3:
            bc_id = _coach_id(coach_counter[0])
            coach_counter[0] += 1
            b_coach_ids.append(bc_id)
            bc_issued_at = _ts(b_coach_days[phase_idx], 10)
            # Causal anchor: only rows scored before this coaching event are eligible.
            trigger = _pick_triggering_conv(traj_rows, rng, before_ts=bc_issued_at)
            bc = CoachingEvent(
                coaching_id=bc_id,
                agent_id=agent_id,
                issued_at=bc_issued_at,
                criterion_targeted="resolution",
                triggering_conversation_id=trigger,
                note_text=_note_text("resolution", phase_idx, rng),
                status=b_statuses[phase_idx],
            )
            coaching.append(bc.model_dump(mode="json"))

    # ---- Criterion C: accuracy (uncoached) ----
    c_rows = per_criterion
    for _ in range(c_rows):
        day = rng.uniform(0, 45)
        row = TrajectoryRow(
            trajectory_id=_traj_id(traj_counter[0]),
            conversation_id=conv_pool[conv_idx % len(conv_pool)],
            agent_id=agent_id,
            ai_score=_jitter(rng, 70, spread=10),
            human_score=None,
            scored_at=_ts(day, rng.randint(8, 17)),
            coaching_phase="uncoached",
            post_coaching_of=None,
            tone_variant=None,
        )
        traj_rows.append(row.model_dump(mode="json"))
        traj_counter[0] += 1
        conv_idx += 1

    # ---- 1–2 disputes ----
    n_disputes = rng.randint(1, 2)
    for _ in range(n_disputes):
        disputed_row = rng.choice(traj_rows)
        criterion_choice = rng.choice(["empathy", "resolution"])
        d = DisputeRecord(
            dispute_id=_disp_id(disp_counter[0]),
            agent_id=agent_id,
            conversation_id=disputed_row["conversation_id"],
            criterion=criterion_choice,
            disputed_score=disputed_row["ai_score"],
            proposed_score=min(100, disputed_row["ai_score"] + rng.randint(5, 15)),
            rationale="Agent believes the score underweights the positive resolution outcome.",
            timestamp=_ts(rng.uniform(10, 44), 14),
        )
        disputes.append(d.model_dump(mode="json"))
        disp_counter[0] += 1

    return traj_rows, coaching, disputes


# ---------------------------------------------------------------------------
# Archetype dispatch table
# ---------------------------------------------------------------------------

_ARCHETYPE_BUILDERS = {
    HistoryType.IMPROVING: _build_improving,
    HistoryType.RECURRING_WEAKNESS: _build_recurring_weakness,
    HistoryType.NEW: _build_new_no_history,
    HistoryType.MIXED: _build_mixed_history,
}


# ---------------------------------------------------------------------------
# File writing helpers
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, records: list[dict]) -> None:
    """Write *records* to *path* as NDJSON using the atomic write utility."""
    atomic_write_jsonl(path, (json.dumps(r, default=str) for r in records))


def _write_profile_json(path: Path, profile_dict: dict) -> None:
    """Write a single agent ``profile.json`` with 2-space indentation atomically."""
    content = json.dumps(profile_dict, indent=2, default=str) + "\n"
    atomic_write_text(path, content)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_agents(
    profile: Profile,
    output_dir: Path,
    seed: int,
    organic_conversation_ids: list[str],
) -> tuple[list[AgentProfile], str]:
    """
    Generate agent fixture directories and write four files per agent.

    Agents are grouped by history type with counts controlled by profile:
    - SaaS: 12 agents (3 per history type × 4 types)
    - PS:   4 agents (1 per history type × 4 types)

    The ``organic_conversation_ids`` list is used to link trajectory rows and
    causal anchors back to conversations that exist in the organic corpus.
    When the list is empty (standalone test mode) synthetic IDs are generated.

    Args:
        profile:                  The Layer 2 profile driving this corpus run.
        output_dir:               Root directory for this profile's corpus output.
                                  Agent fixtures are written to ``output_dir/agents/``.
        seed:                     Deterministic RNG seed; identical seeds produce
                                  identical output across all runs.
        organic_conversation_ids: Flat list of conversation IDs from the organic
                                  corpus, used for causal linkage.

    Returns:
        A tuple of ``(agent_profiles, content_hash)`` where:
        - ``agent_profiles`` is the list of :class:`~resonantforge.schemas.AgentProfile`
          instances, one per generated agent.
        - ``content_hash`` is a ``"sha256:<hex>"`` string computed over all
          JSONL content written (sorted by agent_id), suitable for the manifest.
    """
    rng = random.Random(seed)
    tenant_id = f"tenant_{profile.name}_v1"
    agents_dir = output_dir / "agents"

    n_agents = _AGENT_COUNTS.get(profile.name, 4)
    agents_per_type = n_agents // len(_HISTORY_TYPES)  # 3 for SaaS, 1 for PS

    # Pre-build conversation pool large enough for all agents.
    pool_size = max(500, n_agents * 50)
    conv_pool = _build_conv_pool(organic_conversation_ids, rng, pool_size)

    # Shared global counters (mutable single-element lists so closures can mutate).
    traj_counter: list[int] = [1]
    coach_counter: list[int] = [1]
    disp_counter: list[int] = [1]

    # Assign linked_conversation_id round-robin from organic pool.
    link_source = organic_conversation_ids if organic_conversation_ids else conv_pool
    link_idx = 0

    agent_profiles: list[AgentProfile] = []
    # Accumulate all written JSONL content for hashing (sorted by agent_id at end).
    all_jsonl: dict[str, str] = {}

    agent_index = 1

    for history_type in _HISTORY_TYPES:
        for rep in range(agents_per_type):
            a_id = _agent_id(agent_index)
            name = _NAMES[(agent_index - 1) % len(_NAMES)]

            # Skill profile: integer percentiles in [40, 90].
            skill = SkillProfile(
                empathy_percentile=int(rng.uniform(40, 90)),
                resolution_percentile=int(rng.uniform(40, 90)),
                brand_voice_percentile=int(rng.uniform(40, 90)),
                accuracy_percentile=int(rng.uniform(40, 90)),
            )

            # Tenure: new agents get 5–14 days; others 30–365.
            if history_type == HistoryType.NEW:
                tenure = rng.randint(5, 14)
            else:
                tenure = rng.randint(30, 365)

            linked_conv = link_source[link_idx % len(link_source)]
            link_idx += 1

            agent_profile = AgentProfile(
                agent_id=a_id,
                tenant_id=tenant_id,
                history_type=history_type,
                tenure_days=tenure,
                skill_profile=skill,
                name=name,
                linked_conversation_id=linked_conv,
            )
            agent_profiles.append(agent_profile)

            # --- Build archetype records ---
            builder = _ARCHETYPE_BUILDERS[history_type]
            traj_rows, coaching_events, dispute_records = builder(
                a_id,
                conv_pool,
                rng,
                traj_counter,
                coach_counter,
                disp_counter,
            )

            # --- Write files ---
            agent_dir = agents_dir / a_id
            agent_dir.mkdir(parents=True, exist_ok=True)

            # profile.json
            profile_dict = agent_profile.model_dump(mode="json")
            _write_profile_json(agent_dir / "profile.json", profile_dict)

            # coaching_history.jsonl
            coaching_path = agent_dir / "coaching_history.jsonl"
            _write_jsonl(coaching_path, coaching_events)

            # score_trajectory.jsonl
            traj_path = agent_dir / "score_trajectory.jsonl"
            _write_jsonl(traj_path, traj_rows)

            # dispute_history.jsonl
            disp_path = agent_dir / "dispute_history.jsonl"
            _write_jsonl(disp_path, dispute_records)

            # Accumulate JSONL content for hash computation.
            combined = ""
            for path in [coaching_path, traj_path, disp_path]:
                with path.open("r", encoding="utf-8") as fh:
                    combined += fh.read()
            all_jsonl[a_id] = combined

            agent_index += 1

    # --- Compute content hash over all JSONL, sorted by agent_id ---
    hasher = hashlib.sha256()
    for a_id in sorted(all_jsonl):
        hasher.update(all_jsonl[a_id].encode("utf-8"))
    content_hash = f"sha256:{hasher.hexdigest()}"

    return agent_profiles, content_hash
