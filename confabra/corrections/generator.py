"""
Corrections log artifact generator — produces corrections.jsonl per profile.

Each corrections file contains planted human score correction records designed
to exercise Cat L1 signal-detection categories (few-shot learning loop testing).
The four pattern classes exercise distinct detection requirements:

- ``systematic_upward``: consistent tenant bias in one direction on a criterion —
  the primary signal the intelligence layer must detect and learn from.
- ``counterexample``: opposing corrections that prevent the model from
  over-generalising the systematic direction.
- ``cross_criterion_noise``: corrections on unrelated criteria that should not
  cause spurious cross-criterion learning.
- ``cross_tenant_boundary``: planted "Tenant B" records that must not bleed
  through tenant isolation into Tenant A's learning signal.

Noise injection is seed-deterministic so that the same seed always produces the
same jitter, rationale choices, and timestamp cluster offsets.  This enables
reproducible Cat L1 assertion failures.

Public API
----------
- :func:`generate_corrections` — build and write corrections.jsonl for one profile
"""

from __future__ import annotations

import hashlib
import json
import random
from datetime import datetime, timedelta, timezone
from pathlib import Path

from confabra.profiles.base import CorrectionPattern, Profile
from confabra.schemas import AgentProfile, CorrectionRecord, NoiseClass, PatternClass
from confabra.utils.atomic_write import atomic_write_text

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Base timestamp from which all correction clusters are offset.
# One month after the simulation epoch so corrections post-date conversations.
_BASE_TS = datetime(2025, 12, 1, 9, 0, 0, tzinfo=timezone.utc)

# Target correction counts per profile name.
_TARGET_COUNTS: dict[str, int] = {
    "saas": 25,
    "ps": 5,
}

# Terse rationale options for noise injection (~30% of records).
_TERSE_RATIONALES = ["wrong", "see notes", ""]

# Off-topic rationale templates — mention a criterion OTHER than the one corrected.
# Keys are the criterion being corrected; values are rationale texts about a
# different criterion (used for off_topic_rationale noise injection, ~10% of records).
_OFF_TOPIC_RATIONALE_TEMPLATES: dict[str, str] = {
    "empathy": "The resolution was complete and all steps were covered thoroughly.",
    "resolution": "The brand voice was consistent with the warm-exploratory style guide.",
    "brand_voice": "Accuracy against the KB was high; all claims were well-supported.",
    "accuracy": "Empathy signals were strong — agent acknowledged customer frustration early.",
}

# Probability thresholds for noise injection.
# Sequential check: roll < OFF_TOPIC → off_topic; roll < TERSE_CEILING → terse.
# Terse window = [0.10, 0.40) = ~30%; off_topic window = [0, 0.10) = ~10%.
_P_TERSE_RATIONALE = 0.40          # ceiling: combined off_topic + terse band
_P_OFF_TOPIC_RATIONALE = 0.10      # 10% of all records get an off-topic rationale
_P_CONTRADICTORY = 0.15            # 15% of systematic_upward records are contradictory

# Score parameters per direction.
_DELTA_MIN = 15   # minimum human-AI delta for systematic_upward records
_DELTA_MAX = 25   # maximum human-AI delta for systematic_upward records
_JITTER_MIN = 3   # minimum score jitter
_JITTER_MAX = 5   # maximum score jitter

# Absolute score clamp ranges.
_AI_SCORE_MIN = 30
_AI_SCORE_MAX = 90
_HUMAN_SCORE_MIN = 40
_HUMAN_SCORE_MAX = 95

# Special agent_id prefix for cross-tenant records.
_CROSS_TENANT_AGENT_PREFIX = "tenant_b_agent_"

# PatternClass string-to-enum mapping (profile returns plain strings).
_PATTERN_CLASS_MAP: dict[str, PatternClass] = {
    "systematic_upward": PatternClass.SYSTEMATIC_UPWARD,
    "counterexample": PatternClass.COUNTEREXAMPLE,
    "cross_criterion_noise": PatternClass.CROSS_CRITERION_NOISE,
    "cross_tenant_boundary": PatternClass.CROSS_TENANT_BOUNDARY,
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _clamp(value: int, lo: int, hi: int) -> int:
    """Clamp an integer score to [lo, hi]."""
    return max(lo, min(hi, value))


def _build_timestamp_clusters(
    profile_name: str,
    target_count: int,
    rng: random.Random,
) -> list[datetime]:
    """
    Build a list of bursty timestamps for ``target_count`` correction records.

    Corrections arrive in 2-3 temporal clusters with multi-day gaps between
    them, simulating realistic human reviewer behaviour (batch review sessions).
    Each record within a cluster gets a uniformly-jittered offset so records
    within the same cluster are not identical.

    Cluster layout:
    - Cluster 1: starting at ``_BASE_TS``, spanning 2 hours
    - Cluster 2: starting at ``_BASE_TS + 3 days``, spanning 3 hours
    - Cluster 3 (SaaS only): starting at ``_BASE_TS + 7 days``, spanning 2 hours
    """
    cluster_starts = [
        _BASE_TS,
        _BASE_TS + timedelta(days=3),
    ]
    cluster_durations_minutes = [120, 180]  # hours × 60

    if profile_name == "saas":
        cluster_starts.append(_BASE_TS + timedelta(days=7))
        cluster_durations_minutes.append(120)

    n_clusters = len(cluster_starts)

    # Distribute records across clusters as evenly as possible, rounding up
    # for early clusters.
    base_per_cluster = target_count // n_clusters
    remainder = target_count % n_clusters

    timestamps: list[datetime] = []
    for i, (start, duration_min) in enumerate(
        zip(cluster_starts, cluster_durations_minutes)
    ):
        count = base_per_cluster + (1 if i < remainder else 0)
        for _ in range(count):
            offset_seconds = rng.uniform(0, duration_min * 60)
            timestamps.append(start + timedelta(seconds=offset_seconds))

    # Sort within each cluster so the output is readable, but do NOT sort
    # globally — the inter-cluster gap is the bursty signal.
    # We return in cluster order (already ordered by cluster start).
    return timestamps


def _choose_noise_class(
    pattern_class: PatternClass,
    is_contradictory: bool,
    rationale_noise: str,  # "full" | "terse" | "off_topic"
) -> NoiseClass:
    """
    Assign exactly one noise class per record, using the most specific label.

    Priority order (highest specificity wins):
    1. cross_tenant — always wins for cross_tenant_boundary records
    2. contradictory — systematic_upward records flagged as contradictory
    3. off_topic_rationale — records with a rationale about a different criterion
    4. terse_rationale — records with a terse/empty rationale
    5. score_jitter — cross_criterion_noise records (they are structurally noisy)
    6. clean — the baseline for systematic_upward records that passed all above
    """
    if pattern_class == PatternClass.CROSS_TENANT_BOUNDARY:
        return NoiseClass.CROSS_TENANT
    if is_contradictory:
        return NoiseClass.CONTRADICTORY
    if rationale_noise == "off_topic":
        return NoiseClass.NON_EXPLAINING_RATIONALE
    if rationale_noise == "terse":
        return NoiseClass.TERSE_RATIONALE
    if pattern_class == PatternClass.CROSS_CRITERION_NOISE:
        return NoiseClass.SCORE_JITTER
    # systematic_upward or counterexample with a full rationale — clean baseline
    return NoiseClass.CLEAN


def _generate_ai_score(
    human_score: int,
    pattern_class: PatternClass,
    direction: str,
    is_contradictory: bool,
    rng: random.Random,
) -> int:
    """
    Derive a plausible ai_score from the planted human_score.

    For systematic_upward records the AI under-scores the human: the delta is
    drawn uniformly from [_DELTA_MIN, _DELTA_MAX].  For counterexample records
    the AI and human scores are close (within ±5).  Cross-criterion and
    cross-tenant records get a random base then jitter.

    A seed-deterministic jitter in [_JITTER_MIN, _JITTER_MAX] is always added
    to the base AI score, with sign chosen uniformly at random.

    Contradictory records within systematic_upward are special: the AI score
    is *above* the human score (human corrected downward, i.e. AI was correct).
    """
    jitter = rng.randint(_JITTER_MIN, _JITTER_MAX)
    jitter_sign = rng.choice([-1, 1])

    if pattern_class == PatternClass.SYSTEMATIC_UPWARD:
        if is_contradictory:
            # AI was actually right — scores should be close or AI above human.
            # Capped at 7 (not 12) to prevent too many records pinning at _AI_SCORE_MAX.
            delta = rng.randint(3, 7)
            base_ai = human_score + delta
        else:
            # AI consistently under-scores relative to human (upward direction).
            delta = rng.randint(_DELTA_MIN, _DELTA_MAX)
            base_ai = human_score - delta
    elif pattern_class == PatternClass.COUNTEREXAMPLE:
        # Human corrects downward — AI was too generous.
        delta = rng.randint(_DELTA_MIN, _DELTA_MAX)
        base_ai = human_score + delta
    else:
        # Cross-criterion and cross-tenant: freeform variation.
        base_ai = human_score + rng.randint(-20, 20)

    ai_score = _clamp(base_ai + jitter_sign * jitter, _AI_SCORE_MIN, _AI_SCORE_MAX)
    return ai_score


def _generate_human_score(
    pattern_class: PatternClass,
    direction: str,
    rng: random.Random,
) -> int:
    """
    Generate the planted human_score for a correction record.

    Systematic-upward records use higher human scores (60-90) because the
    tenant consistently believes the AI under-scores.  Counterexample records
    use lower human scores (40-70) as the tenant is correcting downward.
    Cross-criterion and cross-tenant records span the full range.
    """
    if pattern_class == PatternClass.SYSTEMATIC_UPWARD:
        return rng.randint(65, _HUMAN_SCORE_MAX)
    elif pattern_class == PatternClass.COUNTEREXAMPLE:
        return rng.randint(_HUMAN_SCORE_MIN, 65)
    else:
        return rng.randint(_HUMAN_SCORE_MIN, _HUMAN_SCORE_MAX)


def _choose_rationale(
    pattern: CorrectionPattern,
    rationale_noise: str,
    rng: random.Random,
) -> str:
    """
    Build the rationale string for a correction record.

    Three tiers:
    - "full": use the pattern's ``rationale_template`` verbatim (one sentence)
    - "terse": pick from _TERSE_RATIONALES at random
    - "off_topic": use a pre-canned sentence about a DIFFERENT criterion
    """
    if rationale_noise == "full":
        return pattern.rationale_template
    elif rationale_noise == "terse":
        return rng.choice(_TERSE_RATIONALES)
    else:  # off_topic
        return _OFF_TOPIC_RATIONALE_TEMPLATES.get(
            pattern.criterion,
            "See scoring rubric for context.",
        )


def _select_agent_id(
    pattern_class: PatternClass,
    agent_profiles: list[AgentProfile],
    record_index: int,
) -> str:
    """
    Choose an agent_id appropriate to the pattern class.

    Cross-tenant records use a synthetic "tenant_b_agent_NNN" identifier that
    does not exist in the real agent pool.  All other records cycle through
    the provided agent_profiles list.
    """
    if pattern_class == PatternClass.CROSS_TENANT_BOUNDARY:
        # Enumerate synthetic tenant-B agent IDs (001-indexed).
        agent_num = (record_index % 3) + 1
        return f"{_CROSS_TENANT_AGENT_PREFIX}{agent_num:03d}"

    # Cycle through real agent profiles.
    real_agents = [a for a in agent_profiles]
    if not real_agents:
        return "agent_unknown"
    return real_agents[record_index % len(real_agents)].agent_id


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def generate_corrections(
    profile: Profile,
    output_dir: Path,
    seed: int,
    agent_profiles: list[AgentProfile],
    organic_conversation_ids: list[str],
) -> tuple[list[CorrectionRecord], str]:
    """
    Generate ``corrections.jsonl`` for the given profile.

    Each correction record is a planted human score override designed to
    exercise a specific Cat L1 signal-detection pattern.  Noise is injected
    in a seed-deterministic fashion so the output is fully reproducible.

    Steps
    -----
    1. Pull the profile's correction_pattern_set (up to target count).
    2. Pad or truncate the pattern list to the target count by cycling.
    3. Assign bursty timestamps across 2-3 temporal clusters.
    4. For each record: derive human_score, ai_score (with jitter), rationale,
       noise_class, and agent/conversation references.
    5. Inject contradictory records into the systematic_upward subset (~15%).
    6. Write to ``output_dir/corrections.jsonl`` (one JSON object per line).
    7. Return (records, sha256_hex) where the hash covers the written content.

    Parameters
    ----------
    profile:
        The active Layer 2 profile (SaaS or PS).
    output_dir:
        Directory to write ``corrections.jsonl`` into (must already exist).
    seed:
        Deterministic RNG seed — same seed always produces the same output.
    agent_profiles:
        Agent fixtures from Task 12; used for agent_id assignment.
    organic_conversation_ids:
        Conversation IDs from the organic corpus layer; cycled for linkage.

    Returns
    -------
    tuple[list[CorrectionRecord], str]
        The list of generated CorrectionRecord objects and the SHA-256 hex
        digest of the written JSONL content.
    """
    rng = random.Random(seed)

    target_count = _TARGET_COUNTS.get(profile.name, 10)
    patterns: list[CorrectionPattern] = list(profile.correction_pattern_set())

    # Cycle or truncate to hit the target count exactly.
    # If the profile defines fewer patterns than the target, we repeat the
    # pattern list (cycling).  If it defines more, we truncate.
    expanded: list[CorrectionPattern] = []
    if patterns:
        while len(expanded) < target_count:
            expanded.extend(patterns)
        expanded = expanded[:target_count]
    # If somehow correction_pattern_set() is empty we produce zero records.

    # Pre-compute rationale noise allocation:
    # ~70% full, ~10% off_topic, ~20% terse (off_topic is a subset of non-full,
    # and takes precedence over terse when both thresholds fire).
    # We assign noise type per-index deterministically.
    rationale_noise_choices: list[str] = []
    for _ in expanded:
        roll = rng.random()
        if roll < _P_OFF_TOPIC_RATIONALE:
            rationale_noise_choices.append("off_topic")
        elif roll < _P_TERSE_RATIONALE:
            rationale_noise_choices.append("terse")
        else:
            rationale_noise_choices.append("full")

    # Pre-compute which systematic_upward indices are contradictory.
    systematic_indices = [
        i for i, p in enumerate(expanded)
        if p.pattern_class == "systematic_upward"
    ]
    n_contradictory = max(1, round(len(systematic_indices) * _P_CONTRADICTORY))
    contradictory_indices: set[int] = set(
        rng.sample(systematic_indices, min(n_contradictory, len(systematic_indices)))
        if systematic_indices else []
    )

    # Build bursty timestamps (returns exactly target_count timestamps).
    timestamps = _build_timestamp_clusters(profile.name, target_count, rng)
    # Pad in the rare edge case where expanded is shorter than target (empty patterns).
    while len(timestamps) < len(expanded):
        timestamps.append(_BASE_TS + timedelta(hours=rng.uniform(0, 48)))

    # Determine tenant_id from agent_profiles (fall back to profile name).
    tenant_id: str
    if agent_profiles:
        tenant_id = agent_profiles[0].tenant_id
    else:
        tenant_id = f"tenant_{profile.name}"

    # Generate records.
    records: list[CorrectionRecord] = []
    for idx, (pattern, rationale_noise, ts) in enumerate(
        zip(expanded, rationale_noise_choices, timestamps)
    ):
        correction_id = f"corr_{idx + 1:06d}"

        pattern_class_enum = _PATTERN_CLASS_MAP.get(
            pattern.pattern_class, PatternClass.CROSS_CRITERION_NOISE
        )

        is_contradictory = idx in contradictory_indices

        human_score = _generate_human_score(pattern_class_enum, pattern.direction, rng)
        ai_score = _generate_ai_score(
            human_score, pattern_class_enum, pattern.direction, is_contradictory, rng
        )

        rationale = _choose_rationale(pattern, rationale_noise, rng)
        noise_class = _choose_noise_class(pattern_class_enum, is_contradictory, rationale_noise)

        agent_id = _select_agent_id(pattern_class_enum, agent_profiles, idx)

        # Cycle conversation IDs — fall back to a synthetic ID if the list is empty.
        if organic_conversation_ids:
            conversation_id = organic_conversation_ids[idx % len(organic_conversation_ids)]
        else:
            conversation_id = f"conv_{idx + 1:04d}"

        record = CorrectionRecord(
            correction_id=correction_id,
            tenant_id=tenant_id,
            agent_id=agent_id,
            conversation_id=conversation_id,
            criterion=pattern.criterion,
            ai_score=ai_score,
            human_score=human_score,
            rationale=rationale,
            pattern_class=pattern_class_enum,
            timestamp=ts,
            noise_class=noise_class,
        )
        records.append(record)

    # Serialise to JSONL.
    lines: list[str] = []
    for record in records:
        lines.append(record.model_dump_json())
    content = "\n".join(lines) + ("\n" if lines else "")

    # Write output file atomically.
    output_path = output_dir / "corrections.jsonl"
    atomic_write_text(output_path, content)

    # Compute SHA-256 hash over the written content.
    content_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()

    return records, content_hash
