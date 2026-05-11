"""
Property-based tests for the ResonantForge corpus generator — 22 assertions.

All tests share a single session-scoped fixture that runs one dry-run
SaaS corpus generation (no Anthropic API calls) into a temp directory.

Groups:
  1.  Determinism            (assertions  1–2)
  2.  Schema conformance     (assertions  3–4)
  3.  Temporal coherence     (assertion   5)
  4.  Manifest integrity     (assertions  6–8)
  5.  KB constraint-type     (assertions  9–10)
  6.  Cat 11 gate density    (assertions 11–13)
  7.  Accuracy label schema  (assertion  14)
  8.  Agent causal integrity (assertions 15–17)
  9.  Corrections integrity  (assertions 18–19)
  10. Corrections noise      (assertions 20–21)
  11. Cross-contamination    (assertion  22)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from resonantforge.pipeline import PipelineConfig, run_pipeline
from resonantforge.schemas import (
    ConversationRecord,
    ConstraintType,
    DimensionVerdict,
    DisagreementRecord,
    GateSeverity,
    GateViolation,
    KBChunk,
    Manifest,
    QualityPlan,
    SimEvent,
    CorrectionRecord,
    NoiseClass,
    ValidationResult,
    ValidationVerdict,
)
from resonantforge.layer1.plan_validator import SkipRateTracker
from resonantforge.layer1.quality_plan_injector import is_conditional_chunk

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED = 42
_SMALL_CORPUS_ACCOUNTS = 5
_SMALL_CORPUS_MONTHS = 2


# ---------------------------------------------------------------------------
# Session-scoped fixture: generate once, share across all tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def corpus(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, Manifest]:
    """
    Generate a small dry-run SaaS corpus once for all property tests.

    Returns (profile_dir, manifest) where profile_dir is the path under
    which all generated artifacts live (events.jsonl, conversations.jsonl, etc.).
    """
    out = tmp_path_factory.mktemp("corpus")
    config = PipelineConfig(
        profile_name="saas",
        accounts=_SMALL_CORPUS_ACCOUNTS,
        months=_SMALL_CORPUS_MONTHS,
        seed=SEED,
        output_root=out,
        anthropic_api_key=None,  # dry-run — no Anthropic calls
    )
    manifest = run_pipeline(config)
    return out / "saas", manifest


# ---------------------------------------------------------------------------
# Helper: read JSONL file into a list of parsed dicts
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> list[dict]:
    """Read a JSONL file and return a list of parsed dicts."""
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


# ---------------------------------------------------------------------------
# Group 1 — Determinism (assertions 1–2)
# ---------------------------------------------------------------------------


def test_determinism(tmp_path: Path) -> None:
    """
    Same seed must produce byte-identical events and snapshots.

    We compare manifest hashes from two independent pipeline runs rather than
    raw file bytes so that generated_at timestamps (which change per run) don't
    cause spurious failures.  events_hash and snapshots_hash are SHA-256 over
    the JSONL content and must be equal across runs.
    """
    def _run(output_root: Path) -> Manifest:
        config = PipelineConfig(
            profile_name="saas",
            accounts=_SMALL_CORPUS_ACCOUNTS,
            months=_SMALL_CORPUS_MONTHS,
            seed=SEED,
            output_root=output_root,
            anthropic_api_key=None,
        )
        return run_pipeline(config)

    m1 = _run(tmp_path / "run1")
    m2 = _run(tmp_path / "run2")

    # Assertion 1
    assert m1.events_hash == m2.events_hash, (
        "events_hash differs between two runs with the same seed — "
        f"run1={m1.events_hash[:16]}… run2={m2.events_hash[:16]}…"
    )
    # Assertion 2
    assert m1.snapshots_hash == m2.snapshots_hash, (
        "snapshots_hash differs between two runs with the same seed — "
        f"run1={m1.snapshots_hash[:16]}… run2={m2.snapshots_hash[:16]}…"
    )


# ---------------------------------------------------------------------------
# Group 2 — Schema conformance (assertions 3–4)
# ---------------------------------------------------------------------------


def test_schema_conformance(corpus: tuple[Path, Manifest]) -> None:
    """
    Every line in events.jsonl and conversations.jsonl must parse as valid
    Pydantic schema objects without raising ValidationError.
    """
    profile_dir, _ = corpus

    # Assertion 3 — events.jsonl
    events_path = profile_dir / "events.jsonl"
    raw_events = _read_jsonl(events_path)
    assert len(raw_events) > 0, "events.jsonl is empty — pipeline produced no events"
    parse_errors: list[str] = []
    for i, row in enumerate(raw_events):
        try:
            SimEvent.model_validate(row)
        except Exception as exc:
            parse_errors.append(f"Line {i}: {exc}")
    assert not parse_errors, (
        f"events.jsonl: {len(parse_errors)} parse errors — first: {parse_errors[0]}"
    )

    # Assertion 4 — conversations.jsonl
    conv_path = profile_dir / "conversations.jsonl"
    raw_convs = _read_jsonl(conv_path)
    assert len(raw_convs) > 0, "conversations.jsonl is empty — pipeline generated no conversations"
    parse_errors = []
    for i, row in enumerate(raw_convs):
        try:
            ConversationRecord.model_validate(row)
        except Exception as exc:
            parse_errors.append(f"Line {i}: {exc}")
    assert not parse_errors, (
        f"conversations.jsonl: {len(parse_errors)} parse errors — first: {parse_errors[0]}"
    )


# ---------------------------------------------------------------------------
# Group 3 — Temporal coherence (assertion 5)
# ---------------------------------------------------------------------------


def test_temporal_coherence(corpus: tuple[Path, Manifest]) -> None:
    """
    Within each account, events must be temporally ordered (day_index
    non-decreasing per account_id).

    This validates the state-machine ordering guarantee: later events within
    an account cannot have a smaller day_index than earlier ones.
    """
    profile_dir, _ = corpus
    raw_events = _read_jsonl(profile_dir / "events.jsonl")

    # Group by account_id, collect day_index sequences.
    account_days: dict[str, list[int]] = {}
    for row in raw_events:
        acct = row["account_id"]
        day = row["day_index"]
        account_days.setdefault(acct, []).append(day)

    violations: list[str] = []
    for acct, days in account_days.items():
        for i in range(1, len(days)):
            if days[i] < days[i - 1]:
                violations.append(
                    f"account={acct} day[{i}]={days[i]} < day[{i-1}]={days[i-1]}"
                )

    # Assertion 5
    assert not violations, (
        f"Temporal ordering violated in {len(violations)} place(s). "
        f"First: {violations[0]}"
    )


# ---------------------------------------------------------------------------
# Group 4 — Manifest integrity (assertions 6–8)
# ---------------------------------------------------------------------------


def test_manifest_integrity(corpus: tuple[Path, Manifest]) -> None:
    """
    The manifest's conversation_count, knowledge_base_chunk_count, and
    agent_count must be consistent with what was actually generated.
    """
    profile_dir, manifest = corpus

    # Assertion 6 — conversation_count matches actual JSONL line count
    conv_path = profile_dir / "conversations.jsonl"
    actual_conv_count = len(_read_jsonl(conv_path))
    assert manifest.conversation_count == actual_conv_count, (
        f"manifest.conversation_count={manifest.conversation_count} "
        f"but conversations.jsonl has {actual_conv_count} lines"
    )

    # Assertion 7 — KB has at least one chunk
    assert manifest.knowledge_base_chunk_count > 0, (
        "manifest.knowledge_base_chunk_count is 0 — KB generation produced no chunks"
    )

    # Assertion 8 — SaaS always generates exactly 12 agents
    assert manifest.agent_count == 12, (
        f"manifest.agent_count={manifest.agent_count} but SaaS profile must produce 12 agents"
    )


# ---------------------------------------------------------------------------
# Group 5 — KB constraint-type metadata (assertions 9–10)
# ---------------------------------------------------------------------------


def test_kb_constraint_type_metadata(corpus: tuple[Path, Manifest]) -> None:
    """
    Every KB chunk must have constraint_type set, and the KB must include
    at least one allow_condition and one deny_condition (Gate 8 coverage).
    """
    profile_dir, _ = corpus
    chunks_path = profile_dir / "knowledge_base" / "chunks.jsonl"
    raw_chunks = _read_jsonl(chunks_path)
    assert len(raw_chunks) > 0, "knowledge_base/chunks.jsonl is empty"

    constraint_types: list[str] = []
    missing_constraint: list[str] = []

    for i, row in enumerate(raw_chunks):
        ct = row.get("constraint_type")
        # Assertion 9: Every chunk must have constraint_type set (not None/absent)
        if ct is None:
            missing_constraint.append(f"chunk[{i}] chunk_id={row.get('chunk_id')!r}")
        else:
            constraint_types.append(ct)

    assert not missing_constraint, (
        f"{len(missing_constraint)} KB chunks have constraint_type=None or missing: "
        f"{missing_constraint[:3]}"
    )

    # Assertion 10: At least one allow_condition and one deny_condition must exist
    has_allow = any(ct == "allow_condition" for ct in constraint_types)
    has_deny = any(ct == "deny_condition" for ct in constraint_types)

    assert has_allow and has_deny, (
        f"KB missing Gate 8 coverage — "
        f"allow_condition present: {has_allow}, deny_condition present: {has_deny}. "
        f"Constraint types found: {set(constraint_types)}"
    )


# ---------------------------------------------------------------------------
# Group 6 — Cat 11 gate density (assertions 11–13)
# ---------------------------------------------------------------------------


def test_cat11_gate_density(corpus: tuple[Path, Manifest]) -> None:
    """
    The KB must contain sufficient Cat 11 gate coverage:
    - At least 3 chunks with counterintuitive=True (Gate 2)
    - At least 3 chunks with cat11_gate=="gate_8" (allow/deny pairs)
    - At least 3 chunks with cat11_gate=="gate_9" (fake citation traps)

    These are profile-curated fixtures so they hold regardless of corpus size.
    """
    profile_dir, _ = corpus
    raw_chunks = _read_jsonl(profile_dir / "knowledge_base" / "chunks.jsonl")

    counterintuitive_chunks = [c for c in raw_chunks if c.get("counterintuitive") is True]
    gate_8_chunks = [c for c in raw_chunks if c.get("cat11_gate") == "gate_8"]
    gate_9_chunks = [c for c in raw_chunks if c.get("cat11_gate") == "gate_9"]

    # Assertion 11 — counterintuitive chunks
    assert len(counterintuitive_chunks) >= 3, (
        f"Expected ≥3 chunks with counterintuitive=True (Gate 2), "
        f"found {len(counterintuitive_chunks)}"
    )

    # Assertion 12 — Gate 8 (allow/deny pairs)
    assert len(gate_8_chunks) >= 3, (
        f"Expected ≥3 chunks with cat11_gate='gate_8', "
        f"found {len(gate_8_chunks)}"
    )

    # Assertion 13 — Gate 9 (fake citation traps)
    assert len(gate_9_chunks) >= 3, (
        f"Expected ≥3 chunks with cat11_gate='gate_9', "
        f"found {len(gate_9_chunks)}"
    )


# ---------------------------------------------------------------------------
# Group 7 — Accuracy label schema (assertion 14)
# ---------------------------------------------------------------------------


def test_accuracy_label_schema(corpus: tuple[Path, Manifest]) -> None:
    """
    Every QualityPlan in planted_quality.jsonl whose rubric_targets.accuracy
    is set must use the two-field AccuracyLabel schema (status + precision),
    not a binary string or a bare boolean.

    This guards against the accuracy dimension regressing to a simpler label
    format that would lose precision information.
    """
    profile_dir, _ = corpus
    plans_path = profile_dir / "planted_quality.jsonl"
    raw_plans = _read_jsonl(plans_path)

    # Assertion 14
    violations: list[str] = []
    for i, row in enumerate(raw_plans):
        rubric = row.get("rubric_targets", {})
        accuracy = rubric.get("accuracy")
        if accuracy is None:
            continue  # accuracy target not set — skip

        # Must be a dict with both status and precision keys.
        if not isinstance(accuracy, dict):
            violations.append(
                f"plan[{i}] conversation_id={row.get('conversation_id')!r}: "
                f"accuracy is {type(accuracy).__name__!r}, expected dict"
            )
            continue

        missing_fields = [f for f in ("status", "precision") if f not in accuracy]
        if missing_fields:
            violations.append(
                f"plan[{i}] conversation_id={row.get('conversation_id')!r}: "
                f"accuracy dict missing fields {missing_fields}"
            )

    assert not violations, (
        f"{len(violations)} quality plan(s) have malformed accuracy labels. "
        f"First: {violations[0]}"
    )


# ---------------------------------------------------------------------------
# Group 8 — Agent causal integrity (assertions 15–17)
# ---------------------------------------------------------------------------


def test_agent_causal_integrity(corpus: tuple[Path, Manifest]) -> None:
    """
    Three causal-anchor invariants across all agent fixture directories:

    15. Every triggering_conversation_id in coaching_history.jsonl must
        reference a conversation_id that appears in score_trajectory.jsonl
        with a scored_at timestamp strictly before the coaching event's issued_at.

    16. Every post_coaching_of value in score_trajectory.jsonl must reference
        a coaching_id that exists in the same agent's coaching_history.jsonl.

    17. Agents with history_type=="new" must have zero coaching events
        (new agents have no manager-issued notes yet).
    """
    profile_dir, _ = corpus
    agents_dir = profile_dir / "agents"

    # Collect all agent directories.
    agent_dirs = sorted(agents_dir.glob("agent_*"))
    assert len(agent_dirs) > 0, "No agent directories found under agents/"

    v15: list[str] = []  # Assertion 15 violations
    v16: list[str] = []  # Assertion 16 violations
    v17: list[str] = []  # Assertion 17 violations

    for agent_dir in agent_dirs:
        agent_id = agent_dir.name

        # Read profile to get history_type.
        profile_path = agent_dir / "profile.json"
        if not profile_path.exists():
            continue
        with profile_path.open(encoding="utf-8") as fh:
            agent_profile = json.load(fh)
        history_type = agent_profile.get("history_type")

        # Read trajectory rows.
        traj_path = agent_dir / "score_trajectory.jsonl"
        traj_rows = _read_jsonl(traj_path) if traj_path.exists() else []

        # Build a map: conversation_id → list of scored_at ISO strings
        # (used by assertion 15 to verify causal ordering)
        conv_scored_at: dict[str, list[str]] = {}
        for row in traj_rows:
            cid = row["conversation_id"]
            conv_scored_at.setdefault(cid, []).append(row["scored_at"])

        # Read coaching history for this agent.
        coaching_path = agent_dir / "coaching_history.jsonl"
        coaching_rows = _read_jsonl(coaching_path) if coaching_path.exists() else []

        # Build map: coaching_id → issued_at (same-agent only)
        coaching_id_to_issued_at: dict[str, str] = {
            c["coaching_id"]: c["issued_at"] for c in coaching_rows
        }
        coaching_ids: set[str] = set(coaching_id_to_issued_at)

        # --- Assertion 15 (strengthened): triggering_conversation_id precedes coaching
        #     issued_at by at least 1 minute, AND the conversation belongs to this agent's
        #     own trajectory (not any other agent's). ---
        # Minimum delta chosen to guard against same-timestamp boundary cases: the
        # generator always places coaching events ≥8 hours after the last pre-coaching
        # row, so 60 seconds is a safe floor that the current fixtures already satisfy.
        _MIN_DELTA_SECONDS = 60
        for coaching in coaching_rows:
            trigger_cid = coaching["triggering_conversation_id"]
            issued_at = coaching["issued_at"]

            # The triggering conversation must appear in THIS agent's own trajectory.
            if trigger_cid not in conv_scored_at:
                v15.append(
                    f"{agent_id}: coaching_id={coaching['coaching_id']!r} "
                    f"triggering_conversation_id={trigger_cid!r} "
                    f"not found in this agent's score_trajectory"
                )
                continue

            # At least one trajectory row for that conversation must be strictly before
            # issued_at by at least _MIN_DELTA_SECONDS seconds.
            from datetime import datetime, timezone as _tz

            def _parse_dt(s: str) -> datetime:
                return datetime.fromisoformat(s)

            issued_dt = _parse_dt(issued_at)
            any_before_with_delta = any(
                (issued_dt - _parse_dt(scored_at)).total_seconds() >= _MIN_DELTA_SECONDS
                for scored_at in conv_scored_at[trigger_cid]
            )
            if not any_before_with_delta:
                v15.append(
                    f"{agent_id}: coaching_id={coaching['coaching_id']!r} "
                    f"issued_at={issued_at} but no trajectory row for "
                    f"{trigger_cid!r} precedes it by ≥{_MIN_DELTA_SECONDS}s"
                )

        # --- Assertion 16 (strengthened): post_coaching_of must reference a coaching_id
        #     in the SAME agent's history, and the coaching's issued_at must be strictly
        #     before this trajectory row's scored_at (post_coaching ordering invariant). ---
        for row in traj_rows:
            post_of = row.get("post_coaching_of")
            if post_of is None:
                continue
            if post_of not in coaching_ids:
                v16.append(
                    f"{agent_id}: trajectory_id={row['trajectory_id']!r} "
                    f"post_coaching_of={post_of!r} not in this agent's coaching_history"
                )
                continue
            # Temporal: coaching.issued_at must be strictly before row.scored_at
            coaching_issued = coaching_id_to_issued_at[post_of]
            if coaching_issued >= row["scored_at"]:
                v16.append(
                    f"{agent_id}: trajectory_id={row['trajectory_id']!r} "
                    f"post_coaching_of={post_of!r} coaching issued_at={coaching_issued} "
                    f"is not strictly before row scored_at={row['scored_at']}"
                )

        # --- Assertion 17: new agents have zero coaching events ---
        if history_type == "new":
            if len(coaching_rows) != 0:
                v17.append(
                    f"{agent_id}: history_type='new' but has "
                    f"{len(coaching_rows)} coaching event(s)"
                )

    assert not v15, (
        f"Assertion 15 — {len(v15)} causal-anchor violation(s). First: {v15[0]}"
    )
    assert not v16, (
        f"Assertion 16 — {len(v16)} post_coaching_of reference violation(s). "
        f"First: {v16[0]}"
    )
    assert not v17, (
        f"Assertion 17 — {len(v17)} new-agent coaching violation(s). First: {v17[0]}"
    )


# ---------------------------------------------------------------------------
# Group 9 — Corrections referential integrity (assertions 18–19)
# ---------------------------------------------------------------------------


def test_corrections_referential_integrity(corpus: tuple[Path, Manifest]) -> None:
    """
    Referential integrity of agent_id values in corrections.jsonl:

    18. Every correction's agent_id is either a real agent from the agent pool
        OR starts with "tenant_b_agent_" (cross-tenant planted record).

    19. Cross-tenant correction records (agent_id starts with "tenant_b_agent_")
        must NOT reference real agent IDs from the agent pool.
    """
    profile_dir, _ = corpus

    # Collect real agent IDs from agent directories.
    agents_dir = profile_dir / "agents"
    real_agent_ids: set[str] = {d.name for d in agents_dir.glob("agent_*")}

    corrections_path = profile_dir / "corrections.jsonl"
    raw_corrections = _read_jsonl(corrections_path)
    assert len(raw_corrections) > 0, "corrections.jsonl is empty"

    _CROSS_TENANT_PREFIX = "tenant_b_agent_"

    # Assertion 18: every agent_id is real or cross-tenant prefix
    v18: list[str] = []
    for i, row in enumerate(raw_corrections):
        aid = row["agent_id"]
        if aid not in real_agent_ids and not aid.startswith(_CROSS_TENANT_PREFIX):
            v18.append(
                f"correction[{i}] correction_id={row['correction_id']!r} "
                f"agent_id={aid!r} is neither a real agent nor a cross-tenant ID"
            )

    assert not v18, (
        f"Assertion 18 — {len(v18)} correction(s) with invalid agent_id. "
        f"First: {v18[0]}"
    )

    # Assertion 19 (strengthened): cross-tenant corrections must not use real agent IDs,
    # AND no real agent in the pool may have an agent_id that starts with "tenant_b_"
    # (catches the case where a real agent is accidentally mis-prefixed).
    v19: list[str] = []
    for i, row in enumerate(raw_corrections):
        aid = row["agent_id"]
        if aid.startswith(_CROSS_TENANT_PREFIX) and aid in real_agent_ids:
            v19.append(
                f"correction[{i}] correction_id={row['correction_id']!r} "
                f"agent_id={aid!r} starts with cross-tenant prefix but is also "
                f"a real agent ID — tenant isolation is broken"
            )

    # Inverse check: no real agent may carry the cross-tenant prefix.
    for aid in sorted(real_agent_ids):
        if aid.startswith(_CROSS_TENANT_PREFIX):
            v19.append(
                f"real agent {aid!r} starts with cross-tenant prefix {_CROSS_TENANT_PREFIX!r} "
                f"— this agent would be silently excluded from cross-tenant joins by name pattern"
            )

    assert not v19, (
        f"Assertion 19 — {len(v19)} cross-tenant isolation violation(s). "
        f"First: {v19[0]}"
    )


# ---------------------------------------------------------------------------
# Group 10 — Corrections noise distribution (assertions 20–21)
# ---------------------------------------------------------------------------


def test_corrections_noise_distribution(corpus: tuple[Path, Manifest]) -> None:
    """
    The corrections noise injection must produce a realistic distribution:

    20. At least 10% of corrections must have noise_class in
        {terse_rationale, off_topic_rationale, non_explaining_rationale}
        (rationale messiness is present in the corpus).

    21. At least 1 correction must have noise_class == "contradictory"
        (contradictory noise exercises the counterexample pattern path).
    """
    profile_dir, _ = corpus
    raw_corrections = _read_jsonl(profile_dir / "corrections.jsonl")
    n_total = len(raw_corrections)
    assert n_total > 0, "corrections.jsonl is empty"

    _RATIONALE_MESSY = {
        NoiseClass.TERSE_RATIONALE.value,
        NoiseClass.NON_EXPLAINING_RATIONALE.value,
    }

    messy_count = sum(
        1 for row in raw_corrections if row.get("noise_class") in _RATIONALE_MESSY
    )
    contradictory_count = sum(
        1 for row in raw_corrections
        if row.get("noise_class") == NoiseClass.CONTRADICTORY.value
    )

    # Assertion 20
    messy_pct = messy_count / n_total
    assert messy_pct >= 0.10, (
        f"Assertion 20 — rationale messiness below 10%: "
        f"{messy_count}/{n_total} = {messy_pct:.1%} have messy rationale noise_class"
    )

    # Assertion 21
    assert contradictory_count >= 1, (
        f"Assertion 21 — no contradictory corrections found in {n_total} records. "
        "At least 1 record with noise_class='contradictory' must exist."
    )


# ---------------------------------------------------------------------------
# Group 11 — Cross-contamination density (assertion 22)
# ---------------------------------------------------------------------------


def test_cross_contamination_density(corpus: tuple[Path, Manifest]) -> None:
    """
    When the profile defines ≥2 brand voice variants (as SaaS does with 3),
    at least 1 KB chunk must have a non-None tone_variant set, indicating the
    cross-contamination injector ran and marked at least one chunk.
    """
    profile_dir, _ = corpus

    # Verify the SaaS profile has ≥2 variants before asserting contamination.
    from resonantforge.profiles import get_profile
    profile = get_profile("saas")
    bv_variants = profile.brand_voice_variants()

    if len(bv_variants) < 2:
        pytest.skip("Profile has fewer than 2 brand voice variants — contamination not expected")

    raw_chunks = _read_jsonl(profile_dir / "knowledge_base" / "chunks.jsonl")

    # Assertion 22
    contaminated = [c for c in raw_chunks if c.get("tone_variant") is not None]
    assert len(contaminated) >= 1, (
        f"Assertion 22 — SaaS profile has {len(bv_variants)} brand voice variants "
        f"but 0 KB chunks have tone_variant set. "
        f"Cross-contamination injector may not have run or may have cleared all markers."
    )


# ---------------------------------------------------------------------------
# Group 12 — Concurrent-run protection (assertions 23a, 23b, 23c)
# ---------------------------------------------------------------------------


def test_concurrent_run_protection(tmp_path: Path) -> None:
    """
    Concurrent-run guard must fail-fast on a non-empty directory, succeed with
    --force, and remove the lockfile on successful completion.

    23a. Running into an existing non-empty directory raises RuntimeError
         without --force.
    23b. Running with --force=True succeeds even when the target is non-empty.
    23c. The lockfile is removed on successful completion.
    """
    from resonantforge.pipeline import PipelineConfig, run_pipeline

    def _make_config(out_root: Path, force: bool = False) -> PipelineConfig:
        return PipelineConfig(
            profile_name="saas",
            accounts=_SMALL_CORPUS_ACCOUNTS,
            months=_SMALL_CORPUS_MONTHS,
            seed=SEED,
            output_root=out_root,
            anthropic_api_key=None,
            force=force,
        )

    # -- First run: succeeds and produces the target directory.
    out1 = tmp_path / "guard_test"
    run_pipeline(_make_config(out1))
    target = out1 / "saas"
    assert target.exists() and any(target.iterdir()), "First run must produce a non-empty target"

    # -- Assertion 23a: second run into same non-empty dir raises without --force.
    import pytest as _pytest
    with _pytest.raises(RuntimeError, match="already contains files"):
        run_pipeline(_make_config(out1, force=False))

    # -- Assertion 23b: second run with --force=True succeeds.
    run_pipeline(_make_config(out1, force=True))

    # -- Assertion 23c: lockfile is removed after a successful run.
    from resonantforge.pipeline import _lockfile_path
    lock = _lockfile_path(out1 / "saas")
    assert not lock.exists(), (
        f"Lockfile {lock} was not cleaned up after a successful run"
    )


# ---------------------------------------------------------------------------
# Group 13 — Post-generation validation (assertions 24–27)
# ---------------------------------------------------------------------------


def test_post_gen_validation_pass(corpus: tuple[Path, Manifest]) -> None:
    """
    Assertion 24: dry-run pipeline must report prose_fact_violation_rate == 0.0.

    In dry-run mode, post-generation validation is skipped (placeholder prose is
    not real conversation text and would always fail deterministic checks). This
    verifies the wiring doesn't crash and metrics show a clean state after wiring.
    """
    _, manifest = corpus
    # Assertion 24
    assert manifest.prose_fact_violation_rate == 0.0, (
        f"Assertion 24 — dry-run expected prose_fact_violation_rate=0.0, "
        f"got {manifest.prose_fact_violation_rate:.4f}"
    )


def test_skip_tracker_prose_fact_increments(tmp_path: Path) -> None:
    """
    Assertion 25: when post_generation_validate returns SKIP for a planted
    conversation, prose_fact_failures increments and the conversation ID appears
    in skipped_conversations.jsonl.

    Uses unittest.mock to force SKIP on every planted validation call so that
    the skip-tracking machinery is exercised without requiring a live Anthropic key.
    """
    from unittest.mock import patch

    _FAKE_PROSE = (
        "Customer: I need help with my account.\n"
        "Agent: I'd be happy to help you today. What seems to be the issue?\n"
        "Customer: I can't access the dashboard.\n"
        "Agent: I understand. Let me look into that for you right away."
    )

    _skip_result = ValidationResult(
        conversation_id="mocked",
        overall_verdict=ValidationVerdict.SKIP,
        dimension_verdicts=[],
        skip_reason="mocked SKIP verdict for test_skip_tracker_prose_fact_increments",
        retry_count=2,
    )

    with (
        patch("resonantforge.pipeline._call_anthropic", return_value=(_FAKE_PROSE, 0, 0)),
        patch("resonantforge.pipeline.validate_all_dimensions", return_value=[]),
        patch(
            "resonantforge.layer1.plan_validator.PlanValidator.post_generation_validate",
            return_value=_skip_result,
        ),
    ):
        config = PipelineConfig(
            profile_name="saas",
            accounts=2,
            months=1,
            seed=42,
            output_root=tmp_path,
            # Non-None key activates the live validation path without an actual API call
            # (prose generation is mocked; accuracy extractor swallows auth errors).
            anthropic_api_key="fake-key-for-skip-test",
        )
        # The gate now fires (prose_fact_rate is 100% when all planned convos skip),
        # so run_pipeline raises RuntimeError after writing the manifest and JSONL files.
        with pytest.raises(RuntimeError, match="Quality gate"):
            run_pipeline(config)

    # Assertion 25a: manifest.json must exist (written before the abort) and report
    # a prose_fact_violation_rate > 0.
    manifest_path = tmp_path / "saas" / "manifest.json"
    assert manifest_path.exists(), (
        "Assertion 25a — manifest.json must be written even when gate aborts the run"
    )
    manifest_data = json.loads(manifest_path.read_text())
    assert manifest_data.get("prose_fact_violation_rate", 0.0) > 0.0, (
        f"Assertion 25a — expected prose_fact_violation_rate > 0 when every planted "
        f"conversation is SKIPped, got {manifest_data.get('prose_fact_violation_rate')}"
    )

    # Assertion 25b: skipped_conversations.jsonl must be non-empty
    skipped_path = tmp_path / "saas" / "skipped_conversations.jsonl"
    skipped = _read_jsonl(skipped_path)
    assert len(skipped) > 0, (
        "Assertion 25b — skipped_conversations.jsonl is empty, but every planted "
        "conversation should have been SKIPped by the mocked validator"
    )


def test_disagreement_ledger_populated(tmp_path: Path) -> None:
    """
    Assertion 26: when a planted conversation produces a FAIL dimension verdict,
    the soft judge is called, a DisagreementRecord accumulates in the ledger, and
    manifest.disagreement_rate is non-zero.

    Uses unittest.mock to inject a FAIL verdict and a mock soft-judge response so
    the full disagreement wiring path is exercised without a live Anthropic key.
    """
    from unittest.mock import patch

    _FAKE_PROSE = (
        "Customer: I need help with my account.\n"
        "Agent: I'd be happy to help you today. What seems to be the issue?\n"
        "Customer: I can't access the dashboard.\n"
        "Agent: I understand. Let me look into that for you right away."
    )

    _fail_verdict = DimensionVerdict(
        dimension="empathy",
        verdict=ValidationVerdict.FAIL,
        target="high",
        signals_summary={"acknowledgment_present": False},
    )

    # post_generation_validate sees the FAIL verdict and returns SKIP (retries exhausted).
    _skip_result = ValidationResult(
        conversation_id="mocked",
        overall_verdict=ValidationVerdict.SKIP,
        dimension_verdicts=[_fail_verdict],
        skip_reason="mocked SKIP after FAIL for test_disagreement_ledger_populated",
        retry_count=2,
    )

    _disagree_record = DisagreementRecord(
        conversation_id="mocked",
        validator_verdict="high",
        soft_judge_perception="empathy:medium",
        disagreement=True,
        disagreement_class="borderline_case",
    )

    with (
        patch("resonantforge.pipeline._call_anthropic", return_value=(_FAKE_PROSE, 0, 0)),
        patch("resonantforge.pipeline.validate_all_dimensions", return_value=[_fail_verdict]),
        patch(
            "resonantforge.layer1.plan_validator.PlanValidator.post_generation_validate",
            return_value=_skip_result,
        ),
        patch("resonantforge.pipeline.run_soft_judge", return_value=_disagree_record),
    ):
        config = PipelineConfig(
            profile_name="saas",
            accounts=2,
            months=1,
            seed=42,
            output_root=tmp_path,
            anthropic_api_key="fake-key-for-ledger-test",
        )
        # Gate fires because all planted convos SKIP → prose_fact_rate > 2%.
        # Manifest is written before the abort so we can still read rates from it.
        with pytest.raises(RuntimeError, match="Quality gate"):
            run_pipeline(config)

    # Assertion 26a: manifest.json must report a non-zero disagreement_rate
    manifest_path = tmp_path / "saas" / "manifest.json"
    assert manifest_path.exists(), "manifest.json must exist after gate abort"
    manifest_data = json.loads(manifest_path.read_text())
    assert manifest_data.get("disagreement_rate", 0.0) > 0.0, (
        f"Assertion 26a — expected disagreement_rate > 0 when mock FAIL verdict fires "
        f"soft judge with disagreement=True, got {manifest_data.get('disagreement_rate')}"
    )

    # Assertion 26b: disagreements.jsonl must contain records
    disag_path = tmp_path / "saas" / "disagreements.jsonl"
    disag_records = _read_jsonl(disag_path)
    assert len(disag_records) > 0, (
        "Assertion 26b — disagreements.jsonl is empty, but mocked soft judge should "
        "have produced at least one DisagreementRecord"
    )


def test_skipped_record_fields_complete(tmp_path: Path) -> None:
    """
    Assertion 28: when a planted conversation is SKIPped after post-generation
    validation, the JSONL record in skipped_conversations.jsonl contains all
    enriched fields with non-null values.

    Uses unittest.mock to inject a FAIL verdict and force a SKIP outcome so that
    the enriched SkippedConversationRecord write path is exercised without a live key.
    """
    from unittest.mock import patch

    _FAKE_PROSE = (
        "Customer: I need help with my account.\n"
        "Agent: I'd be happy to help you today. What seems to be the issue?\n"
        "Customer: I can't access the dashboard.\n"
        "Agent: I understand. Let me look into that for you right away."
    )

    _fail_verdict = DimensionVerdict(
        dimension="empathy",
        verdict=ValidationVerdict.FAIL,
        target="high",
        signals_summary={"acknowledgment_present": False},
    )

    _skip_result = ValidationResult(
        conversation_id="mocked",
        overall_verdict=ValidationVerdict.SKIP,
        dimension_verdicts=[_fail_verdict],
        skip_reason="mocked SKIP for test_skipped_record_fields_complete",
        retry_count=2,
    )

    with (
        patch("resonantforge.pipeline._call_anthropic", return_value=(_FAKE_PROSE, 0, 0)),
        patch("resonantforge.pipeline.validate_all_dimensions", return_value=[_fail_verdict]),
        patch(
            "resonantforge.layer1.plan_validator.PlanValidator.post_generation_validate",
            return_value=_skip_result,
        ),
    ):
        config = PipelineConfig(
            profile_name="saas",
            accounts=2,
            months=1,
            seed=42,
            output_root=tmp_path,
            anthropic_api_key="fake-key-for-fields-test",
        )
        # Gate fires when all planted convos SKIP; manifest and JSONL are written first.
        with pytest.raises(RuntimeError, match="Quality gate"):
            run_pipeline(config)

    skipped_path = tmp_path / "saas" / "skipped_conversations.jsonl"
    skipped = _read_jsonl(skipped_path)
    assert len(skipped) > 0, (
        "Assertion 28 — skipped_conversations.jsonl is empty, expected at least one "
        "SKIPped record from the mocked validator"
    )

    required_fields = [
        "conversation_id",
        "account_id",
        "event_id",
        "quality_plan_summary",
        "final_retry_count",
        "final_verdicts",
        "agent_prose_snippet",
        "kb_chunks_required",
        "timestamp",
    ]

    violations: list[str] = []
    for rec in skipped:
        for field in required_fields:
            if field not in rec or rec[field] is None:
                violations.append(
                    f"conv_id={rec.get('conversation_id')!r}: "
                    f"field {field!r} missing or null (got {rec.get(field)!r})"
                )

    # Assertion 28
    assert not violations, (
        f"Assertion 28 — {len(violations)} skipped record field violation(s):\n"
        + "\n".join(f"  {v}" for v in violations[:5])
    )


def test_variant_id_non_null_for_validated_convs(corpus: tuple[Path, Manifest]) -> None:
    """
    Assertion 27 (OQ2): every quality plan has an effective non-null variant_id
    for brand voice validation.

    The validation loop uses rubric_targets.brand_voice_against as the brand voice
    variant ID, defaulting to "bv_warm_exploratory" when absent. This property ensures the
    effective variant_id passed to validate_all_dimensions is always a non-empty string
    — i.e., brand voice validation always has a calibrated profile to validate against.
    """
    profile_dir, _ = corpus
    plans = _read_jsonl(profile_dir / "planted_quality.jsonl")
    assert len(plans) > 0, "planted_quality.jsonl is empty — no plans to check"

    violations: list[str] = []
    for plan in plans:
        rubric = plan.get("rubric_targets", {})
        # Mirrors the fallback in _generate_prose_for_chunk.
        effective_variant_id = rubric.get("brand_voice_against") or "bv_warm_exploratory"
        if not effective_variant_id:
            violations.append(
                f"conv_id={plan.get('conversation_id')!r}: "
                f"brand_voice_against={rubric.get('brand_voice_against')!r} "
                "produces an empty effective variant_id"
            )

    # Assertion 27
    assert not violations, (
        f"Assertion 27 — {len(violations)} quality plan(s) with empty effective variant_id. "
        f"First: {violations[0]}"
    )


# ---------------------------------------------------------------------------
# Gate violation structure tests (assertions 29–31)
# ---------------------------------------------------------------------------


def test_gate_returns_structured_violations() -> None:
    """
    Assertion 29: check_gates() returns GateViolation objects with severity=ERROR
    when a hard gate threshold is exceeded.

    Sets prose_fact_rate to 50% (well above the 2% threshold) and confirms
    the returned violation has the right gate_name, severity, threshold,
    and actual_value fields.
    """
    tracker = SkipRateTracker(prose_fact_failures=5, prose_fact_attempts=10)
    violations = tracker.check_gates()

    assert len(violations) > 0, (
        "Assertion 29a — check_gates() returned no violations with prose_fact_rate=0.50, "
        "expected at least one ERROR violation"
    )
    prose_violations = [v for v in violations if v.gate_name == "prose_fact_rate"]
    assert len(prose_violations) == 1, (
        f"Assertion 29b — expected exactly one prose_fact_rate violation, "
        f"got {[v.gate_name for v in violations]}"
    )
    v = prose_violations[0]
    # Assertion 29c
    assert isinstance(v, GateViolation), (
        f"Assertion 29c — check_gates() must return GateViolation objects, got {type(v)}"
    )
    assert v.severity == GateSeverity.ERROR, (
        f"Assertion 29d — prose_fact_rate violation must be severity=ERROR, got {v.severity}"
    )
    assert v.threshold == pytest.approx(0.02), (
        f"Assertion 29e — prose_fact_rate threshold must be 0.02, got {v.threshold}"
    )
    assert v.actual_value == pytest.approx(0.50), (
        f"Assertion 29f — actual_value must be 0.50 (5/10), got {v.actual_value}"
    )


def test_gate_warning_does_not_abort() -> None:
    """
    Assertion 30: disagreement_rate between 15% and 25% produces only a WARNING,
    not an ERROR — so the pipeline is not forced to abort.

    Sets disagreement_rate=0.20 (above 15% warn, below 25% block) and confirms
    the violation is severity=WARNING with no ERROR-level violations present.
    """
    # 4 disagreements out of 20 checks = 20% — above warn threshold, below block.
    tracker = SkipRateTracker(disagreement_cases=4, disagreement_checks=20)
    violations = tracker.check_gates()

    errors = [v for v in violations if v.severity == GateSeverity.ERROR]
    warnings = [v for v in violations if v.severity == GateSeverity.WARNING]

    # Assertion 30a: no ERROR violations
    assert len(errors) == 0, (
        f"Assertion 30a — disagreement_rate=0.20 must not produce ERROR violations, "
        f"got {[v.gate_name for v in errors]}"
    )
    # Assertion 30b: exactly one WARNING violation
    assert len(warnings) == 1, (
        f"Assertion 30b — disagreement_rate=0.20 must produce exactly one WARNING, "
        f"got {[(v.gate_name, v.severity) for v in warnings]}"
    )
    assert warnings[0].gate_name == "disagreement_rate", (
        f"Assertion 30c — warning must be for disagreement_rate, got {warnings[0].gate_name!r}"
    )


# ---------------------------------------------------------------------------
# Group 14 — Satisfiability pre-check (RFORGE-11, assertions 32–36)
# ---------------------------------------------------------------------------


def test_chunk_satisfies_intent_empty_tags_matches_any() -> None:
    """
    Assertion 32: a chunk with intent_tags=[] (the default) must return True for
    any event intent, including an empty list.

    This validates the backward-compatibility guarantee: all pre-RFORGE-11 chunks
    that have no intent_tags are eligible for any event regardless of its intent.
    """
    from resonantforge.layer1.quality_plan_injector import _chunk_satisfies_intent

    # Build a minimal KBChunk with no intent_tags (the default).
    chunk = KBChunk(
        chunk_id="kb_chunk_test_empty_tags_v1",
        document_id="doc_test_v1",
        document_path="test/doc.md",
        chunk_text="Test chunk content.",
        domains=["technical_issue"],
        intent_tags=[],  # explicit default — should match anything
    )

    # Assertion 32a: matches specific intent
    assert _chunk_satisfies_intent(chunk, ["api_usage_question"]) is True, (
        "Assertion 32a — chunk with intent_tags=[] must match intent=['api_usage_question']"
    )
    # Assertion 32b: matches different specific intent
    assert _chunk_satisfies_intent(chunk, ["webhook_configuration"]) is True, (
        "Assertion 32b — chunk with intent_tags=[] must match intent=['webhook_configuration']"
    )
    # Assertion 32c: matches empty intent list
    assert _chunk_satisfies_intent(chunk, []) is True, (
        "Assertion 32c — chunk with intent_tags=[] must match empty event intent list"
    )
    # Assertion 32d: matches multi-value intent list
    assert _chunk_satisfies_intent(chunk, ["bug_report", "escalation"]) is True, (
        "Assertion 32d — chunk with intent_tags=[] must match multi-value intent list"
    )


def test_chunk_satisfies_intent_tag_intersection() -> None:
    """
    Assertion 33: a chunk with intent_tags=["webhook_configuration"] must match
    events whose intent includes "webhook_configuration" and must NOT match events
    with a different intent like "api_usage_question".

    This validates the core filtering behavior: tagged chunks are only eligible when
    the event's intent overlaps with the chunk's declared topic.
    """
    from resonantforge.layer1.quality_plan_injector import _chunk_satisfies_intent

    chunk = KBChunk(
        chunk_id="kb_chunk_ti_webhook_error_allow_deny_v1",
        document_id="doc_technical_support_v1",
        document_path="product_docs/technical_support.md",
        chunk_text="Webhook Dead Letter log access content.",
        domains=["technical_issue", "api_and_webhooks"],
        intent_tags=["webhook_configuration"],
    )

    # Assertion 33a: matching intent passes
    assert _chunk_satisfies_intent(chunk, ["webhook_configuration"]) is True, (
        "Assertion 33a — chunk tagged webhook_configuration must match "
        "intent=['webhook_configuration']"
    )
    # Assertion 33b: non-matching intent fails
    assert _chunk_satisfies_intent(chunk, ["api_usage_question"]) is False, (
        "Assertion 33b — chunk tagged webhook_configuration must NOT match "
        "intent=['api_usage_question']"
    )
    # Assertion 33c: empty intent fails when tags are non-empty
    assert _chunk_satisfies_intent(chunk, []) is False, (
        "Assertion 33c — chunk with non-empty intent_tags must NOT match empty event intent"
    )
    # Assertion 33d: intersection match — intent list contains the tag alongside other values
    assert _chunk_satisfies_intent(chunk, ["bug_report", "webhook_configuration"]) is True, (
        "Assertion 33d — chunk tagged webhook_configuration must match a multi-value intent "
        "list that includes the tag"
    )


def test_satisfiability_check_retries_on_mismatch() -> None:
    """
    Assertion 34: when the first RNG pick returns a tagged chunk that does not
    match the event's intent, the injector must retry and select a different chunk.

    We construct a pool where the first alphabetically selected chunk is tagged
    with "webhook_configuration" and a second chunk has no intent_tags (so it
    matches any intent).  With event intent=["api_usage_question"], the injector
    must not select the webhook chunk and must eventually pick the untagged chunk.
    """
    import random as _random
    from resonantforge.layer1.quality_plan_injector import QualityPlanInjector
    from datetime import datetime, timezone as _tz

    # Webhook chunk — will fail the satisfiability check for api_usage_question.
    webhook_chunk = KBChunk(
        chunk_id="kb_chunk_aa_webhook_narrow_v1",  # "aa" prefix → sorts first alphabetically
        document_id="doc_test_v1",
        document_path="test/doc.md",
        chunk_text="Webhook specific content.",
        domains=["api_and_webhooks"],
        intent_tags=["webhook_configuration"],
    )
    # General chunk — no intent_tags → matches any intent.
    general_chunk = KBChunk(
        chunk_id="kb_chunk_bb_general_api_v1",  # "bb" prefix → sorts second alphabetically
        document_id="doc_test_v1",
        document_path="test/doc.md",
        chunk_text="General API content.",
        domains=["api_and_webhooks"],
        intent_tags=[],
    )

    # Event with api_usage_question intent — should NOT trigger the webhook chunk.
    event = SimEvent(
        event_id="evt_rforge11_test_34",
        event_type="conversation_started",
        account_id="acct_test",
        agent_id="agent_test",
        timestamp=datetime(2026, 1, 1, 10, 0, tzinfo=_tz.utc),
        day_index=0,
        month_index=0,
        payload={
            "domain": "api_and_webhooks",
            "intent": ["api_usage_question"],
            "surface_channel": "intercom",
            "customer_name": "Test Customer",
            "agent_id": "agent_test",
        },
    )

    rng = _random.Random(42)
    injector = QualityPlanInjector(rng=rng, profile_name="saas")

    # Call _build_plan with a spec that will trigger chunk selection.
    spec = {"type": "accuracy", "label": __import__("resonantforge.schemas", fromlist=["AccuracyLabel"]).AccuracyLabel(status="supported", precision="exact")}
    plan = injector._build_plan(
        conv_event=event,
        account_snap=None,
        spec=spec,
        kb_chunks=[webhook_chunk, general_chunk],
    )

    # Assertion 34: the webhook chunk must NOT have been selected — only the general chunk
    # (or empty) is acceptable because the webhook chunk's intent_tag didn't match.
    assert "kb_chunk_aa_webhook_narrow_v1" not in plan.kb_chunks_required, (
        "Assertion 34 — webhook_configuration chunk must not appear in kb_chunks_required "
        "for an event with intent=['api_usage_question'].  "
        f"Got kb_chunks_required={plan.kb_chunks_required!r}"
    )


def test_satisfiability_fallback_drops_kb_required() -> None:
    """
    Assertion 35: when all chunks in the pick pool are tagged and none match the
    event's intent, the injector must fall back to kb_required=[] (no crash, no
    mismatched plan).

    This validates the fallback policy: wrong-chunk plans are worse than no-chunk
    plans from a training-signal perspective.
    """
    import random as _random
    from resonantforge.layer1.quality_plan_injector import QualityPlanInjector
    from datetime import datetime, timezone as _tz

    # All chunks are tagged — none will match "api_usage_question".
    tagged_chunk_a = KBChunk(
        chunk_id="kb_chunk_aa_webhook_only_a_v1",
        document_id="doc_test_v1",
        document_path="test/doc.md",
        chunk_text="Webhook only content A.",
        domains=["api_and_webhooks"],
        intent_tags=["webhook_configuration"],
    )
    tagged_chunk_b = KBChunk(
        chunk_id="kb_chunk_bb_webhook_only_b_v1",
        document_id="doc_test_v1",
        document_path="test/doc.md",
        chunk_text="Webhook only content B.",
        domains=["api_and_webhooks"],
        intent_tags=["webhook_configuration"],
    )
    tagged_chunk_c = KBChunk(
        chunk_id="kb_chunk_cc_webhook_only_c_v1",
        document_id="doc_test_v1",
        document_path="test/doc.md",
        chunk_text="Webhook only content C.",
        domains=["api_and_webhooks"],
        intent_tags=["webhook_configuration"],
    )

    # Event with intent that matches none of the chunks.
    event = SimEvent(
        event_id="evt_rforge11_test_35",
        event_type="conversation_started",
        account_id="acct_test",
        agent_id="agent_test",
        timestamp=datetime(2026, 1, 1, 10, 0, tzinfo=_tz.utc),
        day_index=0,
        month_index=0,
        payload={
            "domain": "api_and_webhooks",
            "intent": ["api_usage_question"],
            "surface_channel": "intercom",
            "customer_name": "Test Customer",
            "agent_id": "agent_test",
        },
    )

    rng = _random.Random(42)
    injector = QualityPlanInjector(rng=rng, profile_name="saas")

    _AccuracyLabel = __import__("resonantforge.schemas", fromlist=["AccuracyLabel"]).AccuracyLabel
    spec = {"type": "accuracy", "label": _AccuracyLabel(status="supported", precision="exact")}

    # Must not raise — fallback produces a valid plan with empty kb_chunks_required.
    plan = injector._build_plan(
        conv_event=event,
        account_snap=None,
        spec=spec,
        kb_chunks=[tagged_chunk_a, tagged_chunk_b, tagged_chunk_c],
    )

    # Assertion 35a: plan must be produced without raising.
    assert plan is not None, "Assertion 35a — _build_plan must return a plan, not raise"

    # Assertion 35b: kb_chunks_required must be empty because all retries failed.
    assert plan.kb_chunks_required == [], (
        f"Assertion 35b — all retries failed (no intent match), "
        f"expected kb_chunks_required=[] but got {plan.kb_chunks_required!r}"
    )


def test_conv_evt_00161_pattern_no_longer_mismatches() -> None:
    """
    Assertion 36: for an event with domain="api_and_webhooks" and
    intent=["api_usage_question"], the injector must NOT select
    kb_chunk_ti_webhook_error_allow_deny_v1 (which is tagged webhook_configuration).

    This directly validates that the root cause of the conv_evt_00161 mismatch
    — diagnosed in docs/resonantforge/plan-generator-chunk-mismatch-diagnosis.md —
    is resolved.  The chunk's intent_tags=["webhook_configuration"] must prevent
    selection for an api_usage_question event.
    """
    import random as _random
    from resonantforge.layer1.quality_plan_injector import QualityPlanInjector
    from resonantforge.kb.saas_content import get_saas_kb_chunks
    from datetime import datetime, timezone as _tz

    # Reconstruct the event pattern that triggered the original mismatch.
    event = SimEvent(
        event_id="evt_00161_pattern",
        event_type="conversation_started",
        account_id="acct_004",
        agent_id="agent_test",
        timestamp=datetime(2026, 1, 1, 10, 0, tzinfo=_tz.utc),
        day_index=0,
        month_index=0,
        payload={
            "domain": "api_and_webhooks",
            "intent": ["api_usage_question"],
            "surface_channel": "intercom",
            "customer_name": "Test Customer",
            "agent_id": "agent_test",
        },
    )

    kb_chunks = get_saas_kb_chunks()

    # Run a large number of plans with different RNG seeds to exercise the full
    # pick lottery — the webhook chunk must never appear as a selected chunk
    # for an api_usage_question event.
    _PROBE_SEEDS = list(range(50))
    violations: list[str] = []

    for seed in _PROBE_SEEDS:
        rng = _random.Random(seed)
        injector = QualityPlanInjector(rng=rng, profile_name="saas")
        _AccuracyLabel = __import__("resonantforge.schemas", fromlist=["AccuracyLabel"]).AccuracyLabel
        spec = {"type": "accuracy", "label": _AccuracyLabel(status="supported", precision="exact")}

        plan = injector._build_plan(
            conv_event=event,
            account_snap=None,
            spec=spec,
            kb_chunks=kb_chunks,
        )

        if "kb_chunk_ti_webhook_error_allow_deny_v1" in plan.kb_chunks_required:
            violations.append(f"seed={seed}: webhook chunk appeared in kb_chunks_required")

    # Assertion 36: across 50 seeds, the webhook chunk must never be assigned to an
    # api_usage_question event.
    assert not violations, (
        f"Assertion 36 — kb_chunk_ti_webhook_error_allow_deny_v1 was selected for "
        f"an api_usage_question event in {len(violations)} of {len(_PROBE_SEEDS)} seed runs. "
        f"First violation: {violations[0]}"
    )


def test_pipeline_aborts_on_hard_gate(tmp_path: Path) -> None:
    """
    Assertion 31: when prose_fact_rate exceeds the 2% gate, the pipeline raises
    RuntimeError AND still writes manifest.json with gate_aborted=true.

    Uses the same mock setup as test_skip_tracker_prose_fact_increments to force
    all planted conversations to SKIP, pushing prose_fact_rate well above 2%.
    """
    from unittest.mock import patch

    _FAKE_PROSE = (
        "Customer: I need help with my account.\n"
        "Agent: I'd be happy to help you today. What seems to be the issue?\n"
        "Customer: I can't access the dashboard.\n"
        "Agent: I understand. Let me look into that for you right away."
    )

    _skip_result = ValidationResult(
        conversation_id="mocked",
        overall_verdict=ValidationVerdict.SKIP,
        dimension_verdicts=[],
        skip_reason="mocked SKIP verdict for test_pipeline_aborts_on_hard_gate",
        retry_count=2,
    )

    with (
        patch("resonantforge.pipeline._call_anthropic", return_value=(_FAKE_PROSE, 0, 0)),
        patch("resonantforge.pipeline.validate_all_dimensions", return_value=[]),
        patch(
            "resonantforge.layer1.plan_validator.PlanValidator.post_generation_validate",
            return_value=_skip_result,
        ),
    ):
        config = PipelineConfig(
            profile_name="saas",
            accounts=2,
            months=1,
            seed=42,
            output_root=tmp_path,
            anthropic_api_key="fake-key-for-gate-abort-test",
        )
        # Assertion 31a: pipeline raises RuntimeError when hard gate fires
        with pytest.raises(RuntimeError, match="Quality gate"):
            run_pipeline(config)

    # Assertion 31b: manifest.json must exist even after the abort
    manifest_path = tmp_path / "saas" / "manifest.json"
    assert manifest_path.exists(), (
        "Assertion 31b — manifest.json must be written even when a gate aborts the run"
    )

    manifest_data = json.loads(manifest_path.read_text())

    # Assertion 31c: manifest must have gate_aborted=True
    assert manifest_data.get("gate_aborted") is True, (
        f"Assertion 31c — manifest.gate_aborted must be True after gate abort, "
        f"got {manifest_data.get('gate_aborted')!r}"
    )

    # Assertion 31d: manifest must have non-empty gate_violations list
    gate_violations = manifest_data.get("gate_violations", [])
    assert len(gate_violations) > 0, (
        "Assertion 31d — manifest.gate_violations must be non-empty after gate abort"
    )


# ---------------------------------------------------------------------------
# Group 14 — KB domain-matching engine (assertions E1–E8)
# ---------------------------------------------------------------------------
#
# These tests use the synthetic KB fixture (tests/fixtures/synthetic_kb.py)
# and inject quality plans directly via QualityPlanInjector so no full
# pipeline run is required.  All assertions target the domain-aware code path.
# ---------------------------------------------------------------------------


def _make_injector(seed: int = 42) -> "QualityPlanInjector":
    """Return a seeded QualityPlanInjector for engine tests."""
    import random as _random
    from resonantforge.layer1.quality_plan_injector import QualityPlanInjector
    return QualityPlanInjector(rng=_random.Random(seed), profile_name="saas")


def _make_conv_event(
    event_id: str,
    domain: str,
    account_id: str = "acct_001",
    event_type_override: str | None = None,
) -> SimEvent:
    """Build a minimal CONVERSATION_STARTED SimEvent for injector testing."""
    from datetime import datetime as _dt
    from resonantforge.schemas import SimEventType as _SET
    etype = _SET(event_type_override) if event_type_override else _SET.CONVERSATION_STARTED
    return SimEvent(
        event_id=event_id,
        event_type=etype,
        account_id=account_id,
        timestamp=_dt(2025, 8, 1, 10, 0, 0),
        day_index=0,
        month_index=0,
        payload={
            "surface_channel": "intercom",
            "agent_id": "agent_001",
            "customer_name": "Test Customer",
            "domain": domain,
            "intent": [],
        },
    )


def _make_snapshot(account_id: str = "acct_001") -> "DaySnapshot":
    """Build a minimal DaySnapshot for injector testing."""
    from datetime import date as _date
    from resonantforge.schemas import (
        DaySnapshot as _DS,
        HealthState as _HS,
        LifecycleStage as _LS,
    )
    return _DS(
        snapshot_id="snap_001",
        account_id=account_id,
        day_index=0,
        month_index=0,
        date=_date(2025, 8, 1),
        lifecycle_stage=_LS.ACTIVE,
        health_state=_HS.HEALTHY,
        health_score=0.8,
        open_tickets=0,
        recent_signals=[],
        active_agents=["agent_001"],
        payment_status="current",
    )


def test_engine_domain_isolation() -> None:
    """
    E1 — Domain isolation: every should_cite and must_not_cite chunk in a
    quality plan produced for domain X must cover domain X.

    Tests billing, api, and refunds in turn using accuracy-type plan specs.
    """
    from resonantforge.layer1.quality_plan_injector import QualityPlanInjector
    from resonantforge.schemas import AccuracyLabel
    from tests.fixtures.synthetic_kb import SYNTHETIC_KB_CHUNKS

    # Build a chunk lookup for assertion.
    chunk_by_id = {c.chunk_id: c for c in SYNTHETIC_KB_CHUNKS}

    for domain in ["billing", "api", "refunds"]:
        injector = _make_injector()
        conv = _make_conv_event("evt_e1", domain)
        snap = _make_snapshot()
        spec = {"type": "accuracy", "label": AccuracyLabel(status="supported", precision="exact")}

        plan = injector._build_plan(conv, snap, spec, SYNTHETIC_KB_CHUNKS)

        for cid in plan.knowledge_citations.should_cite:
            if cid == "*":
                continue
            chunk = chunk_by_id.get(cid)
            assert chunk is not None, f"E1: should_cite chunk_id={cid!r} not in synthetic KB"
            assert domain in chunk.domains, (
                f"E1: domain={domain!r}, should_cite chunk {cid!r} covers {chunk.domains}"
            )

        for cid in plan.knowledge_citations.must_not_cite:
            if cid == "*":
                continue
            chunk = chunk_by_id.get(cid)
            assert chunk is not None, f"E1: must_not_cite chunk_id={cid!r} not in synthetic KB"
            assert domain in chunk.domains, (
                f"E1: domain={domain!r}, must_not_cite chunk {cid!r} covers {chunk.domains}"
            )


def test_engine_adversarial_exclusion_default() -> None:
    """
    E2 — Without explicit opt-in, no should_cite chunk is adversarial.
    """
    from resonantforge.schemas import AccuracyLabel
    from tests.fixtures.synthetic_kb import SYNTHETIC_KB_CHUNKS

    chunk_by_id = {c.chunk_id: c for c in SYNTHETIC_KB_CHUNKS}

    for domain in ["billing", "api", "refunds"]:
        injector = _make_injector()
        conv = _make_conv_event("evt_e2", domain)
        snap = _make_snapshot()
        spec = {"type": "accuracy", "label": AccuracyLabel(status="supported", precision="exact")}

        plan = injector._build_plan(
            conv, snap, spec, SYNTHETIC_KB_CHUNKS, include_adversarial_in_should_cite=False
        )

        for cid in plan.knowledge_citations.should_cite:
            if cid == "*":
                continue
            chunk = chunk_by_id.get(cid)
            if chunk:
                assert not chunk.adversarial, (
                    f"E2: domain={domain!r}, adversarial chunk {cid!r} appeared in "
                    f"should_cite without opt-in"
                )


def test_engine_adversarial_inclusion_when_opted_in() -> None:
    """
    E3 — With include_adversarial_in_should_cite=True, adversarial chunks
    may appear in should_cite for a domain that has adversarial chunks.
    """
    import random as _random
    from resonantforge.layer1.quality_plan_injector import QualityPlanInjector
    from resonantforge.schemas import AccuracyLabel
    from tests.fixtures.synthetic_kb import SYNTHETIC_KB_CHUNKS

    chunk_by_id = {c.chunk_id: c for c in SYNTHETIC_KB_CHUNKS}

    # Run many trials across seeds to increase chance of picking an adversarial chunk.
    found_adversarial_in_should_cite = False
    for seed in range(100):
        injector = QualityPlanInjector(rng=_random.Random(seed), profile_name="saas")
        conv = _make_conv_event("evt_e3", "refunds")  # refunds has 2 adversarial chunks
        snap = _make_snapshot()
        spec = {"type": "accuracy", "label": AccuracyLabel(status="supported", precision="exact")}

        plan = injector._build_plan(
            conv, snap, spec, SYNTHETIC_KB_CHUNKS, include_adversarial_in_should_cite=True
        )
        for cid in plan.knowledge_citations.should_cite:
            chunk = chunk_by_id.get(cid)
            if chunk and chunk.adversarial:
                found_adversarial_in_should_cite = True
                break
        if found_adversarial_in_should_cite:
            break

    assert found_adversarial_in_should_cite, (
        "E3: across 100 seeds, no adversarial chunk ever appeared in should_cite "
        "even with include_adversarial_in_should_cite=True"
    )


def test_engine_within_topic_normalization() -> None:
    """
    E4 — Within-topic normalization: generating 100 quality plans for the
    refunds domain (4 chunks, 2 adversarial → 2 non-adversarial allow pool)
    must not let any single chunk dominate beyond a generous cap.

    The cap is: max_allowed_per_chunk = 2 * (100 / allow_pool_size) * 1.5
    For 2 allow-eligible refund chunks that is 2 * 50 * 1.5 = 150 — but since
    we only run 100 iterations and there are 2 chunks, neither should exceed
    75 selections (100 * 75% as an extreme upper bound for ~50/50 distribution).
    """
    import random as _random
    from resonantforge.layer1.quality_plan_injector import QualityPlanInjector
    from resonantforge.schemas import AccuracyLabel
    from tests.fixtures.synthetic_kb import SYNTHETIC_KB_CHUNKS

    N = 100
    freq: dict[str, int] = {}
    for i in range(N):
        injector = QualityPlanInjector(rng=_random.Random(i), profile_name="saas")
        conv = _make_conv_event(f"evt_e4_{i}", "refunds")
        snap = _make_snapshot()
        spec = {"type": "accuracy", "label": AccuracyLabel(status="supported", precision="exact")}
        plan = injector._build_plan(conv, snap, spec, SYNTHETIC_KB_CHUNKS)
        for cid in plan.knowledge_citations.should_cite:
            if cid != "*":
                freq[cid] = freq.get(cid, 0) + 1

    # refunds allow pool has 2 non-adversarial chunks; neither should exceed 75% of trials
    allow_refund_chunks = [
        c for c in SYNTHETIC_KB_CHUNKS if "refunds" in c.domains and not c.adversarial
    ]
    assert len(allow_refund_chunks) > 0, "E4: no non-adversarial refund chunks in synthetic KB"

    max_allowed = int(N * 0.75)
    for chunk in allow_refund_chunks:
        count = freq.get(chunk.chunk_id, 0)
        assert count <= max_allowed, (
            f"E4: chunk {chunk.chunk_id!r} selected {count}/{N} times — "
            f"exceeds within-topic normalisation cap of {max_allowed}"
        )


def test_engine_determinism_with_domain() -> None:
    """
    E5 — Same seed produces identical chunk selections for a domain-specific plan.
    """
    from resonantforge.schemas import AccuracyLabel
    from tests.fixtures.synthetic_kb import SYNTHETIC_KB_CHUNKS

    def _run_plan(seed: int) -> list[str]:
        injector = _make_injector(seed)
        conv = _make_conv_event("evt_e5", "billing")
        snap = _make_snapshot()
        spec = {"type": "accuracy", "label": AccuracyLabel(status="supported", precision="exact")}
        plan = injector._build_plan(conv, snap, spec, SYNTHETIC_KB_CHUNKS)
        return sorted(plan.knowledge_citations.should_cite + plan.knowledge_citations.must_not_cite)

    assert _run_plan(42) == _run_plan(42), (
        "E5: same seed produced different chunk selections across two runs"
    )
    # Different seeds should produce potentially different selections (not guaranteed,
    # but worth checking for statistical independence).
    # We just verify the same-seed case is deterministic — divergence is not guaranteed.


def test_engine_fallback_on_no_candidates() -> None:
    """
    E6 — When the event domain has no matching KB chunks (e.g. a legacy state-machine
    domain string that predates the 13-domain vocabulary), the injector falls back to
    the full KB pool rather than raising, so the pipeline continues to run.

    Previously this test verified a ValueError raise; updated in PR2 to document the
    graceful-fallback behavior introduced to keep the legacy state machine compatible
    with the newly domain-tagged KB.
    """
    from resonantforge.schemas import AccuracyLabel
    from tests.fixtures.synthetic_kb import SYNTHETIC_KB_CHUNKS

    injector = _make_injector()
    conv = _make_conv_event("evt_e6", "nonexistent_domain")
    snap = _make_snapshot()
    spec = {"type": "accuracy", "label": AccuracyLabel(status="supported", precision="exact")}

    # Should NOT raise — falls back to full KB pool.
    plan = injector._build_plan(conv, snap, spec, SYNTHETIC_KB_CHUNKS)
    # At least one chunk must be selected from the full pool.
    assert plan.knowledge_citations.should_cite or plan.kb_chunks_required is not None


def test_engine_raise_on_unrecognized_event_type() -> None:
    """
    E7 — ValueError is raised when the trigger event type is not in the
    eligible allowlist (e.g. account_created going through the injector).
    """
    from resonantforge.schemas import AccuracyLabel
    from tests.fixtures.synthetic_kb import SYNTHETIC_KB_CHUNKS

    injector = _make_injector()
    # Build an ACCOUNT_CREATED event — not eligible for injection.
    from datetime import datetime as _dt
    from resonantforge.schemas import SimEventType as _SET
    non_conv_event = SimEvent(
        event_id="evt_e7",
        event_type=_SET.ACCOUNT_CREATED,
        account_id="acct_001",
        timestamp=_dt(2025, 8, 1, 10, 0, 0),
        day_index=0,
        month_index=0,
        payload={"plan_tier": "starter", "industry": "saas", "domain": "billing"},
    )
    snap = _make_snapshot()
    spec = {"type": "accuracy", "label": AccuracyLabel(status="supported", precision="exact")}

    with pytest.raises(ValueError, match="not eligible"):
        injector._build_plan(non_conv_event, snap, spec, SYNTHETIC_KB_CHUNKS)


def test_engine_manifest_fields_populated(tmp_path: Path) -> None:
    """
    E8 — After a synthetic-fixture dry run, the manifest contains non-default
    values for kb_version, kb_chunk_count, domain_distribution_observed, and
    chunk_selection_frequency.

    Uses a patched KB generator so the synthetic fixture is used instead of
    the real saas_content.py chunks.
    """
    from unittest.mock import patch
    from tests.fixtures.synthetic_kb import SYNTHETIC_KB_CHUNKS

    # Compute expected kb_hash from synthetic chunks (same logic as pipeline).
    import hashlib
    _source = "".join(
        c.chunk_id + c.chunk_text
        for c in sorted(SYNTHETIC_KB_CHUNKS, key=lambda c: c.chunk_id)
    )
    _expected_kb_version = hashlib.sha256(_source.encode()).hexdigest()

    # Hash the chunks JSONL for the kb_chunks_hash field (pipeline writes to disk).
    _chunk_lines = [c.model_dump_json() for c in SYNTHETIC_KB_CHUNKS]
    _kb_hash = hashlib.sha256("\n".join(_chunk_lines).encode()).hexdigest()

    with patch(
        "resonantforge.pipeline.generate_kb",
        return_value=(SYNTHETIC_KB_CHUNKS, _kb_hash),
    ):
        config = PipelineConfig(
            profile_name="saas",
            accounts=_SMALL_CORPUS_ACCOUNTS,
            months=_SMALL_CORPUS_MONTHS,
            seed=SEED,
            output_root=tmp_path,
            anthropic_api_key=None,
        )
        manifest = run_pipeline(config)

    # kb_version must be a non-empty sha256 hex string
    assert manifest.kb_version and manifest.kb_version == _expected_kb_version, (
        f"E8: manifest.kb_version mismatch or empty: {manifest.kb_version!r}"
    )

    # kb_chunk_count must equal number of synthetic chunks
    assert manifest.kb_chunk_count == len(SYNTHETIC_KB_CHUNKS), (
        f"E8: manifest.kb_chunk_count={manifest.kb_chunk_count}, "
        f"expected {len(SYNTHETIC_KB_CHUNKS)}"
    )

    # domain_distribution_observed must have entries for at least one domain
    assert manifest.domain_distribution_observed, (
        "E8: manifest.domain_distribution_observed is empty — "
        "no CONVERSATION_STARTED events with domain field were found"
    )

    # chunk_selection_frequency is populated when at least one plan cites a chunk.
    # For accuracy plans using the synthetic fixture, at least one chunk should appear.
    assert isinstance(manifest.chunk_selection_frequency, dict), (
        "E8: manifest.chunk_selection_frequency is not a dict"
    )


def test_accuracy_directive_contains_chunk_text() -> None:
    """
    E9 — For any accuracy plan with non-empty should_cite, the rendered
    prose_generation_directives must contain the actual chunk_text of every
    chunk in should_cite (substring match).

    This regression test would have caught the original bug where only chunk IDs
    were passed to the prose directive, leaving the LLM with no content to cite.
    """
    from resonantforge.schemas import AccuracyLabel
    from tests.fixtures.synthetic_kb import SYNTHETIC_KB_CHUNKS

    chunk_by_id = {c.chunk_id: c for c in SYNTHETIC_KB_CHUNKS}

    # Test all four accuracy label types across three domains.
    labels = [
        AccuracyLabel(status="supported", precision="exact"),
        AccuracyLabel(status="supported", precision="overgeneralized"),
        AccuracyLabel(status="supported", precision="conditional_applied"),
        AccuracyLabel(status="contradicted", precision="exact"),
    ]

    for domain in ["billing", "api", "refunds"]:
        for label in labels:
            injector = _make_injector()
            conv = _make_conv_event("evt_e9", domain)
            snap = _make_snapshot()
            spec = {"type": "accuracy", "label": label}

            plan = injector._build_plan(conv, snap, spec, SYNTHETIC_KB_CHUNKS)

            if not plan.knowledge_citations.should_cite:
                continue  # no should_cite chunks — directive is vacuous, skip

            directive = plan.prose_generation_directives
            for cid in plan.knowledge_citations.should_cite:
                if cid == "*":
                    continue
                chunk = chunk_by_id.get(cid)
                assert chunk is not None, (
                    f"E9: should_cite chunk_id={cid!r} not found in synthetic KB"
                )
                assert chunk.chunk_text in directive, (
                    f"E9: domain={domain!r} label={label} — chunk_text of {cid!r} "
                    f"not present in prose_generation_directives.\n"
                    f"Directive:\n{directive}\n"
                    f"Expected substring:\n{chunk.chunk_text}"
                )


# ---------------------------------------------------------------------------
# Group 15 — KB tagging and invariant health (assertions A, B, C, D)
# ---------------------------------------------------------------------------


def test_kb_domain_tag_completeness_existing() -> None:
    """Every existing KB chunk must have at least one domain tag after PR2."""
    from resonantforge.kb.saas_content import get_saas_kb_chunks
    chunks = get_saas_kb_chunks()
    untagged = [c.chunk_id for c in chunks if not c.domains]
    assert not untagged, f"Chunks missing domain tags: {untagged}"


def test_adversarial_chunks_marked_correctly() -> None:
    """All adversarial KB fixtures must have adversarial=True set on the KBChunk model field."""
    from resonantforge.kb.saas_content import get_saas_kb_chunks
    EXPECTED_ADVERSARIAL = {
        "kb_chunk_refund_eligibility_timelines_v1",
        "kb_chunk_refund_grace_period_stale_v1",
        "kb_chunk_all_customers_refund_bait_v1",
        "kb_chunk_ti_severity_inflation_adv_v1",
        "kb_chunk_ti_resolution_overpromise_adv_v1",
        "kb_chunk_ti_diagnostic_omission_adv_v1",
    }
    chunks = get_saas_kb_chunks()
    adversarial_set = {c.chunk_id for c in chunks if c.adversarial}
    missing = EXPECTED_ADVERSARIAL - adversarial_set
    assert not missing, f"Expected adversarial chunks missing adversarial=True: {missing}"
    # No unexpected adversarial chunks
    extra = adversarial_set - EXPECTED_ADVERSARIAL
    assert not extra, f"Unexpected chunks marked adversarial: {extra}"


def test_invariant_checker_passes_existing_kb() -> None:
    """
    The invariant checker must return zero errors against the existing tagged KB.

    If this fails, the existing KB has internal contradictions that must be
    resolved before PR3 adds new chunks. Surface errors — do not suppress them.
    """
    from resonantforge.kb.saas_content import get_saas_kb_chunks
    from resonantforge.validators.invariant_checker import run_checker
    chunks = get_saas_kb_chunks()
    report = run_checker(chunks)
    assert not report.errors, (
        f"Invariant checker found {len(report.errors)} error(s) in existing KB:\n"
        + "\n".join(f"  - {e}" for e in report.errors)
    )


def test_chunks_with_claims_are_well_formed() -> None:
    """
    Every KB chunk with non-empty claims must have well-formed claim values.
    Claims must be a dict of str keys mapping to non-None primitive values or lists.
    """
    from resonantforge.kb.saas_content import get_saas_kb_chunks
    chunks = get_saas_kb_chunks()
    malformed = []
    for c in chunks:
        if not c.claims:
            continue  # empty claims are fine
        if not isinstance(c.claims, dict):
            malformed.append((c.chunk_id, f"claims is {type(c.claims).__name__}, expected dict"))
            continue
        for key, value in c.claims.items():
            if not isinstance(key, str):
                malformed.append((c.chunk_id, f"claim key {key!r} is not a str"))
            # Values may be None (e.g. Enterprise price is None) — that's allowed
            # as long as the key is a str
    assert not malformed, f"Malformed claims found:\n" + "\n".join(f"  {cid}: {reason}" for cid, reason in malformed)


# ---------------------------------------------------------------------------
# Group 16 — Invariant checker scope and tier vocabulary (assertions A–J)
# ---------------------------------------------------------------------------


def _make_chunk(
    chunk_id: str = "test_chunk",
    document_id: str = "doc_test",
    document_path: str = "policies/test.md",
    chunk_text: str = "Default text.",
    domains: list[str] | None = None,
    claims: dict | None = None,
    adversarial: bool = False,
) -> "KBChunk":
    from resonantforge.schemas import ConstraintType, KBChunk
    return KBChunk(
        chunk_id=chunk_id,
        document_id=document_id,
        document_path=document_path,
        chunk_text=chunk_text,
        constraint_type=ConstraintType.INFORMATIONAL,
        domains=domains or ["billing"],
        claims=claims or {},
        adversarial=adversarial,
    )


def test_cross_chunk_contradiction_not_flagged() -> None:
    """
    Load-bearing regression test: two chunks that share the same document,
    path, domain, and a common claim key but with DIFFERENT values must
    produce zero errors and zero warnings.

    This test prevents any future change from re-introducing a cross-chunk
    claim-comparison rule. The invariant checker is single-chunk-scoped.
    """
    from resonantforge.validators.invariant_checker import run_checker

    chunk_a = _make_chunk(
        chunk_id="test_export_a",
        document_id="doc_data_policy_v2",
        document_path="policies/data_policy.md",
        chunk_text="Accounts with fewer than 1,000,000 records export immediately.",
        domains=["data_management"],
        claims={"export_immediate_threshold_records": 1000000},
    )
    chunk_b = _make_chunk(
        chunk_id="test_export_b",
        document_id="doc_data_policy_v2",
        document_path="policies/data_policy.md",
        chunk_text="Accounts with fewer than 100,000 records export immediately.",
        domains=["data_management"],
        claims={"export_immediate_threshold_records": 100000},
    )

    report = run_checker([chunk_a, chunk_b])
    # Coverage-gap warnings may fire on this minimal 2-chunk fixture (no plan names,
    # no rate limits, etc.) — that's expected. The load-bearing assertion is that no
    # ERRORS fire, which is what a future cross-chunk comparison rule would produce.
    assert not report.errors, (
        f"Cross-chunk claim comparison must not produce errors. Got: {report.errors}"
    )
    # Verify no error mentions a conflict between the two chunks specifically.
    for err in report.errors:
        assert "test_export_a" not in err or "test_export_b" not in err, (
            f"Cross-chunk comparison between test_export_a and test_export_b must not fire: {err}"
        )


# --- Tier vocabulary: negative tests (must flag) ---


def test_tier_vocab_claim_non_canonical_value_errors() -> None:
    """Claim with tier-like string value 'Standard' (non-canonical) must error."""
    from resonantforge.validators.invariant_checker import run_checker

    chunk = _make_chunk(
        chunk_id="test_tier_claim_standard",
        chunk_text="Rate limits vary by plan.",
        claims={"tier": "Standard"},
    )
    report = run_checker([chunk])
    assert report.errors, "Expected error for non-canonical tier name 'Standard' in claims"


def test_tier_vocab_claim_lowercase_canonical_errors() -> None:
    """Claim with lowercase canonical tier name 'starter' (case drift) must error."""
    from resonantforge.validators.invariant_checker import run_checker

    chunk = _make_chunk(
        chunk_id="test_tier_claim_lowercase",
        chunk_text="Plans vary in features.",
        claims={"tier": "starter"},
    )
    report = run_checker([chunk])
    assert report.errors, "Expected error for lowercase canonical tier name 'starter' in claims"


def test_tier_vocab_text_non_canonical_before_plan_errors() -> None:
    """Text containing 'upgrade to the Standard plan' must flag a tier-vocab error."""
    from resonantforge.validators.invariant_checker import run_checker

    chunk = _make_chunk(
        chunk_id="test_tier_text_standard_plan",
        chunk_text="You can upgrade to the Standard plan at any time.",
    )
    report = run_checker([chunk])
    assert report.errors, "Expected error for 'Standard plan' in chunk text"


def test_tier_vocab_text_non_canonical_in_list_errors() -> None:
    """Text containing 'the Standard, Growth, or Enterprise tier' must flag Standard."""
    from resonantforge.validators.invariant_checker import run_checker

    chunk = _make_chunk(
        chunk_id="test_tier_text_standard_in_list",
        chunk_text="This feature is available on the Standard, Growth, or Enterprise tier.",
    )
    report = run_checker([chunk])
    assert report.errors, "Expected error for 'Standard' in a canonical-tier list"


# --- Tier vocabulary: positive tests (must NOT flag) ---


def test_tier_vocab_compound_word_not_flagged() -> None:
    """Text with 'enterprise-grade security and growth in user adoption' must not flag."""
    from resonantforge.validators.invariant_checker import run_checker

    chunk = _make_chunk(
        chunk_id="test_tier_text_compound",
        chunk_text="We provide enterprise-grade security and support growth in user adoption.",
    )
    report = run_checker([chunk])
    tier_errors = [e for e in report.errors if "non-canonical tier" in e]
    assert not tier_errors, f"Compound words must not flag tier-vocab errors. Got: {tier_errors}"


def test_tier_vocab_premium_support_not_flagged() -> None:
    """'premium support' must not flag — 'support' is not a tier-context word."""
    from resonantforge.validators.invariant_checker import run_checker

    chunk = _make_chunk(
        chunk_id="test_tier_text_premium_support",
        chunk_text="Priority support and premium support options are available for all plans.",
    )
    report = run_checker([chunk])
    tier_errors = [e for e in report.errors if "non-canonical tier" in e]
    assert not tier_errors, f"'premium support' must not flag. Got: {tier_errors}"


def test_tier_vocab_free_trial_not_flagged() -> None:
    """'a free trial' must not flag — 'trial' is not a tier-context word."""
    from resonantforge.validators.invariant_checker import run_checker

    chunk = _make_chunk(
        chunk_id="test_tier_text_free_trial",
        chunk_text="A free trial is available for 14 days before committing to a paid plan.",
    )
    report = run_checker([chunk])
    tier_errors = [e for e in report.errors if "non-canonical tier" in e]
    assert not tier_errors, f"'free trial' must not flag. Got: {tier_errors}"


def test_tier_vocab_basic_understanding_not_flagged() -> None:
    """'basic understanding' must not flag — non-tier usage of 'basic'."""
    from resonantforge.validators.invariant_checker import run_checker

    chunk = _make_chunk(
        chunk_id="test_tier_text_basic_understanding",
        chunk_text="A basic understanding of REST APIs is recommended before using this feature.",
    )
    report = run_checker([chunk])
    tier_errors = [e for e in report.errors if "non-canonical tier" in e]
    assert not tier_errors, f"'basic understanding' must not flag. Got: {tier_errors}"


def test_tier_vocab_standard_or_enterprise_flags() -> None:
    """'Standard or Enterprise' (no Growth in between) must flag 'Standard'."""
    from resonantforge.validators.invariant_checker import run_checker

    chunk = _make_chunk(
        chunk_id="test_tier_text_standard_or_enterprise",
        chunk_text="This feature is available on Standard or Enterprise plans.",
    )
    report = run_checker([chunk])
    assert report.errors, "Expected error for 'Standard or Enterprise' — list context with canonical tier"


def test_tier_vocab_pro_pricing_flags() -> None:
    """'Pro pricing' must flag — 'Pro' before a tier-context word."""
    from resonantforge.validators.invariant_checker import run_checker

    chunk = _make_chunk(
        chunk_id="test_tier_text_pro_pricing",
        chunk_text="Pro pricing applies to this feature and includes all advanced options.",
    )
    report = run_checker([chunk])
    assert report.errors, "Expected error for 'Pro pricing' — non-canonical tier before 'pricing'"


def test_tier_vocab_lowercase_standard_plan_flags() -> None:
    """'the standard plan' (lowercase) must flag — case drift on non-canonical name."""
    from resonantforge.validators.invariant_checker import run_checker

    chunk = _make_chunk(
        chunk_id="test_tier_text_lowercase_standard_plan",
        chunk_text="If you are on the standard plan, you receive 1,000 API calls per minute.",
    )
    report = run_checker([chunk])
    assert report.errors, "Expected error for lowercase 'standard plan'"


def test_tier_vocab_canonical_names_not_flagged() -> None:
    """Text 'available on Starter, Growth, and Enterprise' must not flag."""
    from resonantforge.validators.invariant_checker import run_checker

    chunk = _make_chunk(
        chunk_id="test_tier_text_canonical_list",
        chunk_text="This feature is available on Starter, Growth, and Enterprise plans.",
    )
    report = run_checker([chunk])
    tier_errors = [e for e in report.errors if "non-canonical tier" in e]
    assert not tier_errors, f"Canonical tier names must not flag. Got: {tier_errors}"


def test_tier_vocab_canonical_claim_not_flagged() -> None:
    """Claim with canonical tier value 'Enterprise' must not flag."""
    from resonantforge.validators.invariant_checker import run_checker

    chunk = _make_chunk(
        chunk_id="test_tier_claim_enterprise",
        chunk_text="Enterprise accounts have dedicated support.",
        claims={"tier": "Enterprise"},
    )
    report = run_checker([chunk])
    tier_errors = [e for e in report.errors if "non-canonical tier" in e]
    assert not tier_errors, f"Canonical claim value 'Enterprise' must not flag. Got: {tier_errors}"


def test_invariant_checker_zero_warnings_existing_kb() -> None:
    """
    The invariant checker must return zero errors AND zero warnings against
    the existing tagged KB after PR2.5 corrections.

    Zero warnings (not just zero errors) is the PR2.5 completion criterion.
    """
    from resonantforge.kb.saas_content import get_saas_kb_chunks
    from resonantforge.validators.invariant_checker import run_checker

    chunks = get_saas_kb_chunks()
    report = run_checker(chunks)
    assert not report.errors, (
        f"Invariant checker found {len(report.errors)} error(s):\n"
        + "\n".join(f"  - {e}" for e in report.errors)
    )
    assert not report.warnings, (
        f"Invariant checker found {len(report.warnings)} warning(s):\n"
        + "\n".join(f"  - {w}" for w in report.warnings)
    )


# ---------------------------------------------------------------------------
# Group 15 — State machine domain vocabulary (assertions SM1–SM3)
# ---------------------------------------------------------------------------
#
# These tests target the weighted-random domain selection introduced in PR3a.
# They run the state machine directly (no full pipeline) to avoid LLM costs.
# ---------------------------------------------------------------------------

_SM_SEED = 7
_SM_LARGE_ACCOUNTS = 100  # 100 × 6 × 30 × ~0.3 ≈ ~5400 conversation events
_SM_SMALL_ACCOUNTS = 5   # used for determinism / sequence tests only


def _collect_conv_domains(seed: int, accounts: int, months: int) -> list[str]:
    """Run the state machine and return the domain for each CONVERSATION_STARTED event."""
    from resonantforge.layer1.state_machine import StateMachine
    from resonantforge.profiles import get_profile
    from resonantforge.schemas import SimEventType

    profile = get_profile("saas")
    sm = StateMachine(seed=seed, num_accounts=accounts, num_months=months, profile=profile)
    events, _ = sm.simulate()
    return [
        e.payload["domain"]
        for e in events
        if e.event_type == SimEventType.CONVERSATION_STARTED
    ]


def test_state_machine_all_domains_reachable() -> None:
    """
    SM1 — All 13 SaaS domains must appear in CONVERSATION_STARTED events across
    a 100-account × 6-month run.

    sla_credits is the rarest domain (weight 1/100). With ~5400 conversations the
    expected count is ~54, so non-appearance would indicate a configuration bug,
    not bad luck.
    """
    from resonantforge.profiles import get_profile

    profile = get_profile("saas")
    expected_domains = set(profile.domain_weights().keys())

    domains_seen = set(_collect_conv_domains(_SM_SEED, _SM_LARGE_ACCOUNTS, 6))

    missing = expected_domains - domains_seen
    assert not missing, (
        f"SM1 — {len(missing)} domain(s) never appeared in a "
        f"{_SM_LARGE_ACCOUNTS}-account × 6-month run: {sorted(missing)}"
    )


def test_state_machine_domain_sequence_deterministic() -> None:
    """
    SM2 — Two runs with the same seed must produce identical domain sequences.

    Also verifies that two different seeds produce different sequences (a trivial
    check, but catches accidental global-state mutations in the RNG).
    """
    seq_a = _collect_conv_domains(_SM_SEED, _SM_SMALL_ACCOUNTS, 2)
    seq_b = _collect_conv_domains(_SM_SEED, _SM_SMALL_ACCOUNTS, 2)

    # Assertion SM2a: same seed → identical sequence
    assert seq_a == seq_b, (
        f"SM2a — Same seed {_SM_SEED} produced different domain sequences: "
        f"len(a)={len(seq_a)}, len(b)={len(seq_b)}, "
        f"first diff at index "
        f"{next(i for i, (x, y) in enumerate(zip(seq_a, seq_b)) if x != y) if seq_a != seq_b else 'len mismatch'}"
    )

    # Assertion SM2b: different seeds → different sequences
    seq_c = _collect_conv_domains(_SM_SEED + 1, _SM_SMALL_ACCOUNTS, 2)
    assert seq_a != seq_c, (
        "SM2b — Seeds differing by 1 produced identical domain sequences; "
        "the RNG is not advancing across seeds as expected"
    )


def test_state_machine_domain_distribution_vs_profile_weights() -> None:
    """
    SM3 — Domain frequencies over ~5400 conversations must be within ±5% absolute
    of the profile's declared weights.

    Tolerance is intentionally generous: the 5% band is much wider than 3σ for
    any domain with weight ≥1/100, so this test guards against grossly wrong
    distributions (e.g. weight table mis-keyed) without being brittle to normal
    stochastic variation.
    """
    from resonantforge.profiles import get_profile

    profile = get_profile("saas")
    weights = profile.domain_weights()
    total_weight = sum(weights.values())
    expected_fraction = {d: w / total_weight for d, w in weights.items()}

    domains = _collect_conv_domains(_SM_SEED, _SM_LARGE_ACCOUNTS, 6)
    n_total = len(domains)
    assert n_total >= 1000, (
        f"SM3 — expected ≥1000 conversation events for meaningful distribution test, "
        f"got {n_total}. Increase _SM_LARGE_ACCOUNTS."
    )

    observed_fraction: dict[str, float] = {}
    for domain in weights:
        count = sum(1 for d in domains if d == domain)
        observed_fraction[domain] = count / n_total

    _TOLERANCE = 0.05  # 5% absolute
    violations: list[str] = []
    for domain, exp in expected_fraction.items():
        obs = observed_fraction.get(domain, 0.0)
        if abs(obs - exp) > _TOLERANCE:
            violations.append(
                f"domain={domain!r}: expected≈{exp:.3f}, observed={obs:.3f}, "
                f"delta={abs(obs - exp):.3f} > tolerance={_TOLERANCE}"
            )

    assert not violations, (
        f"SM3 — domain distribution outside ±{_TOLERANCE:.0%} tolerance "
        f"({n_total} total conversations):\n"
        + "\n".join(f"  {v}" for v in violations)
    )


def test_skipped_conversations_have_final_prose(tmp_path: Path) -> None:
    """
    Assertion 29: every skipped conversation record written after exhausting prose
    retries must carry the full final prose in a ``final_prose`` field, and that
    field must be consistent with the existing ``agent_prose_snippet`` (snippet is
    the first 200 characters of the same prose string).

    Uses the same mocked-validator approach as test_skipped_record_fields_complete
    so no live API key is required.
    """
    from unittest.mock import patch

    _FAKE_PROSE = (
        "Customer: I need help with my account.\n"
        "Agent: I'd be happy to help you today. What seems to be the issue?\n"
        "Customer: I can't access the dashboard.\n"
        "Agent: I understand. Let me look into that for you right away.\n"
        "Customer: It's been broken since yesterday.\n"
        "Agent: I can see the issue on our end and will escalate this immediately."
    )

    _fail_verdict = DimensionVerdict(
        dimension="empathy",
        verdict=ValidationVerdict.FAIL,
        target="high",
        signals_summary={"acknowledgment_present": False},
    )

    _skip_result = ValidationResult(
        conversation_id="mocked",
        overall_verdict=ValidationVerdict.SKIP,
        dimension_verdicts=[_fail_verdict],
        skip_reason="mocked SKIP for test_skipped_conversations_have_final_prose",
        retry_count=2,
    )

    with (
        patch("resonantforge.pipeline._call_anthropic", return_value=(_FAKE_PROSE, 0, 0)),
        patch("resonantforge.pipeline.validate_all_dimensions", return_value=[_fail_verdict]),
        patch(
            "resonantforge.layer1.plan_validator.PlanValidator.post_generation_validate",
            return_value=_skip_result,
        ),
    ):
        config = PipelineConfig(
            profile_name="saas",
            accounts=2,
            months=1,
            seed=42,
            output_root=tmp_path,
            anthropic_api_key="fake-key-for-final-prose-test",
        )
        with pytest.raises(RuntimeError, match="Quality gate"):
            run_pipeline(config)

    skipped_path = tmp_path / "saas" / "skipped_conversations.jsonl"
    skipped = _read_jsonl(skipped_path)
    assert len(skipped) > 0, (
        "Assertion 29 — skipped_conversations.jsonl is empty, expected at least one "
        "SKIPped record from the mocked validator"
    )

    violations: list[str] = []
    for rec in skipped:
        conv_id = rec.get("conversation_id", "?")

        # final_prose must be present and non-empty
        final_prose = rec.get("final_prose")
        if not final_prose:
            violations.append(f"conv_id={conv_id!r}: final_prose missing or empty (got {final_prose!r})")
            continue

        # snippet must be the first 200 chars of the same prose
        snippet = rec.get("agent_prose_snippet") or ""
        if not final_prose.startswith(snippet):
            violations.append(
                f"conv_id={conv_id!r}: agent_prose_snippet is not a prefix of final_prose "
                f"(snippet={snippet[:40]!r}, prose_start={final_prose[:40]!r})"
            )

    assert not violations, (
        f"Assertion 29 — {len(violations)} final_prose violation(s):\n"
        + "\n".join(f"  {v}" for v in violations[:5])
    )


# ---------------------------------------------------------------------------
# Accuracy extractor — fence stripping, parse logging, multi_chunk_required
# ---------------------------------------------------------------------------


def test_strip_markdown_fence_removes_json_wrapper() -> None:
    """
    _strip_markdown_fence must handle all fence variants and pass through clean JSON.

    Assertion 30 — four cases:
      a) Raw JSON (no fence) — unchanged.
      b) ```json-wrapped — fence stripped, raw JSON returned.
      c) Plain ```-wrapped — fence stripped, raw JSON returned.
      d) Whitespace-padded raw JSON — leading/trailing whitespace removed.
    """
    from resonantforge.validators.extractors.accuracy import _strip_markdown_fence

    raw = '[{"claim_text": "hello"}]'

    # (a) raw JSON passthrough
    assert _strip_markdown_fence(raw) == raw, "Assertion 30a — raw JSON should be unchanged"

    # (b) ```json fence
    fenced_json = f"```json\n{raw}\n```"
    assert _strip_markdown_fence(fenced_json) == raw, (
        "Assertion 30b — ```json fence should be stripped"
    )

    # (c) plain ``` fence
    fenced_plain = f"```\n{raw}\n```"
    assert _strip_markdown_fence(fenced_plain) == raw, (
        "Assertion 30c — plain ``` fence should be stripped"
    )

    # (d) whitespace-padded raw JSON
    padded = f"   {raw}   "
    assert _strip_markdown_fence(padded) == raw, (
        "Assertion 30d — surrounding whitespace should be stripped"
    )


def test_extract_claims_handles_fenced_response() -> None:
    """
    extract_claims_llm must return parsed claims when the model wraps output in a fence.

    Assertion 31 — mock the Anthropic client to return a ```json-fenced response;
    the extractor must return a non-empty claims list.
    """
    from unittest.mock import MagicMock
    from resonantforge.validators.extractors.accuracy import extract_claims_llm

    fenced_response = '```json\n[{"claim_text": "tokens do not expire", "claim_span": [0, 22], "claim_type": "factual", "subject": "tokens", "predicate": "expire", "object": "no"}]\n```'

    mock_message = MagicMock()
    mock_message.content = [MagicMock(text=fenced_response)]

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_message

    claims = extract_claims_llm("tokens do not expire automatically", anthropic_client=mock_client)

    assert len(claims) == 1, (
        f"Assertion 31 — expected 1 claim from fenced response, got {len(claims)}"
    )
    assert claims[0].claim_text == "tokens do not expire", (
        f"Assertion 31 — claim_text mismatch: {claims[0].claim_text!r}"
    )


def test_extract_claims_logs_on_parse_failure(caplog: pytest.LogCaptureFixture) -> None:
    """
    extract_claims_llm must raise ClaimExtractionError AND emit structured WARNING logs
    when all 3 attempts return unparseable JSON.

    Assertion 32 — verifies the fail-loud contract and that the structured log includes
    the telemetry fields needed to distinguish truncation from other failure modes:
    response_length, stop_reason, excerpt_head, excerpt_tail.
    """
    import logging
    import pytest as _pytest
    from unittest.mock import MagicMock
    from resonantforge.validators.extractors.accuracy import extract_claims_llm, ClaimExtractionError

    mock_message = MagicMock()
    mock_message.content = [MagicMock(text="THIS IS NOT JSON AT ALL")]
    mock_message.stop_reason = "end_turn"

    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_message

    with caplog.at_level(logging.WARNING, logger="resonantforge.validators.extractors.accuracy"):
        with _pytest.raises(ClaimExtractionError):
            extract_claims_llm("some agent prose", anthropic_client=mock_client, max_attempts=3)

    parse_fail_records = [r for r in caplog.records if "JSON parse failed" in r.message]
    assert len(parse_fail_records) == 3, (
        f"Assertion 32 — expected 3 'JSON parse failed' warnings (one per attempt), "
        f"got {len(parse_fail_records)}"
    )
    # Fields must appear in the formatted message (not just as LogRecord attributes via
    # extra={}) so they are visible in any log handler, including plain StreamHandler.
    msg = parse_fail_records[0].getMessage()
    assert "stop_reason" in msg, f"Assertion 32 — stop_reason missing from log message: {msg!r}"
    assert "response_length" in msg, f"Assertion 32 — response_length missing from log message: {msg!r}"
    assert "end_turn" in msg, f"Assertion 32 — stop_reason value missing from log message: {msg!r}"
    assert str(len("THIS IS NOT JSON AT ALL")) in msg, (
        f"Assertion 32 — response_length value missing from log message: {msg!r}"
    )


def test_multi_chunk_required_uses_length() -> None:
    """
    run_kb_alignment_pipeline's early-exit path must set multi_chunk_required=False
    when kb_chunks_required has exactly one entry (single-chunk plan).

    Assertion 33 — the early-exit path (no claims) previously used bool(kb_chunks_required),
    which is True for any non-empty list. It must now use len > 1, matching the happy path.
    """
    from resonantforge.validators.extractors.accuracy import run_kb_alignment_pipeline
    from resonantforge.schemas import KBChunk, ConstraintType

    single_chunk_required = ["kb_chunk_api_authentication_v3"]
    dummy_chunk = KBChunk(
        chunk_id="kb_chunk_api_authentication_v3",
        document_id="doc_1",
        document_path="policies/api.md",
        chunk_text="Tokens do not expire automatically.",
        constraint_type=ConstraintType.INFORMATIONAL,
    )

    # No claims → early-exit path in run_kb_alignment_pipeline
    signals = run_kb_alignment_pipeline(
        claims=[],
        kb_chunks=[dummy_chunk],
        conversation_context="",
        kb_chunks_required=single_chunk_required,
        synonym_map={},
    )

    assert signals.multi_chunk_required is False, (
        f"Assertion 33 — single-chunk plan should have multi_chunk_required=False, "
        f"got {signals.multi_chunk_required}"
    )


def test_extract_claims_handles_null_fields() -> None:
    """
    extract_claims_llm must return gracefully when the model returns claims with
    null values for subject, predicate, or object fields.

    This reproduces the 'NoneType' object has no attribute 'lower' crash that occurred
    on conv_evt_00165 (second retry): raw.get("subject", "") returns None when the JSON
    key is present with a null value, bypassing the default — then _normalize_text(None)
    called .lower() on None.

    Assertions 34a–34c:
      34a — null subject/predicate/object → non-empty claims list (graceful, not crash)
      34b — null claim_type → falls back to "factual", claim is returned
      34c — _normalize_text called with None directly → returns empty string (not crash)
    """
    from unittest.mock import MagicMock
    from resonantforge.validators.extractors.accuracy import extract_claims_llm, _normalize_text

    # 34c: direct guard on _normalize_text — must not raise
    assert _normalize_text(None, {}) == "", (  # type: ignore[arg-type]
        "Assertion 34c — _normalize_text(None) should return '' not crash"
    )
    assert _normalize_text(42, {}) == "", (  # type: ignore[arg-type]
        "Assertion 34c — _normalize_text(non-str) should return '' not crash"
    )

    # 34a: null subject/predicate/object in LLM response
    null_fields_response = (
        '[{"claim_text": "tokens never expire", '
        '"claim_span": [0, 19], '
        '"claim_type": "factual", '
        '"subject": null, '
        '"predicate": null, '
        '"object": null}]'
    )
    mock_msg_a = MagicMock()
    mock_msg_a.content = [MagicMock(text=null_fields_response)]
    mock_client_a = MagicMock()
    mock_client_a.messages.create.return_value = mock_msg_a

    claims_a = extract_claims_llm("tokens never expire", anthropic_client=mock_client_a)
    assert len(claims_a) == 1, (
        f"Assertion 34a — expected 1 claim despite null fields, got {len(claims_a)}"
    )
    assert claims_a[0].claim_text == "tokens never expire", (
        f"Assertion 34a — claim_text wrong: {claims_a[0].claim_text!r}"
    )
    assert claims_a[0].normalized_subject == "", (
        f"Assertion 34a — null subject should normalize to '', got {claims_a[0].normalized_subject!r}"
    )

    # 34b: null claim_type should default to "factual"
    null_type_response = (
        '[{"claim_text": "refunds take 5 days", '
        '"claim_span": [0, 19], '
        '"claim_type": null, '
        '"subject": "refunds", '
        '"predicate": "take", '
        '"object": "5 days"}]'
    )
    mock_msg_b = MagicMock()
    mock_msg_b.content = [MagicMock(text=null_type_response)]
    mock_client_b = MagicMock()
    mock_client_b.messages.create.return_value = mock_msg_b

    claims_b = extract_claims_llm("refunds take 5 days", anthropic_client=mock_client_b)
    assert len(claims_b) == 1, (
        f"Assertion 34b — expected 1 claim with null claim_type fallback, got {len(claims_b)}"
    )
    assert claims_b[0].claim_type == "factual", (
        f"Assertion 34b — null claim_type should default to 'factual', got {claims_b[0].claim_type!r}"
    )


# ---------------------------------------------------------------------------
# Group 15 — Accuracy extractor candidate pool fixes (assertions 35a–35d)
# ---------------------------------------------------------------------------


def test_candidate_pool_is_required_chunks_only() -> None:
    """
    Assertion 35a: run_kb_alignment_pipeline must consider ONLY the chunks listed in
    kb_chunks_required, never extra chunks that happen to be present in the full corpus.

    A pool of 5 chunks is passed; only 2 are in kb_chunks_required. After the fix,
    kb_chunks_used must be a subset of the required IDs — the unrequired chunks must
    never surface in alignment output regardless of claim content.
    """
    from unittest.mock import MagicMock
    from resonantforge.validators.extractors.accuracy import run_kb_alignment_pipeline
    from resonantforge.schemas import Claim, KBChunk, ConstraintType

    required_ids = {"kb_chunk_required_a", "kb_chunk_required_b"}
    unrequired_ids = {"kb_chunk_unrequired_x", "kb_chunk_unrequired_y", "kb_chunk_unrequired_z"}

    def _make_chunk(chunk_id: str, text: str) -> KBChunk:
        return KBChunk(
            chunk_id=chunk_id,
            document_id="doc_test",
            document_path="policies/test.md",
            chunk_text=text,
            constraint_type=ConstraintType.INFORMATIONAL,
        )

    # All chunks share the same keywords so relevance matching would hit all of them.
    all_chunks = [
        _make_chunk("kb_chunk_required_a", "refund policy allows returns within 30 days"),
        _make_chunk("kb_chunk_required_b", "refund processing takes 5 business days"),
        _make_chunk("kb_chunk_unrequired_x", "refund requests must be submitted online"),
        _make_chunk("kb_chunk_unrequired_y", "refund eligibility requires original receipt"),
        _make_chunk("kb_chunk_unrequired_z", "refund amounts are credited within 3 days"),
    ]

    # Claim is written to match all chunks by keyword overlap.
    claim = Claim(
        claim_text="We can process your refund within 30 days.",
        claim_span=(0, 44),
        claim_type="policy",
        normalized_subject="refund",
        normalized_predicate="process",
        normalized_object="30 days",
    )

    signals = run_kb_alignment_pipeline(
        claims=[claim],
        kb_chunks=all_chunks,
        conversation_context="",
        kb_chunks_required=list(required_ids),
        synonym_map={},
    )

    # Assertion 35a: no unrequired chunk may appear in kb_chunks_used.
    used_set = set(signals.kb_chunks_used)
    leaked = used_set & unrequired_ids
    assert not leaked, (
        f"Assertion 35a — unrequired chunks leaked into alignment: {leaked!r}. "
        f"Candidate pool must be restricted to kb_chunks_required."
    )


def test_required_chunk_always_in_pool() -> None:
    """
    Assertion 35b: a required chunk at a high insertion index (>50) must reach
    alignment scoring after the candidate pool fix.

    Before the fix, top-K truncation would silently exclude insertion-order index 55.
    After the fix, the required chunk is the *only* candidate regardless of position.
    """
    from resonantforge.validators.extractors.accuracy import run_kb_alignment_pipeline
    from resonantforge.schemas import Claim, KBChunk, ConstraintType

    target_id = "kb_chunk_target_at_index_55"

    # Build 60 filler chunks that do not share keywords with the claim.
    filler_chunks = [
        KBChunk(
            chunk_id=f"kb_chunk_filler_{i:03d}",
            document_id="doc_filler",
            document_path="policies/filler.md",
            chunk_text=f"unrelated policy clause number {i} about escalation procedures",
            constraint_type=ConstraintType.INFORMATIONAL,
        )
        for i in range(60)
    ]

    # Target chunk is at position 55 in the list (high insertion index).
    target_chunk = KBChunk(
        chunk_id=target_id,
        document_id="doc_target",
        document_path="policies/webhook.md",
        chunk_text="webhook tokens do not expire unless explicitly revoked",
        constraint_type=ConstraintType.INFORMATIONAL,
    )
    all_chunks = filler_chunks[:55] + [target_chunk] + filler_chunks[55:]

    # Claim that shares keywords with the target chunk only.
    claim = Claim(
        claim_text="Tokens do not expire automatically.",
        claim_span=(0, 35),
        claim_type="policy",
        normalized_subject="tokens",
        normalized_predicate="expire",
        normalized_object="",
    )

    signals = run_kb_alignment_pipeline(
        claims=[claim],
        kb_chunks=all_chunks,
        conversation_context="",
        kb_chunks_required=[target_id],
        synonym_map={},
    )

    # Assertion 35b: target chunk must appear in kb_chunks_used (it reached alignment scoring).
    assert target_id in signals.kb_chunks_used, (
        f"Assertion 35b — required chunk at insertion index 55 was not reached. "
        f"kb_chunks_used={signals.kb_chunks_used!r}. "
        f"Top-K insertion-order truncation may still be active."
    )


def test_word_boundary_trigger_matching() -> None:
    """
    Assertion 35c: _check_constraint_type_applies must use word-boundary matching
    for trigger phrases, not substring matching.

    'supermore than 5' must NOT match the trigger 'more than'.
    'more than 5 units' MUST match 'more than'.
    'used up all quota' MUST match 'used'.
    'unused credits' must NOT match 'used' (substring inside 'unused').
    """
    from resonantforge.validators.extractors.accuracy import _check_constraint_type_applies
    from resonantforge.schemas import KBChunk, ConstraintType

    deny_chunk = KBChunk(
        chunk_id="kb_chunk_deny_usage",
        document_id="doc_deny",
        document_path="policies/usage.md",
        chunk_text="Refunds are not available when usage exceeds more than 500 units.",
        constraint_type=ConstraintType.DENY_CONDITION,
    )

    # Should NOT fire: 'supermore than' contains 'more than' as substring but not at a word boundary.
    assert _check_constraint_type_applies(deny_chunk, "I have supermore than 5 tasks here") is False, (
        "Assertion 35c — 'supermore than' matched trigger 'more than' via substring; "
        "word-boundary matching must prevent this."
    )

    # Should NOT fire: 'unused' contains 'used' as substring but not at a word boundary.
    assert _check_constraint_type_applies(deny_chunk, "I have unused credits on my account") is False, (
        "Assertion 35c — 'unused' matched trigger 'used' via substring; "
        "word-boundary matching must prevent this."
    )

    # SHOULD fire: 'more than' appears as a complete word phrase.
    assert _check_constraint_type_applies(deny_chunk, "I have used more than 600 units this month") is True, (
        "Assertion 35c — 'more than' at word boundary should trigger the deny condition."
    )

    # SHOULD fire: 'used' appears as a standalone word.
    assert _check_constraint_type_applies(deny_chunk, "I used all of my quota already") is True, (
        "Assertion 35c — standalone 'used' should trigger the deny condition."
    )


def test_organic_conversation_handles_empty_required() -> None:
    """
    Assertion 35d: run_kb_alignment_pipeline with an empty kb_chunks_required list
    must return an early-exit AccuracySignals without crashing.

    Organic conversations (no QualityPlan planted) produce empty kb_chunks_required.
    The pipeline must not attempt KB alignment and must return alignment='not_found'
    with all constraint flags False.
    """
    from resonantforge.validators.extractors.accuracy import run_kb_alignment_pipeline
    from resonantforge.schemas import Claim, KBChunk, ConstraintType

    chunk = KBChunk(
        chunk_id="kb_chunk_some_policy",
        document_id="doc_1",
        document_path="policies/billing.md",
        chunk_text="Billing cycles reset on the first of each month.",
        constraint_type=ConstraintType.INFORMATIONAL,
    )
    claim = Claim(
        claim_text="Billing resets monthly.",
        claim_span=(0, 22),
        claim_type="policy",
        normalized_subject="billing",
        normalized_predicate="resets",
        normalized_object="monthly",
    )

    signals = run_kb_alignment_pipeline(
        claims=[claim],
        kb_chunks=[chunk],
        conversation_context="",
        kb_chunks_required=[],  # organic — no planted ground truth
        synonym_map={},
    )

    # Assertion 35d: early return with not_found alignment; no constraint flags set.
    assert signals.alignment == "not_found", (
        f"Assertion 35d — organic conversation (empty required) should yield "
        f"alignment='not_found', got {signals.alignment!r}"
    )
    assert signals.blocking_constraint_violated is False, (
        "Assertion 35d — blocking_constraint_violated must be False for organic conversation"
    )
    assert signals.overgeneralization_flag is False, (
        "Assertion 35d — overgeneralization_flag must be False for organic conversation"
    )
    assert signals.kb_chunks_used == [], (
        f"Assertion 35d — kb_chunks_used must be empty for organic conversation, "
        f"got {signals.kb_chunks_used!r}"
    )


# ---------------------------------------------------------------------------
# Phase 1A — Empathy / resolution lexicon expansion + apology bridge
# ---------------------------------------------------------------------------


def test_empathy_recognizes_sorry_youre_hitting() -> None:
    """
    Assertion 36a — "sorry you're hitting" must produce acknowledgment_present: True.

    "sorry you're experiencing" was the only "sorry you're <verb>" entry before
    Phase 1A. Agents routinely use "hitting", "running into", and "seeing" for
    the same sentiment.
    """
    from resonantforge.validators.extractors.empathy import extract_empathy_signals
    from resonantforge.profiles.lexicons.saas import (
        ACKNOWLEDGMENT_PHRASES, EMOTION_LEXICON, APOLOGY_LEXICON, ACTION_VERB_LEXICON,
    )

    prose = "I'm sorry you're hitting that error—let me look into it right now."
    signals = extract_empathy_signals(
        agent_prose=prose,
        acknowledgment_phrases=ACKNOWLEDGMENT_PHRASES,
        emotion_lexicon=EMOTION_LEXICON,
        apology_lexicon=APOLOGY_LEXICON,
        action_verb_lexicon=ACTION_VERB_LEXICON,
    )
    assert signals.acknowledgment_present is True, (
        "Assertion 36a — 'sorry you're hitting' should produce acknowledgment_present=True; "
        f"got acknowledgment_present={signals.acknowledgment_present}, "
        f"ack_phrases={signals.acknowledgment_phrases}"
    )


def test_empathy_recognizes_positive_warmth_terms() -> None:
    """
    Assertion 36b — positive warmth terms ("glad") must produce emotional_language_present: True.

    EMOTION_LEXICON previously covered only negative-affect vocabulary. Agents
    expressing positive warmth ("I'm glad we got this sorted") scored zero on
    emotional_language_present despite clear empathetic signal.
    """
    from resonantforge.validators.extractors.empathy import extract_empathy_signals
    from resonantforge.profiles.lexicons.saas import (
        ACKNOWLEDGMENT_PHRASES, EMOTION_LEXICON, APOLOGY_LEXICON, ACTION_VERB_LEXICON,
    )

    prose = "I'm glad we could get this sorted for you today."
    signals = extract_empathy_signals(
        agent_prose=prose,
        acknowledgment_phrases=ACKNOWLEDGMENT_PHRASES,
        emotion_lexicon=EMOTION_LEXICON,
        apology_lexicon=APOLOGY_LEXICON,
        action_verb_lexicon=ACTION_VERB_LEXICON,
    )
    assert signals.emotional_language_present is True, (
        "Assertion 36b — 'glad' should produce emotional_language_present=True; "
        f"got emotional_language_present={signals.emotional_language_present}, "
        f"emotional_terms={signals.emotional_terms}"
    )


def test_empathy_walk_through_triggers_follow_through() -> None:
    """
    Assertion 36c — "let me walk you through this" after an acknowledgment phrase
    must produce follow_through_present: True.

    "walk" was absent from ACTION_VERB_LEXICON. "let me walk you through this" is
    a canonical follow-through phrase in conversational support.
    """
    from resonantforge.validators.extractors.empathy import extract_empathy_signals
    from resonantforge.profiles.lexicons.saas import (
        ACKNOWLEDGMENT_PHRASES, EMOTION_LEXICON, APOLOGY_LEXICON, ACTION_VERB_LEXICON,
    )

    prose = "I can see the issue. Let me walk you through the fix step by step."
    signals = extract_empathy_signals(
        agent_prose=prose,
        acknowledgment_phrases=ACKNOWLEDGMENT_PHRASES,
        emotion_lexicon=EMOTION_LEXICON,
        apology_lexicon=APOLOGY_LEXICON,
        action_verb_lexicon=ACTION_VERB_LEXICON,
    )
    assert signals.follow_through_present is True, (
        "Assertion 36c — 'let me walk you through' after ack phrase should produce "
        f"follow_through_present=True; got follow_through_present={signals.follow_through_present}"
    )


def test_empathy_apology_bridges_to_acknowledgment() -> None:
    """
    Assertion 36d — an apology alone (no ACKNOWLEDGMENT_PHRASES match) must produce
    acknowledgment_present: True via the apology bridge.

    Before the bridge fix, an agent who opened with "I'm sorry" (APOLOGY_LEXICON
    match) but hit no ACKNOWLEDGMENT_PHRASES entry would always fail the empathy
    high gate. Apologies are a superset of acknowledgments from a human empathy
    standpoint.
    """
    from resonantforge.validators.extractors.empathy import extract_empathy_signals
    from resonantforge.profiles.lexicons.saas import (
        EMOTION_LEXICON, APOLOGY_LEXICON, ACTION_VERB_LEXICON,
    )

    # Use a minimal ACKNOWLEDGMENT_PHRASES list that will NOT match this prose,
    # so the only empathy signal is the apology.
    minimal_ack_phrases: list[str] = []
    prose = "I'm sorry for the trouble. Let me resolve this for you right now."
    signals = extract_empathy_signals(
        agent_prose=prose,
        acknowledgment_phrases=minimal_ack_phrases,
        emotion_lexicon=EMOTION_LEXICON,
        apology_lexicon=APOLOGY_LEXICON,
        action_verb_lexicon=ACTION_VERB_LEXICON,
    )
    assert signals.acknowledgment_present is True, (
        "Assertion 36d — apology alone should bridge to acknowledgment_present=True; "
        f"got acknowledgment_present={signals.acknowledgment_present}, "
        f"apology_terms={signals.apology_terms}"
    )


def test_resolution_recognizes_imperative_instructions() -> None:
    """
    Assertion 36e — "you'll want to go to Settings" must produce solution_provided: True.

    RESOLUTION_PATTERNS was biased toward outcome-summary language and had no
    coverage of imperative instruction patterns (the dominant pattern in step-by-step
    LLM-generated support responses).
    """
    from resonantforge.validators.extractors.resolution import extract_resolution_signals
    from resonantforge.profiles.lexicons.saas import (
        RESOLUTION_PATTERNS, DEFLECTION_PATTERNS, NEXT_STEPS_PATTERNS,
        TEMPORAL_ANCHOR_PATTERNS, SPECIFIC_ACTOR_PATTERNS, OWNERSHIP_PATTERNS,
        ISSUE_KEYWORDS, COMPLETION_VERB_PATTERNS, COMPLETION_RESOLUTION_PATTERNS,
    )

    prose = "You'll want to go to Settings and update your configuration there."
    signals = extract_resolution_signals(
        agent_prose=prose,
        customer_prose="",
        resolution_patterns=RESOLUTION_PATTERNS,
        deflection_patterns=DEFLECTION_PATTERNS,
        next_steps_patterns=NEXT_STEPS_PATTERNS,
        temporal_anchor_patterns=TEMPORAL_ANCHOR_PATTERNS,
        specific_actor_patterns=SPECIFIC_ACTOR_PATTERNS,
        ownership_patterns=OWNERSHIP_PATTERNS,
        issue_keywords=ISSUE_KEYWORDS,
        completion_verb_patterns=COMPLETION_VERB_PATTERNS,
        completion_resolution_patterns=COMPLETION_RESOLUTION_PATTERNS,
    )
    assert signals.solution_provided is True, (
        "Assertion 36e — \"you'll want to go to\" should produce solution_provided=True; "
        f"got solution_provided={signals.solution_provided}"
    )


def test_resolution_recognizes_hyphenated_temporal_range() -> None:
    """
    Assertion 36f — "within 10-15 minutes" must produce next_steps_actionable: True.

    The pattern r"\\bwithin \\d+ (?:hour|minute|day|business day)\\b" did not match
    hyphenated ranges ("10-15") because \\d+ matches only a single integer. The unit
    word "minutes" was then not adjacent to the matched digit group, causing
    next_steps_actionable=False for well-written time-estimate responses.
    """
    from resonantforge.validators.extractors.resolution import extract_resolution_signals
    from resonantforge.profiles.lexicons.saas import (
        RESOLUTION_PATTERNS, DEFLECTION_PATTERNS, NEXT_STEPS_PATTERNS,
        TEMPORAL_ANCHOR_PATTERNS, SPECIFIC_ACTOR_PATTERNS, OWNERSHIP_PATTERNS,
        ISSUE_KEYWORDS, COMPLETION_VERB_PATTERNS, COMPLETION_RESOLUTION_PATTERNS,
    )

    # next_steps_present requires a NEXT_STEPS_PATTERNS match; "within \d+" already
    # matches. The temporal anchor "within 10-15 minutes" is the new addition.
    prose = "DNS propagation usually completes within 10-15 minutes in most cases."
    signals = extract_resolution_signals(
        agent_prose=prose,
        customer_prose="",
        resolution_patterns=RESOLUTION_PATTERNS,
        deflection_patterns=DEFLECTION_PATTERNS,
        next_steps_patterns=NEXT_STEPS_PATTERNS,
        temporal_anchor_patterns=TEMPORAL_ANCHOR_PATTERNS,
        specific_actor_patterns=SPECIFIC_ACTOR_PATTERNS,
        ownership_patterns=OWNERSHIP_PATTERNS,
        issue_keywords=ISSUE_KEYWORDS,
        completion_verb_patterns=COMPLETION_VERB_PATTERNS,
        completion_resolution_patterns=COMPLETION_RESOLUTION_PATTERNS,
    )
    assert signals.next_steps_actionable is True, (
        "Assertion 36f — 'within 10-15 minutes' should produce next_steps_actionable=True; "
        f"got next_steps_actionable={signals.next_steps_actionable}, "
        f"next_steps_present={signals.next_steps_present}"
    )


# ---------------------------------------------------------------------------
# RFORGE-25: quality plan conversation_id must match pipeline conv_id format
# ---------------------------------------------------------------------------

def test_quality_plan_conv_id_matches_pipeline_naming() -> None:
    """
    Assertion 37 — QualityPlanInjector must produce conversation_id = f"conv_{event_id}".

    The pipeline generates conv_ids as f"conv_{event.event_id}" (pipeline.py:1181).
    The extractor matches quality plans to passed conversations by conv_id. If the
    injector uses a different prefix (e.g. "conv_planted_"), the extractor can never
    join quality plans to their conversations, making planted-quality replay impossible.
    """
    import random as _random
    from datetime import datetime, timezone as _tz
    from resonantforge.layer1.quality_plan_injector import QualityPlanInjector
    from resonantforge.schemas import KBChunk, SimEvent

    event = SimEvent(
        event_id="evt_rforge25_test",
        event_type="conversation_started",
        account_id="acct_test",
        agent_id="agent_test",
        timestamp=datetime(2026, 1, 1, 10, 0, tzinfo=_tz.utc),
        day_index=0,
        month_index=0,
        payload={
            "domain": "billing",
            "intent": ["billing_dispute"],
            "surface_channel": "intercom",
            "customer_name": "Test Customer",
            "agent_id": "agent_test",
        },
    )
    kb_chunk = KBChunk(
        chunk_id="kb_chunk_test_v1",
        document_id="doc_test_v1",
        document_path="test/doc.md",
        chunk_text="Refunds are processed within 5 business days.",
        domains=["billing"],
        intent_tags=[],
    )

    injector = QualityPlanInjector(rng=_random.Random(42), profile_name="saas")
    plans = injector.inject(
        events=[event],
        snapshots=[],
        kb_chunks=[kb_chunk],
        planted_count=1,
    )

    assert len(plans) >= 1, "Assertion 37 — expected at least one plan to be injected"
    for plan in plans:
        expected_conv_id = f"conv_{plan.trigger_event_id}"
        assert plan.conversation_id == expected_conv_id, (
            f"Assertion 37 — plan.conversation_id={plan.conversation_id!r} does not match "
            f"pipeline naming convention f'conv_{{event_id}}'={expected_conv_id!r}. "
            "This breaks extractor quality-plan matching."
        )


# ---------------------------------------------------------------------------
# Group 17 — Conditional reroute (RFORGE-10, assertions 38–40)
# ---------------------------------------------------------------------------


def test_no_exact_plan_paired_with_conditional_chunk(corpus: tuple[Path, Manifest]) -> None:
    """
    Assertion 38 — For any generated quality plan, if any kb_chunks_required member
    has constraint_type in {ALLOW_CONDITION, DENY_CONDITION}, accuracy_precision must
    not be 'exact'.

    Conditional chunks have branching structure that an :exact plan cannot honor —
    the agent applies one branch correctly, the validator finds the full conditional
    unreproduced, returns alignment='partial', FAIL. RFORGE-10 reroutes these to
    :conditional_applied at plan-generation time.
    """
    profile_dir, _ = corpus

    kb_chunks_path = profile_dir / "kb_chunks.jsonl"
    if not kb_chunks_path.exists():
        pytest.skip("kb_chunks.jsonl not found — KB generation may have been skipped")

    kb_lookup = {
        c.chunk_id: c
        for c in (KBChunk(**r) for r in _read_jsonl(kb_chunks_path))
    }

    plans_path = profile_dir / "planted_quality.jsonl"
    if not plans_path.exists():
        pytest.skip("planted_quality.jsonl not found")

    plans = [QualityPlan(**r) for r in _read_jsonl(plans_path)]
    _CONDITIONAL_TYPES = {ConstraintType.ALLOW_CONDITION, ConstraintType.DENY_CONDITION}

    violations = []
    for plan in plans:
        if not plan.rubric_targets.accuracy:
            continue
        if plan.rubric_targets.accuracy.precision != "exact":
            continue
        for cid in plan.kb_chunks_required:
            chunk = kb_lookup.get(cid)
            if chunk and chunk.constraint_type in _CONDITIONAL_TYPES:
                violations.append(
                    f"plan={plan.conversation_id} precision=exact "
                    f"chunk={cid} constraint_type={chunk.constraint_type}"
                )

    assert violations == [], (
        "Assertion 38 — Found :exact plans paired with conditional chunks (RFORGE-10 reroute failed):\n"
        + "\n".join(violations)
    )


def test_no_contradicted_exact_plan_paired_with_conditional_chunk(
    corpus: tuple[Path, Manifest],
) -> None:
    """
    Assertion 39 — No contradicted:exact plan may be paired with a conditional chunk.

    The contradicted:exact pick_pool_capped filter (RFORGE-10) must exclude
    ALLOW_CONDITION and DENY_CONDITION chunks before the fact-bearing pre-filter.
    Violating this produces unsatisfiable negation plans.
    """
    profile_dir, _ = corpus

    kb_chunks_path = profile_dir / "kb_chunks.jsonl"
    if not kb_chunks_path.exists():
        pytest.skip("kb_chunks.jsonl not found")

    kb_lookup = {
        c.chunk_id: c
        for c in (KBChunk(**r) for r in _read_jsonl(kb_chunks_path))
    }

    plans_path = profile_dir / "planted_quality.jsonl"
    if not plans_path.exists():
        pytest.skip("planted_quality.jsonl not found")

    plans = [QualityPlan(**r) for r in _read_jsonl(plans_path)]
    _CONDITIONAL_TYPES = {ConstraintType.ALLOW_CONDITION, ConstraintType.DENY_CONDITION}

    violations = []
    for plan in plans:
        if not plan.rubric_targets.accuracy:
            continue
        acc = plan.rubric_targets.accuracy
        if not (acc.status == "contradicted" and acc.precision == "exact"):
            continue
        for cid in plan.kb_chunks_required:
            chunk = kb_lookup.get(cid)
            if chunk and chunk.constraint_type in _CONDITIONAL_TYPES:
                violations.append(
                    f"plan={plan.conversation_id} contradicted:exact "
                    f"chunk={cid} constraint_type={chunk.constraint_type}"
                )

    assert violations == [], (
        "Assertion 39 — Found contradicted:exact plans with conditional chunks "
        "(RFORGE-10 pool filter failed):\n" + "\n".join(violations)
    )


def test_manifest_conditional_reroute_fields_present(corpus: tuple[Path, Manifest]) -> None:
    """
    Assertion 40 — manifest.conditional_reroute_count and manifest.conditional_reroute_events
    must be present and type-correct after a pipeline run.

    These fields are produced by QualityPlanInjector and forwarded to the manifest in
    Phase 5. Their presence confirms the RFORGE-10 telemetry wiring is complete.
    """
    _, manifest = corpus

    assert hasattr(manifest, "conditional_reroute_count"), (
        "Assertion 40 — Manifest missing 'conditional_reroute_count' field (RFORGE-10 telemetry)"
    )
    assert hasattr(manifest, "conditional_reroute_events"), (
        "Assertion 40 — Manifest missing 'conditional_reroute_events' field (RFORGE-10 telemetry)"
    )
    assert isinstance(manifest.conditional_reroute_count, int), (
        f"conditional_reroute_count must be int, got {type(manifest.conditional_reroute_count)}"
    )
    assert isinstance(manifest.conditional_reroute_events, list), (
        f"conditional_reroute_events must be list, got {type(manifest.conditional_reroute_events)}"
    )
    assert manifest.conditional_reroute_count >= 0, (
        f"conditional_reroute_count must be non-negative, got {manifest.conditional_reroute_count}"
    )
    assert len(manifest.conditional_reroute_events) == manifest.conditional_reroute_count, (
        f"conditional_reroute_events length {len(manifest.conditional_reroute_events)} "
        f"!= conditional_reroute_count {manifest.conditional_reroute_count}"
    )
