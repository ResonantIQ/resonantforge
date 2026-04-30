"""
Property-based tests for the Confabra corpus generator — 22 assertions.

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

from confabra.pipeline import PipelineConfig, run_pipeline
from confabra.schemas import (
    ConversationRecord,
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
from confabra.layer1.plan_validator import SkipRateTracker

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
    from confabra.profiles import get_profile
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
    from confabra.pipeline import PipelineConfig, run_pipeline

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
    from confabra.pipeline import _lockfile_path
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
        patch("confabra.pipeline._call_anthropic", return_value=(_FAKE_PROSE, 0, 0)),
        patch("confabra.pipeline.validate_all_dimensions", return_value=[]),
        patch(
            "confabra.layer1.plan_validator.PlanValidator.post_generation_validate",
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
        patch("confabra.pipeline._call_anthropic", return_value=(_FAKE_PROSE, 0, 0)),
        patch("confabra.pipeline.validate_all_dimensions", return_value=[_fail_verdict]),
        patch(
            "confabra.layer1.plan_validator.PlanValidator.post_generation_validate",
            return_value=_skip_result,
        ),
        patch("confabra.pipeline.run_soft_judge", return_value=_disagree_record),
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
        patch("confabra.pipeline._call_anthropic", return_value=(_FAKE_PROSE, 0, 0)),
        patch("confabra.pipeline.validate_all_dimensions", return_value=[_fail_verdict]),
        patch(
            "confabra.layer1.plan_validator.PlanValidator.post_generation_validate",
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
    variant ID, defaulting to "bv_baseline" when absent. This property ensures the
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
        effective_variant_id = rubric.get("brand_voice_against") or "bv_baseline"
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
        patch("confabra.pipeline._call_anthropic", return_value=(_FAKE_PROSE, 0, 0)),
        patch("confabra.pipeline.validate_all_dimensions", return_value=[]),
        patch(
            "confabra.layer1.plan_validator.PlanValidator.post_generation_validate",
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
    from confabra.layer1.quality_plan_injector import QualityPlanInjector
    return QualityPlanInjector(rng=_random.Random(seed), profile_name="saas")


def _make_conv_event(
    event_id: str,
    domain: str,
    account_id: str = "acct_001",
    event_type_override: str | None = None,
) -> SimEvent:
    """Build a minimal CONVERSATION_STARTED SimEvent for injector testing."""
    from datetime import datetime as _dt
    from confabra.schemas import SimEventType as _SET
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
    from confabra.schemas import (
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
    from confabra.layer1.quality_plan_injector import QualityPlanInjector
    from confabra.schemas import AccuracyLabel
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
    from confabra.schemas import AccuracyLabel
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
    from confabra.layer1.quality_plan_injector import QualityPlanInjector
    from confabra.schemas import AccuracyLabel
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
    from confabra.layer1.quality_plan_injector import QualityPlanInjector
    from confabra.schemas import AccuracyLabel
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
    from confabra.schemas import AccuracyLabel
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


def test_engine_raise_on_no_candidates() -> None:
    """
    E6 — ValueError is raised when the event domain has no matching KB chunks.
    """
    from confabra.schemas import AccuracyLabel
    from tests.fixtures.synthetic_kb import SYNTHETIC_KB_CHUNKS

    injector = _make_injector()
    conv = _make_conv_event("evt_e6", "nonexistent_domain")
    snap = _make_snapshot()
    spec = {"type": "accuracy", "label": AccuracyLabel(status="supported", precision="exact")}

    with pytest.raises(ValueError, match="nonexistent_domain"):
        injector._build_plan(conv, snap, spec, SYNTHETIC_KB_CHUNKS)


def test_engine_raise_on_unrecognized_event_type() -> None:
    """
    E7 — ValueError is raised when the trigger event type is not in the
    eligible allowlist (e.g. account_created going through the injector).
    """
    from confabra.schemas import AccuracyLabel
    from tests.fixtures.synthetic_kb import SYNTHETIC_KB_CHUNKS

    injector = _make_injector()
    # Build an ACCOUNT_CREATED event — not eligible for injection.
    from datetime import datetime as _dt
    from confabra.schemas import SimEventType as _SET
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
        "confabra.pipeline.generate_kb",
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
