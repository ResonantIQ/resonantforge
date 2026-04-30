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
    KBChunk,
    Manifest,
    QualityPlan,
    SimEvent,
    CorrectionRecord,
    NoiseClass,
)

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
        conv_scored_at: dict[str, list[str]] = {}
        for row in traj_rows:
            cid = row["conversation_id"]
            conv_scored_at.setdefault(cid, []).append(row["scored_at"])

        # Build set of known coaching_ids from coaching_history.
        coaching_path = agent_dir / "coaching_history.jsonl"
        coaching_rows = _read_jsonl(coaching_path) if coaching_path.exists() else []
        coaching_ids: set[str] = {c["coaching_id"] for c in coaching_rows}

        # --- Assertion 15: triggering_conversation_id precedes coaching issued_at ---
        for coaching in coaching_rows:
            trigger_cid = coaching["triggering_conversation_id"]
            issued_at = coaching["issued_at"]

            # The triggering conversation must appear in the trajectory.
            if trigger_cid not in conv_scored_at:
                v15.append(
                    f"{agent_id}: coaching_id={coaching['coaching_id']!r} "
                    f"triggering_conversation_id={trigger_cid!r} "
                    f"not found in score_trajectory"
                )
                continue

            # At least one trajectory row for that conversation must predate issued_at.
            any_before = any(
                scored_at < issued_at for scored_at in conv_scored_at[trigger_cid]
            )
            if not any_before:
                v15.append(
                    f"{agent_id}: coaching_id={coaching['coaching_id']!r} "
                    f"issued_at={issued_at} but all trajectory rows for "
                    f"{trigger_cid!r} are at or after that timestamp"
                )

        # --- Assertion 16: post_coaching_of references a known coaching_id ---
        for row in traj_rows:
            post_of = row.get("post_coaching_of")
            if post_of is not None and post_of not in coaching_ids:
                v16.append(
                    f"{agent_id}: trajectory_id={row['trajectory_id']!r} "
                    f"post_coaching_of={post_of!r} not in coaching_history"
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

    # Assertion 19: cross-tenant corrections must not use real agent IDs
    v19: list[str] = []
    for i, row in enumerate(raw_corrections):
        aid = row["agent_id"]
        if aid.startswith(_CROSS_TENANT_PREFIX) and aid in real_agent_ids:
            v19.append(
                f"correction[{i}] correction_id={row['correction_id']!r} "
                f"agent_id={aid!r} starts with cross-tenant prefix but is also "
                f"a real agent ID — tenant isolation is broken"
            )

    assert not v19, (
        f"Assertion 19 — {len(v19)} cross-tenant correction(s) reference real agents. "
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
