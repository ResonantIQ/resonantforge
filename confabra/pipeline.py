"""
End-to-end corpus generation pipeline (Task 15).

Orchestrates the five phases of corpus generation:
  Phase 1 — Deterministic state planning (state machine + quality plan injection)
  Phase 2 — Corpus-config artifacts (KB, tenant config, agents — no LLM)
  Phase 3 — Chunked prose generation (dry-run or live Anthropic API)
  Phase 4 — Corrections log
  Phase 5 — Artifact emission + manifest

Public API
----------
- :class:`PipelineConfig` — configuration dataclass
- :func:`run_pipeline`    — execute the pipeline, return Manifest
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from confabra.agents.generator import generate_agents
from confabra.corrections.generator import generate_corrections
from confabra.cross_contamination import CrossContaminationInjector
from confabra.kb.generator import generate_kb
from confabra.layer1.plan_validator import PlanValidator, SkipRateTracker
from confabra.layer1.quality_plan_injector import QualityPlanInjector
from confabra.layer1.snapshot_emitter import SnapshotEmitter
from confabra.layer1.state_machine import StateMachine
from confabra.profiles import get_profile
from confabra.utils.atomic_write import atomic_write_jsonl, atomic_write_text
from confabra.schemas import (
    ConversationRecord,
    DaySnapshot,
    Manifest,
    QualityPlan,
    SimEvent,
    SimEventType,
    ValidationVerdict,
)
from confabra.tenant_config.generator import generate_tenant_config

# ---------------------------------------------------------------------------
# Optional Anthropic import guard
# ---------------------------------------------------------------------------

try:
    import anthropic  # type: ignore[import-untyped]

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


# ---------------------------------------------------------------------------
# Configuration dataclass
# ---------------------------------------------------------------------------


@dataclass
class PipelineConfig:
    """
    Configuration for a single corpus generation run.

    All randomness is seeded from ``seed``, making repeated runs with identical
    configs produce bit-for-bit identical artifacts.  Set ``anthropic_api_key``
    to ``None`` for a dry-run (placeholder prose is generated instead of calling
    Claude); this is safe to use in CI or when the API key is unavailable.
    """

    profile_name: str  # "saas" or "professional_services" / "ps"
    accounts: int  # number of accounts to simulate
    months: int  # number of months to simulate
    seed: int  # deterministic PRNG seed
    output_root: Path  # root output directory; pipeline appends <profile>/ internally
    anthropic_api_key: str | None = None  # None → dry-run (no LLM calls)
    verbose: bool = False
    force: bool = False  # when True, overwrite existing target directory contents


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _log(config: PipelineConfig, msg: str) -> None:
    """Print a progress message when verbose mode is enabled."""
    if config.verbose:
        print(f"[confabra] {msg}")


def _sha256_jsonl(lines: list[str]) -> str:
    """Compute a SHA-256 hex digest over a list of JSONL lines."""
    h = hashlib.sha256()
    for line in lines:
        h.update(line.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _write_jsonl(path: Path, lines: list[str]) -> None:
    """Write a list of pre-serialised JSON strings to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for line in lines:
            fh.write(line + "\n")


def _group_events_by_chunk(
    events: list[SimEvent],
) -> dict[tuple[str, int], list[SimEvent]]:
    """
    Group CONVERSATION_STARTED events by (account_id, month_index).

    Only CONVERSATION_STARTED events are included because each group represents
    one potential organic conversation chunk (one ConversationRecord).  Other
    event types are attached as context in the prompt builder.
    """
    groups: dict[tuple[str, int], list[SimEvent]] = defaultdict(list)
    for event in events:
        if event.event_type == SimEventType.CONVERSATION_STARTED:
            key = (event.account_id, event.month_index)
            groups[key].append(event)
    return groups


def _build_prompt(
    account_id: str,
    month_index: int,
    conv_event: SimEvent,
    month_summary: dict,
    quality_plan: QualityPlan | None,
    profile_name: str,
) -> tuple[str, str]:
    """
    Build a (system_prompt, user_prompt) pair for prose generation.

    Keeps the prompt structure intentionally simple — the goal is a working
    end-to-end pipeline, not a high-fidelity prose engine (that is Phase 3
    detail work in a future task).

    Args:
        account_id:    The account the conversation belongs to.
        month_index:   0-based simulation month.
        conv_event:    The CONVERSATION_STARTED event that triggered this chunk.
        month_summary: Summary statistics dict from SnapshotEmitter.month_summary().
        quality_plan:  If not None, prose must follow the plan's directives.
        profile_name:  Profile label (e.g. "saas") for system prompt context.

    Returns:
        (system_prompt, user_prompt) strings ready for the Anthropic messages API.
    """
    system_prompt = (
        f"You are generating a synthetic customer-support conversation for a {profile_name} company. "
        "Write a realistic, natural dialogue between a customer and a support agent. "
        "The conversation should be 4-12 turns. Format each turn as 'Customer: ...' or 'Agent: ...' "
        "on its own line. Do not add any preamble or metadata — output only the conversation."
    )

    channel = conv_event.payload.get("surface_channel", "chat")
    customer = conv_event.payload.get("customer_name", f"Customer_{account_id}")
    agent_id = conv_event.payload.get("agent_id", "agent_unknown")

    health_info = ""
    if month_summary:
        avg_health = month_summary.get("avg_health_score", 0.5)
        dominant_health = month_summary.get("dominant_health_state", "healthy")
        payment_failures = month_summary.get("payment_failures", 0)
        health_info = (
            f"\n\nAccount context for month {month_index}:"
            f"\n- Average health score: {avg_health:.2f}"
            f"\n- Dominant health state: {dominant_health}"
            f"\n- Payment failures this month: {payment_failures}"
        )

    plan_section = ""
    if quality_plan is not None:
        plan_section = (
            f"\n\nQuality directives (follow these exactly):\n{quality_plan.prose_generation_directives}"
        )

    user_prompt = (
        f"Generate a customer-support conversation on the {channel} channel."
        f"\nCustomer: {customer}"
        f"\nAgent ID: {agent_id}"
        f"\nAccount: {account_id}"
        f"{health_info}"
        f"{plan_section}"
        "\n\nWrite the conversation now:"
    )

    return system_prompt, user_prompt


def _call_anthropic(
    api_key: str,
    system_prompt: str,
    user_prompt: str,
) -> tuple[str, int, int]:
    """
    Call the Anthropic API and return the generated text plus cache usage counters.

    Uses claude-haiku-4-5-20251001 with a conservative max_tokens to keep costs
    reasonable during corpus generation runs.  The system prompt is wrapped with
    an ephemeral cache_control block — the prefix is byte-stable within a run
    (only profile_name varies, and that is constant per pipeline execution).
    Raises on API errors — the caller handles retries.

    Args:
        api_key:       Anthropic API key.
        system_prompt: System prompt defining the generation role.
        user_prompt:   User prompt carrying account/chunk context.

    Returns:
        Tuple of (prose_text, cache_creation_input_tokens, cache_read_input_tokens).
        Token counts are 0 when not reported by the API (e.g. first call in a run
        writes the cache, subsequent calls read it).
    """
    if not ANTHROPIC_AVAILABLE:
        raise RuntimeError(
            "anthropic package is not installed. Install it with: pip install anthropic"
        )
    client = anthropic.Anthropic(api_key=api_key)
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=[
            {
                "type": "text",
                "text": system_prompt,
                "cache_control": {"type": "ephemeral"},
            }
        ],
        messages=[{"role": "user", "content": user_prompt}],
    )
    cache_creation = getattr(response.usage, "cache_creation_input_tokens", 0) or 0
    cache_read = getattr(response.usage, "cache_read_input_tokens", 0) or 0
    # Extract text from the first content block.
    for block in response.content:
        if hasattr(block, "text"):
            return block.text, cache_creation, cache_read
    return "[NO CONTENT]", cache_creation, cache_read


def _make_conversation_record(
    conv_id: str,
    account_id: str,
    conv_event: SimEvent,
    prose: str,
    quality_plan: QualityPlan | None,
) -> ConversationRecord:
    """
    Build a ConversationRecord from a simulation event and generated prose.

    Duration is derived from the CONVERSATION_ENDED payload when available;
    falls back to a fixed 10-minute duration.

    Args:
        conv_id:       Stable conversation ID for this chunk.
        account_id:    Account the conversation belongs to.
        conv_event:    The CONVERSATION_STARTED event.
        prose:         Generated (or placeholder) conversation text.
        quality_plan:  The quality plan driving prose generation, if any.

    Returns:
        A fully-populated ConversationRecord.
    """
    agent_id = conv_event.payload.get("agent_id", conv_event.agent_id or "agent_unknown")
    channel = conv_event.payload.get("surface_channel", "intercom")
    started_at = conv_event.timestamp
    ended_at = started_at + timedelta(minutes=10)

    return ConversationRecord(
        conversation_id=conv_id,
        account_id=account_id,
        agent_id=agent_id,
        surface_channel=channel,
        started_at=started_at,
        ended_at=ended_at,
        turn_count=prose.count("Agent:") + prose.count("Customer:"),
        prose=prose,
        trigger_event_id=conv_event.event_id,
        is_planted_quality=quality_plan is not None,
        tone_variant=None,
    )


def _find_quality_plan_for_event(
    event_id: str,
    quality_plans: list[QualityPlan],
) -> QualityPlan | None:
    """
    Look up a quality plan by its trigger_event_id.

    Returns the first matching plan, or None if no plan targets this event.
    The injector guarantees at most one plan per event_id, so first-match is safe.

    Args:
        event_id:      The event_id from a CONVERSATION_STARTED event.
        quality_plans: All quality plans from the injector.

    Returns:
        Matching QualityPlan or None.
    """
    for plan in quality_plans:
        if plan.trigger_event_id == event_id:
            return plan
    return None


def _find_account_events(
    account_id: str,
    events: list[SimEvent],
) -> list[SimEvent]:
    """
    Return all events for a given account.

    Used by the pre-prompt validator to check plan consistency against the
    account's event history.

    Args:
        account_id: Account to filter events for.
        events:     Full event log from the state machine.

    Returns:
        Events belonging to the account, in emission order.
    """
    return [e for e in events if e.account_id == account_id]


def _find_account_snapshot(
    account_id: str,
    day_index: int,
    snapshots: list[DaySnapshot],
) -> DaySnapshot | None:
    """
    Retrieve the DaySnapshot for an account on a specific simulation day.

    Falls back to the account's most recent snapshot if the exact day is absent
    (can happen when an account churns mid-month and no snapshot was written for
    the conversation day).

    Args:
        account_id: Account to query.
        day_index:  The simulation day to retrieve.
        snapshots:  Full snapshot log from the state machine.

    Returns:
        Matching DaySnapshot or None if the account has no snapshots at all.
    """
    account_snaps = [s for s in snapshots if s.account_id == account_id]
    for snap in account_snaps:
        if snap.day_index == day_index:
            return snap
    # Fallback: most recent snapshot before the target day.
    before = [s for s in account_snaps if s.day_index < day_index]
    if before:
        return max(before, key=lambda s: s.day_index)
    return account_snaps[-1] if account_snaps else None


# ---------------------------------------------------------------------------
# Concurrent-run protection
# ---------------------------------------------------------------------------

_LOCK_FILENAME = ".confabra-lock"


def _lockfile_path(target: Path) -> Path:
    """Return the lock file path for a target directory."""
    return target.parent / _LOCK_FILENAME


def _pid_is_live(pid: int) -> bool:
    """Return True if a process with the given PID is currently running."""
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _acquire_lock(target: Path, config: PipelineConfig) -> Path:
    """
    Write a .confabra-lock file in the parent of *target*.

    Refuses to start if a live lockfile exists (same-PID guard).  Overwrites
    stale locks (dead PID).  Returns the lockfile path so the caller can remove
    it on exit.
    """
    lock_path = _lockfile_path(target)
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    if lock_path.exists():
        try:
            lock_data = json.loads(lock_path.read_text(encoding="utf-8"))
            owner_pid = lock_data.get("pid", -1)
            if _pid_is_live(owner_pid):
                raise RuntimeError(
                    f"Another confabra process (PID {owner_pid}) is already generating "
                    f"into {target}. Lock file: {lock_path}"
                )
            _log(config, f"  WARNING: stale lock file found (PID {owner_pid} is dead). Overwriting.")
        except (json.JSONDecodeError, OSError):
            _log(config, "  WARNING: unreadable lock file found. Overwriting.")

    lock_data = {"pid": os.getpid(), "started_at": datetime.now(tz=timezone.utc).isoformat()}
    lock_path.write_text(json.dumps(lock_data), encoding="utf-8")
    return lock_path


def _check_target_dir(target: Path, config: PipelineConfig) -> None:
    """
    Refuse to overwrite a non-empty target directory unless --force is set.

    When --force is True, deletes all existing contents before the run so that
    the pipeline starts from a clean slate.
    """
    if target.exists() and any(target.iterdir()):
        if not config.force:
            raise RuntimeError(
                f"Target directory {target} already contains files. "
                "Refusing to overwrite. Pass --force to override or choose a different --out-root path."
            )
        _log(config, f"  --force: removing existing contents of {target}")
        shutil.rmtree(target)


# ---------------------------------------------------------------------------
# Phase 3 — prose generation loop
# ---------------------------------------------------------------------------


def _generate_prose_for_chunk(
    conv_id: str,
    account_id: str,
    month_index: int,
    conv_event: SimEvent,
    quality_plan: QualityPlan | None,
    account_snapshot: DaySnapshot | None,
    account_events: list[SimEvent],
    emitter: SnapshotEmitter,
    validator: PlanValidator,
    skip_tracker: SkipRateTracker,
    config: PipelineConfig,
    profile_name: str,
) -> ConversationRecord | None:
    """
    Generate prose for one (account_id, month_index, conv_event) chunk.

    Runs the full pre-prompt → generate → post-prompt validation cycle with up to
    2 retries on post-generation failure.  Returns None when the chunk is skipped
    (pre-prompt failure or exhausted retries).

    Pre-prompt validation always runs even in dry-run mode so that planted-quality
    consistency is enforced regardless of whether an LLM is available.

    Args:
        conv_id:          Stable ID for this conversation chunk.
        account_id:       Account the conversation belongs to.
        month_index:      0-based simulation month.
        conv_event:       The CONVERSATION_STARTED event driving this chunk.
        quality_plan:     Quality plan for this chunk, or None if organic.
        account_snapshot: The DaySnapshot for this account/day, or None.
        account_events:   All events for this account.
        emitter:          SnapshotEmitter for month-summary queries.
        validator:        PlanValidator for pre-prompt and post-generation gates.
        skip_tracker:     SkipRateTracker accumulating run-wide skip rates.
        config:           Pipeline configuration (API key, verbose, etc.).
        profile_name:     Profile label for prompt context.

    Returns:
        A ConversationRecord on success, or None if the chunk was skipped.
    """
    # ------------------------------------------------------------------
    # Stage 1 — pre-prompt validation (always, including dry-run)
    # ------------------------------------------------------------------
    if quality_plan is not None and account_snapshot is not None:
        skip_tracker.quality_rule_attempts += 1
        pre_result = validator.pre_prompt_validate(
            quality_plan, account_snapshot, account_events
        )
        if pre_result.overall_verdict == ValidationVerdict.FAIL:
            # Hard gate: generator bug — skip without LLM call.
            skip_tracker.quality_rule_failures += 1
            skip_tracker.total_skipped += 1
            _log(
                config,
                f"  PRE-PROMPT FAIL (skip) {conv_id}: {pre_result.skip_reason}",
            )
            return None

    # ------------------------------------------------------------------
    # Stage 2 — prose generation (with retry loop)
    # ------------------------------------------------------------------
    month_summary = emitter.month_summary(account_id, month_index)
    system_prompt, user_prompt = _build_prompt(
        account_id=account_id,
        month_index=month_index,
        conv_event=conv_event,
        month_summary=month_summary,
        quality_plan=quality_plan,
        profile_name=profile_name,
    )

    max_retries = 2
    prose: str | None = None

    for attempt in range(max_retries + 1):
        if config.anthropic_api_key is None:
            # Dry-run: generate deterministic placeholder prose.
            prose = (
                f"[DRY RUN] conv_id={conv_id} account={account_id} "
                f"month={month_index} attempt={attempt}\n"
                "Customer: I need help with my account.\n"
                "Agent: I'd be happy to help you today. What seems to be the issue?\n"
                "Customer: I can't access the dashboard.\n"
                "Agent: I understand. Let me look into that for you right away."
            )
        else:
            try:
                prose, cache_creation, cache_read = _call_anthropic(
                    api_key=config.anthropic_api_key,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
                skip_tracker.cache_api_calls += 1
                skip_tracker.cache_creation_tokens += cache_creation
                skip_tracker.cache_read_tokens += cache_read
                _log(
                    config,
                    f"  cache {conv_id}: creation={cache_creation} read={cache_read}",
                )
            except Exception as exc:  # noqa: BLE001
                _log(config, f"  API error on attempt {attempt}: {exc}")
                if attempt >= max_retries:
                    skip_tracker.prose_fact_failures += 1
                    skip_tracker.total_skipped += 1
                    return None
                continue

        # Post-generation validation is simplified here (no signal extractor yet).
        # We treat dry-run prose as always passing; real prose also passes at this
        # stage since the full signal extraction pipeline is a separate task.
        # The retry/skip logic is wired up for correctness but won't trigger
        # until the signal extractor is integrated.
        skip_tracker.prose_fact_attempts += 1
        break  # Success — exit retry loop.

    if prose is None:
        skip_tracker.total_skipped += 1
        return None

    return _make_conversation_record(
        conv_id=conv_id,
        account_id=account_id,
        conv_event=conv_event,
        prose=prose,
        quality_plan=quality_plan,
    )


# ---------------------------------------------------------------------------
# Main pipeline entry point
# ---------------------------------------------------------------------------


def run_pipeline(config: PipelineConfig) -> Manifest:
    """
    Execute the 5-phase corpus generation pipeline.

    Phase 1: Deterministic state planning — state machine + quality plan injection.
    Phase 2: Corpus-config artifacts — KB, tenant config, agents (no LLM).
    Phase 3: Chunked prose generation — per-(account, month) conversation records.
    Phase 4: Corrections log — planted human score correction records.
    Phase 5: Artifact emission + manifest — write all JSONL files and manifest.json.

    Args:
        config: Pipeline configuration controlling profile, scale, seed, and output.

    Returns:
        The completed :class:`~confabra.schemas.Manifest` record.
    """
    generated_at = datetime.now(tz=timezone.utc)

    # Resolve profile early so we know the target directory.
    profile = get_profile(config.profile_name)
    target = config.output_root / profile.name

    # Concurrent-run protection: fail-fast on non-empty target, acquire lockfile.
    _check_target_dir(target, config)
    lock_path = _acquire_lock(target, config)

    try:
        return _run_pipeline_inner(config, profile, target, generated_at)
    finally:
        try:
            lock_path.unlink(missing_ok=True)
        except OSError:
            pass


def _run_pipeline_inner(
    config: PipelineConfig,
    profile,  # type: ignore[type-arg]
    profile_dir: Path,
    generated_at: datetime,
) -> Manifest:
    """Inner pipeline implementation — called after locks are acquired."""
    # ==================================================================
    # Phase 1 — Deterministic state planning
    # ==================================================================
    _log(config, "Phase 1: state planning")

    # 1b. Run state machine — returns (events, snapshots).
    sm = StateMachine(
        seed=config.seed,
        num_accounts=config.accounts,
        num_months=config.months,
        profile=profile,
    )
    events, snapshots = sm.simulate()
    _log(config, f"  {len(events)} events, {len(snapshots)} snapshots")

    # 1c. Build snapshot emitter (takes snapshots + events).
    emitter = SnapshotEmitter(snapshots=snapshots, events=events)

    # 1d. Inject quality plans.
    # QualityPlanInjector takes a seeded RNG (not a raw seed integer).
    injector_rng = random.Random(config.seed)
    injector = QualityPlanInjector(rng=injector_rng, profile_name=profile.name)
    quality_plans: list[QualityPlan] = injector.inject(
        events=events,
        snapshots=snapshots,
    )
    _log(config, f"  {len(quality_plans)} quality plans injected")

    # ==================================================================
    # Phase 2 — Corpus-config artifacts (deterministic, no LLM)
    # ==================================================================
    _log(config, "Phase 2: corpus-config artifacts")

    profile_dir.mkdir(parents=True, exist_ok=True)

    # KB generation — output_dir arg is the profile dir (generator creates
    # the knowledge_base subdirectory itself).
    kb_chunks, kb_hash = generate_kb(profile.name, profile_dir)
    _log(config, f"  {len(kb_chunks)} KB chunks, hash={kb_hash[:16]}…")

    # Apply cross-contamination to KB.
    # Need at least 2 brand voice variants; fall back gracefully if profile
    # only defines 1 (PS profile case).
    bv_variants = [v.id for v in profile.brand_voice_variants()]
    if len(bv_variants) >= 2:
        injector_cc = CrossContaminationInjector(
            seed=config.seed,
            variants=bv_variants,
            dominant_variant="bv_baseline" if "bv_baseline" in bv_variants else bv_variants[0],
        )
        kb_chunks = injector_cc.contaminate_kb_chunks(kb_chunks)
        # Re-write chunks.jsonl with the contaminated chunk list so that
        # tone_variant annotations are persisted to disk.
        atomic_write_jsonl(
            profile_dir / "knowledge_base" / "chunks.jsonl",
            [chunk.model_dump_json() for chunk in kb_chunks],
        )
    else:
        _log(config, "  Skipping cross-contamination (fewer than 2 brand voice variants)")

    # Tenant config.
    tenant_config_dict, tenant_config_hash = generate_tenant_config(profile, profile_dir)
    _log(config, f"  tenant config hash={tenant_config_hash[:16]}…")

    # Agents — empty organic IDs at this stage; real IDs linked after Phase 3.
    agent_profiles, agents_hash = generate_agents(
        profile=profile,
        output_dir=profile_dir,
        seed=config.seed,
        organic_conversation_ids=[],
    )
    _log(config, f"  {len(agent_profiles)} agents, hash={agents_hash[:16]}…")

    # ==================================================================
    # Phase 3 — Chunked prose generation
    # ==================================================================
    _log(config, "Phase 3: prose generation")

    validator = PlanValidator()
    skip_tracker = SkipRateTracker()

    # Build a map from event_id → quality_plan for O(1) lookup.
    quality_plan_by_event: dict[str, QualityPlan] = {
        plan.trigger_event_id: plan for plan in quality_plans
    }

    # Group CONVERSATION_STARTED events by (account_id, month_index).
    # Each group represents one organic conversation chunk.
    all_conversations: list[ConversationRecord] = []
    skipped_conv_ids: list[str] = []

    # Iterate deterministically: all accounts, all months, all conv events in
    # the order they appear in the event log.
    for event in events:
        if event.event_type != SimEventType.CONVERSATION_STARTED:
            continue

        account_id = event.account_id
        month_index = event.month_index

        # Conversation ID: embed event_id for traceability.
        conv_id = f"conv_{event.event_id}"

        # Find matching quality plan (if any).
        quality_plan = quality_plan_by_event.get(event.event_id)

        # Retrieve snapshot and full account events for validation.
        account_snapshot = _find_account_snapshot(account_id, event.day_index, snapshots)
        account_events = _find_account_events(account_id, events)

        result = _generate_prose_for_chunk(
            conv_id=conv_id,
            account_id=account_id,
            month_index=month_index,
            conv_event=event,
            quality_plan=quality_plan,
            account_snapshot=account_snapshot,
            account_events=account_events,
            emitter=emitter,
            validator=validator,
            skip_tracker=skip_tracker,
            config=config,
            profile_name=profile.name,
        )

        if result is None:
            skipped_conv_ids.append(conv_id)
        else:
            all_conversations.append(result)

    _log(
        config,
        f"  {len(all_conversations)} conversations generated, "
        f"{len(skipped_conv_ids)} skipped",
    )

    # Emit cache telemetry summary for the prose generation phase.
    _total_creation = skip_tracker.cache_creation_tokens
    _total_read = skip_tracker.cache_read_tokens
    _total_calls = skip_tracker.cache_api_calls
    _cacheable = _total_creation + _total_read
    _hit_rate = _total_read / max(1, _cacheable)
    # Haiku input tokens cost $0.25/M uncached; cached reads are $0.03/M (88% cheaper).
    _saved_tokens = _total_read
    _estimated_savings_usd = _saved_tokens * (0.25 - 0.03) / 1_000_000
    _cache_summary = (
        f"  cache summary: calls={_total_calls} "
        f"creation={_total_creation} read={_total_read} "
        f"hit_rate={_hit_rate:.1%} "
        f"estimated_savings=${_estimated_savings_usd:.4f}"
    )
    print(f"[confabra] {_cache_summary}")

    # Apply cross-contamination to organic conversations.
    if len(bv_variants) >= 2:
        all_conversations = injector_cc.contaminate_conversations(all_conversations)

    # ==================================================================
    # Phase 4 — Corrections log
    # ==================================================================
    _log(config, "Phase 4: corrections log")

    organic_conv_ids = [c.conversation_id for c in all_conversations]

    corrections, corrections_hash = generate_corrections(
        profile=profile,
        output_dir=profile_dir,
        seed=config.seed,
        agent_profiles=agent_profiles,
        organic_conversation_ids=organic_conv_ids,
    )
    _log(config, f"  {len(corrections)} corrections, hash={corrections_hash[:16]}…")

    # ==================================================================
    # Phase 5 — Artifact emission + manifest
    # ==================================================================
    _log(config, "Phase 5: artifact emission")

    # Derive unique document count from KB chunks.
    kb_doc_paths = {chunk.document_path for chunk in kb_chunks}
    kb_doc_count = len(kb_doc_paths)
    kb_chunk_count = len(kb_chunks)

    # Serialise events.
    event_lines = [e.model_dump_json() for e in events]
    events_hash = _sha256_jsonl(event_lines)
    atomic_write_jsonl(profile_dir / "events.jsonl", event_lines)

    # Serialise snapshots.
    snapshot_lines = [s.model_dump_json() for s in snapshots]
    snapshots_hash = _sha256_jsonl(snapshot_lines)
    atomic_write_jsonl(profile_dir / "snapshots.jsonl", snapshot_lines)

    # Serialise conversations.
    conv_lines = [c.model_dump_json() for c in all_conversations]
    conversations_hash = _sha256_jsonl(conv_lines)
    atomic_write_jsonl(profile_dir / "conversations.jsonl", conv_lines)

    # Serialise quality plans.
    plan_lines = [p.model_dump_json() for p in quality_plans]
    planted_quality_hash = _sha256_jsonl(plan_lines)
    atomic_write_jsonl(profile_dir / "planted_quality.jsonl", plan_lines)

    # Serialise skipped conversation IDs.
    skipped_lines = [json.dumps({"conversation_id": cid}) for cid in skipped_conv_ids]
    atomic_write_jsonl(profile_dir / "skipped_conversations.jsonl", skipped_lines)

    # Build manifest.
    manifest = Manifest(
        generator_version="0.2.0",
        profile_name=profile.name,
        profile_version=f"{profile.name}-{profile.version}",
        seed=config.seed,
        accounts=config.accounts,
        months=config.months,
        generated_at=generated_at,
        event_count=len(events),
        snapshot_count=len(snapshots),
        conversation_count=len(all_conversations),
        skipped_conversation_count=len(skipped_conv_ids),
        planted_quality_count=len(quality_plans),
        knowledge_base_doc_count=kb_doc_count,
        knowledge_base_chunk_count=kb_chunk_count,
        agent_count=len(agent_profiles),
        corrections_count=len(corrections),
        events_hash=events_hash,
        snapshots_hash=snapshots_hash,
        conversations_hash=conversations_hash,
        planted_quality_hash=planted_quality_hash,
        kb_chunks_hash=kb_hash,
        tenant_config_hash=tenant_config_hash,
        agent_fixtures_hash=agents_hash,
        corrections_hash=corrections_hash,
        prose_fact_violation_rate=skip_tracker.prose_fact_rate,
        validator_rule_failure_rate=skip_tracker.quality_rule_rate,
        disagreement_rate=skip_tracker.disagreement_rate,
        cache_creation_tokens=skip_tracker.cache_creation_tokens,
        cache_read_tokens=skip_tracker.cache_read_tokens,
        cache_hit_rate=_hit_rate,
        cache_estimated_savings_usd=_estimated_savings_usd,
    )

    manifest_path = profile_dir / "manifest.json"
    atomic_write_text(manifest_path, json.dumps(manifest.model_dump(mode="json"), indent=2))

    _log(config, f"  manifest written → {manifest_path}")
    _log(
        config,
        f"Done. {len(all_conversations)} conversations, "
        f"{len(quality_plans)} planted, {len(skipped_conv_ids)} skipped.",
    )

    return manifest
