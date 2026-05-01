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
from typing import TYPE_CHECKING, Callable

from confabra.agents.generator import generate_agents
from confabra.corrections.generator import generate_corrections
from confabra.cross_contamination import CrossContaminationInjector
from confabra.kb.generator import generate_kb
from confabra.layer1.coverage_backfill import CoverageBackfill
from confabra.layer1.plan_validator import PlanValidator, SkipRateTracker
from confabra.layer1.quality_plan_injector import QualityPlanInjector
from confabra.layer1.snapshot_emitter import SnapshotEmitter
from confabra.layer1.state_machine import StateMachine
from confabra.profiles import get_profile
from confabra.utils.atomic_write import atomic_write_jsonl, atomic_write_text
from confabra.schemas import (
    ConversationRecord,
    ConversationSignals,
    DaySnapshot,
    GateSeverity,
    GateViolation,
    KBChunk,
    Manifest,
    QualityPlan,
    SimEvent,
    SimEventType,
    SkippedConversationRecord,
    ValidationVerdict,
)
from confabra.tenant_config.generator import generate_tenant_config
from confabra.validators.disagreement_ledger import DisagreementLedger, run_soft_judge
from confabra.validators.extractors.accuracy import extract_accuracy_signals
from confabra.validators.extractors.brand_voice import extract_brand_voice_signals
from confabra.validators.extractors.empathy import extract_empathy_signals
from confabra.validators.extractors.resolution import extract_resolution_signals
from confabra.validators.rule_engine import validate_all_dimensions

# ---------------------------------------------------------------------------
# Optional Anthropic import guard
# ---------------------------------------------------------------------------

try:
    import anthropic  # type: ignore[import-untyped]

    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

from rich.console import Console, Group
from rich.live import Live
from rich.text import Text


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
    client: "anthropic.Anthropic",
    system_prompt: str,
    user_prompt: str,
) -> tuple[str, int, int]:
    """
    Call the Anthropic API and return the generated text plus cache usage counters.

    Uses claude-haiku-4-5-20251001 with a conservative max_tokens to keep costs
    reasonable during corpus generation runs.  Raises on API errors — the caller
    handles retries.

    Prompt caching was evaluated and removed: the system prompt at ~83 tokens is
    25× below Haiku's 2048-token minimum cache threshold, so cache_control blocks
    were silently ignored by the API.  See docs/resonantforge/cache-diagnosis.md.
    The return shape still includes (text, 0, 0) so the cache telemetry shell in
    _run_pipeline_inner compiles without change — re-enabling caching only requires
    updating this function and crossing the token threshold.

    Args:
        client:        Pre-initialized Anthropic client (created once per run).
        system_prompt: System prompt defining the generation role.
        user_prompt:   User prompt carrying account/chunk context.

    Returns:
        Tuple of (prose_text, cache_creation_input_tokens, cache_read_input_tokens).
        Cache token counts are always 0 — caching is not active (see above).
    """
    if not ANTHROPIC_AVAILABLE:
        raise RuntimeError(
            "anthropic package is not installed. Install it with: pip install anthropic"
        )
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1024,
        system=system_prompt,
        messages=[{"role": "user", "content": user_prompt}],
    )
    # Caching removed — always return 0 for cache counters.
    # See function docstring for full rationale.
    cache_creation, cache_read = 0, 0
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
# Post-generation validation helpers
# ---------------------------------------------------------------------------


@dataclass
class LexiconsBundle:
    """
    All lexicon lists needed by the four post-generation signal extractors.

    Loaded once per call to _generate_prose_for_chunk via _load_lexicons() so
    that the extractors receive pure lists with no profile coupling.
    """

    acknowledgment_phrases: list[str]
    emotion_lexicon: list[str]
    apology_lexicon: list[str]
    action_verb_lexicon: list[str]
    hedging_lexicon: list[str]
    directive_lexicon: list[str]
    warm_terms: list[str]
    clinical_terms: list[str]
    contraction_patterns: list[str]
    resolution_patterns: list[str]
    deflection_patterns: list[str]
    next_steps_patterns: list[str]
    temporal_anchor_patterns: list[str]
    specific_actor_patterns: list[str]
    ownership_patterns: list[str]
    issue_keywords: list[str]
    synonym_map: dict[str, str]


def _split_prose_turns(prose: str) -> tuple[str, str]:
    """
    Split a full conversation into agent-only and customer-only strings.

    Splits on "Agent:" and "Customer:" line prefixes.  Returns concatenated
    agent turns and concatenated customer turns respectively.  Lines that don't
    start with a recognised speaker prefix are attributed to neither speaker.

    Args:
        prose: Full conversation text with "Agent: ..." and "Customer: ..." lines.

    Returns:
        (agent_prose, customer_prose) — each is a single string with speaker
        prefix lines joined by newlines.
    """
    agent_lines: list[str] = []
    customer_lines: list[str] = []
    for line in prose.splitlines():
        stripped = line.strip()
        if stripped.startswith("Agent:"):
            agent_lines.append(stripped[len("Agent:"):].strip())
        elif stripped.startswith("Customer:"):
            customer_lines.append(stripped[len("Customer:"):].strip())
    return "\n".join(agent_lines), "\n".join(customer_lines)


def _load_lexicons(profile) -> LexiconsBundle:  # type: ignore[type-arg]
    """
    Load all extractor lexicons from a profile's accessor methods.

    Calling this once per conversation avoids repeated attribute lookups on the
    profile object and provides a single typed bundle to pass to all extractors.

    Args:
        profile: A concrete Profile instance (SaaSProfile, PSProfile, etc.).

    Returns:
        LexiconsBundle with all lists populated.
    """
    return LexiconsBundle(
        acknowledgment_phrases=profile.acknowledgment_phrases(),
        emotion_lexicon=profile.emotion_lexicon(),
        apology_lexicon=profile.apology_lexicon(),
        action_verb_lexicon=profile.action_verb_lexicon(),
        hedging_lexicon=profile.hedging_lexicon(),
        directive_lexicon=profile.directive_lexicon(),
        warm_terms=profile.warm_terms(),
        clinical_terms=profile.clinical_terms(),
        contraction_patterns=profile.contraction_patterns(),
        resolution_patterns=profile.resolution_patterns(),
        deflection_patterns=profile.deflection_patterns(),
        next_steps_patterns=profile.next_steps_patterns(),
        temporal_anchor_patterns=profile.temporal_anchor_patterns(),
        specific_actor_patterns=profile.specific_actor_patterns(),
        ownership_patterns=profile.ownership_patterns(),
        issue_keywords=profile.issue_keywords(),
        synonym_map=profile.synonym_map(),
    )


def _build_conversation_signals(
    conv_id: str,
    empathy: "confabra.schemas.EmpathySignals",
    resolution: "confabra.schemas.ResolutionSignals",
    brand_voice: "confabra.schemas.BrandVoiceSignals",
    accuracy: "confabra.schemas.AccuracySignals",
) -> ConversationSignals:
    """
    Assemble the four dimension signal objects into a ConversationSignals record.

    Args:
        conv_id:    Conversation ID for provenance.
        empathy:    Output of extract_empathy_signals.
        resolution: Output of extract_resolution_signals.
        brand_voice: Output of extract_brand_voice_signals.
        accuracy:   Output of extract_accuracy_signals.

    Returns:
        ConversationSignals ready for post_generation_validate.
    """
    return ConversationSignals(
        conversation_id=conv_id,
        empathy=empathy,
        resolution=resolution,
        brand_voice=brand_voice,
        accuracy=accuracy,
    )


# ---------------------------------------------------------------------------
# Phase 3 — prose generation loop
# ---------------------------------------------------------------------------


def _make_skipped_record(
    conv_id: str,
    account_id: str,
    conv_event: SimEvent,
    quality_plan: QualityPlan | None,
    final_retry_count: int,
    validator_verdicts: list,
    prose: str | None,
) -> SkippedConversationRecord:
    """
    Build a SkippedConversationRecord from the state at a skip point.

    Args:
        conv_id:            Conversation ID being skipped.
        account_id:         Account the conversation belongs to.
        conv_event:         The CONVERSATION_STARTED event for this chunk.
        quality_plan:       Quality plan if planted, None if organic.
        final_retry_count:  Number of the attempt on which the skip occurred.
        validator_verdicts: Per-dimension verdicts from the last validation run.
        prose:              Generated prose from the last attempt, or None.

    Returns:
        SkippedConversationRecord with all available context populated.
    """
    quality_plan_summary: dict | None = None
    kb_chunks_required: list | None = None
    if quality_plan is not None:
        rt = quality_plan.rubric_targets
        quality_plan_summary = {
            "empathy": rt.empathy,
            "resolution": rt.resolution,
            "brand_voice_target": rt.brand_voice_target,
            "accuracy_status": rt.accuracy.status if rt.accuracy else None,
            "accuracy_precision": rt.accuracy.precision if rt.accuracy else None,
        }
        kb_chunks_required = quality_plan.kb_chunks_required

    return SkippedConversationRecord(
        conversation_id=conv_id,
        account_id=account_id,
        event_id=quality_plan.trigger_event_id if quality_plan else conv_event.event_id,
        quality_plan_summary=quality_plan_summary,
        final_retry_count=final_retry_count,
        final_verdicts=[
            {
                "dimension": v.dimension,
                "verdict": v.verdict.value if hasattr(v.verdict, "value") else str(v.verdict),
                "target": v.target,
                "signals_summary": v.signals_summary,
            }
            for v in validator_verdicts
        ],
        agent_prose_snippet=prose[:200] if prose else None,
        kb_chunks_required=kb_chunks_required,
        timestamp=datetime.now(tz=timezone.utc).isoformat(),
    )


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
    profile,  # type: ignore[type-arg]  # Profile instance
    anthropic_client: "anthropic.Anthropic | None",
    kb_chunks: list,
    ledger: DisagreementLedger,
    skipped_records: "list[SkippedConversationRecord]",
    progress_cb: Callable[[str], None] | None = None,
) -> ConversationRecord | None:
    """
    Generate prose for one (account_id, month_index, conv_event) chunk.

    Runs the full pre-prompt → generate → post-generation validation cycle with up
    to 2 retries on post-generation failure.  Returns None when the chunk is skipped
    (pre-prompt failure or exhausted retries after validation).

    Pre-prompt validation always runs even in dry-run mode so that planted-quality
    consistency is enforced regardless of whether an LLM is available.

    Post-generation validation (signal extraction + rule engine) runs only for
    planted conversations (quality_plan is not None).  Organic conversations are
    accepted on first generation attempt.

    Args:
        conv_id:           Stable ID for this conversation chunk.
        account_id:        Account the conversation belongs to.
        month_index:       0-based simulation month.
        conv_event:        The CONVERSATION_STARTED event driving this chunk.
        quality_plan:      Quality plan for this chunk, or None if organic.
        account_snapshot:  The DaySnapshot for this account/day, or None.
        account_events:    All events for this account.
        emitter:           SnapshotEmitter for month-summary queries.
        validator:         PlanValidator for pre-prompt and post-generation gates.
        skip_tracker:      SkipRateTracker accumulating run-wide skip rates.
        config:            Pipeline configuration (verbose flag, etc.).
        profile:           Concrete Profile instance for lexicon access.
        anthropic_client:  Pre-initialized Anthropic client, or None for dry-run.
        kb_chunks:         All KB chunks from Phase 2 (used by accuracy extractor).
        ledger:            DisagreementLedger accumulating soft-judge records.
        skipped_records:   Accumulator list; a SkippedConversationRecord is appended
                           here whenever this function returns None.

    Returns:
        A ConversationRecord on success, or None if the chunk was skipped.
    """
    def _progress(status: str) -> None:
        if progress_cb is not None:
            progress_cb(status)

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
            _progress("skipped: pre-prompt fail")
            skipped_records.append(_make_skipped_record(
                conv_id, account_id, conv_event, quality_plan,
                final_retry_count=0, validator_verdicts=[], prose=None,
            ))
            return None

    # ------------------------------------------------------------------
    # Stage 2 — prose generation with post-generation validation loop
    # ------------------------------------------------------------------
    month_summary = emitter.month_summary(account_id, month_index)
    system_prompt, user_prompt = _build_prompt(
        account_id=account_id,
        month_index=month_index,
        conv_event=conv_event,
        month_summary=month_summary,
        quality_plan=quality_plan,
        profile_name=profile.name,
    )

    max_retries = 2
    prose: str | None = None
    last_validator_verdicts: list = []  # updated after each validation run; used for skip records

    for attempt in range(max_retries + 1):
        if attempt > 0:
            _progress(f"retry {attempt} · generating prose")
        if anthropic_client is None:
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
                    client=anthropic_client,
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                )
                skip_tracker.cache_api_calls += 1
                skip_tracker.cache_creation_tokens += cache_creation
                skip_tracker.cache_read_tokens += cache_read
            except Exception as exc:  # noqa: BLE001
                _log(config, f"  API error on attempt {attempt}: {exc}")
                if attempt >= max_retries:
                    skip_tracker.prose_fact_failures += 1
                    skip_tracker.total_skipped += 1
                    _progress("skipped: API error")
                    skipped_records.append(_make_skipped_record(
                        conv_id, account_id, conv_event, quality_plan,
                        final_retry_count=attempt, validator_verdicts=last_validator_verdicts,
                        prose=prose,
                    ))
                    return None
                continue

        # ------------------------------------------------------------------
        # Stage 3 — post-generation validation (planted conversations only)
        # ------------------------------------------------------------------
        skip_tracker.prose_fact_attempts += 1

        if quality_plan is not None and anthropic_client is not None:
            # Post-generation validation only runs for live LLM-generated prose.
            # Dry-run placeholder prose is always accepted — it doesn't represent
            # real conversations so validating it would always fail and pollute
            # prose_fact_violation_rate metrics (per Section 7 test expectation).
            assert prose is not None
            agent_prose, customer_prose = _split_prose_turns(prose)
            lexicons = _load_lexicons(profile)

            _progress("validating empathy")
            empathy_signals = extract_empathy_signals(
                agent_prose,
                lexicons.acknowledgment_phrases,
                lexicons.emotion_lexicon,
                lexicons.apology_lexicon,
                lexicons.action_verb_lexicon,
            )
            _progress("validating resolution")
            resolution_signals = extract_resolution_signals(
                agent_prose,
                customer_prose,
                lexicons.resolution_patterns,
                lexicons.deflection_patterns,
                lexicons.next_steps_patterns,
                lexicons.temporal_anchor_patterns,
                lexicons.specific_actor_patterns,
                lexicons.ownership_patterns,
                lexicons.issue_keywords,
            )
            _progress("validating brand_voice")
            brand_voice_signals = extract_brand_voice_signals(
                agent_prose,
                lexicons.hedging_lexicon,
                lexicons.directive_lexicon,
                lexicons.warm_terms,
                lexicons.clinical_terms,
                lexicons.contraction_patterns,
            )
            _progress("validating accuracy")
            accuracy_signals = extract_accuracy_signals(
                agent_prose,
                customer_prose,
                kb_chunks,
                quality_plan.kb_chunks_required,
                lexicons.synonym_map,
                anthropic_client=anthropic_client,
            )

            signals = _build_conversation_signals(
                conv_id, empathy_signals, resolution_signals, brand_voice_signals, accuracy_signals
            )

            # OQ2: use brand_voice_against from quality plan, default to bv_baseline.
            variant_id = quality_plan.rubric_targets.brand_voice_against or "bv_baseline"
            feature_profiles = profile.brand_voice_feature_profiles()

            validator_verdicts = validate_all_dimensions(
                empathy_signals,
                resolution_signals,
                brand_voice_signals,
                accuracy_signals,
                quality_plan.rubric_targets,
                variant_id,
                feature_profiles,
            )
            last_validator_verdicts = validator_verdicts

            # Soft-judge on FAIL verdicts — non-gating, diagnostic only.
            for verdict in validator_verdicts:
                if verdict.verdict == ValidationVerdict.FAIL:
                    record = run_soft_judge(
                        conv_id,
                        agent_prose,
                        verdict.dimension,
                        verdict.target,
                        anthropic_client,
                    )
                    ledger.add_record(record)

            post_result = validator.post_generation_validate(
                quality_plan,
                prose,
                signals,
                validator_verdicts,
                retry_count=attempt,
            )

            if post_result.overall_verdict == ValidationVerdict.PASS:
                _progress("passed")
                break
            elif post_result.overall_verdict == ValidationVerdict.SKIP:
                skip_tracker.prose_fact_failures += 1
                skip_tracker.total_skipped += 1
                _log(config, f"  POST-GEN SKIP {conv_id}: {post_result.skip_reason}")
                _progress(f"skipped: {post_result.skip_reason or 'validation'}")
                skipped_records.append(_make_skipped_record(
                    conv_id, account_id, conv_event, quality_plan,
                    final_retry_count=attempt, validator_verdicts=validator_verdicts,
                    prose=prose,
                ))
                return None
            else:  # FAIL — retry on next attempt
                _log(config, f"  POST-GEN FAIL (attempt {attempt + 1}) {conv_id}")
                continue
        else:
            # Organic conversation, or dry-run mode — no post-generation validation.
            _progress("passed")
            break
    else:
        # Retry loop exhausted without break (should not occur — SKIP path returns above).
        skip_tracker.prose_fact_failures += 1
        skip_tracker.total_skipped += 1
        _progress("skipped: retries exhausted")
        skipped_records.append(_make_skipped_record(
            conv_id, account_id, conv_event, quality_plan,
            final_retry_count=max_retries, validator_verdicts=last_validator_verdicts,
            prose=prose,
        ))
        return None

    if prose is None:
        skip_tracker.total_skipped += 1
        _progress("skipped: no prose generated")
        skipped_records.append(_make_skipped_record(
            conv_id, account_id, conv_event, quality_plan,
            final_retry_count=0, validator_verdicts=[], prose=None,
        ))
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

    # Create the profile output directory early so KB generation (which runs
    # before quality plan injection) can write its subdirectory.
    profile_dir.mkdir(parents=True, exist_ok=True)

    # 1a. Build coverage backfill from the profile's KB chunks.
    # We call profile.knowledge_base_content() directly (not generate_kb) so
    # we can derive authored cells before the state machine runs — generate_kb
    # writes to disk but the chunk objects are identical to what the profile
    # already owns.  Any profile that returns [] here gets no backfill (which
    # is safe — the state machine falls back to pure weighted-random).
    _kb_chunks_for_cells = getattr(profile, "knowledge_base_content", lambda: [])()
    _authored_cells: list[tuple[str, str]] = []
    _seen_cells: set[tuple[str, str]] = set()
    for _chunk in _kb_chunks_for_cells:
        _gate = getattr(_chunk, "cat11_gate", None)
        if not _gate:
            continue
        for _dom in getattr(_chunk, "domains", []):
            _cell = (_dom, _gate)
            if _cell not in _seen_cells:
                _authored_cells.append(_cell)
                _seen_cells.add(_cell)

    _cell_min_overrides: dict[tuple[str, str], int] = getattr(
        profile, "cell_min_events_overrides", lambda: {}
    )()
    backfill = CoverageBackfill(
        cells=_authored_cells,
        min_events_override=_cell_min_overrides if _cell_min_overrides else None,
    )
    _log(config, f"  {len(_authored_cells)} authored (domain×gate) cells registered for backfill")

    # 1b. Run state machine — returns (events, snapshots).
    sm = StateMachine(
        seed=config.seed,
        num_accounts=config.accounts,
        num_months=config.months,
        profile=profile,
        backfill=backfill,
    )
    events, snapshots = sm.simulate()
    _log(config, f"  {len(events)} events, {len(snapshots)} snapshots")

    # 1c. Build snapshot emitter (takes snapshots + events).
    emitter = SnapshotEmitter(snapshots=snapshots, events=events)

    # 1e. Generate KB chunks now (before injection) so that domain-based chunk
    # selection in the injector can reference real chunk metadata.  The chunks
    # list is generated once here and re-used in Phase 2 for serialisation —
    # no second generation pass occurs.
    kb_chunks, kb_hash = generate_kb(profile.name, profile_dir)
    _log(config, f"  {len(kb_chunks)} KB chunks pre-generated for injection, hash={kb_hash[:16]}…")

    # 1d. Inject quality plans — passes real KB chunks so domain-based selection works.
    # QualityPlanInjector takes a seeded RNG (not a raw seed integer).
    injector_rng = random.Random(config.seed)
    injector = QualityPlanInjector(rng=injector_rng, profile_name=profile.name)
    quality_plans: list[QualityPlan] = injector.inject(
        events=events,
        snapshots=snapshots,
        kb_chunks=kb_chunks,
    )
    _log(config, f"  {len(quality_plans)} quality plans injected")

    # ==================================================================
    # Phase 2 — Corpus-config artifacts (deterministic, no LLM)
    # ==================================================================
    _log(config, "Phase 2: corpus-config artifacts")

    # KB chunks were already generated in Phase 1 (before injection).
    # profile_dir was also created in Phase 1.
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

    # Create a single Anthropic client for all LLM calls in Phase 3 (prose
    # generation + accuracy extraction).  None in dry-run mode.
    if ANTHROPIC_AVAILABLE and config.anthropic_api_key is not None:
        anthropic_client = anthropic.Anthropic(api_key=config.anthropic_api_key)
    else:
        anthropic_client = None

    # Disagreement ledger: accumulates soft-judge records from every FAIL verdict.
    ledger = DisagreementLedger()

    # Build a map from event_id → quality_plan for O(1) lookup.
    quality_plan_by_event: dict[str, QualityPlan] = {
        plan.trigger_event_id: plan for plan in quality_plans
    }

    # Group CONVERSATION_STARTED events by (account_id, month_index).
    # Each group represents one organic conversation chunk.
    all_conversations: list[ConversationRecord] = []
    skipped_records: list[SkippedConversationRecord] = []

    # Phase 3 progress display setup.
    total_convs = sum(1 for e in events if e.event_type == SimEventType.CONVERSATION_STARTED)
    _p3_passed = 0
    _p3_skipped = 0
    _progress_console = Console()
    _is_tty = _progress_console.is_terminal

    def _p3_header_text() -> Text:
        pending = total_convs - _p3_passed - _p3_skipped
        return Text(
            f"  [{_p3_passed}/{total_convs}] passed · "
            f"[{_p3_skipped}/{total_convs}] skipped · "
            f"[{pending}/{total_convs}] pending"
        )

    # Iterate deterministically: all accounts, all months, all conv events in
    # the order they appear in the event log.
    with Live(console=_progress_console, refresh_per_second=10) as live:
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

            conv_index = _p3_passed + _p3_skipped + 1

            def _make_progress_cb(
                _idx: int = conv_index,
                _cid: str = conv_id,
                _aid: str = account_id,
            ) -> Callable[[str], None]:
                def _cb(status: str) -> None:
                    line = f"[{_idx}/{total_convs}] {_cid} · {_aid} · {status}"
                    if _is_tty:
                        live.update(Group(_p3_header_text(), Text(f"  {line}")))
                    else:
                        _progress_console.print(line)
                return _cb

            progress_cb = _make_progress_cb()
            progress_cb("generating prose")

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
                profile=profile,
                anthropic_client=anthropic_client,
                kb_chunks=kb_chunks,
                ledger=ledger,
                skipped_records=skipped_records,
                progress_cb=progress_cb,
            )

            if result is None:
                _p3_skipped += 1
            else:
                _p3_passed += 1
                all_conversations.append(result)

            if _is_tty:
                live.update(_p3_header_text())

    _log(
        config,
        f"  {len(all_conversations)} conversations generated, "
        f"{len(skipped_records)} skipped",
    )

    # Wire disagreement ledger stats into skip_tracker for manifest output.
    skip_tracker.disagreement_checks = len(ledger.records)
    skip_tracker.disagreement_cases = sum(1 for r in ledger.records if r.disagreement)

    # Log disagreement ledger health summary.
    ledger_status, ledger_msg = ledger.check_health()
    _log(config, f"  disagreement ledger: {ledger_msg}")
    if ledger_status == "blocking":
        _log(config, "  WARNING: disagreement rate exceeds 25% — corpus extraction may be unreliable")

    # Collect gate violations (structured). Abort on errors *after* manifest is written.
    gate_violations: list[GateViolation] = skip_tracker.check_gates()
    gate_errors = [v for v in gate_violations if v.severity == GateSeverity.ERROR]
    gate_warnings = [v for v in gate_violations if v.severity == GateSeverity.WARNING]
    for w in gate_warnings:
        print(f"[confabra]   GATE WARNING: {w.message}")
        _log(config, f"  GATE WARNING: {w.message}")
    for e in gate_errors:
        print(f"[confabra]   GATE ERROR: {e.message}")

    # Cache telemetry summary — placeholder for future re-enable of prompt caching.
    # Caching was evaluated but removed because the system prompt (~83 tokens) is
    # 25× below Haiku's 2048-token minimum threshold. See cache-diagnosis.md.
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

    # Serialise skipped conversation records (enriched with verdict/prose context).
    skipped_lines = [r.model_dump_json() for r in skipped_records]
    atomic_write_jsonl(profile_dir / "skipped_conversations.jsonl", skipped_lines)

    # Serialise disagreement ledger (empty in dry-run or when no FAIL verdicts occurred).
    disagreement_lines = [r.model_dump_json() for r in ledger.records]
    atomic_write_jsonl(profile_dir / "disagreements.jsonl", disagreement_lines)

    # --- KB domain telemetry ---
    # kb_version: deterministic hash of sorted chunk_ids + chunk_text.
    _kb_version_source = "".join(
        c.chunk_id + c.chunk_text
        for c in sorted(kb_chunks, key=lambda c: c.chunk_id)
    )
    _kb_version = hashlib.sha256(_kb_version_source.encode()).hexdigest()

    # domain_distribution_observed: count CONVERSATION_STARTED events per domain.
    _domain_dist: dict[str, int] = {}
    for e in events:
        if e.event_type == SimEventType.CONVERSATION_STARTED:
            _d = e.payload.get("domain", "")
            if _d:
                _domain_dist[_d] = _domain_dist.get(_d, 0) + 1

    # chunk_selection_frequency: count how many times each chunk_id appeared
    # in any quality plan's should_cite or must_not_cite (excluding the wildcard "*").
    _chunk_freq: dict[str, int] = {}
    for qp in quality_plans:
        for cid in qp.knowledge_citations.should_cite + qp.knowledge_citations.must_not_cite:
            if cid != "*":
                _chunk_freq[cid] = _chunk_freq.get(cid, 0) + 1

    # Build manifest.  gate_aborted=True and gate_violations are set when a hard
    # gate fires; the manifest is always written before raising so the run is
    # inspectable even on abort.
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
        skipped_conversation_count=len(skipped_records),
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
        gate_aborted=bool(gate_errors),
        gate_violations=[v.model_dump() for v in gate_violations],
        kb_version=_kb_version,
        kb_chunk_count=kb_chunk_count,
        domain_distribution_observed=_domain_dist,
        chunk_selection_frequency=_chunk_freq,
        backfill_activations=backfill.get_backfill_activations(),
        cells_requiring_backfill=backfill.get_cells_requiring_backfill(),
        cells_satisfied_by_normal=backfill.get_cells_satisfied_by_normal(),
        deficit_at_run_end=backfill.get_deficit_at_run_end(),
    )

    manifest_path = profile_dir / "manifest.json"
    atomic_write_text(manifest_path, json.dumps(manifest.model_dump(mode="json"), indent=2))

    _log(config, f"  manifest written → {manifest_path}")
    _log(
        config,
        f"Done. {len(all_conversations)} conversations, "
        f"{len(quality_plans)} planted, {len(skipped_records)} skipped.",
    )

    # Abort after manifest write so the run is inspectable.
    if gate_errors:
        error_messages = "\n".join(f"  • {e.message}" for e in gate_errors)
        raise RuntimeError(
            f"Quality gate(s) exceeded — aborting corpus run:\n{error_messages}"
        )

    return manifest
