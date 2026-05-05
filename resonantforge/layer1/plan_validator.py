"""Two-stage quality plan validator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from resonantforge.schemas import (
    DaySnapshot,
    DimensionVerdict,
    GateSeverity,
    GateViolation,
    LifecycleStage,
    QualityPlan,
    SimEvent,
    ValidationResult,
    ValidationVerdict,
)

if TYPE_CHECKING:
    from resonantforge.schemas import ConversationSignals


# ---------------------------------------------------------------------------
# Skip-rate tracker
# ---------------------------------------------------------------------------


@dataclass
class SkipRateTracker:
    """
    Track three separate skip/failure rates (Section 10.1).

    Three independent rate counters correspond to three quality gates:
    1. prose_fact — event-log facts missing from generated prose (hard: 2% gate)
    2. quality_rule — validator rule failures on planted-quality conversations (hard: 2% gate)
    3. disagreement — rule-based validator vs. soft-judge divergence (warn: 15%, block: 25%)

    ``total_skipped`` is incremented by the caller each time a conversation is
    skipped after exhausting retries; it is not derived from the other counters
    because a single skip may be attributed to more than one failure class.
    """

    # Prose-fact violations (post-gen validation on event-log facts): gate at 2%
    prose_fact_attempts: int = 0
    prose_fact_failures: int = 0
    # Validator rule failures on planted-quality (gate at 2%)
    quality_rule_attempts: int = 0
    quality_rule_failures: int = 0
    # Disagreement ledger (15% warn / 25% block)
    disagreement_checks: int = 0
    disagreement_cases: int = 0
    # Total skipped conversations
    total_skipped: int = 0
    # Cache telemetry (Anthropic ephemeral prompt caching)
    cache_creation_tokens: int = 0
    cache_read_tokens: int = 0
    cache_api_calls: int = 0

    @property
    def prose_fact_rate(self) -> float:
        """Failure rate for prose-fact checks; 0.0 when no attempts recorded."""
        return self.prose_fact_failures / max(1, self.prose_fact_attempts)

    @property
    def quality_rule_rate(self) -> float:
        """Failure rate for quality-rule checks; 0.0 when no attempts recorded."""
        return self.quality_rule_failures / max(1, self.quality_rule_attempts)

    @property
    def disagreement_rate(self) -> float:
        """Disagreement rate between rule validator and soft judge; 0.0 when no checks."""
        return self.disagreement_cases / max(1, self.disagreement_checks)

    def check_gates(self) -> list[GateViolation]:
        """
        Return a list of gate violations (empty list = all gates healthy).

        Thresholds (Section 10.1):
        - prose_fact_rate > 2% → ERROR: generator reliability problem, abort run
        - quality_rule_rate > 2% → ERROR: planted-quality consistency problem, abort run
        - disagreement_rate > 25% → ERROR: extraction/validator divergence blocks extraction
        - disagreement_rate > 15% (and ≤ 25%) → WARNING: approaching block threshold
        """
        violations: list[GateViolation] = []
        if self.prose_fact_rate > 0.02:
            violations.append(GateViolation(
                gate_name="prose_fact_rate",
                severity=GateSeverity.ERROR,
                actual_value=self.prose_fact_rate,
                threshold=0.02,
                message=f"prose_fact_rate={self.prose_fact_rate:.3f} exceeds 0.02",
            ))
        if self.quality_rule_rate > 0.02:
            violations.append(GateViolation(
                gate_name="quality_rule_rate",
                severity=GateSeverity.ERROR,
                actual_value=self.quality_rule_rate,
                threshold=0.02,
                message=f"quality_rule_rate={self.quality_rule_rate:.3f} exceeds 0.02",
            ))
        if 0.15 < self.disagreement_rate <= 0.25:
            violations.append(GateViolation(
                gate_name="disagreement_rate",
                severity=GateSeverity.WARNING,
                actual_value=self.disagreement_rate,
                threshold=0.15,
                message=(
                    f"disagreement_rate={self.disagreement_rate:.3f} exceeds 0.15 "
                    "(approaching 25% block threshold)"
                ),
            ))
        if self.disagreement_rate > 0.25:
            violations.append(GateViolation(
                gate_name="disagreement_rate",
                severity=GateSeverity.ERROR,
                actual_value=self.disagreement_rate,
                threshold=0.25,
                message=f"disagreement_rate={self.disagreement_rate:.3f} exceeds 0.25 (blocks extraction)",
            ))
        return violations


# ---------------------------------------------------------------------------
# Plan validator
# ---------------------------------------------------------------------------


class PlanValidator:
    """
    Two-stage quality plan validator. Stateless; takes context as arguments.

    Stage 1 — pre_prompt_validate: hard gate run *before* prose generation.
    A failure here means the quality plan is internally inconsistent or
    contradicts the simulation state — this is always a generator bug, so the
    caller should abort the run rather than retry.

    Stage 2 — post_generation_validate: soft gate run *after* prose generation.
    Failures here are expected occasionally (LLM non-compliance); the caller
    retries up to 2 times before issuing a SKIP verdict.
    """

    # ------------------------------------------------------------------
    # Stage 1 — pre-prompt validation
    # ------------------------------------------------------------------

    def pre_prompt_validate(
        self,
        quality_plan: QualityPlan,
        account_snapshot: DaySnapshot,
        account_events: list[SimEvent],
    ) -> ValidationResult:
        """
        Hard gate: check quality plan is consistent with simulation state.

        A failure here indicates a generator bug — the caller should abort the
        corpus run rather than skip or retry.

        Checks performed:
        1. trigger_event_id must reference a real event in account_events.
        2. A ``resolution:strong`` rubric target contradicts an account that has
           already churned (LifecycleStage.CHURNED) — there is no customer to
           resolve for.
        3. If the plan's knowledge_citations has should_cite entries, the plan's
           kb_chunks_required list must be populated (the two fields are required
           to agree so the post-generation validator can check coverage).

        Args:
            quality_plan:     The QualityPlan authored by the corpus planner.
            account_snapshot: The DaySnapshot for the account on the conversation day.
            account_events:   All SimEvents for the account on the conversation day.

        Returns:
            ValidationResult with PASS or FAIL overall_verdict.
        """
        errors: list[str] = []

        # Check 1: trigger event must exist in the day's event log.
        event_ids = {e.event_id for e in account_events}
        if quality_plan.trigger_event_id not in event_ids:
            errors.append(
                f"trigger_event_id {quality_plan.trigger_event_id!r} not found in account events"
            )

        # Check 2: resolution:strong is semantically invalid for churned accounts.
        if account_snapshot.lifecycle_stage == LifecycleStage.CHURNED:
            if quality_plan.rubric_targets.resolution == "strong":
                errors.append(
                    "resolution:strong plan contradicts CHURNED account lifecycle stage"
                )

        # Check 3: should_cite implies kb_chunks_required must be populated.
        if quality_plan.knowledge_citations.should_cite and not quality_plan.kb_chunks_required:
            errors.append(
                "quality plan has should_cite but kb_chunks_required is empty"
            )

        if errors:
            return ValidationResult(
                conversation_id=quality_plan.conversation_id,
                overall_verdict=ValidationVerdict.FAIL,
                dimension_verdicts=[],
                skip_reason="; ".join(errors),
            )

        return ValidationResult(
            conversation_id=quality_plan.conversation_id,
            overall_verdict=ValidationVerdict.PASS,
            dimension_verdicts=[],
        )

    # ------------------------------------------------------------------
    # Stage 2 — post-generation validation
    # ------------------------------------------------------------------

    def post_generation_validate(
        self,
        quality_plan: QualityPlan,
        generated_prose: str,
        signals: "ConversationSignals | None",
        validator_verdicts: list[DimensionVerdict],
        retry_count: int = 0,
    ) -> ValidationResult:
        """
        Soft gate: check generated prose passes validator rules.

        Called after the prose generator has produced conversation text and the
        signal extractor has run. If signal extraction failed (signals is None)
        the prose cannot be evaluated — treat as FAIL and retry.

        Retry logic:
        - retry_count < 2 → return FAIL (caller re-generates prose and re-runs)
        - retry_count >= 2 → return SKIP (caller records skip, moves on)

        Args:
            quality_plan:       The QualityPlan that drove prose generation.
            generated_prose:    The raw generated conversation text.
            signals:            Extracted ConversationSignals, or None if extraction failed.
            validator_verdicts: Per-dimension verdicts from the rule-based validator.
            retry_count:        Number of retries already attempted (0-based).

        Returns:
            ValidationResult with PASS, FAIL, or SKIP overall_verdict.
        """
        # Signal extraction must succeed for any evaluation to proceed.
        if signals is None:
            return ValidationResult(
                conversation_id=quality_plan.conversation_id,
                overall_verdict=(
                    ValidationVerdict.FAIL if retry_count < 2 else ValidationVerdict.SKIP
                ),
                dimension_verdicts=[],
                skip_reason="signal extraction failed",
                retry_count=retry_count,
            )

        failed = [
            v for v in validator_verdicts if v.verdict == ValidationVerdict.FAIL
        ]

        if failed:
            if retry_count >= 2:
                # Exhausted retries — skip this conversation.
                return ValidationResult(
                    conversation_id=quality_plan.conversation_id,
                    overall_verdict=ValidationVerdict.SKIP,
                    dimension_verdicts=validator_verdicts,
                    skip_reason=(
                        f"validator rule failures after {retry_count + 1} attempts: "
                        f"{[v.dimension for v in failed]}"
                    ),
                    retry_count=retry_count,
                )
            # Still have retries remaining — signal caller to regenerate.
            return ValidationResult(
                conversation_id=quality_plan.conversation_id,
                overall_verdict=ValidationVerdict.FAIL,
                dimension_verdicts=validator_verdicts,
                retry_count=retry_count,
            )

        return ValidationResult(
            conversation_id=quality_plan.conversation_id,
            overall_verdict=ValidationVerdict.PASS,
            dimension_verdicts=validator_verdicts,
            retry_count=retry_count,
        )

    # ------------------------------------------------------------------
    # Prose-fact check helper
    # ------------------------------------------------------------------

    def validate_prose_facts(
        self,
        prose: str,
        expected_facts: dict[str, str],
    ) -> list[str]:
        """
        Check that generated prose contains required factual references.

        Performs case-insensitive substring search for each expected value.
        The ``expected_facts`` dict maps a human-readable fact key (used only
        in violation messages) to the substring that must appear in the prose.

        Args:
            prose:          The full generated conversation text.
            expected_facts: Mapping of {fact_key: expected_value_substring}.

        Returns:
            List of violation strings (empty list = all facts present).
        """
        violations: list[str] = []
        for key, expected in expected_facts.items():
            if expected.lower() not in prose.lower():
                violations.append(
                    f"fact '{key}' (expected '{expected}') not found in prose"
                )
        return violations
