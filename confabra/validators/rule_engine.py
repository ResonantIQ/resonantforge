# confabra/validators/rule_engine.py
"""
Deterministic validator rule engine. Pure logic over extracted signals.
No LLM in the gating path. Per Section 5.1.2.
"""
from __future__ import annotations
from confabra.schemas import (
    EmpathySignals, ResolutionSignals, BrandVoiceSignals, AccuracySignals,
    DimensionVerdict, ValidationVerdict, AccuracyLabel
)
from confabra.validators.extractors.brand_voice import brand_voice_feature_profile


def _within_range(value: float | int, range_tuple: tuple) -> bool:
    """Check if value falls within (min, max) range inclusive."""
    lo, hi = range_tuple
    return lo <= value <= hi


def validate_empathy(signals: EmpathySignals, target: str) -> DimensionVerdict:
    """
    Validate empathy dimension against target.

    Targets: "low" | "high"

    For target "low":
        - acknowledgment_present MUST be False
        - emotional_language_present MUST be False

    For target "high":
        - acknowledgment_present MUST be True
        - emotional_language_present MUST be True
        - follow_through_present MUST be True
        - fake_empathy_flag MUST be False (anti-cheat)
    """
    if target == "empathy:low" or target == "low":
        passed = (
            not signals.acknowledgment_present and
            not signals.emotional_language_present
        )
    elif target == "empathy:high" or target == "high":
        passed = (
            signals.acknowledgment_present and
            signals.emotional_language_present and
            signals.follow_through_present and
            not signals.fake_empathy_flag  # anti-cheat: no fake empathy
        )
    else:
        # Unknown target — skip this dimension
        return DimensionVerdict(
            dimension="empathy",
            verdict=ValidationVerdict.SKIP,
            target=target,
            signals_summary=signals.model_dump(),
        )

    return DimensionVerdict(
        dimension="empathy",
        verdict=ValidationVerdict.PASS if passed else ValidationVerdict.FAIL,
        target=target,
        signals_summary={
            "acknowledgment_present": signals.acknowledgment_present,
            "emotional_language_present": signals.emotional_language_present,
            "follow_through_present": signals.follow_through_present,
            "fake_empathy_flag": signals.fake_empathy_flag,
            "response_length_tokens": signals.response_length_tokens,
        },
    )


def validate_resolution(signals: ResolutionSignals, target: str) -> DimensionVerdict:
    """
    Validate resolution dimension against target.

    Targets: "weak" | "strong"

    For target "weak":
        - solution_provided MUST be False OR resolution_blocked MUST be True

    For target "strong":
        - solution_provided MUST be True
        - solution_type MUST be "complete"
        - next_steps_actionable MUST be True
        - ownership_language_present MUST be True
        - deflection_present MUST be False
    """
    if target == "resolution:weak" or target == "weak":
        passed = (
            not signals.solution_provided or
            signals.resolution_blocked
        )
    elif target == "resolution:strong" or target == "strong":
        passed = (
            signals.solution_provided and
            signals.solution_type == "complete" and
            signals.next_steps_actionable and
            signals.ownership_language_present and
            not signals.deflection_present
        )
    else:
        return DimensionVerdict(
            dimension="resolution",
            verdict=ValidationVerdict.SKIP,
            target=target,
            signals_summary=signals.model_dump(),
        )

    return DimensionVerdict(
        dimension="resolution",
        verdict=ValidationVerdict.PASS if passed else ValidationVerdict.FAIL,
        target=target,
        signals_summary={
            "solution_provided": signals.solution_provided,
            "solution_type": signals.solution_type,
            "next_steps_actionable": signals.next_steps_actionable,
            "ownership_language_present": signals.ownership_language_present,
            "deflection_present": signals.deflection_present,
            "resolution_blocked": signals.resolution_blocked,
        },
    )


def validate_brand_voice(
    signals: BrandVoiceSignals,
    variant_id: str,
    target: str,
    feature_profiles: dict,
) -> DimensionVerdict:
    """
    Validate brand voice dimension against target and variant.

    Targets: "on_brand" | "off_brand"

    For "on_brand": all of the variant's calibrated feature ranges must be satisfied,
    AND the aligned term count must meet the minimum.

    For "off_brand": at least one range must be outside the variant's expected range.
    (A deliberate off-brand conversation should fail at least one feature check.)
    """
    profile = brand_voice_feature_profile(variant_id, feature_profiles)

    if not profile:
        return DimensionVerdict(
            dimension="brand_voice",
            verdict=ValidationVerdict.SKIP,
            target=target,
            signals_summary=signals.model_dump(),
        )

    # Evaluate all feature checks
    checks = {
        "avg_sentence_length": _within_range(signals.avg_sentence_length, profile["sentence_length_range"]),
        "question_count": _within_range(signals.question_count, profile["question_count_range"]),
        "hedging_terms": _within_range(signals.hedging_terms_count, profile["hedging_range"]),
        "directive_terms": _within_range(signals.directive_terms_count, profile["directive_range"]),
        "aligned_vocabulary": signals.vocabulary_match.get(profile["aligned_term_field"], 0) >= profile["aligned_terms_min"],
    }

    all_pass = all(checks.values())
    any_fail = not all_pass

    if target == "brand_voice:on_brand" or target == "on_brand":
        passed = all_pass
    elif target == "brand_voice:off_brand" or target == "off_brand":
        # Off-brand should fail at least one feature check
        passed = any_fail
    else:
        return DimensionVerdict(
            dimension="brand_voice",
            verdict=ValidationVerdict.SKIP,
            target=target,
            signals_summary=signals.model_dump(),
        )

    return DimensionVerdict(
        dimension="brand_voice",
        verdict=ValidationVerdict.PASS if passed else ValidationVerdict.FAIL,
        target=target,
        signals_summary={
            "feature_checks": checks,
            "avg_sentence_length": signals.avg_sentence_length,
            "question_count": signals.question_count,
            "hedging_terms_count": signals.hedging_terms_count,
            "directive_terms_count": signals.directive_terms_count,
            "vocabulary_match": signals.vocabulary_match,
        },
    )


def validate_accuracy(signals: AccuracySignals, target: AccuracyLabel) -> DimensionVerdict:
    """
    Validate accuracy dimension against target.

    Target uses structured AccuracyLabel: {status, precision}

    For supported + exact:
        alignment == "supported" AND constraint_preserved AND NOT overgeneralization_flag
        AND NOT blocking_constraint_violated AND (not multi_chunk_required OR multi_chunk_satisfied)

    For supported + conditional_applied:
        alignment == "supported" AND constraint_preserved (condition was applied correctly)

    For supported + overgeneralized:
        overgeneralization_flag is True (test that we planted this correctly)

    For contradicted + exact:
        alignment == "contradicted"

    For insufficient_information:
        alignment == "not_found" AND no claims (len(claims) == 0)
    """
    status = target.status
    precision = target.precision

    if status == "supported" and precision == "exact":
        passed = (
            signals.alignment == "supported" and
            signals.constraint_preserved and
            not signals.overgeneralization_flag and
            not signals.blocking_constraint_violated and
            (not signals.multi_chunk_required or signals.multi_chunk_satisfied)
        )
    elif status == "supported" and precision == "conditional_applied":
        # Condition was correctly applied — claim incorporates the conditional
        passed = (
            signals.alignment == "supported" and
            signals.constraint_preserved
        )
    elif status == "supported" and precision == "overgeneralized":
        # The planted case should show overgeneralization
        passed = signals.overgeneralization_flag
    elif status == "supported" and precision == "missing_constraint":
        passed = (
            signals.alignment == "partial" and
            not signals.constraint_preserved
        )
    elif status == "contradicted" and precision == "exact":
        passed = signals.alignment == "contradicted"
    elif status == "insufficient_information":
        passed = (
            signals.alignment == "not_found" and
            len(signals.claims) == 0
        )
    else:
        return DimensionVerdict(
            dimension="accuracy",
            verdict=ValidationVerdict.SKIP,
            target=f"{status}:{precision}",
            signals_summary=signals.model_dump(),
        )

    return DimensionVerdict(
        dimension="accuracy",
        verdict=ValidationVerdict.PASS if passed else ValidationVerdict.FAIL,
        target=f"{status}:{precision}",
        signals_summary={
            "alignment": signals.alignment,
            "constraint_preserved": signals.constraint_preserved,
            "overgeneralization_flag": signals.overgeneralization_flag,
            "blocking_constraint_violated": signals.blocking_constraint_violated,
            "multi_chunk_required": signals.multi_chunk_required,
            "multi_chunk_satisfied": signals.multi_chunk_satisfied,
            "claim_count": len(signals.claims),
        },
    )


def validate_all_dimensions(
    empathy_signals: EmpathySignals | None,
    resolution_signals: ResolutionSignals | None,
    brand_voice_signals: BrandVoiceSignals | None,
    accuracy_signals: AccuracySignals | None,
    rubric_targets: "RubricTarget",
    brand_voice_variant_id: str,
    feature_profiles: dict,
) -> list[DimensionVerdict]:
    """
    Run all applicable dimension validators against their targets.
    Only validates dimensions that have targets in rubric_targets.
    """
    verdicts = []

    if empathy_signals and rubric_targets.empathy:
        verdicts.append(validate_empathy(empathy_signals, rubric_targets.empathy))

    if resolution_signals and rubric_targets.resolution:
        verdicts.append(validate_resolution(resolution_signals, rubric_targets.resolution))

    if brand_voice_signals and rubric_targets.brand_voice_target:
        verdicts.append(validate_brand_voice(
            brand_voice_signals,
            brand_voice_variant_id,
            rubric_targets.brand_voice_target,
            feature_profiles,
        ))

    if accuracy_signals and rubric_targets.accuracy:
        verdicts.append(validate_accuracy(accuracy_signals, rubric_targets.accuracy))

    return verdicts
