"""
Three-layer agreement computation between validator verdicts and human labels.

Layer 1: outcome_match  — overall pass/fail matches expected_outcome
Layer 2: rule_match     — exact set of failed rules matches expected_failures
Layer 3: false positives / missed rules — which rules diverged
"""
from __future__ import annotations

from resonantforge.replay.schemas import AgreementResult, ReplayLabels
from resonantforge.schemas import DimensionVerdict, ValidationVerdict


def compute_agreement(
    dimension_verdicts: list[DimensionVerdict],
    claim_extraction_ok: bool,
    overall_outcome: str,
    labels: ReplayLabels,
) -> AgreementResult:
    """
    Compare validator output against human labels on three layers.

    Args:
        dimension_verdicts: per-dimension validator results
        claim_extraction_ok: True when claim extraction succeeded (always True in replay)
        overall_outcome: "pass" or "fail" — the validator's overall verdict
        labels: human-authored ground truth

    Returns:
        AgreementResult capturing outcome_match, rule_match, and diverging rules
    """
    # Layer 1: outcome_match
    if labels.expected_outcome == "uncertain":
        outcome_match = None
    else:
        outcome_match = overall_outcome == labels.expected_outcome

    # Build the set of rules the validator flagged as FAIL
    validator_failed: set[str] = set()
    for verdict in dimension_verdicts:
        if verdict.verdict == ValidationVerdict.FAIL:
            validator_failed.add(verdict.dimension)

    if not claim_extraction_ok:
        validator_failed.add("claim_extraction")

    # Build the set of rules the label says should fail
    label_failed: set[str] = {
        key for key, should_fail in labels.expected_failures.items()
        if should_fail
    }

    # Layer 2: rule_match — exact set equality
    rule_match = validator_failed == label_failed

    # Layer 3: diverging rules
    false_positive_rules = sorted(validator_failed - label_failed)  # stable output
    missed_rules = sorted(label_failed - validator_failed)          # stable output

    return AgreementResult(
        outcome_match=outcome_match,
        rule_match=rule_match,
        false_positive_rules=false_positive_rules,
        missed_rules=missed_rules,
    )
