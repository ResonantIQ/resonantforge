"""
Human-readable diff output for replay results.

Used by `rforge replay run` to render per-conversation progress lines and
the final summary table. All output is deterministic (no timestamps, no
colors that vary by terminal state).
"""
from __future__ import annotations

from typing import Literal

from resonantforge.replay.schemas import AgreementResult, ReplayResult

ChangeCategory = Literal["improvement", "regression", "rule_shift", "new_failure", "new_pass"]


def classify_change(result: ReplayResult) -> ChangeCategory:
    """
    Classify a replay result into a change category for diff attribution.

    Categories:
      improvement   outcome_match=True  AND rule_match=True  (full agreement)
      regression    outcome_match=False AND overall_outcome=fail (validator now fails a conv labeled pass)
      new_failure   outcome_match=False AND overall_outcome=fail AND label was fail (outcome correct but was previously passing)
      new_pass      outcome_match=False AND overall_outcome=pass AND label was pass (unexpected pass, was failing)
      rule_shift    outcome_match=True  but rule_match=False (overall outcome right but wrong rules fired)

    For uncertain labels (outcome_match=None) the result always falls through to rule_shift
    if rules disagree, or improvement if they agree.
    """
    ag = result.agreement
    expected = result.labels.expected_outcome

    # Full agreement
    if ag.outcome_match is not False and ag.rule_match:
        return "improvement"

    # outcome_match True but rules disagree → rule changed without flipping outcome
    if ag.outcome_match is True and not ag.rule_match:
        return "rule_shift"

    # Uncertain outcome with rule disagreement
    if ag.outcome_match is None:
        return "rule_shift"

    # outcome_match is False from here
    if result.overall_outcome == "fail":
        if expected == "pass":
            return "regression"
        return "new_failure"

    # overall_outcome == "pass"
    if expected == "fail":
        return "new_pass"

    return "regression"


def format_conv_line(result: ReplayResult) -> str:
    """
    Single-line progress output for one conversation, always emitted live.

    Format:
      [PASS|FAIL] conv_id  outcome_match=Y/N  rule_match=Y/N  category=...  fp=<false_pos>  miss=<missed>

    Args:
        result: completed ReplayResult

    Returns:
        A single line string (no trailing newline)
    """
    ag = result.agreement
    outcome_icon = "PASS" if result.overall_outcome == "pass" else "FAIL"

    if ag.outcome_match is None:
        om_str = "uncertain"
    else:
        om_str = "Y" if ag.outcome_match else "N"

    rm_str = "Y" if ag.rule_match else "N"
    category = classify_change(result)

    fp_str = ",".join(ag.false_positive_rules) if ag.false_positive_rules else "-"
    miss_str = ",".join(ag.missed_rules) if ag.missed_rules else "-"

    return (
        f"[{outcome_icon}] {result.conv_id}"
        f"  outcome_match={om_str}"
        f"  rule_match={rm_str}"
        f"  category={category}"
        f"  fp={fp_str}"
        f"  miss={miss_str}"
    )


def format_summary(results: list[ReplayResult]) -> str:
    """
    Multi-line summary block for a completed corpus run.

    Includes:
      - Total conv count
      - Outcome match rate, headline (high+medium confidence only)
      - Outcome match rate including low-confidence and uncertain
      - Rule match rate (excluding uncertain)
      - Counts per change category
      - Per-rule false positive and missed counts
      - List of disagreeing conv_ids for each diverging rule

    Args:
        results: all ReplayResult objects from the run

    Returns:
        Multi-line string (no trailing newline)
    """
    if not results:
        return "No results."

    total = len(results)

    # Headline: high + medium confidence, non-uncertain only
    headline = [
        r for r in results
        if r.labels.confidence in ("high", "medium") and r.agreement.outcome_match is not None
    ]
    headline_matches = sum(1 for r in headline if r.agreement.outcome_match)

    # Full: all non-uncertain
    certain = [r for r in results if r.agreement.outcome_match is not None]
    outcome_matches = sum(1 for r in certain if r.agreement.outcome_match)
    rule_matches = sum(1 for r in certain if r.agreement.rule_match)

    n_headline = len(headline)
    n_certain = len(certain)
    headline_rate = f"{headline_matches}/{n_headline}" if n_headline else "n/a"
    outcome_rate = f"{outcome_matches}/{n_certain}" if n_certain else "n/a"
    rule_rate = f"{rule_matches}/{n_certain}" if n_certain else "n/a"

    # Per-category counts
    categories: dict[str, list[str]] = {}
    for r in results:
        cat = classify_change(r)
        categories.setdefault(cat, []).append(r.conv_id)

    # Per-rule divergence tallies
    fp_by_rule: dict[str, list[str]] = {}
    miss_by_rule: dict[str, list[str]] = {}

    for r in results:
        for rule in r.agreement.false_positive_rules:
            fp_by_rule.setdefault(rule, []).append(r.conv_id)
        for rule in r.agreement.missed_rules:
            miss_by_rule.setdefault(rule, []).append(r.conv_id)

    lines = [
        f"Replay complete: {total} conversation(s)",
        f"  Outcome match (high+med): {headline_rate} ({_pct(headline_matches, n_headline)})",
        f"  Outcome match (all)     : {outcome_rate} ({_pct(outcome_matches, n_certain)})",
        f"  Rule match              : {rule_rate} ({_pct(rule_matches, n_certain)})",
    ]

    # Change category breakdown
    cat_order: list[ChangeCategory] = ["improvement", "regression", "new_failure", "new_pass", "rule_shift"]
    lines.append("")
    lines.append("Change categories:")
    for cat in cat_order:
        convs = categories.get(cat, [])
        lines.append(f"  {cat:<14} {len(convs):>3}")

    all_rules = sorted(set(fp_by_rule) | set(miss_by_rule))
    if all_rules:
        lines.append("")
        lines.append("Diverging rules:")
        for rule in all_rules:
            fps = fp_by_rule.get(rule, [])
            misses = miss_by_rule.get(rule, [])
            if fps:
                lines.append(f"  {rule}  false-positive ({len(fps)}): {', '.join(fps)}")
            if misses:
                lines.append(f"  {rule}  missed ({len(misses)}): {', '.join(misses)}")
    else:
        lines.append("  All rules matched labels exactly.")

    return "\n".join(lines)


def _pct(num: int, denom: int) -> str:
    if denom == 0:
        return "n/a"
    return f"{100 * num / denom:.1f}%"
