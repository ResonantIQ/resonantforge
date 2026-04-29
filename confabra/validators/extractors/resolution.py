# confabra/validators/extractors/resolution.py
"""
Resolution signal extractor — fully deterministic (regex/structural, no LLM).
Per Section 5.1 implementation-grade signal schema.
"""
import re
from confabra.schemas import ResolutionSignals


def _regex_any(text: str, patterns: list[str]) -> bool:
    """Return True if any pattern matches (case-insensitive)."""
    text_lower = text.lower()
    return any(re.search(p, text_lower) for p in patterns)


def _regex_matches_list(text: str, patterns: list[str]) -> list[str]:
    """Return list of all matched patterns."""
    text_lower = text.lower()
    return [p for p in patterns if re.search(p, text_lower)]


def _classify_solution_type(
    agent_prose: str,
    customer_prose: str,
    resolution_patterns: list[str],
    issue_keywords: list[str],
) -> str:
    """
    Classify solution type: complete | partial | none

    complete: solution_provided AND solution references the customer's stated issue
    partial: solution_provided but only directional
    none: no action verb, no next-step language
    """
    if not _regex_any(agent_prose, resolution_patterns):
        return "none"

    # Check if solution references the customer's specific issue (keyword overlap)
    if customer_prose:
        customer_lower = customer_prose.lower()
        agent_lower = agent_prose.lower()
        issue_kw_in_customer = [kw for kw in issue_keywords if kw in customer_lower]
        if issue_kw_in_customer:
            # Check if any of those keywords also appear in the agent response
            referenced_in_agent = [kw for kw in issue_kw_in_customer if kw in agent_lower]
            if referenced_in_agent:
                return "complete"

    return "partial"


def _check_deflection_without_help(
    agent_prose: str,
    deflection_patterns: list[str],
    resolution_patterns: list[str],
) -> bool:
    """
    Deflection is present when the agent uses deflection phrases WITHOUT
    providing additional substantive help immediately after.
    """
    has_deflection = _regex_any(agent_prose, deflection_patterns)
    if not has_deflection:
        return False

    # If there's also a resolution pattern, the deflection is accompanied by help
    has_resolution = _regex_any(agent_prose, resolution_patterns)
    return not has_resolution


def extract_resolution_signals(
    agent_prose: str,
    customer_prose: str,
    resolution_patterns: list[str],
    deflection_patterns: list[str],
    next_steps_patterns: list[str],
    temporal_anchor_patterns: list[str],
    specific_actor_patterns: list[str],
    ownership_patterns: list[str],
    issue_keywords: list[str],
) -> ResolutionSignals:
    """
    Extract resolution signals from conversation prose.

    Args:
        agent_prose: agent turns concatenated
        customer_prose: customer turns concatenated (for issue keyword detection)
        All other args: from profile lexicon
    """
    # 1. Solution provided
    solution_provided = _regex_any(agent_prose, resolution_patterns)

    # 2. Solution type
    solution_type = _classify_solution_type(
        agent_prose, customer_prose, resolution_patterns, issue_keywords
    )

    # 3. Next steps present
    next_steps_present = _regex_any(agent_prose, next_steps_patterns)

    # 4. Next steps actionable (temporal anchor OR specific actor)
    has_temporal = _regex_any(agent_prose, temporal_anchor_patterns)
    has_specific_actor = _regex_any(agent_prose, specific_actor_patterns)
    next_steps_actionable = next_steps_present and (has_temporal or has_specific_actor)

    # 5. Ownership language
    ownership_matches = _regex_matches_list(agent_prose, ownership_patterns)
    ownership_language_present = len(ownership_matches) > 0

    # 6. Deflection present (without substantive help)
    deflection_present = _check_deflection_without_help(
        agent_prose, deflection_patterns, resolution_patterns
    )

    # 7. Resolution blocked (derived)
    resolution_blocked = not solution_provided and deflection_present

    return ResolutionSignals(
        solution_provided=solution_provided,
        solution_type=solution_type,
        next_steps_present=next_steps_present,
        next_steps_actionable=next_steps_actionable,
        ownership_language_present=ownership_language_present,
        ownership_phrases=ownership_matches,
        deflection_present=deflection_present,
        resolution_blocked=resolution_blocked,
    )
