"""
Constraint extraction from KB chunk text.

Extracts concrete qualifying phrases (numeric limits, time windows, plan/tier
restrictions, conditionals) that a planted overgeneralization event is
designed to drop. The extracted Constraint is stored on the QualityPlan so
the validator can do a deterministic presence check instead of regex guessing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Constraint:
    raw: str        # phrase as matched in source text
    normalized: str  # lowercased, whitespace-collapsed — use this for presence checks


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


# Patterns applied in order; more specific patterns first so they consume the
# right tokens before the generic bare-numeric sweep runs.
# Each entry: (compiled_pattern,)
_PATTERNS: list[re.Pattern[str]] = [
    # Category 2a: "up to N [unit]"
    re.compile(r'\bup\s+to\s+\d+(?:\s+(?:seat|user|device|license|gb|mb|tb)s?)?\b', re.IGNORECASE),
    # Category 2b: "maximum / max / no more than / at most [of] N [unit]"
    re.compile(r'\b(?:maximum|max|no\s+more\s+than|at\s+most)\s+(?:of\s+)?\d+(?:\s+(?:seat|user|device|license|gb|mb|tb)s?)?\b', re.IGNORECASE),
    # Category 2c: "at least N"
    re.compile(r'\bat\s+least\s+\d+(?:\s+\w+)?\b', re.IGNORECASE),
    # Category 3a: "within N days/hours/etc."
    re.compile(r'\bwithin\s+\d+\s+(?:day|hour|minute|week|month)s?\b', re.IGNORECASE),
    # Category 3b: "first N days/hours/etc."
    re.compile(r'\bfirst\s+\d+\s+(?:day|hour|minute|week|month)s?\b', re.IGNORECASE),
    # Category 4a: "[plan-tier] plan/tier/subscription only"
    re.compile(r'\b(?:enterprise|pro|growth|starter|free|paid|annual|monthly)\s+(?:plan|tier|account|subscription)s?\s+only\b', re.IGNORECASE),
    # Category 4b: "for/on [plan-tier] plan/tier/subscription"
    re.compile(r'\b(?:for|on)\s+(?:enterprise|pro|growth|starter|free|paid|annual|monthly)\s+(?:plan|tier|account|subscription)s?\b', re.IGNORECASE),
    # Category 4c: "available for annual/monthly subscribers only" etc.
    re.compile(r'\b(?:enterprise|pro|growth|starter|free|paid|annual|monthly)\s+(?:subscriber|customer|user|member)s?\s+only\b', re.IGNORECASE),
    # Category 5: "only if ..." — capture up to 60 chars before end-of-phrase
    re.compile(r'\bonly\s+if\s+[^.!?,\n]{3,60}', re.IGNORECASE),
    # Category 1 (last — most permissive): bare "N seats/users/devices/etc."
    re.compile(r'\b\d+\s+(?:seat|user|device|license|request|call|gb|mb|tb)s?\b', re.IGNORECASE),
]


def extract_constraints(chunk_text: str) -> list[Constraint]:
    """
    Extract qualifying constraint phrases from KB chunk text.

    Returns a deduplicated list ordered by pattern specificity. Use
    ``Constraint.normalized`` for substring presence checks against agent claims.
    """
    if not chunk_text:
        return []

    seen: set[str] = set()
    results: list[Constraint] = []

    for pattern in _PATTERNS:
        for match in pattern.finditer(chunk_text):
            raw = match.group(0).strip()
            normalized = _normalize(raw)
            if normalized not in seen:
                seen.add(normalized)
                results.append(Constraint(raw=raw, normalized=normalized))

    return results
