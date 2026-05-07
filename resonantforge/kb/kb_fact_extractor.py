"""
Fact extraction from KB chunk text for the planted_contradiction pipeline.

Precision over recall: returns high-quality atomic facts only. Reject cases
(comparatives, vague quantifiers, nested conditionals, world-knowledge,
marketing prose) are as important as positive cases — garbage facts pollute
the contradicted:exact generation pipeline.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class KbFact:
    source_text: str     # phrase as extracted from source text
    normalized: str      # lowercased, whitespace-collapsed — use for dedup
    fact_category: str   # "numeric" | "categorical" | "capability" | "policy"


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


# ── Sentence-level rejection ───────────────────────────────────────────────
# If any pattern matches, the entire sentence is skipped. These cover cases
# where no reliable, contradictable fact can be extracted.

_REJECT_SENTENCE: list[re.Pattern[str]] = [
    # Nested conditionals: "if ... and ... then" or "if ... unless"
    re.compile(r"\bif\b.{0,200}\bunless\b", re.IGNORECASE | re.DOTALL),
    re.compile(r"\bif\b.{0,100}\band\b.{0,100}\bthen\b", re.IGNORECASE),
    # World-knowledge-dependent claims — no fact to contradict without external context
    re.compile(r"\ball\s+applicable\b", re.IGNORECASE),
    re.compile(r"\brelevant\s+regulatory\b", re.IGNORECASE),
    re.compile(r"\bindustry\s+standards?\b", re.IGNORECASE),
    re.compile(r"\bregulatory\s+requirements?\b", re.IGNORECASE),
    # Pure marketing prose — no actionable fact possible
    re.compile(r"\btransform(?:s|ing)?\s+your\b", re.IGNORECASE),
    re.compile(r"\bunlock\s+the\s+power\s+of\b", re.IGNORECASE),
]

# ── Positive extraction patterns by category ───────────────────────────────
# Source text = the matched string (not the full sentence), so vague words
# in the same sentence don't contaminate the extracted fact.

_NUMERIC: list[re.Pattern[str]] = [
    # Rate: "1000 requests per hour" (most specific — before bare count)
    re.compile(r"\b\d+\s+(?:request|call|api\s+call)s?\s+per\s+\w+\b", re.IGNORECASE),
    # Capacity with qualifier: "up to 500 users"
    re.compile(r"\bup\s+to\s+\d+(?:\s+(?:user|seat|device|license|gb|mb|tb|request)s?)?\b", re.IGNORECASE),
    # Currency: $99, $99/month, $99.99/yr
    re.compile(r"\$\d+(?:\.\d+)?(?:/(?:month|year|mo|yr|user|seat))?\b"),
    # Percentage: 99.9%, 99% (no trailing \b — % is non-word, boundary check fails before space)
    re.compile(r"\b\d+(?:\.\d+)?\s*%"),
    # Duration: "90 days", "24 hours"
    re.compile(r"\b\d+\s+(?:day|hour|minute|week|month|year)s?\b", re.IGNORECASE),
    # Count with product unit: "500 users", "1000 requests"
    re.compile(r"\b\d+\s+(?:user|seat|device|license|request|call|gb|mb|tb)s?\b", re.IGNORECASE),
]

_CATEGORICAL: list[re.Pattern[str]] = [
    # "available on Enterprise plans only" — most specific first
    re.compile(
        r"\b(?:available\s+on\s+)?(?:enterprise|pro|growth|starter|free|paid)\s+"
        r"(?:plan|account|tier|subscription)s?\s+only\b",
        re.IGNORECASE,
    ),
    # "for/on Enterprise plans"
    re.compile(
        r"\b(?:for|on)\s+(?:enterprise|pro|growth|starter|free|paid)\s+"
        r"(?:plan|account|tier|subscription)s?\b",
        re.IGNORECASE,
    ),
    # "available on all plans"
    re.compile(r"\bavailable\s+on\s+all\s+(?:plan|tier|account|subscription)s?\b", re.IGNORECASE),
    # "on/for all plans"
    re.compile(r"\b(?:on|for)\s+all\s+(?:plan|tier|account|subscription)s?\b", re.IGNORECASE),
    # Protocol/auth support: "supports SAML SSO for Enterprise accounts"
    re.compile(
        r"\bsupports?\s+(?:SAML(?:\s+SSO)?|OIDC|OAuth\s*\d*|LDAP|SCIM|MFA|2FA|WebAuthn|SSO)\b"
        r"(?:\s+for\s+\w+(?:\s+\w+)?\b)?",
        re.IGNORECASE,
    ),
    # Data storage location: "stored in US-East data centers"
    re.compile(
        r"\bstored\s+in\s+[A-Za-z0-9][A-Za-z0-9\s\-]{1,30}\b",
        re.IGNORECASE,
    ),
]

_CAPABILITY: list[re.Pattern[str]] = [
    # "export [data] to CSV" — specific named format required
    re.compile(r"\bexport(?:s|ing)?\b[^.!?,]{0,30}\b(?:CSV|JSON|Excel|PDF|XML|XLSX)\b", re.IGNORECASE),
    # "integrates with [named system]"
    re.compile(r"\bintegrates?\s+with\s+\w+(?:\s+\w+){0,2}\b", re.IGNORECASE),
    # "supports [outbound] webhooks"
    re.compile(r"\bsupports?\s+(?:outbound\s+)?webhooks?\b", re.IGNORECASE),
    # "webhook support"
    re.compile(r"\bwebhook\s+support\b", re.IGNORECASE),
]

_POLICY: list[re.Pattern[str]] = [
    # Refund policy — capture clause including any numeric window
    re.compile(r"\brefunds?\b[^.!?]*", re.IGNORECASE),
    # "All sales are final"
    re.compile(r"\ball\s+sales\s+are\s+final\b[^.!?]*", re.IGNORECASE),
    # Authentication requirement
    re.compile(r"\b(?:all\s+)?api\s+endpoints?\s+require[^.!?]*", re.IGNORECASE),
    re.compile(r"\brequire(?:s|d)?\s+authentication\b[^.!?]*", re.IGNORECASE),
    # SLA / credit policy
    re.compile(r"\bsla\s+credits?\b[^.!?]*", re.IGNORECASE),
    re.compile(r"\bcredits?\s+(?:are\s+)?(?:issued|applied|processed|calculated)[^.!?]*", re.IGNORECASE),
]

_CATEGORY_PATTERNS: list[tuple[list[re.Pattern[str]], str]] = [
    (_NUMERIC, "numeric"),
    (_CATEGORICAL, "categorical"),
    (_CAPABILITY, "capability"),
    (_POLICY, "policy"),
]


def extract_facts(chunk_text: str) -> list[KbFact]:
    """
    Extract concrete, contradictable facts from a KB chunk.

    Returns a deduplicated list of KbFact instances. Empty input or purely
    vague/marketing text returns [].
    """
    if not chunk_text or not chunk_text.strip():
        return []

    sentences = re.split(r"(?<=[.!?])\s+", chunk_text.strip())

    seen: set[str] = set()
    results: list[KbFact] = []

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if any(pat.search(sentence) for pat in _REJECT_SENTENCE):
            continue

        for patterns, category in _CATEGORY_PATTERNS:
            for pattern in patterns:
                for match in pattern.finditer(sentence):
                    raw = match.group(0).strip()
                    if not raw:
                        continue
                    norm = _normalize(raw)
                    if norm and norm not in seen:
                        seen.add(norm)
                        results.append(KbFact(
                            source_text=raw,
                            normalized=norm,
                            fact_category=category,
                        ))

    return results
