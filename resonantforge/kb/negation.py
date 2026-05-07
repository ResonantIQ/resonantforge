"""
Fact negation for the planted_contradiction pipeline.

Each mutation band produces an unmistakable contradiction — not a subtle tweak.
The negated form is stored on the QualityPlan so the validator can use
substring presence/absence instead of semantic reasoning.

Mutation bands (from spec):
  numeric    — 10× the original value, OR polarity flip for SLA-style percentages
               (> 80%). Reject any mutation within 50% of original.
  categorical — polarity inversion: restriction → broad, broad → restriction
  capability — outright negation ("does not support X")
  policy     — inversion ("refunds available" ↔ "no refunds")
"""
from __future__ import annotations

import re

from resonantforge.kb.kb_fact_extractor import KbFact


def negate_fact(fact: KbFact) -> str:
    """
    Return a negated string form of fact that is obviously incompatible with
    the original. Raises ValueError for unknown fact_category.
    """
    if fact.fact_category == "numeric":
        return _negate_numeric(fact.source_text)
    if fact.fact_category == "categorical":
        return _negate_categorical(fact.source_text)
    if fact.fact_category == "capability":
        return _negate_capability(fact.source_text)
    if fact.fact_category == "policy":
        return _negate_policy(fact.source_text)
    raise ValueError(f"Unknown fact_category: {fact.fact_category!r}")


# ── Numeric ───────────────────────────────────────────────────────────────────


def _negate_numeric(source_text: str) -> str:
    # SLA-style percentage: high percentage → dramatic polarity flip
    pct = re.search(r"(\d+(?:\.\d+)?)\s*%", source_text)
    if pct:
        original = float(pct.group(1))
        if original > 50:
            # Flip to complement; guaranteed > 50% difference from original
            negated = round(100.0 - original, 4)
        else:
            negated = round(original * 10, 4)
        negated_str = str(int(negated)) if negated == int(negated) else str(negated)
        start, end = pct.span(1)
        return source_text[:start] + negated_str + source_text[end:]

    # All other numerics: multiply by 10
    num = re.search(r"(\d+(?:\.\d+)?)", source_text)
    if not num:
        return source_text + " (not applicable)"
    original = float(num.group(1))
    mutated = original * 10
    # Invariant: 10× is always > 50% different from original
    mutated_str = str(int(mutated)) if mutated == int(mutated) else str(mutated)
    start, end = num.span(1)
    return source_text[:start] + mutated_str + source_text[end:]


# ── Categorical ───────────────────────────────────────────────────────────────


def _negate_categorical(source_text: str) -> str:
    lower = source_text.lower()

    # Plan restriction (Enterprise/Pro/etc only) → widen to all plans
    if re.search(
        r"\b(?:enterprise|pro|growth|starter|free|paid)\s+(?:plan|account|tier|subscription)s?\s+only\b",
        lower,
    ):
        return re.sub(
            r"\b(?:enterprise|pro|growth|starter|free|paid)\s+(?:plan|account|tier|subscription)s?\s+only\b",
            "all plans",
            source_text,
            flags=re.IGNORECASE,
        )

    # For/on a specific plan tier (without "only") → widen
    if re.search(
        r"\b(?:for|on)\s+(?:enterprise|pro|growth|starter|free|paid)\s+(?:plan|account|tier|subscription)s?\b",
        lower,
    ):
        return re.sub(
            r"\b(?:for|on)\s+(?:enterprise|pro|growth|starter|free|paid)\s+(?:plan|account|tier|subscription)s?\b",
            "for all plans",
            source_text,
            flags=re.IGNORECASE,
        )

    # "all plans" → restrict to enterprise only
    if re.search(r"\ball\s+(?:plan|tier|account|subscription)s?\b", lower):
        return re.sub(
            r"\ball\s+(?:plan|tier|account|subscription)s?\b",
            "Enterprise plans only",
            source_text,
            flags=re.IGNORECASE,
        )

    # Protocol/auth support: SAML, SSO, etc. → "does not support"
    if re.search(
        r"\bsupports?\s+(?:SAML|OIDC|OAuth|SSO|LDAP|SCIM|MFA|2FA|WebAuthn)\b",
        source_text,
        re.IGNORECASE,
    ):
        return re.sub(r"\bsupports?\b", "does not support", source_text, flags=re.IGNORECASE, count=1)

    # Data location: "stored in X" → "not stored in X"
    if re.search(r"\bstored\s+in\b", lower):
        return re.sub(r"\bstored\s+in\b", "not stored in", source_text, flags=re.IGNORECASE, count=1)

    return "This is not the case: " + source_text


# ── Capability ────────────────────────────────────────────────────────────────


def _negate_capability(source_text: str) -> str:
    # "can X" → "cannot X"
    result = re.sub(r"\bcan\s+", "cannot ", source_text, flags=re.IGNORECASE, count=1)
    if result != source_text:
        return result

    # "supports/support X" → "does not support X"
    result = re.sub(r"\bsupports?\b", "does not support", source_text, flags=re.IGNORECASE, count=1)
    if result != source_text:
        return result

    # "integrates/integrate with X" → "does not integrate with X"
    result = re.sub(r"\bintegrates?\b", "does not integrate", source_text, flags=re.IGNORECASE, count=1)
    if result != source_text:
        return result

    return "This feature is not available: " + source_text


# ── Policy ────────────────────────────────────────────────────────────────────


def _negate_policy(source_text: str) -> str:
    lower = source_text.lower()

    # "refunds are available [within N days]" → no refunds
    if re.search(r"\brefunds?\s+(?:are\s+)?available\b", lower):
        return "No refunds are issued under any circumstances."

    # "all sales are final" or "no refunds" → refunds available
    if re.search(r"\ball\s+sales\s+are\s+final\b", lower) or re.search(r"\bno\s+refunds?\b", lower):
        return "Refunds are available and eligible within 30 days of purchase."

    # Authentication requirement → not required
    if re.search(r"\brequire(?:s|d)?\s+authentication\b", lower) or re.search(
        r"\bapi\s+endpoints?\s+require\b", lower
    ):
        return re.sub(
            r"\brequire(?:s|d)?\b", "do not require", source_text, flags=re.IGNORECASE, count=1
        )

    # SLA credits issued automatically → never / manual
    if re.search(r"\bsla\s+credits?\b", lower) or re.search(
        r"\bcredits?\s+are\s+(?:issued|applied)\b", lower
    ):
        if "automatically" in lower:
            return re.sub(
                r"\bissued\s+automatically\b",
                "never issued automatically — credits must be requested manually",
                source_text,
                flags=re.IGNORECASE,
            )
        return re.sub(r"\bissued\b", "not issued", source_text, flags=re.IGNORECASE, count=1)

    return "This policy does not apply: " + source_text
