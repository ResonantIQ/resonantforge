"""
Deterministic invariant checker for the SaaS KB.

NOTE: The invariant checker operates strictly at the single-chunk level, with the
sole exception of controlled-vocabulary checks. Cross-chunk claim comparison is
intentionally NOT supported. Forge KB chunks are test fixtures, and chunks tagged
to different gates may legitimately contradict each other on numeric claims, time
windows, conditional thresholds, and similar test material. See
docs/resonantforge/saas-invariants.md for the design rationale.

Two named concerns:
  validate_chunk_structure(chunk)  — per-chunk structural rules
  validate_controlled_vocab(chunk) — per-chunk controlled-vocabulary rules

The global-state aggregation below serves only coverage checks (e.g., "does any
chunk define a refund window?"), never cross-chunk claim comparison.

No NLP, no LLM, no embeddings — pure structured analysis.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from resonantforge.profiles.saas import CANONICAL_TIER_NAMES
from resonantforge.schemas import KBChunk


# ---------------------------------------------------------------------------
# Tier vocabulary constants
# ---------------------------------------------------------------------------

# Non-canonical words that are sometimes used in place of canonical tier names.
# Case-insensitive matching is applied at check time.
_NON_CANONICAL_TIER_WORDS: frozenset[str] = frozenset(
    {"standard", "pro", "basic", "premium", "free", "plus", "business", "team"}
)

# Canonical tier names in lowercase for case-drift detection in claims.
_CANONICAL_LOWER: frozenset[str] = frozenset(t.lower() for t in CANONICAL_TIER_NAMES)

# Combined set: anything that looks like a tier name token.
_ALL_TIER_LIKE: frozenset[str] = _NON_CANONICAL_TIER_WORDS | _CANONICAL_LOWER | frozenset(
    t.lower() for t in CANONICAL_TIER_NAMES
)

# Words that signal a tier-name context in free text.
_TIER_CTX = r"(?:plans?|tiers?|pricing|subscriptions?|accounts?|customers?)"

# Non-canonical tier pattern (case-insensitive word boundary, not hyphenated).
_NC = r"(?:" + "|".join(sorted(_NON_CANONICAL_TIER_WORDS)) + r")"

_TEXT_PATTERNS: list[re.Pattern[str]] = [
    # Non-canonical word immediately before a tier-context word (0–1 intervening words).
    re.compile(rf"(?i)\b{_NC}(?!-)\b(?:\s+\w+)?\s+{_TIER_CTX}\b"),
    # Non-canonical word immediately after a tier-context word.
    re.compile(rf"(?i)\b{_TIER_CTX}\s+\b{_NC}(?!-)\b"),
    # Preposition + non-canonical tier name.
    re.compile(rf"(?i)\b(?:upgrade to|on|for|using|switch to|downgrade to|available (?:on|for))\s+\b{_NC}(?!-)\b"),
    # Non-canonical in a list alongside a canonical tier name.
    re.compile(
        rf"(?i)\b{_NC}(?!-)\b\s*(?:,\s*|\s+or\s+|\s+and\s+)"
        rf"(?:{'|'.join(t.lower() for t in CANONICAL_TIER_NAMES)})\b"
    ),
    # Canonical tier name listed alongside a non-canonical word.
    re.compile(
        rf"(?i)\b(?:{'|'.join(t.lower() for t in CANONICAL_TIER_NAMES)})\b"
        rf"\s*(?:,\s*|\s+or\s+|\s+and\s+)\b{_NC}(?!-)\b"
    ),
]


# ---------------------------------------------------------------------------
# Dataclasses for global coverage aggregation
# ---------------------------------------------------------------------------


@dataclass
class FeatureGateClaims:
    """Plan-tier feature availability assertions."""

    sso_available_plans: list[str] | None = None
    sso_org_level_enforced: bool | None = None
    priority_support_included_in_enterprise: bool | None = None
    priority_support_price_monthly_usd: float | None = None
    plan_names_seen: list[str] = field(default_factory=list)


@dataclass
class RefundPolicyClaims:
    """Refund window, eligibility, and processing time assertions."""

    refund_window_days: int | None = None
    annual_monetary_refund_eligible: bool | None = None
    refund_denied_if_api_credits_exceeded: int | None = None
    processing_days_current_min: int | None = None
    processing_days_current_max: int | None = None
    exceptions_after_window: bool | None = None


@dataclass
class ApiLimitsClaims:
    """API rate limits and webhook configuration assertions."""

    rate_limit_starter_per_min: int | None = None
    rate_limit_enterprise_per_min: int | None = None
    webhook_response_timeout_seconds: int | None = None
    webhook_protocol: str | None = None


@dataclass
class DataPolicyClaims:
    """Data retention assertions."""

    retention_days_post_cancellation: int | None = None


@dataclass
class GlobalState:
    """Aggregated claims across all KB chunks — used only for coverage checks."""

    feature_gate: FeatureGateClaims = field(default_factory=FeatureGateClaims)
    refund: RefundPolicyClaims = field(default_factory=RefundPolicyClaims)
    api: ApiLimitsClaims = field(default_factory=ApiLimitsClaims)
    data: DataPolicyClaims = field(default_factory=DataPolicyClaims)
    sources: dict[str, list[str]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# InvariantReport
# ---------------------------------------------------------------------------


@dataclass
class InvariantReport:
    """Result of running the invariant checker against a set of KB chunks."""

    errors: list[str]
    warnings: list[str]


# ---------------------------------------------------------------------------
# Per-chunk validators
# ---------------------------------------------------------------------------


def validate_chunk_structure(chunk: KBChunk) -> list[str]:
    """
    Per-chunk structural validation. Returns a list of error messages.

    Currently a thin extension point — structural rules (domain presence, gate
    references) are enforced by test_kb_lint.py and test_properties.py.
    Add rules here as the checker grows.
    """
    return []


def validate_controlled_vocab(chunk: KBChunk) -> list[str]:
    """
    Per-chunk controlled-vocabulary validation. Returns a list of error messages.

    Two-part check with explicit asymmetry:
      - Claims (strict): any string claim value that looks like a tier name must
        be an exact member of CANONICAL_TIER_NAMES.
      - Text (heuristic): pattern-based proximity matching detects non-canonical
        tier words used in a tier-name context. See _TEXT_PATTERNS for the
        contracts; tests/test_invariant_checker.py is the source of truth.

    Adversarial chunks are excluded — their text is intentionally misleading.
    """
    if chunk.adversarial:
        return []

    errors: list[str] = []
    cid = chunk.chunk_id

    # --- Claims (strict) ---
    for key, value in (chunk.claims or {}).items():
        if not isinstance(value, str):
            continue
        lower = value.lower()
        if lower in _ALL_TIER_LIKE and value not in CANONICAL_TIER_NAMES:
            errors.append(
                f"{cid}: claim '{key}' has non-canonical tier name '{value}' "
                f"— use one of {sorted(CANONICAL_TIER_NAMES)}"
            )

    # --- Text (heuristic) ---
    for pattern in _TEXT_PATTERNS:
        match = pattern.search(chunk.chunk_text)
        if match:
            errors.append(
                f"{cid}: chunk text contains non-canonical tier reference "
                f"'{match.group(0).strip()}' — use canonical names "
                f"{sorted(CANONICAL_TIER_NAMES)}"
            )
            break  # one error per chunk; first match is sufficient

    return errors


# ---------------------------------------------------------------------------
# Global coverage helpers
# ---------------------------------------------------------------------------


def _fmt_sources(state: GlobalState, *field_names: str) -> str:
    all_sources: list[str] = []
    for name in field_names:
        all_sources.extend(state.sources.get(name, []))
    return f"(sources: {', '.join(sorted(set(all_sources)))})"


def _track(state: GlobalState, field_name: str, chunk_id: str) -> None:
    state.sources.setdefault(field_name, [])
    state.sources[field_name].append(chunk_id)


def _ingest(chunk_id: str, claims: dict[str, Any], state: GlobalState) -> None:
    """
    Merge a single chunk's claims into the global state for coverage tracking.

    Only updates fields — no cross-chunk comparison is performed here.
    The global state is used exclusively to detect coverage gaps.
    """
    fg = state.feature_gate
    rf = state.refund
    ap = state.api
    da = state.data

    if "sso_available_plans" in claims:
        fg.sso_available_plans = claims["sso_available_plans"]
        _track(state, "sso_available_plans", chunk_id)

    if "sso_org_level_enforced" in claims:
        fg.sso_org_level_enforced = claims["sso_org_level_enforced"]
        _track(state, "sso_org_level_enforced", chunk_id)

    if "priority_support_included_in_enterprise" in claims:
        fg.priority_support_included_in_enterprise = claims["priority_support_included_in_enterprise"]
        _track(state, "priority_support_included_in_enterprise", chunk_id)

    if "priority_support_price_monthly_usd" in claims:
        fg.priority_support_price_monthly_usd = claims["priority_support_price_monthly_usd"]
        _track(state, "priority_support_price_monthly_usd", chunk_id)

    if "plan_name" in claims and claims["plan_name"] is not None:
        fg.plan_names_seen.append(str(claims["plan_name"]))
        _track(state, "plan_names_seen", chunk_id)

    if "refund_window_days" in claims:
        rf.refund_window_days = claims["refund_window_days"]
        _track(state, "refund_window_days", chunk_id)

    if "annual_monetary_refund_eligible" in claims:
        rf.annual_monetary_refund_eligible = claims["annual_monetary_refund_eligible"]
        _track(state, "annual_monetary_refund_eligible", chunk_id)

    if "refund_denied_if_api_credits_exceeded" in claims:
        rf.refund_denied_if_api_credits_exceeded = claims["refund_denied_if_api_credits_exceeded"]
        _track(state, "refund_denied_if_api_credits_exceeded", chunk_id)

    if "exceptions_after_window" in claims:
        rf.exceptions_after_window = claims["exceptions_after_window"]
        _track(state, "exceptions_after_window", chunk_id)

    is_current = claims.get("policy_status") == "current"
    if is_current:
        for key_min in ("processing_days_min", "refund_processing_days_min"):
            if key_min in claims:
                rf.processing_days_current_min = claims[key_min]
                _track(state, "processing_days_current_min", chunk_id)
                break
        for key_max in ("processing_days_max", "refund_processing_days_max"):
            if key_max in claims:
                rf.processing_days_current_max = claims[key_max]
                _track(state, "processing_days_current_max", chunk_id)
                break

    # rate_limit_standard_per_min is retained as claim key for backward
    # compatibility; the canonical tier name "Starter" lives in chunk text.
    if "rate_limit_standard_per_min" in claims:
        ap.rate_limit_starter_per_min = claims["rate_limit_standard_per_min"]
        _track(state, "rate_limit_starter_per_min", chunk_id)

    if "rate_limit_enterprise_per_min" in claims:
        ap.rate_limit_enterprise_per_min = claims["rate_limit_enterprise_per_min"]
        _track(state, "rate_limit_enterprise_per_min", chunk_id)

    if "webhook_response_timeout_seconds" in claims:
        ap.webhook_response_timeout_seconds = claims["webhook_response_timeout_seconds"]
        _track(state, "webhook_response_timeout_seconds", chunk_id)

    if "webhook_protocol" in claims:
        ap.webhook_protocol = claims["webhook_protocol"]
        _track(state, "webhook_protocol", chunk_id)

    if "retention_days_post_cancellation" in claims:
        da.retention_days_post_cancellation = claims["retention_days_post_cancellation"]
        _track(state, "retention_days_post_cancellation", chunk_id)


# ---------------------------------------------------------------------------
# Global coverage checks (not cross-chunk claim comparison)
# ---------------------------------------------------------------------------


def _check_feature_gates(state: GlobalState) -> tuple[list[str], list[str]]:
    """
    Check plan-tier feature gating consistency.

    Detects: priority support simultaneously marked as included in Enterprise
    AND carrying a separate monthly price (mutually exclusive states).
    Warns: no plan names seen across all chunks.
    """
    errors, warnings = [], []

    if (
        state.feature_gate.priority_support_included_in_enterprise is True
        and state.feature_gate.priority_support_price_monthly_usd is not None
    ):
        sources = _fmt_sources(
            state,
            "priority_support_included_in_enterprise",
            "priority_support_price_monthly_usd",
        )
        errors.append(
            f"Priority support contradiction: marked as included in Enterprise AND "
            f"priced at ${state.feature_gate.priority_support_price_monthly_usd}/mo. {sources}"
        )

    if not state.feature_gate.plan_names_seen:
        warnings.append("No plan names (Starter/Growth/Enterprise) found in KB claims")

    return errors, warnings


def _check_refund_policy(state: GlobalState) -> tuple[list[str], list[str]]:
    """
    Check refund window coverage.

    Warns only when no chunk defines a refund window length (coverage gap).
    Multiple chunks may define this field without conflict — cross-chunk
    comparison is not performed.
    """
    errors, warnings = [], []

    if state.refund.refund_window_days is None:
        warnings.append("No chunk defines a refund window length")

    return errors, warnings


def _check_api_limits(state: GlobalState) -> tuple[list[str], list[str]]:
    """
    Check API rate limit coverage.

    Warns when either the Starter or Enterprise plan rate limit is absent from
    the KB claims — both are required for agents to answer tier-specific queries.
    """
    errors, warnings = [], []

    if state.api.rate_limit_starter_per_min is None:
        warnings.append("No Starter plan API rate limit defined in KB claims")
    if state.api.rate_limit_enterprise_per_min is None:
        warnings.append("No Enterprise plan API rate limit defined in KB claims")

    return errors, warnings


def _check_coverage(state: GlobalState) -> tuple[list[str], list[str]]:
    """
    Check that critical invariant groups have at least one claim.

    Coverage gaps are warnings (not errors) — a missing claim means the checker
    has no data, not that the KB contains a contradiction.
    """
    errors, warnings = [], []

    if state.refund.refund_window_days is None:
        warnings.append("No refund window defined — refund policy coverage may be insufficient")
    if state.data.retention_days_post_cancellation is None:
        warnings.append("No post-cancellation data retention period defined")

    return errors, warnings


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_checker(chunks: list[KBChunk]) -> InvariantReport:
    """
    Run the invariant checker against a list of KB chunks.

    Returns an InvariantReport with errors (contradictions or vocabulary
    violations) and warnings (coverage gaps).

    Per-chunk validation (validate_chunk_structure + validate_controlled_vocab)
    runs for every non-adversarial chunk. Global coverage checks aggregate claim
    presence across all non-adversarial chunks but never compare claim values
    across chunks.
    """
    state = GlobalState()
    errors: list[str] = []
    warnings: list[str] = []

    for chunk in chunks:
        if chunk.adversarial:
            continue

        errors.extend(validate_chunk_structure(chunk))
        errors.extend(validate_controlled_vocab(chunk))

        if chunk.claims:
            _ingest(chunk.chunk_id, chunk.claims, state)

    for checker in [_check_feature_gates, _check_refund_policy, _check_api_limits, _check_coverage]:
        e, w = checker(state)
        errors.extend(e)
        warnings.extend(w)

    return InvariantReport(errors=errors, warnings=warnings)
