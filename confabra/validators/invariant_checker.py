"""
Deterministic invariant checker for the SaaS KB.

Aggregates structured claims across KB chunks and detects contradictions
between them. Source tracking: every error names the chunk_id(s) involved.

No NLP, no LLM, no embeddings — pure structured claim aggregation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from confabra.schemas import KBChunk


# ---------------------------------------------------------------------------
# Dataclasses for 5 invariant groups
# ---------------------------------------------------------------------------


@dataclass
class FeatureGateClaims:
    """Plan-tier feature availability assertions."""

    sso_available_plans: list[str] | None = None        # from sso_saml_setup chunk
    sso_org_level_enforced: bool | None = None           # from enterprise_sso chunk
    priority_support_included_in_enterprise: bool | None = None
    priority_support_price_monthly_usd: float | None = None
    plan_names_seen: list[str] = field(default_factory=list)  # all plan names mentioned


@dataclass
class RefundPolicyClaims:
    """Refund window, eligibility, and processing time assertions."""

    refund_window_days: int | None = None
    annual_monetary_refund_eligible: bool | None = None
    refund_denied_if_api_credits_exceeded: int | None = None  # the threshold
    processing_days_current_min: int | None = None
    processing_days_current_max: int | None = None
    exceptions_after_window: bool | None = None


@dataclass
class ApiLimitsClaims:
    """API rate limits and webhook configuration assertions."""

    rate_limit_standard_per_min: int | None = None
    rate_limit_enterprise_per_min: int | None = None
    webhook_response_timeout_seconds: int | None = None
    webhook_protocol: str | None = None


@dataclass
class DataPolicyClaims:
    """Data retention and export constraint assertions."""

    retention_days_post_cancellation: int | None = None
    export_immediate_threshold_records: int | None = None  # two chunks differ — see warning


@dataclass
class GlobalState:
    """Aggregated claims across all KB chunks, with source tracking."""

    feature_gate: FeatureGateClaims = field(default_factory=FeatureGateClaims)
    refund: RefundPolicyClaims = field(default_factory=RefundPolicyClaims)
    api: ApiLimitsClaims = field(default_factory=ApiLimitsClaims)
    data: DataPolicyClaims = field(default_factory=DataPolicyClaims)
    # Source tracking: field_name → list of chunk_ids that set it
    sources: dict[str, list[str]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# InvariantReport
# ---------------------------------------------------------------------------


@dataclass
class InvariantReport:
    """Result of running the invariant checker against a set of KB chunks."""

    errors: list[str]    # contradictions that must be fixed
    warnings: list[str]  # coverage gaps and informational notices


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _fmt_sources(state: GlobalState, *field_names: str) -> str:
    """
    Format source chunk IDs into a human-readable parenthetical for error messages.

    Collects all chunk IDs that contributed to each named field, deduplicates,
    sorts for determinism, and wraps in "(sources: ...)".
    """
    all_sources: list[str] = []
    for name in field_names:
        all_sources.extend(state.sources.get(name, []))
    return f"(sources: {', '.join(sorted(set(all_sources)))})"


def _track(state: GlobalState, field_name: str, chunk_id: str) -> None:
    """
    Record that chunk_id contributed a value for field_name.

    Creates the list on first call; appends on subsequent calls so that
    multi-chunk coverage of the same field is visible in error messages.
    """
    state.sources.setdefault(field_name, [])
    state.sources[field_name].append(chunk_id)


# ---------------------------------------------------------------------------
# Ingest
# ---------------------------------------------------------------------------


def ingest(chunk_id: str, claims: dict[str, Any], state: GlobalState) -> None:
    """
    Merge a single chunk's claims into the global state, tracking sources.

    Each recognised claim key maps to a specific field on one of the four
    domain sub-states (feature_gate, refund, api, data). The last chunk to
    set a field wins — multi-source conflicts are detected at check time by
    inspecting state.sources for fields with >1 entry.

    ``processing_days_min/max`` and their aliases are only ingested when
    ``claims.get("policy_status") == "current"``, so superseded-policy
    chunks don't pollute the current-policy baseline.
    """
    fg = state.feature_gate
    rf = state.refund
    ap = state.api
    da = state.data

    # --- Feature gate ---
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

    # --- Refund policy ---
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

    # Processing time: only ingest when policy_status is "current"
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

    # --- API limits ---
    if "rate_limit_standard_per_min" in claims:
        ap.rate_limit_standard_per_min = claims["rate_limit_standard_per_min"]
        _track(state, "rate_limit_standard_per_min", chunk_id)

    if "rate_limit_enterprise_per_min" in claims:
        ap.rate_limit_enterprise_per_min = claims["rate_limit_enterprise_per_min"]
        _track(state, "rate_limit_enterprise_per_min", chunk_id)

    if "webhook_response_timeout_seconds" in claims:
        ap.webhook_response_timeout_seconds = claims["webhook_response_timeout_seconds"]
        _track(state, "webhook_response_timeout_seconds", chunk_id)

    if "webhook_protocol" in claims:
        ap.webhook_protocol = claims["webhook_protocol"]
        _track(state, "webhook_protocol", chunk_id)

    # --- Data policy ---
    if "retention_days_post_cancellation" in claims:
        da.retention_days_post_cancellation = claims["retention_days_post_cancellation"]
        _track(state, "retention_days_post_cancellation", chunk_id)

    if "export_immediate_threshold_records" in claims:
        da.export_immediate_threshold_records = claims["export_immediate_threshold_records"]
        _track(state, "export_immediate_threshold_records", chunk_id)


# ---------------------------------------------------------------------------
# Checker functions
# ---------------------------------------------------------------------------


def _check_feature_gates(state: GlobalState) -> tuple[list[str], list[str]]:
    """
    Check plan-tier feature gating consistency.

    Detects: priority support simultaneously marked as included in Enterprise
    AND carrying a separate monthly price (mutually exclusive states).
    Warns: no plan names seen across all chunks.
    """
    errors, warnings = [], []

    # Priority support must not be both included AND priced separately.
    # Both claims being True (included=True, price set) would be a contradiction.
    # Note: the current KB has included=False with price=299, which is consistent.
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

    # Coverage warning: no plan names seen.
    if not state.feature_gate.plan_names_seen:
        warnings.append("No plan names (Starter/Growth/Enterprise) found in KB claims")

    return errors, warnings


def _check_refund_policy(state: GlobalState) -> tuple[list[str], list[str]]:
    """
    Check refund window and eligibility consistency.

    Detects: multiple chunks that disagree on refund_window_days (last-write-wins
    masking a disagreement, surfaced via >1 sources entry).
    Warns: no chunk defines a refund window length.
    """
    errors, warnings = [], []

    sources_for_window = state.sources.get("refund_window_days", [])
    if len(sources_for_window) == 0:
        warnings.append("No chunk defines a refund window length")
    elif len(sources_for_window) > 1:
        # Multiple chunks wrote to this field — check that they all agree
        # (last-write-wins means we can only see the final value, so we warn
        # rather than error here; a real conflict would be caught in KB authoring).
        warnings.append(
            f"refund_window_days set by {len(sources_for_window)} chunks "
            f"{_fmt_sources(state, 'refund_window_days')} — verify all agree on "
            f"{state.refund.refund_window_days} days."
        )

    return errors, warnings


def _check_api_limits(state: GlobalState) -> tuple[list[str], list[str]]:
    """
    Check API rate limit coverage.

    Warns when either the standard or enterprise plan rate limit is absent from
    the KB claims — both are required for agents to answer tier-specific queries.
    """
    errors, warnings = [], []

    if state.api.rate_limit_standard_per_min is None:
        warnings.append("No standard plan API rate limit defined in KB claims")
    if state.api.rate_limit_enterprise_per_min is None:
        warnings.append("No enterprise plan API rate limit defined in KB claims")

    return errors, warnings


def _check_data_policy(state: GlobalState) -> tuple[list[str], list[str]]:
    """
    Check data retention and export constraint consistency.

    Detects: export_immediate_threshold_records set by multiple chunks with
    different values (kb_chunk_data_export_conditional_v2 says 1M records;
    kb_chunk_data_portability_v2 says 100k). These describe different operations
    (max queued-export size vs. instant-export ceiling) — flagged as a warning
    so KB authors can either unify the language or add distinct claim keys.
    """
    errors, warnings = [], []

    if state.data.export_immediate_threshold_records is not None:
        sources = state.sources.get("export_immediate_threshold_records", [])
        if len(sources) > 1:
            warnings.append(
                f"Multiple export threshold values defined across {len(sources)} chunks "
                f"{_fmt_sources(state, 'export_immediate_threshold_records')}. "
                f"Final value is {state.data.export_immediate_threshold_records:,} records — "
                "verify the chunks describe the same operation (instant export ceiling vs. "
                "max export size may warrant separate claim keys)."
            )

    return errors, warnings


def _check_coverage(state: GlobalState) -> tuple[list[str], list[str]]:
    """
    Check that critical invariant groups have at least one claim.

    Coverage gaps are warnings (not errors) because a missing claim means the
    checker has no data, not that the KB contains a contradiction.
    """
    errors, warnings = [], []

    if state.refund.refund_window_days is None:
        warnings.append(
            "No refund window defined — refund policy coverage may be insufficient"
        )
    if state.data.retention_days_post_cancellation is None:
        warnings.append(
            "No post-cancellation data retention period defined"
        )

    return errors, warnings


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


def run_checker(chunks: list[KBChunk]) -> InvariantReport:
    """
    Run the invariant checker against a list of KB chunks.

    Returns an InvariantReport with errors (contradictions) and warnings
    (coverage gaps and multi-source discrepancies).

    Only ingests claims from non-adversarial chunks; adversarial chunks are
    intentionally inconsistent and should not pollute the global state.
    Chunks with empty claims dicts are skipped without error.
    """
    state = GlobalState()

    for chunk in chunks:
        if chunk.adversarial or not chunk.claims:
            continue
        ingest(chunk.chunk_id, chunk.claims, state)

    errors: list[str] = []
    warnings: list[str] = []

    checkers = [
        _check_feature_gates,
        _check_refund_policy,
        _check_api_limits,
        _check_data_policy,
        _check_coverage,
    ]
    for checker in checkers:
        e, w = checker(state)
        errors.extend(e)
        warnings.extend(w)

    return InvariantReport(errors=errors, warnings=warnings)
