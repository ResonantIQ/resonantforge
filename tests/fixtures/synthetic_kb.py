"""
Synthetic KB fixture for PR1 engine tests.

10 chunks across 3 domains: billing (3), api (3), refunds (4).
The refunds domain is intentionally chunk-count-skewed to test within-topic
normalization — selection should not over-weight refunds just because the domain
has more chunks.

Adversarial distribution:
  billing:  1 adversarial (1/3 = 33%)
  api:      1 adversarial (1/3 = 33%)
  refunds:  2 adversarial (2/4 = 50%)
"""

from __future__ import annotations

from resonantforge.schemas import ConstraintType, KBChunk

SYNTHETIC_KB_CHUNKS: list[KBChunk] = [
    # -----------------------------------------------------------------------
    # billing: 3 chunks (1 adversarial)
    # -----------------------------------------------------------------------
    KBChunk(
        chunk_id="syn_billing_cycle_v1",
        document_id="doc_syn_billing",
        document_path="policies/billing.md",
        chunk_text=(
            "Billing is processed monthly on the first of each calendar month. "
            "Invoices are sent 5 days before the billing date."
        ),
        constraint_type=ConstraintType.INFORMATIONAL,
        domains=["billing"],
        adversarial=False,
    ),
    KBChunk(
        chunk_id="syn_billing_error_refund_v1",
        document_id="doc_syn_billing",
        document_path="policies/billing.md",
        chunk_text=(
            "Billing errors are eligible for a full refund if reported within "
            "30 days of the invoice date. Refunds are processed within 5 business days."
        ),
        constraint_type=ConstraintType.ALLOW_CONDITION,
        domains=["billing"],
        adversarial=False,
    ),
    KBChunk(
        chunk_id="syn_billing_adversarial_v1",
        document_id="doc_syn_billing",
        document_path="policies/billing.md",
        chunk_text=(
            "All billing disputes are automatically resolved in the customer's favour "
            "with a full refund issued within 24 hours."  # trap: overgeneralisation
        ),
        constraint_type=ConstraintType.DENY_CONDITION,
        domains=["billing"],
        adversarial=True,
    ),

    # -----------------------------------------------------------------------
    # api: 3 chunks (1 adversarial)
    # -----------------------------------------------------------------------
    KBChunk(
        chunk_id="syn_api_rate_limits_v1",
        document_id="doc_syn_api",
        document_path="docs/api.md",
        chunk_text=(
            "API rate limits are enforced per API key. Starter plans allow 100 requests "
            "per minute; Growth plans allow 1,000; Enterprise plans allow 10,000."
        ),
        constraint_type=ConstraintType.ALLOW_CONDITION,
        domains=["api"],
        adversarial=False,
    ),
    KBChunk(
        chunk_id="syn_api_auth_v1",
        document_id="doc_syn_api",
        document_path="docs/api.md",
        chunk_text=(
            "API authentication requires a Bearer token in the Authorization header. "
            "Tokens expire after 24 hours and must be refreshed using the /auth/refresh endpoint."
        ),
        constraint_type=ConstraintType.INFORMATIONAL,
        domains=["api"],
        adversarial=False,
    ),
    KBChunk(
        chunk_id="syn_api_adversarial_v1",
        document_id="doc_syn_api",
        document_path="docs/api.md",
        chunk_text=(
            "There are no rate limits on API requests for any plan. "
            "Customers can make unlimited requests at any time."  # trap: false
        ),
        constraint_type=ConstraintType.DENY_CONDITION,
        domains=["api"],
        adversarial=True,
    ),

    # -----------------------------------------------------------------------
    # refunds: 4 chunks (2 adversarial) — intentionally dense to test normalisation
    # -----------------------------------------------------------------------
    KBChunk(
        chunk_id="syn_refund_policy_v1",
        document_id="doc_syn_refunds",
        document_path="policies/refund_policy.md",
        chunk_text=(
            "Refunds are available for paid plans only. Free-tier accounts are not "
            "eligible for refunds under any circumstances."
        ),
        constraint_type=ConstraintType.ALLOW_CONDITION,
        domains=["refunds"],
        adversarial=False,
    ),
    KBChunk(
        chunk_id="syn_refund_window_v1",
        document_id="doc_syn_refunds",
        document_path="policies/refund_policy.md",
        chunk_text=(
            "Refund requests must be submitted within 14 days of the charge date. "
            "Requests outside this window are not eligible."
        ),
        constraint_type=ConstraintType.ALLOW_CONDITION,
        domains=["refunds"],
        adversarial=False,
    ),
    KBChunk(
        chunk_id="syn_refund_adversarial_overgeneralise_v1",
        document_id="doc_syn_refunds",
        document_path="policies/refund_policy.md",
        chunk_text=(
            "All customers are eligible for a refund at any time, for any reason, "
            "with no time limit."  # trap: omits paid-plan-only and 14-day constraints
        ),
        constraint_type=ConstraintType.DENY_CONDITION,
        domains=["refunds"],
        adversarial=True,
    ),
    KBChunk(
        chunk_id="syn_refund_adversarial_stale_v1",
        document_id="doc_syn_refunds",
        document_path="policies/refund_policy.md",
        chunk_text=(
            "Per the legacy policy (deprecated 2023-01-01): refunds were available "
            "within 30 days of purchase for all plans."  # trap: stale policy
        ),
        constraint_type=ConstraintType.DENY_CONDITION,
        domains=["refunds"],
        adversarial=True,
    ),
]
