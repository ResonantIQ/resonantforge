"""Tests for negate_fact.

Each mutation band must produce an unmistakable contradiction — not a subtle
tweak. Validators use substring presence/absence, so the negated form must
be obviously incompatible with the original.
"""

import pytest

from resonantforge.kb.kb_fact_extractor import KbFact
from resonantforge.kb.negation import negate_fact


def _fact(source_text: str, category: str) -> KbFact:
    return KbFact(
        source_text=source_text,
        normalized=" ".join(source_text.lower().split()),
        fact_category=category,
    )


# ── Numeric mutations ─────────────────────────────────────────────────────────


def test_numeric_multiplied_by_10():
    """Standard numeric: multiply by 10."""
    result = negate_fact(_fact("up to 500 users", "numeric"))
    assert "5000" in result


def test_numeric_retention_multiplied():
    result = negate_fact(_fact("Data is retained for 90 days", "numeric"))
    assert "900" in result


def test_numeric_rate_limit_multiplied():
    result = negate_fact(_fact("1000 requests per hour", "numeric"))
    assert "10000" in result


def test_numeric_currency_multiplied():
    result = negate_fact(_fact("$99/month", "numeric"))
    assert "990" in result


def test_numeric_percentage_polarity_flip():
    """SLA-style percentage: flip to unmistakably low value (not within 50%)."""
    result = negate_fact(_fact("99.9% uptime", "numeric"))
    # Must not be within 50% of 99.9 (i.e., must be < 49.95 or > 149.85)
    # Polarity flip strategy: negate to something like 50% or invert
    import re
    nums = re.findall(r"\d+(?:\.\d+)?", result)
    assert nums, "negated form must contain a number"
    val = float(nums[0])
    assert abs(val - 99.9) / 99.9 > 0.5, f"mutation {val} is within 50% of original 99.9"


def test_numeric_rejects_within_50_percent():
    """Any numeric mutation within 50% of the original is rejected."""
    # 500 → 510 would be within 50% (only 2% difference) — must not happen
    result = negate_fact(_fact("500 users", "numeric"))
    import re
    nums = re.findall(r"\d+", result)
    assert nums, "negated form must contain a number"
    val = int(nums[0])
    assert abs(val - 500) / 500 > 0.5, f"mutation {val} is within 50% of original 500"


def test_numeric_produces_string():
    result = negate_fact(_fact("$99/month", "numeric"))
    assert isinstance(result, str)
    assert result.strip()


# ── Categorical mutations ─────────────────────────────────────────────────────


def test_categorical_plan_restriction_inverted():
    """'Enterprise plans only' → claim it's available on all plans."""
    result = negate_fact(_fact("API access is available on Enterprise plans only", "categorical"))
    lower = result.lower()
    assert "all" in lower or "every" in lower or "any" in lower


def test_categorical_all_plans_inverted():
    """'Available on all plans' → restrict to enterprise only."""
    result = negate_fact(_fact("Two-factor authentication is available on all plans", "categorical"))
    lower = result.lower()
    assert "enterprise" in lower or "only" in lower or "paid" in lower


def test_categorical_data_location_inverted():
    """Data location: negate the stated region."""
    result = negate_fact(_fact("All customer data is stored in US-East data centers", "categorical"))
    lower = result.lower()
    # Should claim a different or no specific region
    assert "us-east" not in lower or "not" in lower or "eu" in lower or "asia" in lower


def test_categorical_sso_inverted():
    """SSO support: negate availability."""
    result = negate_fact(_fact("The platform supports SAML SSO for Enterprise accounts", "categorical"))
    lower = result.lower()
    assert "not" in lower or "does not" in lower or "no" in lower or "all" in lower


def test_categorical_produces_string():
    result = negate_fact(_fact("Enterprise plans only", "categorical"))
    assert isinstance(result, str)
    assert result.strip()


# ── Capability mutations ──────────────────────────────────────────────────────


def test_capability_csv_export_negated():
    """CSV export capability: outright negation."""
    result = negate_fact(_fact("You can export your data to CSV from the Reports tab", "capability"))
    lower = result.lower()
    assert "not" in lower or "cannot" in lower or "no" in lower or "does not" in lower


def test_capability_integration_negated():
    """CRM integration: outright negation."""
    result = negate_fact(_fact("Resonant IQ integrates with Salesforce CRM out of the box", "capability"))
    lower = result.lower()
    assert "not" in lower or "cannot" in lower or "no" in lower or "does not" in lower


def test_capability_webhook_negated():
    """Webhook support: outright negation."""
    result = negate_fact(_fact("The platform supports outbound webhooks for event notifications", "capability"))
    lower = result.lower()
    assert "not" in lower or "cannot" in lower or "no" in lower or "does not" in lower


def test_capability_produces_string():
    result = negate_fact(_fact("supports outbound webhooks", "capability"))
    assert isinstance(result, str)
    assert result.strip()


# ── Policy mutations ──────────────────────────────────────────────────────────


def test_policy_refund_inverted():
    """Refund-available policy → no refunds."""
    result = negate_fact(_fact("Refunds are available within 30 days of purchase", "policy"))
    lower = result.lower()
    assert "no" in lower or "not" in lower or "final" in lower


def test_policy_no_refund_inverted():
    """No-refund policy → refunds available."""
    result = negate_fact(_fact("All sales are final — no refunds are issued", "policy"))
    lower = result.lower()
    assert "refund" in lower and ("available" in lower or "eligible" in lower or "within" in lower)


def test_policy_auth_requirement_inverted():
    """Auth requirement → no authentication needed."""
    result = negate_fact(_fact("All API endpoints require authentication via API key", "policy"))
    lower = result.lower()
    assert "not" in lower or "no" in lower or "without" in lower or "optional" in lower


def test_policy_sla_credit_inverted():
    """SLA credit policy → credits not issued / manual process."""
    result = negate_fact(_fact("SLA credits are issued automatically for downtime exceeding 99.9%", "policy"))
    lower = result.lower()
    assert "not" in lower or "no" in lower or "manual" in lower or "never" in lower


def test_policy_produces_string():
    result = negate_fact(_fact("Refunds are available within 30 days", "policy"))
    assert isinstance(result, str)
    assert result.strip()


# ── Shared invariants ─────────────────────────────────────────────────────────


def test_negation_differs_from_source():
    """Negated form must not be identical to the source text."""
    fact = _fact("up to 500 users", "numeric")
    result = negate_fact(fact)
    assert result.lower() != fact.source_text.lower()


def test_negation_returns_nonempty():
    for source, category in [
        ("$99/month", "numeric"),
        ("Enterprise plans only", "categorical"),
        ("supports outbound webhooks", "capability"),
        ("Refunds are available within 30 days", "policy"),
    ]:
        result = negate_fact(_fact(source, category))
        assert result and result.strip(), f"empty result for {category!r}: {source!r}"
