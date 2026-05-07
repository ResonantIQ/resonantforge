"""Tests for kb_fact_extractor.extract_facts.

Precision over recall: the extractor must return high-quality candidates only.
Reject cases are as important as the positive cases — they prevent garbage from
polluting the contradicted:exact generation pipeline.
"""

import pytest

from resonantforge.kb.kb_fact_extractor import KbFact, extract_facts


# ── Category: numeric ─────────────────────────────────────────────────────────


def test_extracts_numeric_seat_limit():
    facts = extract_facts("Your plan supports up to 500 users.")
    assert any(f.fact_category == "numeric" and "500" in f.source_text for f in facts)


def test_extracts_numeric_retention_period():
    facts = extract_facts("Data is retained for 90 days after account closure.")
    assert any(f.fact_category == "numeric" and "90" in f.source_text for f in facts)


def test_extracts_numeric_monthly_price():
    facts = extract_facts("The Growth plan costs $99/month.")
    assert any(f.fact_category == "numeric" and "$99" in f.source_text for f in facts)


def test_extracts_numeric_sla_percentage():
    facts = extract_facts("We guarantee 99.9% uptime per calendar month.")
    assert any(f.fact_category == "numeric" and "99.9" in f.source_text for f in facts)


def test_extracts_numeric_rate_limit():
    facts = extract_facts("The API rate limit is 1000 requests per hour.")
    assert any(f.fact_category == "numeric" and "1000" in f.source_text for f in facts)


# ── Category: categorical ─────────────────────────────────────────────────────


def test_extracts_categorical_plan_restriction():
    facts = extract_facts("API access is available on Enterprise plans only.")
    assert any(f.fact_category == "categorical" for f in facts)


def test_extracts_categorical_sso_support():
    facts = extract_facts("The platform supports SAML SSO for Enterprise accounts.")
    assert any(f.fact_category == "categorical" and "SAML" in f.source_text for f in facts)


def test_extracts_categorical_data_location():
    facts = extract_facts("All customer data is stored in US-East data centers.")
    assert any(f.fact_category == "categorical" for f in facts)


def test_extracts_categorical_all_plans_available():
    facts = extract_facts("Two-factor authentication is available on all plans.")
    assert any(f.fact_category == "categorical" for f in facts)


# ── Category: capability ──────────────────────────────────────────────────────


def test_extracts_capability_csv_export():
    facts = extract_facts("You can export your data to CSV from the Reports tab.")
    assert any(f.fact_category == "capability" and "CSV" in f.source_text for f in facts)


def test_extracts_capability_crm_integration():
    facts = extract_facts("Resonant IQ integrates with Salesforce CRM out of the box.")
    assert any(f.fact_category == "capability" and "Salesforce" in f.source_text for f in facts)


def test_extracts_capability_webhook_support():
    facts = extract_facts("The platform supports outbound webhooks for event notifications.")
    assert any(f.fact_category == "capability" and "webhook" in f.source_text.lower() for f in facts)


# ── Category: policy ─────────────────────────────────────────────────────────


def test_extracts_policy_refund_window():
    facts = extract_facts("Refunds are available within 30 days of purchase.")
    assert any(f.fact_category == "policy" and "30" in f.source_text for f in facts)


def test_extracts_policy_auth_requirement():
    facts = extract_facts("All API endpoints require authentication via API key.")
    assert any(f.fact_category == "policy" for f in facts)


def test_extracts_policy_no_refund():
    facts = extract_facts("All sales are final — no refunds are issued.")
    assert any(f.fact_category == "policy" for f in facts)


def test_extracts_policy_sla_credit():
    facts = extract_facts("SLA credits are issued automatically for downtime exceeding 99.9%.")
    assert any(f.fact_category == "policy" for f in facts)


# ── Reject: comparative adjectives ───────────────────────────────────────────


def test_rejects_best_in_class():
    facts = extract_facts("Our platform is best-in-class for enterprise customers.")
    assert len(facts) == 0


def test_rejects_industry_leading():
    facts = extract_facts("We provide industry-leading support and reliability.")
    assert len(facts) == 0


def test_rejects_superior_performance():
    facts = extract_facts("Experience superior performance and unmatched scalability.")
    assert len(facts) == 0


def test_rejects_award_winning():
    facts = extract_facts("Our award-winning platform transforms customer support.")
    assert len(facts) == 0


# ── Reject: vague quantifiers ─────────────────────────────────────────────────


def test_rejects_vague_fast_reliable():
    facts = extract_facts("Our system is fast and reliable.")
    assert len(facts) == 0


def test_rejects_vague_many():
    facts = extract_facts("Many customers rely on our platform daily.")
    assert len(facts) == 0


def test_rejects_vague_highly_reliable():
    facts = extract_facts("We are highly reliable and always available.")
    assert len(facts) == 0


def test_rejects_vague_robust():
    facts = extract_facts("A robust and scalable solution for modern enterprises.")
    assert len(facts) == 0


# ── Reject: nested conditionals ───────────────────────────────────────────────


def test_rejects_multi_clause_nested_conditional():
    chunk = (
        "If you are on an Enterprise plan and your contract includes premium support, "
        "then you qualify for dedicated SLAs unless you opted out during onboarding."
    )
    facts = extract_facts(chunk)
    assert len(facts) == 0


# ── Reject: world-knowledge-dependent facts ───────────────────────────────────


def test_rejects_compliance_without_specifics():
    facts = extract_facts("Our platform complies with all applicable industry standards.")
    assert len(facts) == 0


def test_rejects_meets_regulatory_requirements():
    facts = extract_facts("We meet all relevant regulatory requirements for data handling.")
    assert len(facts) == 0


# ── Reject: marketing prose ───────────────────────────────────────────────────


def test_rejects_transform_your_experience():
    facts = extract_facts("Transform your customer experience with our powerful platform.")
    assert len(facts) == 0


def test_rejects_unlock_the_power():
    facts = extract_facts("Unlock the power of AI-driven insights with seamless integration.")
    assert len(facts) == 0


# ── Edge cases ────────────────────────────────────────────────────────────────


def test_empty_chunk_returns_empty():
    assert extract_facts("") == []


def test_whitespace_only_returns_empty():
    assert extract_facts("   \n\t  ") == []


def test_returns_kbfact_instances():
    facts = extract_facts("The plan retains data for 90 days.")
    assert all(isinstance(f, KbFact) for f in facts)


def test_normalized_field_is_lowercase_collapsed():
    facts = extract_facts("Data is retained for  90  days  after closure.")
    for f in facts:
        assert f.normalized == " ".join(f.normalized.lower().split())


def test_mixed_chunk_extracts_concrete_rejects_vague():
    """Chunk with both extractable and vague content: concrete facts come through."""
    chunk = "Our best-in-class platform supports up to 500 users and offers industry-leading reliability."
    facts = extract_facts(chunk)
    assert any("500" in f.source_text for f in facts)
    assert not any("best-in-class" in f.source_text.lower() for f in facts)
    assert not any("industry-leading" in f.source_text.lower() for f in facts)


def test_deduplicates_identical_normalized_facts():
    chunk = "Supports up to 500 users. You can have up to 500 users on this plan."
    facts = extract_facts(chunk)
    normalized_texts = [f.normalized for f in facts]
    assert len(normalized_texts) == len(set(normalized_texts))
