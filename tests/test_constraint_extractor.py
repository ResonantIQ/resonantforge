"""
Unit tests for resonantforge.kb.constraint_extractor (RFORGE-29 TDD Step 1).

Covers all four constraint categories + conv_evt_00114-class bare numeric
+ empty chunk + deduplication + multiple constraints in one chunk.
"""
from __future__ import annotations

import pytest

from resonantforge.kb.constraint_extractor import Constraint, extract_constraints


class TestExtractConstraints:

    # --- Category 1: bare numeric limits (conv_evt_00114 class) ---

    def test_bare_numeric_seats(self):
        """Chunk says 'N seats' without 'up to' prefix — the original validator miss."""
        result = extract_constraints("The plan includes 10 seats per account.")
        assert any("10 seats" in c.normalized for c in result), (
            f"Expected '10 seats' in {[c.normalized for c in result]}"
        )

    def test_bare_numeric_users(self):
        result = extract_constraints("Access is limited to 5 users on this plan.")
        assert any("5 users" in c.normalized for c in result)

    def test_bare_numeric_devices(self):
        result = extract_constraints("Up to 3 devices may be registered per account.")
        assert any("3 devices" in c.normalized or "up to 3" in c.normalized for c in result)

    # --- Category 2: "up to N" / "max N" / quantified limits ---

    def test_up_to_n_with_unit(self):
        result = extract_constraints("Storage up to 50 GB per workspace is included.")
        normalized = [c.normalized for c in result]
        assert any("up to 50" in n for n in normalized), f"Got: {normalized}"

    def test_maximum_n(self):
        result = extract_constraints("A maximum of 3 devices may be registered per account.")
        normalized = [c.normalized for c in result]
        assert any("maximum" in n and "3" in n for n in normalized), f"Got: {normalized}"

    def test_no_more_than_n(self):
        result = extract_constraints("Customers may have no more than 5 active sessions.")
        normalized = [c.normalized for c in result]
        assert any("5" in n for n in normalized), f"Got: {normalized}"

    # --- Category 3: date/time windows ---

    def test_within_n_days(self):
        result = extract_constraints("Refunds are available within 30 days of purchase.")
        assert any("within 30 days" in c.normalized for c in result)

    def test_within_n_hours(self):
        result = extract_constraints("Response guaranteed within 24 hours.")
        assert any("within 24 hours" in c.normalized for c in result)

    def test_first_n_days(self):
        result = extract_constraints("Feature available during the first 14 days of your trial.")
        normalized = [c.normalized for c in result]
        assert any("14 days" in n for n in normalized), f"Got: {normalized}"

    # --- Category 4: plan/tier restrictions ---

    def test_paid_plans_only(self):
        result = extract_constraints("This feature is available on paid plans only.")
        normalized = [c.normalized for c in result]
        assert any("paid plan" in n for n in normalized), f"Got: {normalized}"

    def test_enterprise_plan_only(self):
        result = extract_constraints("Enterprise plan only — not available on starter tier.")
        normalized = [c.normalized for c in result]
        assert any("enterprise" in n for n in normalized), f"Got: {normalized}"

    def test_annual_subscription_only(self):
        result = extract_constraints("Available for annual subscription only.")
        normalized = [c.normalized for c in result]
        assert any("annual" in n for n in normalized), f"Got: {normalized}"

    # --- Category 5: "only if" conditionals ---

    def test_only_if_conditional(self):
        result = extract_constraints("Eligible only if account is on an annual subscription.")
        normalized = [c.normalized for c in result]
        assert any("only if" in n for n in normalized), f"Got: {normalized}"

    # --- Empty / unconstrained ---

    def test_no_constraints_returns_empty(self):
        result = extract_constraints("We offer great customer support around the clock.")
        assert result == []

    def test_generic_statement_returns_empty(self):
        result = extract_constraints("Our team is here to help you succeed.")
        assert result == []

    def test_empty_string_returns_empty(self):
        result = extract_constraints("")
        assert result == []

    # --- Structural / contract ---

    def test_constraint_has_raw_and_normalized_fields(self):
        result = extract_constraints("Refunds available within 30 days of purchase.")
        assert len(result) >= 1
        c = result[0]
        assert isinstance(c, Constraint)
        assert isinstance(c.raw, str) and c.raw
        assert isinstance(c.normalized, str) and c.normalized
        assert c.normalized == " ".join(c.raw.lower().split())

    def test_no_duplicates_on_repeated_phrase(self):
        """Same phrase appearing twice in a chunk must not yield duplicate constraints."""
        result = extract_constraints(
            "Refunds within 30 days. Only valid within 30 days of purchase."
        )
        normalized = [c.normalized for c in result]
        assert len(normalized) == len(set(normalized)), (
            f"Duplicate constraints returned: {normalized}"
        )

    def test_multiple_constraints_in_one_chunk(self):
        """A chunk with multiple constraints should return at least 2."""
        result = extract_constraints(
            "Refunds available within 30 days for paid plans only."
        )
        assert len(result) >= 2, (
            f"Expected ≥2 constraints, got {len(result)}: {[c.normalized for c in result]}"
        )
