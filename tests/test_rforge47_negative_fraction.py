"""
RFORGE-47 — Tests for negative_fraction parameter on QualityPlanInjector.

TDD step: RED — tests should fail until negative_fraction is wired through
_build_schedule() and the injector __init__.
"""
from __future__ import annotations

import random

import pytest

from resonantforge.layer1.quality_plan_injector import QualityPlanInjector


def _empathy_targets(plans) -> list[str]:
    return [
        p.rubric_targets.empathy
        for p in plans
        if p.rubric_targets.empathy is not None
    ]


def _resolution_targets(plans) -> list[str]:
    return [
        p.rubric_targets.resolution
        for p in plans
        if p.rubric_targets.resolution is not None
    ]


def _brand_voice_targets(plans) -> list[str]:
    return [
        p.rubric_targets.brand_voice_target
        for p in plans
        if p.rubric_targets.brand_voice_target is not None
    ]


class TestNegativeFractionParameter:
    """Verifies that negative_fraction controls the proportion of failing rubric targets."""

    def test_default_negative_fraction_is_0_5(self):
        """Injector must accept negative_fraction as a keyword arg defaulting to 0.5."""
        inj = QualityPlanInjector(rng=random.Random(42), negative_fraction=0.5)
        assert inj.negative_fraction == 0.5

    def test_all_positive_fraction_zero_produces_no_low_empathy(self):
        """negative_fraction=0.0 → all empathy targets are 'high'."""
        inj = QualityPlanInjector(rng=random.Random(42), negative_fraction=0.0)
        schedule = inj._build_schedule(20)
        empathy_specs = [s for s in schedule if s["type"] == "empathy"]
        assert all(s["target"] == "high" for s in empathy_specs), (
            f"Expected all 'high' but got: {empathy_specs}"
        )

    def test_all_negative_fraction_one_produces_all_low_empathy(self):
        """negative_fraction=1.0 → all empathy targets are 'low'."""
        inj = QualityPlanInjector(rng=random.Random(42), negative_fraction=1.0)
        schedule = inj._build_schedule(20)
        empathy_specs = [s for s in schedule if s["type"] == "empathy"]
        assert all(s["target"] == "low" for s in empathy_specs), (
            f"Expected all 'low' but got: {empathy_specs}"
        )

    def test_all_positive_fraction_zero_produces_no_off_brand(self):
        """negative_fraction=0.0 → all brand_voice targets are 'on_brand'."""
        inj = QualityPlanInjector(rng=random.Random(42), negative_fraction=0.0)
        schedule = inj._build_schedule(20)
        bv_specs = [s for s in schedule if s["type"] == "brand_voice"]
        assert all(s["target"] == "on_brand" for s in bv_specs), (
            f"Expected all 'on_brand' but got: {bv_specs}"
        )

    def test_all_negative_fraction_one_produces_all_off_brand(self):
        """negative_fraction=1.0 → all brand_voice targets are 'off_brand'."""
        inj = QualityPlanInjector(rng=random.Random(42), negative_fraction=1.0)
        schedule = inj._build_schedule(20)
        bv_specs = [s for s in schedule if s["type"] == "brand_voice"]
        assert all(s["target"] == "off_brand" for s in bv_specs), (
            f"Expected all 'off_brand' but got: {bv_specs}"
        )

    def test_all_positive_fraction_zero_produces_no_weak_resolution(self):
        """negative_fraction=0.0 → no resolution targets are 'weak'."""
        inj = QualityPlanInjector(rng=random.Random(42), negative_fraction=0.0)
        schedule = inj._build_schedule(20)
        res_specs = [s for s in schedule if s["type"] == "resolution"]
        assert all(s["target"] != "weak" for s in res_specs), (
            f"Got unexpected 'weak': {res_specs}"
        )

    def test_all_negative_fraction_one_produces_all_weak_resolution(self):
        """negative_fraction=1.0 → all resolution targets are 'weak'."""
        inj = QualityPlanInjector(rng=random.Random(42), negative_fraction=1.0)
        schedule = inj._build_schedule(20)
        res_specs = [s for s in schedule if s["type"] == "resolution"]
        assert all(s["target"] == "weak" for s in res_specs), (
            f"Expected all 'weak' but got: {res_specs}"
        )

    def test_default_0_5_produces_roughly_half_negative(self):
        """Default 0.5 → roughly half of empathy specs are 'low'."""
        inj = QualityPlanInjector(rng=random.Random(42))
        schedule = inj._build_schedule(40)
        empathy_specs = [s for s in schedule if s["type"] == "empathy"]
        low_count = sum(1 for s in empathy_specs if s["target"] == "low")
        total = len(empathy_specs)
        assert total > 0
        # At 50/50 with integer rounding, allow ±1 to account for odd totals
        assert abs(low_count - total / 2) <= 1, (
            f"Expected ~50% low but got {low_count}/{total}"
        )

    def test_injector_without_explicit_fraction_uses_default(self):
        """Constructor without negative_fraction= should behave identically to 0.5."""
        inj_default = QualityPlanInjector(rng=random.Random(42))
        inj_explicit = QualityPlanInjector(rng=random.Random(42), negative_fraction=0.5)
        s1 = inj_default._build_schedule(20)
        s2 = inj_explicit._build_schedule(20)
        assert s1 == s2
