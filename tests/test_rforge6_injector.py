"""
Tests for RFORGE-6: injector changes — guided resolution directive and 3-way rotation.
"""
from __future__ import annotations

from resonantforge.layer1.quality_plan_injector import _prose_directive_for_resolution


# ── Directive content tests ───────────────────────────────────────────────────


def test_guided_directive_contains_step_instructions():
    """Guided directive must describe step-by-step walkthrough with imperative language."""
    directive = _prose_directive_for_resolution("guided")
    lower = directive.lower()
    # Must contain step-by-step or walkthrough language
    assert any(word in lower for word in ["step", "walk", "instructions"]), (
        f"Guided directive must describe step-by-step instructions. Got: {directive}"
    )


def test_guided_directive_no_ownership_language():
    """Guided directive must NOT instruct the LLM to add ownership language (customer executes)."""
    directive = _prose_directive_for_resolution("guided")
    lower = directive.lower()
    # Ownership signals like 'i will', 'let me', 'i'll fix' should not be required
    assert "i will" not in lower and "let me" not in lower and "i'll fix" not in lower, (
        f"Guided directive must not require agent ownership language. Got: {directive}"
    )


def test_weak_directive_unchanged():
    """Regression: weak directive content must be unchanged."""
    directive = _prose_directive_for_resolution("weak")
    assert "deflect" in directive.lower() or "vague" in directive.lower(), (
        f"Weak directive must still describe deflection/vagueness. Got: {directive}"
    )


def test_strong_directive_unchanged():
    """Regression: strong directive must still require ownership and temporal anchor."""
    directive = _prose_directive_for_resolution("strong")
    lower = directive.lower()
    assert "ownership" in lower or "i will" in lower or "let me" in lower, (
        f"Strong directive must still require ownership language. Got: {directive}"
    )
    assert "temporal" in lower or "tomorrow" in lower or "24 hours" in lower or "within" in lower, (
        f"Strong directive must still require temporal anchors. Got: {directive}"
    )


# ── 3-way rotation test ───────────────────────────────────────────────────────


def test_resolution_rotation_covers_all_three_targets():
    """
    Resolution target rotation must produce weak, guided, and strong in a 3-item cycle.
    Targets at positions 0,1,2,3,4,5 must cover all three.
    """
    # Import the module-level cycle list or simulate the rotation logic
    # We test behavior by calling the function with known indices
    # The rotation is: i%3==0 → weak, i%3==1 → guided, i%3==2 → strong
    from resonantforge.layer1.quality_plan_injector import _resolution_target_for_index

    targets = [_resolution_target_for_index(i) for i in range(6)]
    assert "weak" in targets, "Rotation must include 'weak'"
    assert "guided" in targets, "Rotation must include 'guided'"
    assert "strong" in targets, "Rotation must include 'strong'"
    # Even distribution: 2 of each in 6
    assert targets.count("weak") == 2
    assert targets.count("guided") == 2
    assert targets.count("strong") == 2
