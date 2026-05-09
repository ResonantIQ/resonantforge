"""
Tests for RFORGE-43: resolution extractor false positives.

Bug 1 — \\byou should\\b too broad:
  "You should see some movement on this soon" matches the resolution pattern and
  sets solution_provided=True, even though the agent has provided no actual fix.
  Fix: remove \\byou should\\b from RESOLUTION_PATTERNS.

Bug 2 — question-form deflection not detected:
  "Have you checked our integration documentation?" does not match
  \\bcheck (?:the )?documentation\\b because (a) \\bcheck\\b does not match
  inside "checked" (word-boundary mismatch) and (b) the phrasing is a question
  to the customer rather than an agent imperative.
  Fix: add \\bhave you (?:checked|tried|looked at)\\b to DEFLECTION_PATTERNS.
"""
from __future__ import annotations

from resonantforge.profiles.lexicons import saas as saas_lex
from resonantforge.validators.extractors.resolution import extract_resolution_signals

# ── Helpers ───────────────────────────────────────────────────────────────────


def _extract(agent_prose: str, customer_prose: str = "") -> object:
    return extract_resolution_signals(
        agent_prose=agent_prose,
        customer_prose=customer_prose,
        resolution_patterns=saas_lex.RESOLUTION_PATTERNS,
        deflection_patterns=saas_lex.DEFLECTION_PATTERNS,
        next_steps_patterns=saas_lex.NEXT_STEPS_PATTERNS,
        temporal_anchor_patterns=saas_lex.TEMPORAL_ANCHOR_PATTERNS,
        specific_actor_patterns=saas_lex.SPECIFIC_ACTOR_PATTERNS,
        ownership_patterns=saas_lex.OWNERSHIP_PATTERNS,
        issue_keywords=saas_lex.ISSUE_KEYWORDS,
        completion_verb_patterns=saas_lex.COMPLETION_VERB_PATTERNS,
        completion_resolution_patterns=saas_lex.COMPLETION_RESOLUTION_PATTERNS,
    )


# ── RFORGE-43 Bug 1: vague promise "you should see movement" must not be a solution ──


def test_vague_movement_promise_not_a_solution():
    """
    "You should see some movement on this soon" must not set solution_provided=True.
    It is a non-committal status promise, not an actionable fix.
    """
    signals = _extract(
        agent_prose=(
            "Our team will review your case and follow up with any insights. "
            "You should see some movement on this soon. "
            "Feel free to reach out if you have any other questions in the meantime."
        )
    )
    assert signals.solution_provided is False, (
        "'You should see some movement on this soon' must not set solution_provided=True. "
        f"Got solution_provided={signals.solution_provided}"
    )


def test_real_solution_still_detected():
    """
    Regression guard: a response with a concrete fix must still set solution_provided=True.
    """
    signals = _extract(
        agent_prose=(
            "Here's how to fix the 401: go to Settings > Integrations and copy the API key. "
            "Paste it into your CRM's webhook configuration and the 401 should clear."
        ),
        customer_prose="I'm getting a 401 error on my webhook integration.",
    )
    assert signals.solution_provided is True, (
        "A response with a concrete fix must still set solution_provided=True. "
        f"Got solution_provided={signals.solution_provided}"
    )


# ── RFORGE-43 Bug 2: question-form deflection must be detected ────────────────


def test_question_form_documentation_redirect_detected_as_deflection():
    """
    "Have you checked our integration documentation?" must set deflection_present=True.
    The agent is redirecting the customer to docs without solving the problem.
    """
    signals = _extract(
        agent_prose=(
            "Have you checked our integration documentation? "
            "There are some great resources there that might address what you're seeing."
        )
    )
    assert signals.deflection_present is True, (
        "'Have you checked our integration documentation?' must be detected as deflection. "
        f"Got deflection_present={signals.deflection_present}"
    )


def test_question_form_have_you_tried_detected_as_deflection():
    """
    "Have you tried restarting the connection?" must set deflection_present=True.
    """
    signals = _extract(
        agent_prose=(
            "Have you tried restarting the connection from your end? "
            "That often clears up authentication issues like this."
        )
    )
    assert signals.deflection_present is True, (
        "'Have you tried X?' must be detected as deflection. "
        f"Got deflection_present={signals.deflection_present}"
    )


def test_imperative_deflection_still_detected():
    """
    Regression guard: existing imperative deflection patterns must still fire.
    "Please contact our support team" must still set deflection_present=True.
    """
    signals = _extract(
        agent_prose=(
            "Please contact our support team for further assistance with this issue."
        )
    )
    assert signals.deflection_present is True, (
        "Existing imperative deflection pattern must still work. "
        f"Got deflection_present={signals.deflection_present}"
    )


# ── Combined: conv_evt_00121 profile — pure deflection scores as weak ─────────


def test_pure_deflection_conversation_scores_as_weak_resolution():
    """
    End-to-end test mirroring conv_evt_00121.

    Agent repeatedly redirects to docs the customer already read, gives no
    actual fix, and ends with a vague promise. After RFORGE-43 fixes:
      - solution_provided must be False (vague promise removed from patterns)
      - deflection_present must be True (question-form deflection detected)
    These produce resolution_blocked=True, so validate_resolution would pass
    for target='weak'.
    """
    customer_prose = (
        "I'm trying to set up the new webhook integration for our CRM "
        "but I keep getting a 401 error when I test it. "
        "I looked at that already. The docs don't explain what's causing the 401. "
        "This is frustrating. Can you escalate this to someone who can help?"
    )
    agent_prose = (
        "Thanks for reaching out! Have you checked our integration documentation? "
        "There are some great resources there that might address what you're seeing. "
        "Good question. It could be a few different things on that front. "
        "Our team will definitely look into this for you. "
        "That's in the integration docs under the Authentication section. "
        "I totally understand your frustration. "
        "Our team will definitely look into your specific setup and see what's going on. "
        "Great, thanks for checking on that. "
        "Our team will review your case and follow up with any insights. "
        "You should see some movement on this soon. "
        "Feel free to reach out if you have any other questions in the meantime."
    )
    signals = _extract(agent_prose=agent_prose, customer_prose=customer_prose)

    assert signals.solution_provided is False, (
        f"Pure-deflection agent must not be scored as solution_provided=True. "
        f"Got solution_provided={signals.solution_provided}"
    )
    assert signals.deflection_present is True, (
        f"Pure-deflection agent must have deflection_present=True. "
        f"Got deflection_present={signals.deflection_present}"
    )
    assert signals.resolution_blocked is True, (
        f"resolution_blocked must be True when solution_provided=False and deflection_present=True. "
        f"Got resolution_blocked={signals.resolution_blocked}"
    )
