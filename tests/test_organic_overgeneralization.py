"""
Tests for organic overgeneralization detection (RFORGE-12).

Root cause: accuracy.py Step 5 gate is dead code.  _check_chunk_relevance
returns alignment="partial" when constraint_preserved=False, so the condition
    if alignment == "supported" and not result["constraint_preserved"]
is never true — "supported" and "partial" are mutually exclusive here.
overgeneralization_flag is therefore never set via the organic path; only
planted_constraint conversations (with the Step 5 override) get flagged.

Fix (Option B): change the Step 5 condition to
    if not result["constraint_preserved"] and alignment != "contradicted":
The `alignment != "contradicted"` guard prevents DENY_CONDITION chunks from
being double-counted as both contradicted and overgeneralized.

Corollary: AccuracySignals.constraint_preserved = not overall_overgeneralization,
so the field was also wrong for organic overgeneralization — reporting True when
the claim demonstrably dropped a chunk constraint.

Tests:
  1. Organic overgeneralization (no planted_constraint) → flag fires    [RED before fix]
  2. Contradicted chunk is NOT flagged as overgeneralization             [guard regression]
  3. Supported:exact (constraint preserved) → no flag                   [regression guard]
  4. AccuracySignals.constraint_preserved is False post-fix             [RED before fix]
"""
from __future__ import annotations

from resonantforge.schemas import Claim, ConstraintType, KBChunk
from resonantforge.validators.extractors.accuracy import run_kb_alignment_pipeline


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

# Chunk has "up to 5" — a constraint the regex catches.
_SEATS_CONSTRAINED_CHUNK = KBChunk(
    chunk_id="kb_workspace_seats",
    document_id="doc_workspace",
    document_path="policies/workspace.md",
    chunk_text="Teams can add up to 5 members to a shared workspace.",
    constraint_type=ConstraintType.ALLOW_CONDITION,
    domains=["workspace"],
)

# DENY_CONDITION chunk — triggers the contradicted path, not the overgeneralization path.
_REFUND_DENY_CHUNK = KBChunk(
    chunk_id="kb_refund_deny",
    document_id="doc_refund",
    document_path="policies/refunds.md",
    chunk_text="Refunds are not allowed after 60 days from purchase.",
    constraint_type=ConstraintType.DENY_CONDITION,
    domains=["billing"],
)


def _organic_overgen_claim() -> Claim:
    """Agent drops 'up to 5' and says 'unlimited' — clean organic overgeneralization."""
    return Claim(
        claim_text="Teams can add unlimited members to a shared workspace.",
        claim_span=(0, 54),
        claim_type="policy",
        normalized_subject="teams workspace",
        normalized_predicate="can add",
        normalized_object="members workspace",
    )


def _preserved_constraint_claim() -> Claim:
    """Agent faithfully repeats the 'up to 5' constraint."""
    return Claim(
        claim_text="Teams can add up to 5 members to a shared workspace.",
        claim_span=(0, 52),
        claim_type="policy",
        normalized_subject="teams workspace",
        normalized_predicate="can add",
        normalized_object="members workspace",
    )


def _contradicting_claim() -> Claim:
    """Agent asserts refunds ARE available — contradicts the DENY_CONDITION chunk."""
    return Claim(
        claim_text="Refunds are available after 60 days from purchase.",
        claim_span=(0, 50),
        claim_type="policy",
        normalized_subject="refunds purchase",
        normalized_predicate="are available",
        normalized_object="refunds days",
    )


# ---------------------------------------------------------------------------
# Test 1 [RED before fix]: organic overgeneralization flags correctly
# ---------------------------------------------------------------------------


def test_organic_overgeneralization_flags_without_planted_constraint():
    """
    Chunk has 'up to 5'; claim says 'unlimited'. planted_constraint=None.

    Pre-fix: overgeneralization_flag=False (Step 5 gate is dead code).
    Post-fix: overgeneralization_flag=True (condition now checks constraint_preserved).
    """
    signals = run_kb_alignment_pipeline(
        claims=[_organic_overgen_claim()],
        kb_chunks=[_SEATS_CONSTRAINED_CHUNK],
        conversation_context="",
        kb_chunks_required=["kb_workspace_seats"],
        synonym_map={},
        planted_constraint=None,
    )
    assert signals.overgeneralization_flag is True, (
        "Organic overgeneralization not detected: chunk has 'up to 5' constraint "
        "but claim says 'unlimited members' and planted_constraint=None"
    )


# ---------------------------------------------------------------------------
# Test 2 [guard regression]: contradicted chunk must NOT set overgeneralization
# ---------------------------------------------------------------------------


def test_contradicted_chunk_not_flagged_as_overgeneralization():
    """
    DENY_CONDITION chunk with negation → alignment='contradicted'.
    The 'alignment != "contradicted"' guard must prevent overgeneralization_flag=True.

    Pre-fix: passes trivially (flag was always False without planted_constraint).
    Post-fix: must still pass — guard prevents double-flagging.
    """
    signals = run_kb_alignment_pipeline(
        claims=[_contradicting_claim()],
        kb_chunks=[_REFUND_DENY_CHUNK],
        conversation_context="",
        kb_chunks_required=["kb_refund_deny"],
        synonym_map={},
        planted_constraint=None,
    )
    assert signals.alignment == "contradicted", (
        f"Expected contradicted alignment, got {signals.alignment!r}"
    )
    assert signals.overgeneralization_flag is False, (
        "Contradicted claim incorrectly flagged as overgeneralization — guard failed"
    )


# ---------------------------------------------------------------------------
# Test 3 [regression guard]: supported:exact (constraint preserved) → no flag
# ---------------------------------------------------------------------------


def test_supported_exact_with_constraint_preserved_no_flag():
    """
    Agent faithfully includes 'up to 5' — constraint preserved.
    Neither overgeneralization_flag nor constraint_preserved should indicate a problem.
    """
    signals = run_kb_alignment_pipeline(
        claims=[_preserved_constraint_claim()],
        kb_chunks=[_SEATS_CONSTRAINED_CHUNK],
        conversation_context="",
        kb_chunks_required=["kb_workspace_seats"],
        synonym_map={},
        planted_constraint=None,
    )
    assert signals.overgeneralization_flag is False, (
        "overgeneralization_flag fired on a claim that preserved the constraint"
    )
    assert signals.constraint_preserved is True, (
        "constraint_preserved=False for a claim that included the chunk constraint"
    )


# ---------------------------------------------------------------------------
# Test 4 [RED before fix]: constraint_preserved field correct post-fix
# ---------------------------------------------------------------------------


def test_constraint_preserved_false_for_organic_overgeneralization():
    """
    AccuracySignals.constraint_preserved = not overall_overgeneralization.

    Pre-fix: overall_overgeneralization is never set True organically, so
    constraint_preserved=True even when the claim drops a chunk constraint.

    Post-fix: overall_overgeneralization=True → constraint_preserved=False.
    """
    signals = run_kb_alignment_pipeline(
        claims=[_organic_overgen_claim()],
        kb_chunks=[_SEATS_CONSTRAINED_CHUNK],
        conversation_context="",
        kb_chunks_required=["kb_workspace_seats"],
        synonym_map={},
        planted_constraint=None,
    )
    assert signals.constraint_preserved is False, (
        "constraint_preserved=True even though claim dropped the 'up to 5' chunk "
        "constraint — corollary of the dead-code bug"
    )
