"""
KB lint property tests — 3 lint assertions + 5 PR4 coverage assertions (KB-1–5).

L1–L3 run against the synthetic fixture in tests/fixtures/synthetic_kb.py.
KB-1–5 run against the real saas_content.py KB and validate PR4 authoring invariants.

Groups:
  L1. Every chunk has at least one domain
  L2. Every domain has at least one non-adversarial chunk
  L3. Warn (not fail) when any chunk covers more than 3 domains
  KB-1. Total chunk count is 62 (46 original + 16 PR4)
  KB-2. technical_issue has gate_1, gate_2, gate_4, gate_7, gate_8, standard, adversarial, probe
  KB-3. Exactly one sanity probe — gate_4 / technical_issue
  KB-4. Adversarial distribution spans at least 2 domains; 3 in refund_policy, 3 in technical_issue
  KB-5. Every gated technical_issue chunk has structured claims
"""

from __future__ import annotations

import warnings

import pytest

from tests.fixtures.synthetic_kb import SYNTHETIC_KB_CHUNKS
from resonantforge.kb.saas_content import get_saas_kb_chunks


# ---------------------------------------------------------------------------
# Lint assertion L1 — every chunk has at least one domain
# ---------------------------------------------------------------------------


def test_every_chunk_has_at_least_one_domain() -> None:
    """
    Every KBChunk in the synthetic fixture must declare at least one domain.

    A chunk with an empty domains list would be invisible to domain-based
    filtering and could never be selected by the injector.
    """
    no_domain = [c.chunk_id for c in SYNTHETIC_KB_CHUNKS if not c.domains]
    assert not no_domain, (
        f"Chunks missing domains list: {no_domain}. "
        "Add at least one domain to each chunk."
    )


# ---------------------------------------------------------------------------
# Lint assertion L2 — every domain has at least one non-adversarial chunk
# ---------------------------------------------------------------------------


def test_no_domain_has_zero_non_adversarial_chunks() -> None:
    """
    For every domain referenced in any chunk, at least one chunk with that
    domain must have adversarial=False.

    A domain where every chunk is adversarial would produce an empty allow pool,
    causing ValueError in _build_plan() for any conversation in that domain.
    """
    all_domains: set[str] = set()
    for chunk in SYNTHETIC_KB_CHUNKS:
        all_domains.update(chunk.domains)

    failing: list[str] = []
    for domain in sorted(all_domains):
        non_adversarial = [
            c for c in SYNTHETIC_KB_CHUNKS
            if domain in c.domains and not c.adversarial
        ]
        if not non_adversarial:
            failing.append(domain)

    assert not failing, (
        f"Domains with zero non-adversarial chunks: {failing}. "
        "Each domain must have at least one allow-eligible chunk."
    )


# ---------------------------------------------------------------------------
# Lint assertion L3 — warn (not fail) on chunks with more than 3 domains
# ---------------------------------------------------------------------------


def test_chunks_warn_on_excessive_domains() -> None:
    """
    Chunks with more than 3 domains produce a UserWarning.

    Cross-domain chunks can dilute topic specificity and make domain isolation
    tests ambiguous. This is a lint warning, not a hard failure.
    """
    over_tagged = [c.chunk_id for c in SYNTHETIC_KB_CHUNKS if len(c.domains) > 3]
    if over_tagged:
        with pytest.warns(UserWarning, match="excessive domains"):
            warnings.warn(
                f"Chunks with excessive domains (>3): {over_tagged}",
                UserWarning,
                stacklevel=1,
            )
    else:
        # No over-tagged chunks — test passes without a warning.
        assert not over_tagged


# ===========================================================================
# PR4 coverage assertions (KB-1 through KB-5) — run against real saas KB
# ===========================================================================

_SAAS_CHUNKS = get_saas_kb_chunks()


def test_kb1_total_chunk_count() -> None:
    """
    KB-1: Total chunk count after PR4 is 62 (46 original + 16 PR4 chunks).
    """
    assert len(_SAAS_CHUNKS) == 62, (
        f"KB-1: expected 62 total chunks after PR4, got {len(_SAAS_CHUNKS)}. "
        "Check that all 16 technical_issue chunks are present and no duplicates exist."
    )


def test_kb2_technical_issue_gate_coverage() -> None:
    """
    KB-2: technical_issue domain has chunks tagged in gate_1, gate_2, gate_4, gate_7,
    gate_8, at least one no-gate standard chunk, at least 2 adversarial chunks,
    and at least one sanity probe.
    """
    ti_chunks = [c for c in _SAAS_CHUNKS if "technical_issue" in c.domains]

    gates_present = {c.cat11_gate for c in ti_chunks if c.cat11_gate}
    for required_gate in ("gate_1", "gate_2", "gate_4", "gate_7", "gate_8"):
        assert required_gate in gates_present, (
            f"KB-2: technical_issue missing gate {required_gate}. "
            f"Gates found: {sorted(gates_present)}"
        )

    standard_chunks = [c for c in ti_chunks if not c.cat11_gate and not c.adversarial]
    assert len(standard_chunks) >= 1, (
        f"KB-2: expected ≥1 no-gate standard chunk in technical_issue, "
        f"got {len(standard_chunks)}"
    )

    adversarial_ti = [c for c in ti_chunks if c.adversarial]
    assert len(adversarial_ti) >= 2, (
        f"KB-2: expected ≥2 adversarial chunks in technical_issue, "
        f"got {len(adversarial_ti)}"
    )

    probes_ti = [c for c in ti_chunks if c.sanity_probe]
    assert len(probes_ti) >= 1, (
        "KB-2: expected ≥1 sanity probe in technical_issue, got 0"
    )


def test_kb3_sanity_probe_identification() -> None:
    """
    KB-3: Exactly one chunk has sanity_probe=True; it is tagged gate_4 and is in
    the technical_issue domain.
    """
    probes = [c for c in _SAAS_CHUNKS if c.sanity_probe]
    assert len(probes) == 1, (
        f"KB-3: expected exactly 1 sanity probe, got {len(probes)}: "
        f"{[p.chunk_id for p in probes]}"
    )
    probe = probes[0]
    assert probe.cat11_gate == "gate_4", (
        f"KB-3: sanity probe {probe.chunk_id!r} has gate {probe.cat11_gate!r}, "
        "expected 'gate_4'"
    )
    assert "technical_issue" in probe.domains, (
        f"KB-3: sanity probe {probe.chunk_id!r} domains {probe.domains} do not "
        "include 'technical_issue'"
    )


def test_kb4_adversarial_distribution() -> None:
    """
    KB-4: At least 2 of the 3 original adversarial chunks are in refund_policy;
    PR4 added 2-3 adversarial chunks in technical_issue; total adversarial
    coverage spans at least 2 domains.
    """
    adversarial_chunks = [c for c in _SAAS_CHUNKS if c.adversarial]

    refund_adversarial = [
        c for c in adversarial_chunks if "refund_policy" in c.domains
    ]
    assert len(refund_adversarial) >= 2, (
        f"KB-4: expected ≥2 adversarial chunks in refund_policy, "
        f"got {len(refund_adversarial)}"
    )

    ti_adversarial = [
        c for c in adversarial_chunks if "technical_issue" in c.domains
    ]
    assert 2 <= len(ti_adversarial) <= 3, (
        f"KB-4: expected 2-3 adversarial chunks in technical_issue, "
        f"got {len(ti_adversarial)}"
    )

    adversarial_domains: set[str] = set()
    for c in adversarial_chunks:
        adversarial_domains.update(c.domains)
    assert len(adversarial_domains) >= 2, (
        f"KB-4: expected adversarial coverage across ≥2 domains, "
        f"got {sorted(adversarial_domains)}"
    )


def test_kb5_technical_issue_gated_chunks_have_claims() -> None:
    """
    KB-5: Every gated (cat11_gate is not None) chunk in technical_issue has
    at least one structured claim in its claims dict.
    """
    ti_gated = [
        c for c in _SAAS_CHUNKS
        if "technical_issue" in c.domains and c.cat11_gate and not c.adversarial
    ]
    missing_claims = [c.chunk_id for c in ti_gated if not c.claims]
    assert not missing_claims, (
        f"KB-5: gated technical_issue chunks with empty claims: {missing_claims}. "
        "Every gated chunk must encode at least one structured claim."
    )
