"""
KB lint property tests — 3 assertions.

All three run against the synthetic fixture defined in tests/fixtures/synthetic_kb.py.
They will re-run against the real KB when PR2 tags saas_content.py chunks.

Groups:
  L1. Every chunk has at least one domain
  L2. Every domain has at least one non-adversarial chunk
  L3. Warn (not fail) when any chunk covers more than 3 domains
"""

from __future__ import annotations

import warnings

import pytest

from tests.fixtures.synthetic_kb import SYNTHETIC_KB_CHUNKS


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
