"""
PR3b coverage backfill tests — 10 assertions.

Tests validate the invariant, not the sampler.  The backfill is tested both
directly (CoverageBackfill in isolation) and through the state machine
(integration / distribution tests).

Groups:
  INV — Invariant holds (4 tests)
  NORM — Normal mode dominates when slack is available (1 test)
  DET — Determinism (2 tests)
  DIST — Distribution preservation / documented shift (2 tests)
"""

from __future__ import annotations

import random
from collections import Counter

import pytest

from resonantforge.layer1.coverage_backfill import CoverageBackfill, DEFAULT_MIN_EVENTS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_backfill(
    domains: list[str],
    gates: list[str],
    min_events: int = DEFAULT_MIN_EVENTS,
    min_events_override: dict[tuple[str, str], int] | None = None,
) -> CoverageBackfill:
    """Build a CoverageBackfill for a full domains × gates grid."""
    cells = [(d, g) for d in domains for g in gates]
    overrides = min_events_override or {(d, g): min_events for d in domains for g in gates}
    return CoverageBackfill(cells=cells, min_events_override=overrides)


def _run_backfill(
    backfill: CoverageBackfill,
    rng: random.Random,
    domain_names: list[str],
    domain_weights: list[int],
    n_steps: int,
) -> list[tuple[str, str | None]]:
    """
    Drive the backfill for ``n_steps`` conversation events.

    Returns the ordered list of (domain, gate_hint) selections made.
    """
    selections: list[tuple[str, str | None]] = []
    for _ in range(n_steps):
        domain, gate_hint = backfill.select(rng, domain_names, domain_weights)
        backfill.record_event(domain, gate_hint)
        selections.append((domain, gate_hint))
    return selections


# ---------------------------------------------------------------------------
# INV-1: invariant holds under a balanced synthetic profile
# ---------------------------------------------------------------------------


def test_inv1_invariant_holds_basic() -> None:
    """
    INV-1: Run 200 steps with a 3-domain × 3-gate synthetic profile.

    Expected: deficit_at_run_end is all zeros after 200 steps.

    With 9 cells × min_events=6 = 54 minimum events and 200 steps available,
    the backfill has ample budget to satisfy every cell.
    """
    domains = ["billing", "technical", "onboarding"]
    gates = ["gate_1", "gate_3", "gate_7"]
    weights = [3, 5, 2]  # non-uniform to make some domains rarer

    backfill = _make_backfill(domains, gates)
    rng = random.Random(42)
    _run_backfill(backfill, rng, domains, weights, n_steps=200)

    deficit = backfill.get_deficit_at_run_end()
    non_zero = {k: v for k, v in deficit.items() if v != 0}
    assert not non_zero, (
        f"INV-1: deficit_at_run_end has non-zero entries after 200 steps: {non_zero}"
    )


# ---------------------------------------------------------------------------
# INV-2: invariant holds under a stress distribution
# ---------------------------------------------------------------------------


def test_inv2_invariant_holds_under_stress() -> None:
    """
    INV-2: Unfavorable weight distribution — one domain at 95%, others at ~1.67% each.

    Without backfill, rare cells would almost never be sampled.  With backfill,
    the invariant must hold.
    """
    domains = ["dominant", "rare_a", "rare_b", "rare_c"]
    gates = ["gate_1", "gate_4", "gate_7"]
    # weights: dominant=95, others=5 total → each rare domain has ~1.67% probability
    weights = [95, 2, 2, 1]

    backfill = _make_backfill(domains, gates)
    rng = random.Random(99)
    _run_backfill(backfill, rng, domains, weights, n_steps=400)

    deficit = backfill.get_deficit_at_run_end()
    non_zero = {k: v for k, v in deficit.items() if v != 0}
    assert not non_zero, (
        f"INV-2: deficit_at_run_end has non-zero entries after 400 steps "
        f"under stressed weights: {non_zero}"
    )

    # Backfill must have activated (rare cells would have been starved)
    assert backfill.get_cells_requiring_backfill(), (
        "INV-2: expected backfill to activate for rare cells under stressed weights, "
        "but cells_requiring_backfill is empty"
    )


# ---------------------------------------------------------------------------
# INV-3: invariant holds at minimum required run length
# ---------------------------------------------------------------------------


def test_inv3_invariant_holds_at_minimum_run_length() -> None:
    """
    INV-3: Run exactly the minimum steps needed to satisfy the invariant.

    With N cells × min_events steps in backfill-only mode, every cell gets
    exactly min_events forced events.  We verify this holds by using a
    pathologically skewed weight (1 domain = 99%) and running exactly
    N_cells × min_events steps.

    With skewed weights and only that many steps, weighted-random alone would
    give the rare cells 0 events.  The backfill must deliver every event.
    """
    domains = ["dominant", "rare"]
    gates = ["gate_1", "gate_2", "gate_3"]
    min_e = 4  # use a small value so the minimum run is tractable
    n_cells = len(domains) * len(gates)
    n_steps = n_cells * min_e  # exactly the minimum needed

    # Extreme weights so weighted-random alone would never reach rare cells
    weights = [99, 1]

    backfill = _make_backfill(domains, gates, min_events=min_e)
    rng = random.Random(7)
    _run_backfill(backfill, rng, domains, weights, n_steps=n_steps)

    deficit = backfill.get_deficit_at_run_end()
    non_zero = {k: v for k, v in deficit.items() if v != 0}
    assert not non_zero, (
        f"INV-3: deficit_at_run_end has non-zero entries at minimum run length "
        f"({n_steps} steps, {n_cells} cells × {min_e} min_events): {non_zero}"
    )


# ---------------------------------------------------------------------------
# INV-4: no oscillation
# ---------------------------------------------------------------------------


def test_inv4_no_oscillation() -> None:
    """
    INV-4: No cell appears in consecutive backfill_activations entries.

    When record_event is wired correctly, every forced step reduces the target
    cell's deficit by 1.  The tie-breaking in select() (insertion order) means
    a different cell becomes the highest-deficit cell on the next step — so no
    cell should appear in back-to-back activations.

    Oscillation (same cell twice in a row in backfill mode) would indicate
    that record_event is not being called after forced steps, leaving the
    deficit unchanged.
    """
    # 3 domains × 3 gates, all starting at 0 with equal min_events.
    # With balanced starting deficits, no cell should dominate consecutively
    # more than min_events times (once other cells drop their deficits to 0,
    # only the remaining cell can appear; that's expected, not oscillation).
    domains = ["alpha", "beta", "gamma"]
    gates = ["gate_1", "gate_2", "gate_3"]
    min_e = 3

    backfill = _make_backfill(domains, gates, min_events=min_e)
    rng = random.Random(13)
    _run_backfill(backfill, rng, domains, [1, 1, 1], n_steps=150)

    activations = backfill.get_backfill_activations()
    if len(activations) < 2:
        return  # nothing to check

    consecutive_repeats = [
        (activations[i]["target_cell"], activations[i + 1]["target_cell"])
        for i in range(len(activations) - 1)
        if activations[i]["target_cell"] == activations[i + 1]["target_cell"]
    ]
    assert not consecutive_repeats, (
        f"INV-4: {len(consecutive_repeats)} case(s) of a cell appearing in "
        f"consecutive backfill activations (oscillation indicator). "
        f"First repeat: {consecutive_repeats[0]}"
    )


# ---------------------------------------------------------------------------
# NORM-1: normal mode dominates when slack is available
# ---------------------------------------------------------------------------


def test_norm1_normal_mode_dominates_when_slack_available() -> None:
    """
    NORM-1: When min_events is small relative to the run length, the initial
    backfill phase is brief and normal mode dominates the rest of the run.

    Profile: 2 domains × 2 gates = 4 cells.  min_events=3.  500 steps.

    All cells start at deficit=min_events, so backfill fires exactly
    n_cells × min_events times (once per cell per quota unit), then every
    remaining step runs in normal mode.  This validates that backfill is a
    short bootstrap, not a persistent drag on the run.

    Expected:
    - deficit_at_run_end is all zero
    - backfill_activations has exactly n_cells * min_events entries
    - backfill ratio (forced / total) is < 5 %
    """
    domains = ["billing", "technical"]
    gates = ["gate_1", "gate_7"]
    n_cells = len(domains) * len(gates)
    min_e = 3
    n_steps = 500

    backfill = _make_backfill(domains, gates, min_events=min_e)
    rng = random.Random(55)
    _run_backfill(backfill, rng, domains, [1, 1], n_steps=n_steps)

    # Invariant must hold
    deficit = backfill.get_deficit_at_run_end()
    assert all(v == 0 for v in deficit.values()), (
        f"NORM-1: deficit non-zero at run end: {deficit}"
    )

    # Backfill phase is exactly n_cells × min_events forced steps
    n_backfill = len(backfill.get_backfill_activations())
    assert n_backfill == n_cells * min_e, (
        f"NORM-1: expected exactly {n_cells * min_e} backfill activations "
        f"(one per cell per quota unit), got {n_backfill}"
    )

    # Normal mode dominates: < 5 % of steps are forced
    assert n_backfill / n_steps < 0.05, (
        f"NORM-1: backfill ratio {n_backfill / n_steps:.1%} exceeds 5 % — "
        "normal mode is not dominating as expected"
    )


# ---------------------------------------------------------------------------
# DET-1: determinism — same seed produces identical sequences
# ---------------------------------------------------------------------------


def test_det1_same_seed_identical_sequence() -> None:
    """
    DET-1: Two runs with the same seed produce identical selection sequences,
    including identical backfill_activations.
    """
    domains = ["billing", "technical", "onboarding"]
    gates = ["gate_1", "gate_4"]
    weights = [60, 30, 10]

    def _run(seed: int) -> tuple[list[tuple[str, str | None]], list[dict]]:
        backfill = _make_backfill(domains, gates)
        rng = random.Random(seed)
        selections = _run_backfill(backfill, rng, domains, weights, n_steps=200)
        return selections, backfill.get_backfill_activations()

    sel_a, act_a = _run(42)
    sel_b, act_b = _run(42)

    assert sel_a == sel_b, (
        "DET-1: same seed produced different selection sequences"
    )
    assert act_a == act_b, (
        "DET-1: same seed produced different backfill_activations"
    )


# ---------------------------------------------------------------------------
# DET-2: different seeds produce different sequences
# ---------------------------------------------------------------------------


def test_det2_different_seeds_produce_different_sequences() -> None:
    """
    DET-2: Different seeds produce different selection sequences, but all
    runs still satisfy the invariant.
    """
    domains = ["billing", "technical", "onboarding"]
    gates = ["gate_1", "gate_4"]
    weights = [60, 30, 10]

    def _run(seed: int) -> tuple[list[tuple[str, str | None]], dict]:
        backfill = _make_backfill(domains, gates)
        rng = random.Random(seed)
        selections = _run_backfill(backfill, rng, domains, weights, n_steps=200)
        return selections, backfill.get_deficit_at_run_end()

    sel_42, def_42 = _run(42)
    sel_99, def_99 = _run(99)

    # Different seeds → different sequences (not guaranteed in theory, but
    # deterministic enough for distinct seeds that the probability is negligible)
    assert sel_42 != sel_99, (
        "DET-2: different seeds produced identical selection sequences; "
        "this is statistically improbable and likely indicates a seeding bug"
    )

    # Both runs must still satisfy the invariant
    non_zero_42 = {k: v for k, v in def_42.items() if v != 0}
    non_zero_99 = {k: v for k, v in def_99.items() if v != 0}
    assert not non_zero_42, f"DET-2: seed=42 deficit non-zero: {non_zero_42}"
    assert not non_zero_99, f"DET-2: seed=99 deficit non-zero: {non_zero_99}"


# ---------------------------------------------------------------------------
# DIST-1: distribution preserved when backfill not engaged
# ---------------------------------------------------------------------------


def test_dist1_distribution_preserved_when_backfill_not_engaged() -> None:
    """
    DIST-1: When no cells are configured, select() is permanently in normal
    mode and calls rng.choices() exactly once per step.  The resulting
    selection sequence is byte-identical to a pure-weighted-random run with
    the same seed.

    This verifies the core RNG-preservation guarantee: normal-mode select()
    consumes exactly one RNG draw, no more, no fewer.
    """
    domains = ["billing", "technical", "onboarding"]
    weights = [3, 5, 2]
    n_steps = 600

    # No authored cells → select() is always in normal mode
    backfill = CoverageBackfill(cells=[])
    rng_with = random.Random(7)
    selections_with = _run_backfill(backfill, rng_with, domains, weights, n_steps)

    # Verify no backfill activations occurred
    assert backfill.get_backfill_activations() == [], (
        "DIST-1: unexpected backfill activations with empty cells list"
    )

    # Pure weighted-random run (same seed)
    rng_pure = random.Random(7)
    selections_pure: list[tuple[str, str | None]] = []
    for _ in range(n_steps):
        domain = rng_pure.choices(domains, weights=weights, k=1)[0]
        selections_pure.append((domain, None))

    # Sequences must be byte-identical (same single RNG draw every step)
    assert selections_with == selections_pure, (
        "DIST-1: normal-mode backfill produced a different sequence than pure "
        "random with the same seed — select() is consuming extra RNG draws"
    )


# ---------------------------------------------------------------------------
# DIST-2: distribution shift documented when backfill heavily engages
# ---------------------------------------------------------------------------


def test_dist2_distribution_shift_documented_when_backfill_engages() -> None:
    """
    DIST-2: When backfill heavily engages (stressed weights), the per-domain
    distribution of the backfill run differs from a pure run, and we document
    which cells absorbed the backfill.

    This is expected and correct behaviour — the backfill deliberately shifts
    the distribution to satisfy rare-cell quotas.
    """
    domains = ["dominant", "rare_a", "rare_b"]
    gates = ["gate_4"]
    weights = [98, 1, 1]
    n_steps = 300
    min_e = 6

    # Backfill run
    backfill = _make_backfill(domains, gates, min_events=min_e)
    rng_bf = random.Random(42)
    selections_bf = _run_backfill(backfill, rng_bf, domains, weights, n_steps)

    # Pure run (same seed)
    rng_pure = random.Random(42)
    selections_pure: list[tuple[str, str | None]] = [
        (rng_pure.choices(domains, weights=weights, k=1)[0], None)
        for _ in range(n_steps)
    ]

    dist_bf = Counter(d for d, _ in selections_bf)
    dist_pure = Counter(d for d, _ in selections_pure)

    # Distributions must differ (backfill pushed rare_a and rare_b up)
    assert dist_bf != dist_pure, (
        "DIST-2: expected distributions to differ when backfill engages, "
        "but they are identical. Check that backfill is actually engaging."
    )

    # The cells that required backfill should include the rare domains
    cells_needing = backfill.get_cells_requiring_backfill()
    assert cells_needing, (
        "DIST-2: cells_requiring_backfill is empty — backfill never engaged "
        "under stressed weights. Increase n_steps or weights ratio."
    )

    # Deficit must be zero (the invariant still holds)
    deficit = backfill.get_deficit_at_run_end()
    non_zero = {k: v for k, v in deficit.items() if v != 0}
    assert not non_zero, (
        f"DIST-2: deficit_at_run_end is non-zero despite backfill: {non_zero}"
    )


# ---------------------------------------------------------------------------
# Supplementary: manifest field types and content
# ---------------------------------------------------------------------------


def test_manifest_field_structure() -> None:
    """
    Verify that all four manifest-facing fields have the correct types and
    content after a short run.
    """
    domains = ["billing", "technical"]
    gates = ["gate_1", "gate_7"]
    weights = [70, 30]
    min_e = 3

    backfill = _make_backfill(domains, gates, min_events=min_e)
    rng = random.Random(0)
    _run_backfill(backfill, rng, domains, weights, n_steps=200)

    activations = backfill.get_backfill_activations()
    requiring = backfill.get_cells_requiring_backfill()
    satisfied = backfill.get_cells_satisfied_by_normal()
    deficit = backfill.get_deficit_at_run_end()

    # backfill_activations is a list of dicts with required keys
    assert isinstance(activations, list)
    for entry in activations:
        assert "step_index" in entry
        assert "target_cell" in entry
        assert "deficit_at_activation" in entry
        assert isinstance(entry["step_index"], int)
        assert isinstance(entry["target_cell"], str)
        assert ":" in entry["target_cell"]
        assert isinstance(entry["deficit_at_activation"], int)
        assert entry["deficit_at_activation"] > 0

    # cells_requiring_backfill is a sorted list of "domain:gate" strings
    assert isinstance(requiring, list)
    for cell_str in requiring:
        assert isinstance(cell_str, str)
        assert ":" in cell_str

    # cells_satisfied_by_normal is a sorted list of "domain:gate" strings
    assert isinstance(satisfied, list)
    for cell_str in satisfied:
        assert isinstance(cell_str, str)
        assert ":" in cell_str

    # requiring and satisfied are disjoint
    assert not (set(requiring) & set(satisfied)), (
        f"A cell appears in both cells_requiring_backfill and cells_satisfied_by_normal: "
        f"{set(requiring) & set(satisfied)}"
    )

    # deficit_at_run_end is all zeros after 200 steps
    assert isinstance(deficit, dict)
    non_zero = {k: v for k, v in deficit.items() if v != 0}
    assert not non_zero, f"deficit_at_run_end non-zero: {non_zero}"

    # Every authored cell appears in deficit_at_run_end
    expected_keys = {f"{d}:{g}" for d in domains for g in gates}
    assert set(deficit.keys()) == expected_keys, (
        f"deficit_at_run_end keys mismatch: {set(deficit.keys())} vs {expected_keys}"
    )


# ---------------------------------------------------------------------------
# Supplementary: pipeline integration — manifest fields are populated
# ---------------------------------------------------------------------------


def test_pipeline_manifest_has_backfill_fields(tmp_path) -> None:
    """
    Verify that after a full pipeline dry-run the manifest contains the four
    PR3b fields with the correct types.

    This is an integration test: it runs the actual pipeline (no LLM calls)
    and checks the returned Manifest object.
    """
    from resonantforge.pipeline import PipelineConfig, run_pipeline

    config = PipelineConfig(
        profile_name="saas",
        accounts=3,
        months=1,
        seed=42,
        output_root=tmp_path,
        anthropic_api_key=None,
    )
    manifest = run_pipeline(config)

    # All four PR3b fields must be present with correct types.
    assert isinstance(manifest.backfill_activations, list), (
        "manifest.backfill_activations must be a list"
    )
    assert isinstance(manifest.cells_requiring_backfill, list), (
        "manifest.cells_requiring_backfill must be a list"
    )
    assert isinstance(manifest.cells_satisfied_by_normal, list), (
        "manifest.cells_satisfied_by_normal must be a list"
    )
    assert isinstance(manifest.deficit_at_run_end, dict), (
        "manifest.deficit_at_run_end must be a dict"
    )

    # deficit_at_run_end values must all be integers
    for k, v in manifest.deficit_at_run_end.items():
        assert isinstance(v, int), (
            f"deficit_at_run_end[{k!r}] must be int, got {type(v).__name__}"
        )

    # requiring and satisfied must be disjoint
    requiring = set(manifest.cells_requiring_backfill)
    satisfied = set(manifest.cells_satisfied_by_normal)
    overlap = requiring & satisfied
    assert not overlap, (
        f"cells_requiring_backfill and cells_satisfied_by_normal overlap: {overlap}"
    )
