"""
Observability backfill layer for the corpus generator (PR3b).

Guarantees that every authored (domain × gate) cell reaches its minimum
event count by the end of a generation run. This is a constraint layer on
top of the weighted-random state machine — not a sampler redesign, not
adaptive weighting, not rotation.

Mechanism
---------
During each conversation event generation step the caller asks
``CoverageBackfill.select()`` for the next (domain, gate_hint) to use.

- **Normal mode** (all deficits at zero): ``select()`` calls
  ``rng.choices()`` exactly once and returns ``(domain, None)``.  The RNG
  stream is byte-identical to a pure-weighted-random run.

- **Backfill mode** (any deficit > 0): ``select()`` returns the
  ``(domain, gate)`` of the highest-deficit cell without consuming an RNG
  draw.  The activation is recorded in ``backfill_activations``.

After generating the event, the caller calls ``record_event(domain, gate_hint)``:

- **Backfill event** (gate_hint is not None): only cell (domain, gate_hint)
  receives credit — one unit toward its ``min_events`` quota.
- **Normal event** (gate_hint is None): every cell whose domain matches
  receives credit.  This is the path by which cells can satisfy their quota
  through ordinary weighted-random sampling without ever entering deficit.

Deficit accounting is cumulative within a run.  Counts only increase; a cell
that reached ``min_events`` stays satisfied.  Each new run must create a fresh
``CoverageBackfill`` instance.

Constants
---------
``DEFAULT_MIN_EVENTS`` = 6. Derived from the signal observability spec's
"≥3 comparable pairs per gate per checkpoint" requirement with a 2× safety
margin to absorb events that don't pair cleanly.
"""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

DEFAULT_MIN_EVENTS: int = 6


@dataclass
class AuthoredCell:
    """A single authored (domain × gate) cell with its minimum event requirement."""

    domain: str
    gate: str
    min_events: int = DEFAULT_MIN_EVENTS


class CoverageBackfill:
    """
    Deficit-driven backfill that ensures every authored cell reaches its
    minimum event count.  Not thread-safe; one instance per generation run.

    Usage::

        backfill = CoverageBackfill(cells=[("billing_and_invoicing", "gate_3")], ...)
        # inside the state machine conversation loop:
        domain, gate_hint = backfill.select(rng, domain_names, domain_weights)
        # ... generate the conversation event with domain ...
        backfill.record_event(domain, gate_hint)
        # after the full run:
        assert all(v == 0 for v in backfill.get_deficit_at_run_end().values())
    """

    def __init__(
        self,
        cells: list[tuple[str, str]],
        min_events_override: dict[tuple[str, str], int] | None = None,
    ) -> None:
        """
        Args:
            cells: Deduplicated list of (domain, gate) authored cells.  Order
                determines tie-breaking when two cells share the highest deficit.
            min_events_override: Per-cell overrides for ``min_events``.  Cells
                not present use ``DEFAULT_MIN_EVENTS``.
        """
        overrides: dict[tuple[str, str], int] = min_events_override or {}
        self._cells: list[AuthoredCell] = [
            AuthoredCell(
                domain=d,
                gate=g,
                min_events=overrides.get((d, g), DEFAULT_MIN_EVENTS),
            )
            for d, g in cells
        ]

        # domain → list of cells, for O(1) normal-mode attribution.
        self._cells_by_domain: dict[str, list[AuthoredCell]] = defaultdict(list)
        for cell in self._cells:
            self._cells_by_domain[cell.domain].append(cell)

        # Cumulative observed events per (domain, gate) cell.
        self._observed: dict[tuple[str, str], int] = {
            (c.domain, c.gate): 0 for c in self._cells
        }

        # Run-level telemetry (populated by select() / record_event()).
        self.backfill_activations: list[dict[str, Any]] = []
        self._cells_entered_deficit: set[tuple[str, str]] = set()

        # Internal step counter — incremented on every select() call.
        self._step: int = 0

    # ------------------------------------------------------------------
    # Core interface
    # ------------------------------------------------------------------

    def select(
        self,
        rng: random.Random,
        domain_names: list[str],
        domain_weights: list[int],
    ) -> tuple[str, str | None]:
        """
        Return the (domain, gate_hint) for the next conversation event.

        In **normal mode**: calls ``rng.choices(domain_names, weights=…)`` once
        (RNG-identical to pure weighted-random) and returns ``(domain, None)``.

        In **backfill mode**: returns ``(forced_domain, forced_gate)`` for the
        highest-deficit cell without consuming an RNG draw.

        Args:
            rng: The caller's PRNG.  Used *only* in normal mode to keep the
                RNG stream consistent with pure weighted-random runs.
            domain_names: Ordered domain identifiers matching ``domain_weights``.
            domain_weights: Positive integer weights parallel to ``domain_names``.

        Returns:
            ``(domain, gate_hint)`` where ``gate_hint`` is ``None`` in normal mode.
        """
        self._step += 1

        if not self._cells:
            # No authored cells configured — always normal mode.
            domain = rng.choices(domain_names, weights=domain_weights, k=1)[0]
            return domain, None

        deficits = self._deficits()
        max_deficit = max(deficits.values())

        if max_deficit > 0:
            # Backfill mode: pick highest-deficit cell (insertion-order tie-break).
            target_key = max(deficits, key=lambda k: deficits[k])
            self.backfill_activations.append(
                {
                    "step_index": self._step,
                    "target_cell": f"{target_key[0]}:{target_key[1]}",
                    "deficit_at_activation": deficits[target_key],
                }
            )
            self._cells_entered_deficit.add(target_key)
            return target_key[0], target_key[1]
        else:
            # Normal mode: weighted-random, identical RNG draw to pure PR3a.
            domain = rng.choices(domain_names, weights=domain_weights, k=1)[0]
            return domain, None

    def record_event(self, domain: str, gate_hint: str | None) -> None:
        """
        Update observed counts after a conversation event is emitted.

        **Backfill event** (``gate_hint`` is not ``None``): only cell
        ``(domain, gate_hint)`` receives credit.

        **Normal event** (``gate_hint`` is ``None``): all cells whose domain
        matches receive credit — this is how cells satisfy their quota through
        ordinary weighted-random sampling.

        Args:
            domain: Domain of the just-emitted conversation event.
            gate_hint: Gate forced for this step, or ``None`` if normal mode.
        """
        if gate_hint is not None:
            key = (domain, gate_hint)
            if key in self._observed:
                self._observed[key] += 1
        else:
            for cell in self._cells_by_domain.get(domain, []):
                self._observed[(cell.domain, cell.gate)] += 1

    # ------------------------------------------------------------------
    # Manifest field accessors (read-only after the run completes)
    # ------------------------------------------------------------------

    def get_backfill_activations(self) -> list[dict[str, Any]]:
        """Ordered list of backfill activation records for this run."""
        return list(self.backfill_activations)

    def get_cells_requiring_backfill(self) -> list[str]:
        """
        Cells that needed at least one forced step, serialised as
        ``"domain:gate"`` strings, sorted for deterministic output.
        """
        return sorted(f"{d}:{g}" for d, g in self._cells_entered_deficit)

    def get_cells_satisfied_by_normal(self) -> list[str]:
        """
        Cells that reached ``min_events`` exclusively through normal-mode
        weighted-random events (never entered deficit), sorted.
        """
        result = []
        for cell in self._cells:
            key = (cell.domain, cell.gate)
            if (
                key not in self._cells_entered_deficit
                and self._observed[key] >= cell.min_events
            ):
                result.append(f"{cell.domain}:{cell.gate}")
        return sorted(result)

    def get_deficit_at_run_end(self) -> dict[str, int]:
        """
        Final deficit per cell after the run completes.

        All values should be 0 when the invariant holds.  Non-zero values
        indicate the run ended before every cell reached ``min_events`` — either
        the run was too short or the backfill mechanism has a bug.

        Keys are ``"domain:gate"`` strings.
        """
        return {
            f"{c.domain}:{c.gate}": max(
                0, c.min_events - self._observed[(c.domain, c.gate)]
            )
            for c in self._cells
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _deficits(self) -> dict[tuple[str, str], int]:
        """Return current deficit per cell (0 when satisfied or over-satisfied)."""
        return {
            (c.domain, c.gate): max(
                0, c.min_events - self._observed[(c.domain, c.gate)]
            )
            for c in self._cells
        }
