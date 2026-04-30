"""
Cross-contamination injector — metadata-only tone drift for KB chunks,
agent trajectory rows, and organic conversation records.

Cross-contamination forces the downstream scorer to consult the brand voice
spec rather than keying off a single dominant tone that pervades every artifact.
By marking ~20% of KB chunks, ~25% of trajectory rows, and ~15% of conversations
with a non-dominant brand voice variant id, the harness ensures that the corpus
cannot be solved by a model that memorises surface-level tone patterns.

Design principles
-----------------
- **Metadata-only** — chunk/row/conversation *content* is never rewritten.
  ``tone_variant`` and ``tone_variant`` are annotation fields that
  describe what voice the artifact was *written in*, for use by the evaluation
  harness.  The prose generator does not read them.
- **Deterministic** — all randomness flows through a single ``random.Random``
  seeded at construction time.  Identical seeds produce identical contamination
  decisions across runs.
- **Independent per item** — each contamination decision is an independent
  Bernoulli draw; the contamination rate is the *expected* fraction, not an
  exact count.
- **Non-dominant only** — when contaminating, the injected variant is chosen
  uniformly from all variants *except* the dominant one.

Public API
----------
- :class:`CrossContaminationInjector`
"""
from __future__ import annotations

import random
from typing import Sequence

from confabra.schemas import ConversationRecord, KBChunk, TrajectoryRow


class CrossContaminationInjector:
    """
    Inject metadata-only tone-drift markers into corpus artifacts.

    Parameters
    ----------
    seed:
        RNG seed for deterministic contamination decisions.
    variants:
        Complete list of brand voice variant ids available in this corpus
        (e.g. ``["bv_warm_exploratory", "bv_direct_clinical", "bv_baseline"]``).
    dominant_variant:
        The "primary" variant id for this corpus run.  Contamination markers
        are always drawn from the complement of this set (i.e. ``variants``
        minus ``dominant_variant``).

    Raises
    ------
    ValueError
        If ``dominant_variant`` is not in ``variants``, or fewer than 2 variants
        are provided (contamination requires at least one alternative).
    """

    #: Fraction of KB chunks to mark with a non-dominant tone variant.
    KB_CONTAMINATION_RATE: float = 0.20

    #: Fraction of agent trajectory rows to mark with a non-dominant tone marker.
    TRAJECTORY_CONTAMINATION_RATE: float = 0.25

    #: Fraction of organic conversation records to mark with a non-dominant tone marker.
    CONVERSATION_CONTAMINATION_RATE: float = 0.15

    def __init__(
        self,
        seed: int,
        variants: list[str],
        dominant_variant: str,
    ) -> None:
        if dominant_variant not in variants:
            raise ValueError(
                f"dominant_variant {dominant_variant!r} must be present in variants list "
                f"(got: {variants!r})"
            )
        if len(variants) < 2:
            raise ValueError(
                "At least two brand voice variants are required for cross-contamination "
                f"(got {len(variants)}: {variants!r})"
            )

        self._rng = random.Random(seed)
        self._variants = list(variants)
        self._dominant = dominant_variant
        # Pre-compute the pool of non-dominant variants to draw from.
        self._alternatives: list[str] = [v for v in variants if v != dominant_variant]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _pick_alternative(self) -> str:
        """
        Choose a non-dominant variant uniformly at random.

        Uses the instance RNG so that the sequence of choices is determined
        solely by the seed and the order of items processed.
        """
        return self._rng.choice(self._alternatives)

    def _should_contaminate(self, rate: float) -> bool:
        """Return True with probability *rate* using the instance RNG."""
        return self._rng.random() < rate

    # ------------------------------------------------------------------
    # Public contamination methods
    # ------------------------------------------------------------------

    def contaminate_kb_chunks(self, chunks: Sequence[KBChunk]) -> list[KBChunk]:
        """
        Mark ~20% of KB chunks with a non-dominant ``tone_variant``.

        Returns a new list; the input sequence is not mutated.  Each item that
        is *not* contaminated retains ``tone_variant=None``.  Contaminated items
        receive a variant id drawn uniformly from the non-dominant alternatives.

        Parameters
        ----------
        chunks:
            KB chunk objects to process.  Must be :class:`~confabra.schemas.KBChunk`
            instances.

        Returns
        -------
        list[KBChunk]
            New list with contamination markers applied.
        """
        result: list[KBChunk] = []
        for chunk in chunks:
            if self._should_contaminate(self.KB_CONTAMINATION_RATE):
                contaminated = chunk.model_copy(
                    update={"tone_variant": self._pick_alternative()}
                )
            else:
                # Ensure field is explicitly None even if the source had a stale value.
                contaminated = chunk.model_copy(update={"tone_variant": None})
            result.append(contaminated)
        return result

    def contaminate_trajectory_rows(
        self, rows: Sequence[TrajectoryRow]
    ) -> list[TrajectoryRow]:
        """
        Mark ~25% of agent trajectory rows with a non-dominant ``tone_variant``.

        Returns a new list; the input sequence is not mutated.  Each item that
        is *not* contaminated retains ``tone_variant=None``.

        Parameters
        ----------
        rows:
            Trajectory row objects to process.  Must be
            :class:`~confabra.schemas.TrajectoryRow` instances.

        Returns
        -------
        list[TrajectoryRow]
            New list with contamination markers applied.
        """
        result: list[TrajectoryRow] = []
        for row in rows:
            if self._should_contaminate(self.TRAJECTORY_CONTAMINATION_RATE):
                contaminated = row.model_copy(
                    update={"tone_variant": self._pick_alternative()}
                )
            else:
                contaminated = row.model_copy(update={"tone_variant": None})
            result.append(contaminated)
        return result

    def contaminate_conversations(
        self, convs: Sequence[ConversationRecord]
    ) -> list[ConversationRecord]:
        """
        Mark ~15% of organic conversation records with a non-dominant ``tone_variant``.

        Returns a new list; the input sequence is not mutated.  Each item that
        is *not* contaminated retains ``tone_variant=None``.

        Parameters
        ----------
        convs:
            Conversation record objects to process.  Must be
            :class:`~confabra.schemas.ConversationRecord` instances.

        Returns
        -------
        list[ConversationRecord]
            New list with contamination markers applied.
        """
        result: list[ConversationRecord] = []
        for conv in convs:
            if self._should_contaminate(self.CONVERSATION_CONTAMINATION_RATE):
                contaminated = conv.model_copy(
                    update={"tone_variant": self._pick_alternative()}
                )
            else:
                contaminated = conv.model_copy(update={"tone_variant": None})
            result.append(contaminated)
        return result
