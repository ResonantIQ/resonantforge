"""
resonantforge.replay — frozen-replay harness for zero-cost validator iteration.

Public API:
  load_envelope   load and validate an envelope.json
  load_labels     load and validate a labels.json
  replay_one      run validators against a frozen envelope + labels pair

Error types:
  EnvelopeSchemaError   schema / conv_id / version mismatch in an envelope
  LabelSchemaError      schema / conv_id / version mismatch or unknown tag in labels
  ReplayModeError       prohibited call (LLM, network) detected during replay

Determinism guarantees (RFORGE_REPLAY_MODE=1):
  - extract_claims_llm is patched to raise ReplayModeError if called
  - All 4 extractors are deterministic; no RNG, no datetime.now()
  - Stable sort throughout (synonym map, KB chunk order)
  - Same envelope + same code → byte-identical fingerprint every run
"""

from resonantforge.replay.engine import (
    EnvelopeSchemaError,
    LabelSchemaError,
    ReplayModeError,
    load_envelope,
    load_labels,
    replay_one,
)
from resonantforge.replay.agreement import compute_agreement
from resonantforge.replay.diff import classify_change, format_conv_line, format_summary
from resonantforge.replay.fingerprint import (
    RunFingerprint,
    compute_corpus_fingerprint,
    compute_fingerprint,
    compute_run_fingerprint,
)
from resonantforge.replay.schemas import (
    AgreementResult,
    ExtractionMeta,
    ReplayEnvelope,
    ReplayLabels,
    ReplayResult,
)

__all__ = [
    # Engine
    "load_envelope",
    "load_labels",
    "replay_one",
    # Error types
    "EnvelopeSchemaError",
    "LabelSchemaError",
    "ReplayModeError",
    # Agreement
    "compute_agreement",
    # Output
    "classify_change",
    "format_conv_line",
    "format_summary",
    # Fingerprint
    "compute_fingerprint",
    "compute_corpus_fingerprint",
    "compute_run_fingerprint",
    "RunFingerprint",
    # Schema types
    "AgreementResult",
    "ExtractionMeta",
    "ReplayEnvelope",
    "ReplayLabels",
    "ReplayResult",
]
