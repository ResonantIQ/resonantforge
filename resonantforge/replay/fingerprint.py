"""
Determinism fingerprint for replay results.

A fingerprint is a SHA-256 hex digest over the stable JSON serialization of
a replay result. Given an unchanged envelope + unchanged code, repeated runs
must produce byte-identical fingerprints.

The test_determinism test asserts this property by running replay_one twice
against the same envelope and comparing fingerprints.

`RunFingerprint` extends the per-run fingerprint with attribution hashes so
you can tell *why* a fingerprint changed: validator code changed, lexicons
changed, or the results themselves changed.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from resonantforge.replay.schemas import ReplayEnvelope, ReplayResult


def compute_fingerprint(result: ReplayResult) -> str:
    """
    Compute a SHA-256 fingerprint of a ReplayResult.

    Serialization is deterministic: Pydantic model_dump produces a stable dict,
    json.dumps with sort_keys=True removes key-ordering variance.

    Args:
        result: a completed ReplayResult from replay_one()

    Returns:
        64-character lowercase hex digest
    """
    payload = result.model_dump(mode="json")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def compute_corpus_fingerprint(results: list[ReplayResult]) -> str:
    """
    Compute a single fingerprint over an ordered list of replay results.

    Used by the determinism test to assert stability across full corpus runs.
    Results are sorted by conv_id before hashing so run order doesn't matter.

    Args:
        results: all ReplayResult objects from a corpus run

    Returns:
        64-character lowercase hex digest
    """
    sorted_results = sorted(results, key=lambda r: r.conv_id)
    h = hashlib.sha256()
    for r in sorted_results:
        payload = r.model_dump(mode="json")
        line = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        h.update(line.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


@dataclass
class RunFingerprint:
    """
    Full attribution fingerprint for one `rforge replay run` invocation.

    Three independent hashes let you diagnose *why* a combined fingerprint changed:
      - results_hash   : the replay outcomes changed (validator logic or envelope content)
      - code_hash      : the validator extractor source files changed
      - lexicons_hash  : the lexicons frozen in the envelopes changed

    Use combined_hash to detect any change at all.
    """

    results_hash: str
    code_hash: str
    lexicons_hash: str
    envelope_schema_version: int
    label_schema_version: int
    combined_hash: str


def _hash_validators_tree() -> str:
    """
    Walk resonantforge/validators/extractors/ and hash all .py source files.

    Files are sorted by path so directory enumeration order doesn't matter.
    Returns a 64-char hex digest.
    """
    extractors_dir = Path(__file__).parent.parent / "validators" / "extractors"
    h = hashlib.sha256()
    if extractors_dir.exists():
        for py_file in sorted(extractors_dir.rglob("*.py")):
            h.update(py_file.as_posix().encode("utf-8"))
            h.update(b"\x00")
            h.update(py_file.read_bytes())
            h.update(b"\x00")
    return h.hexdigest()


def _hash_lexicons(envelopes: list[ReplayEnvelope]) -> str:
    """
    Hash the lexicons block from every envelope, sorted by conv_id.

    Detects drift in the frozen lexicons without having to diff the full envelope.
    Returns a 64-char hex digest.
    """
    h = hashlib.sha256()
    for env in sorted(envelopes, key=lambda e: e.conv_id):
        lexicons_dict = env.lexicons.model_dump(mode="json")
        canonical = json.dumps(lexicons_dict, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        h.update(env.conv_id.encode("utf-8"))
        h.update(b"\x00")
        h.update(canonical.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def compute_run_fingerprint(
    results: list[ReplayResult],
    envelopes: list[ReplayEnvelope],
) -> RunFingerprint:
    """
    Compute a full attribution fingerprint for a replay run.

    Args:
        results: all ReplayResult objects from the run
        envelopes: the corresponding ReplayEnvelope objects (same order or any order — sorted internally)

    Returns:
        RunFingerprint with independent hashes for results, code, and lexicons
    """
    results_hash = compute_corpus_fingerprint(results)
    code_hash = _hash_validators_tree()
    lexicons_hash = _hash_lexicons(envelopes)

    envelope_schema_version = envelopes[0].schema_version if envelopes else 0
    label_schema_version = results[0].labels.schema_version if results else 0

    combined = hashlib.sha256(
        f"{results_hash}:{code_hash}:{lexicons_hash}:{envelope_schema_version}:{label_schema_version}".encode("utf-8")
    ).hexdigest()

    return RunFingerprint(
        results_hash=results_hash,
        code_hash=code_hash,
        lexicons_hash=lexicons_hash,
        envelope_schema_version=envelope_schema_version,
        label_schema_version=label_schema_version,
        combined_hash=combined,
    )
