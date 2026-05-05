"""
Tests for RFORGE-17: per-conv persistence in pipeline Phase 3.

Covers:
  1. Mid-Phase-3 interrupt leaves recoverable JSONL with exactly N records.
  2. Robust JSONL reader skips a corrupt trailing line, returns prior records intact.
  3. Full pipeline still produces a valid manifest.json after the change.
  4. --force wipes the corpus directory at Phase 1 start.
"""

from __future__ import annotations

import json
import unittest.mock as mock
from pathlib import Path

import pytest

from confabra.pipeline import PipelineConfig, run_pipeline
from confabra.utils.atomic_write import append_jsonl_line, read_jsonl_robust

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SEED = 42
_SMALL_ACCOUNTS = 3
_SMALL_MONTHS = 2


def _minimal_config(output_root: Path, *, force: bool = False) -> PipelineConfig:
    return PipelineConfig(
        profile_name="saas",
        accounts=_SMALL_ACCOUNTS,
        months=_SMALL_MONTHS,
        seed=_SEED,
        output_root=output_root,
        anthropic_api_key=None,  # dry-run — no Anthropic calls
        force=force,
    )


# ---------------------------------------------------------------------------
# Test 1 — Mid-Phase-3 interrupt leaves recoverable JSONL
# ---------------------------------------------------------------------------

_INTERRUPT_AFTER = 3  # raise after this many passed conversations


def test_mid_phase3_interrupt_leaves_recoverable_corpus(tmp_path: Path) -> None:
    """
    Simulating a mid-Phase-3 interrupt by patching append_jsonl_line to raise
    after N successful convs.  Verifies conversations.jsonl on disk contains
    exactly N parseable records.
    """
    convs_path = tmp_path / "saas" / "conversations.jsonl"

    _pass_count = 0
    _original_append = append_jsonl_line

    def _patched_append(path: Path, line: str) -> None:
        nonlocal _pass_count
        # Only count flushes to the conversations file (not skipped/disagreements).
        if path.name == "conversations.jsonl":
            _pass_count += 1
            if _pass_count > _INTERRUPT_AFTER:
                raise KeyboardInterrupt("simulated mid-run interrupt")
        _original_append(path, line)

    with mock.patch(
        "confabra.pipeline.append_jsonl_line", side_effect=_patched_append
    ):
        with pytest.raises(KeyboardInterrupt):
            run_pipeline(_minimal_config(tmp_path))

    # The file must exist and contain exactly _INTERRUPT_AFTER parseable records.
    assert convs_path.exists(), "conversations.jsonl was not created before interrupt"
    records = read_jsonl_robust(convs_path)
    assert len(records) == _INTERRUPT_AFTER, (
        f"Expected {_INTERRUPT_AFTER} records on disk after interrupt, "
        f"got {len(records)}"
    )
    # Every record must be a valid JSON object with a conversation_id.
    for i, rec in enumerate(records):
        assert "conversation_id" in rec, f"Record {i} missing conversation_id: {rec}"


# ---------------------------------------------------------------------------
# Test 2 — Robust JSONL reader skips corrupt trailing line
# ---------------------------------------------------------------------------


def test_read_jsonl_robust_skips_corrupt_trailing_line(tmp_path: Path) -> None:
    """
    Manually write 3 valid records then a truncated (corrupt) 4th line.
    read_jsonl_robust must return the 3 good records and skip the corrupt one.
    """
    path = tmp_path / "test.jsonl"
    good_records = [
        {"id": "a", "value": 1},
        {"id": "b", "value": 2},
        {"id": "c", "value": 3},
    ]
    # Write good records.
    for rec in good_records:
        append_jsonl_line(path, json.dumps(rec))

    # Append a truncated (unparseable) line directly.
    with open(path, "a", encoding="utf-8") as fh:
        fh.write('{"id": "d", "val')  # truncated mid-record, no closing brace or newline

    result = read_jsonl_robust(path)
    assert len(result) == 3, f"Expected 3 good records, got {len(result)}"
    assert [r["id"] for r in result] == ["a", "b", "c"]


def test_read_jsonl_robust_returns_empty_for_missing_file(tmp_path: Path) -> None:
    result = read_jsonl_robust(tmp_path / "nonexistent.jsonl")
    assert result == []


# ---------------------------------------------------------------------------
# Test 3 — Integration: full pipeline produces a valid manifest
# ---------------------------------------------------------------------------


def test_full_pipeline_produces_valid_manifest(tmp_path: Path) -> None:
    """
    After the RFORGE-17 change, a complete dry-run pipeline still produces
    manifest.json with consistent counts.  This guards against regressions
    where the per-conv flush path breaks the manifest or hash computation.
    """
    manifest = run_pipeline(_minimal_config(tmp_path))
    profile_dir = tmp_path / "saas"

    assert (profile_dir / "manifest.json").exists()

    # conversations.jsonl line count must match manifest.conversation_count.
    convs = read_jsonl_robust(profile_dir / "conversations.jsonl")
    assert len(convs) == manifest.conversation_count, (
        f"manifest.conversation_count={manifest.conversation_count} but "
        f"conversations.jsonl has {len(convs)} lines"
    )

    # skipped_conversations.jsonl line count must match manifest.
    skipped = read_jsonl_robust(profile_dir / "skipped_conversations.jsonl")
    assert len(skipped) == manifest.skipped_conversation_count, (
        f"manifest.skipped_conversation_count={manifest.skipped_conversation_count} but "
        f"skipped_conversations.jsonl has {len(skipped)} lines"
    )

    # manifest.json must be valid JSON with expected top-level keys.
    manifest_raw = json.loads((profile_dir / "manifest.json").read_text())
    for key in ("conversation_count", "events_hash", "conversations_hash", "seed"):
        assert key in manifest_raw, f"manifest.json missing key: {key}"


# ---------------------------------------------------------------------------
# Test 4 — --force wipes corpus directory at Phase 1 start
# ---------------------------------------------------------------------------


def test_force_flag_wipes_corpus_directory(tmp_path: Path) -> None:
    """
    --force must delete all existing contents of the profile directory before
    the run starts.  A sentinel file placed before the run must be gone after.
    """
    profile_dir = tmp_path / "saas"
    profile_dir.mkdir(parents=True)

    sentinel = profile_dir / "sentinel_should_be_deleted.txt"
    sentinel.write_text("leftover from prior run")

    # Without --force, the pipeline refuses to run into a non-empty directory.
    with pytest.raises(RuntimeError, match="Refusing to overwrite"):
        run_pipeline(_minimal_config(tmp_path, force=False))

    assert sentinel.exists(), "sentinel should still exist — run was rejected"

    # With --force, the directory is wiped and the pipeline succeeds.
    run_pipeline(_minimal_config(tmp_path, force=True))

    assert not sentinel.exists(), (
        "--force should have wiped the profile directory before starting"
    )
