"""
Tests for the frozen-directory guard in ``rforge generate`` (RFORGE-60).

The guard prevents ``rforge generate`` from silently overwriting frozen replay
envelopes when the default ``./replay_corpus`` path already contains envelopes
for the requested profile.  This is the CLI-level footgun introduced in
RFORGE-41 where running ``rforge generate`` from the ``harness/`` directory
would resolve ``./replay_corpus`` to the checked-in reference envelope tree and
silently destroy it.

Covers:
  - Guard fires when ``resolved_replay_dir/<profile>/*/envelope.json`` exists →
    sys.exit(1) with an informative message naming the directory and listing the
    safe alternatives (--no-replay, --replay-out, --force-replay-out).
  - Guard is bypassed by ``--force-replay-out`` → no error, pipeline proceeds.
  - Guard is skipped entirely when ``--no-replay`` is passed → no error.
  - Guard does not fire when the replay directory is empty (no envelopes) → no
    error, pipeline proceeds normally.
  - ``--force-replay-out`` appears in ``rforge generate --help``.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from resonantforge.cli import cli


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fake_run_pipeline(config_capture: list):
    """Return a fake ``run_pipeline`` that records the PipelineConfig and returns a
    minimal mock manifest so the CLI summary table renders without error."""

    def _fake(config):
        config_capture.append(config)
        m = MagicMock()
        m.profile_name = config.profile_name
        m.seed = config.seed
        m.accounts = config.accounts
        m.months = config.months
        m.event_count = 0
        m.conversation_count = 0
        m.planted_quality_count = 0
        m.skipped_conversation_count = 0
        m.corrections_count = 0
        m.agent_count = 0
        m.knowledge_base_doc_count = 0
        m.knowledge_base_chunk_count = 0
        return m

    return _fake


def _plant_envelopes(base_dir: Path, profile: str, n: int = 1) -> None:
    """Create ``n`` fake envelope.json files under ``base_dir/<profile>/conv_NNN/``."""
    for i in range(n):
        conv_dir = base_dir / profile / f"conv_{i:05d}"
        conv_dir.mkdir(parents=True, exist_ok=True)
        (conv_dir / "envelope.json").write_text(
            json.dumps({"schema_version": 1, "conv_id": f"conv_{i:05d}"})
        )


# ---------------------------------------------------------------------------
# Guard fires when envelopes already exist
# ---------------------------------------------------------------------------


def test_guard_fires_when_envelopes_exist(tmp_path: Path) -> None:
    """Guard must exit(1) when frozen envelopes already exist at the replay output path."""
    _plant_envelopes(tmp_path / "replay_corpus", profile="saas", n=3)

    runner = CliRunner()
    # Run from tmp_path so ./replay_corpus resolves to our planted tree.
    with runner.isolated_filesystem(temp_dir=tmp_path):
        # Copy envelope tree into the isolated filesystem CWD.
        import shutil
        shutil.copytree(tmp_path / "replay_corpus", Path.cwd() / "replay_corpus")

        result = runner.invoke(
            cli,
            ["generate", "--dry-run", "--out-root", str(tmp_path / "corpus")],
        )

    assert result.exit_code == 1, (
        "Guard must exit(1) when frozen envelopes exist; "
        f"got exit_code={result.exit_code}\nOutput:\n{result.output}"
    )
    # The error message must name the directory that would be overwritten.
    assert "replay_corpus" in result.output, (
        "Error message must mention the replay corpus directory"
    )
    # The error message must list the three safe alternatives.
    assert "--no-replay" in result.output, "Error must list --no-replay as an alternative"
    assert "--replay-out" in result.output, "Error must list --replay-out as an alternative"
    assert "--force-replay-out" in result.output, "Error must list --force-replay-out as an alternative"


def test_guard_error_message_names_exact_directory(tmp_path: Path) -> None:
    """Error message must include the exact replay directory path that would be overwritten."""
    replay_dir = tmp_path / "my_replay_dir"
    _plant_envelopes(replay_dir, profile="saas", n=1)

    runner = CliRunner()
    result = runner.invoke(
        cli,
        [
            "generate",
            "--dry-run",
            "--out-root", str(tmp_path / "corpus"),
            "--replay-out", str(replay_dir),
        ],
    )

    assert result.exit_code == 1, (
        f"Guard must fire for --replay-out pointing at populated dir; "
        f"got exit_code={result.exit_code}\nOutput:\n{result.output}"
    )
    # The resolved directory path must appear in the error.
    assert str(replay_dir) in result.output or "my_replay_dir" in result.output, (
        "Error message must name the exact directory that would be overwritten"
    )


# ---------------------------------------------------------------------------
# Guard bypassed by --force-replay-out
# ---------------------------------------------------------------------------


def test_force_replay_out_bypasses_guard(tmp_path: Path) -> None:
    """--force-replay-out must suppress the guard and allow the pipeline to run."""
    replay_dir = tmp_path / "replay_corpus"
    _plant_envelopes(replay_dir, profile="saas", n=2)

    captured: list = []

    runner = CliRunner()
    with patch("resonantforge.pipeline.run_pipeline", side_effect=_make_fake_run_pipeline(captured)):
        result = runner.invoke(
            cli,
            [
                "generate",
                "--dry-run",
                "--out-root", str(tmp_path / "corpus"),
                "--replay-out", str(replay_dir),
                "--force-replay-out",
            ],
        )

    assert result.exit_code == 0, (
        "--force-replay-out must bypass the guard; "
        f"got exit_code={result.exit_code}\nOutput:\n{result.output}"
    )
    assert captured, "run_pipeline must be called when guard is bypassed"


# ---------------------------------------------------------------------------
# Guard skipped when --no-replay is passed
# ---------------------------------------------------------------------------


def test_no_replay_skips_guard(tmp_path: Path) -> None:
    """--no-replay must skip the guard entirely (replay_dir is None, nothing to protect)."""
    # Even if replay_corpus exists with envelopes, --no-replay means we're not writing
    # to it at all — so the guard must not fire.
    replay_dir = tmp_path / "replay_corpus"
    _plant_envelopes(replay_dir, profile="saas", n=5)

    captured: list = []

    runner = CliRunner()
    with patch("resonantforge.pipeline.run_pipeline", side_effect=_make_fake_run_pipeline(captured)):
        result = runner.invoke(
            cli,
            [
                "generate",
                "--dry-run",
                "--out-root", str(tmp_path / "corpus"),
                "--no-replay",
            ],
        )

    assert result.exit_code == 0, (
        "--no-replay must skip the guard; "
        f"got exit_code={result.exit_code}\nOutput:\n{result.output}"
    )
    assert captured, "run_pipeline must be called when --no-replay is passed"


# ---------------------------------------------------------------------------
# Guard does not fire on empty replay directory
# ---------------------------------------------------------------------------


def test_guard_does_not_fire_when_replay_dir_is_empty(tmp_path: Path) -> None:
    """Guard must not fire when the replay directory exists but contains no envelopes."""
    replay_dir = tmp_path / "replay_corpus"
    # Create the directory but plant no envelopes.
    (replay_dir / "saas").mkdir(parents=True, exist_ok=True)

    captured: list = []

    runner = CliRunner()
    with patch("resonantforge.pipeline.run_pipeline", side_effect=_make_fake_run_pipeline(captured)):
        result = runner.invoke(
            cli,
            [
                "generate",
                "--dry-run",
                "--out-root", str(tmp_path / "corpus"),
                "--replay-out", str(replay_dir),
            ],
        )

    assert result.exit_code == 0, (
        "Guard must not fire on an empty replay directory; "
        f"got exit_code={result.exit_code}\nOutput:\n{result.output}"
    )
    assert captured, "run_pipeline must be called when replay dir has no envelopes"


def test_guard_does_not_fire_when_replay_dir_missing(tmp_path: Path) -> None:
    """Guard must not fire when the replay directory does not exist at all."""
    replay_dir = tmp_path / "replay_corpus_nonexistent"
    # Do not create the directory.

    captured: list = []

    runner = CliRunner()
    with patch("resonantforge.pipeline.run_pipeline", side_effect=_make_fake_run_pipeline(captured)):
        result = runner.invoke(
            cli,
            [
                "generate",
                "--dry-run",
                "--out-root", str(tmp_path / "corpus"),
                "--replay-out", str(replay_dir),
            ],
        )

    assert result.exit_code == 0, (
        "Guard must not fire when replay dir does not exist; "
        f"got exit_code={result.exit_code}\nOutput:\n{result.output}"
    )
    assert captured, "run_pipeline must be called when replay dir does not exist"


# ---------------------------------------------------------------------------
# Help output includes --force-replay-out
# ---------------------------------------------------------------------------


def test_force_replay_out_appears_in_help() -> None:
    """--force-replay-out must be present in rforge generate --help."""
    runner = CliRunner()
    result = runner.invoke(cli, ["generate", "--help"])
    assert "--force-replay-out" in result.output, (
        "--force-replay-out must appear in rforge generate --help"
    )
