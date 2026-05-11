"""
Tests for the --smoke CLI flag (RFORGE-44).

The --smoke flag is a shorthand for a cheap validation run:
  --accounts=5 --months=4 (~107 conversations, ~7.5x cheaper than full run).

Covers:
  - --smoke appears in rforge generate --help
  - --smoke resolves to _SMOKE_ACCOUNTS and _SMOKE_MONTHS via PipelineConfig
  - --smoke is mutually exclusive with --accounts and --months
  - --smoke with --accounts exits with an error
  - --smoke with --months exits with an error
  - --smoke + --accounts + --months exits with an error (all three together)
  - SMOKE MODE banner is printed when --smoke is active
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch
from pathlib import Path

import pytest


# ── Help output ────────────────────────────────────────────────────────────────


def test_smoke_flag_appears_in_help() -> None:
    """--smoke must be present in rforge generate --help."""
    from click.testing import CliRunner
    from resonantforge.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["generate", "--help"])
    assert "--smoke" in result.output, "--smoke flag must appear in rforge generate --help"


# ── Mutual exclusion ───────────────────────────────────────────────────────────


def test_smoke_and_accounts_is_error(tmp_path: Path) -> None:
    """--smoke with --accounts must exit non-zero (mutually exclusive)."""
    from click.testing import CliRunner
    from resonantforge.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["generate", "--smoke", "--accounts", "10", "--dry-run",
                                 "--out-root", str(tmp_path)])
    assert result.exit_code != 0, "--smoke --accounts must exit non-zero"


def test_smoke_and_months_is_error(tmp_path: Path) -> None:
    """--smoke with --months must exit non-zero (mutually exclusive)."""
    from click.testing import CliRunner
    from resonantforge.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["generate", "--smoke", "--months", "2", "--dry-run",
                                 "--out-root", str(tmp_path)])
    assert result.exit_code != 0, "--smoke --months must exit non-zero"


def test_smoke_and_accounts_and_months_is_error(tmp_path: Path) -> None:
    """--smoke with both --accounts and --months must exit non-zero."""
    from click.testing import CliRunner
    from resonantforge.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["generate", "--smoke", "--accounts", "3", "--months", "2",
                                 "--dry-run", "--out-root", str(tmp_path)])
    assert result.exit_code != 0, "--smoke --accounts --months must exit non-zero"


# ── Resolved dimensions ────────────────────────────────────────────────────────


def test_smoke_resolves_to_correct_accounts_and_months(tmp_path: Path) -> None:
    """--smoke must build a PipelineConfig with _SMOKE_ACCOUNTS and _SMOKE_MONTHS."""
    from resonantforge.cli import _SMOKE_ACCOUNTS, _SMOKE_MONTHS

    captured: list = []

    def fake_run_pipeline(config):
        captured.append(config)
        return MagicMock(total_conversations=0, skipped_count=0,
                         planted_rate=0.0, kb_version="test",
                         profile="saas", seed=42,
                         accounts=config.accounts, months=config.months,
                         output_root=str(config.output_root))

    from click.testing import CliRunner
    from resonantforge.cli import cli

    runner = CliRunner()
    with patch("resonantforge.pipeline.run_pipeline", side_effect=fake_run_pipeline):
        result = runner.invoke(cli, ["generate", "--smoke", "--dry-run",
                                     "--out-root", str(tmp_path), "--no-replay"])

    assert captured, "run_pipeline was not called"
    cfg = captured[0]
    assert cfg.accounts == _SMOKE_ACCOUNTS, (
        f"--smoke must set accounts={_SMOKE_ACCOUNTS}, got {cfg.accounts}"
    )
    assert cfg.months == _SMOKE_MONTHS, (
        f"--smoke must set months={_SMOKE_MONTHS}, got {cfg.months}"
    )


# ── Banner ─────────────────────────────────────────────────────────────────────


def test_smoke_prints_banner(tmp_path: Path) -> None:
    """--smoke must print a SMOKE MODE banner so runs are obviously identifiable."""
    def fake_run_pipeline(config):
        return MagicMock(total_conversations=0, skipped_count=0,
                         planted_rate=0.0, kb_version="test",
                         profile="saas", seed=42,
                         accounts=config.accounts, months=config.months,
                         output_root=str(config.output_root))

    from click.testing import CliRunner
    from resonantforge.cli import cli

    runner = CliRunner()
    with patch("resonantforge.pipeline.run_pipeline", side_effect=fake_run_pipeline):
        result = runner.invoke(cli, ["generate", "--smoke", "--dry-run",
                                     "--out-root", str(tmp_path), "--no-replay"])

    assert "SMOKE" in result.output, (
        "--smoke must print a SMOKE MODE banner; got:\n" + result.output
    )
