"""
Confabra CLI entry point.

Provides four subcommands:
  generate  — run the corpus generation pipeline
  validate  — verify a generated corpus directory against its manifest
  inspect   — sample records from each corpus artifact
  stats     — print manifest statistics in a Rich panel
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from typing import Optional

import click
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

console = Console()

# ---------------------------------------------------------------------------
# Profile defaults
# ---------------------------------------------------------------------------

_PROFILE_DEFAULTS: dict[str, dict[str, int]] = {
    "saas": {"accounts": 25, "months": 6},
    "professional_services": {"accounts": 5, "months": 3},
    "ps": {"accounts": 5, "months": 3},
}

_DEFAULT_PROFILE = "saas"


def _resolve_profile_defaults(profile: str, accounts: Optional[int], months: Optional[int]) -> tuple[int, int]:
    """
    Resolve ``--accounts`` and ``--months`` to concrete integers.

    Click does not support dynamic defaults tied to another option, so both
    parameters use ``default=None`` and we fill in profile-appropriate values
    here.  Falls back to the SaaS profile defaults for unrecognised profile
    names.

    Args:
        profile:  Profile name supplied by the user (e.g. "saas", "ps").
        accounts: Explicit account count, or None if not supplied.
        months:   Explicit month count, or None if not supplied.

    Returns:
        A ``(accounts, months)`` tuple with concrete integer values.
    """
    defaults = _PROFILE_DEFAULTS.get(profile, _PROFILE_DEFAULTS[_DEFAULT_PROFILE])
    return (
        accounts if accounts is not None else defaults["accounts"],
        months if months is not None else defaults["months"],
    )


# ---------------------------------------------------------------------------
# Root group
# ---------------------------------------------------------------------------


@click.group()
def cli() -> None:
    """Confabra — deterministic synthetic customer-conversation corpus generator."""


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


@cli.command("generate")
@click.option("--profile", default="saas", show_default=True, help='Profile name: "saas" or "professional_services"')
@click.option("--accounts", default=None, type=int, help="Number of accounts to simulate (default: profile-dependent)")
@click.option("--months", default=None, type=int, help="Number of months to simulate (default: profile-dependent)")
@click.option("--seed", default=42, show_default=True, type=int, help="Deterministic PRNG seed")
@click.option("--out-root", "out_root", default=None, type=click.Path(), help="Output root directory; corpus written to <out-root>/<profile>/ (default: ./corpus)")
@click.option("--out", "out_legacy", default=None, type=click.Path(), hidden=True, help="[DEPRECATED] Use --out-root instead.")
@click.option("--api-key", default=None, envvar="ANTHROPIC_API_KEY", help="[DEPRECATED] Set ANTHROPIC_API_KEY environment variable instead. Anthropic API key.")
@click.option("--verbose", is_flag=True, default=False, help="Enable verbose output")
@click.option("--dry-run", "dry_run", is_flag=True, default=False, help="Skip LLM calls; generate deterministic placeholder prose")
@click.option("--force", is_flag=True, default=False, help="Overwrite existing output directory contents instead of failing fast.")
def generate(
    profile: str,
    accounts: Optional[int],
    months: Optional[int],
    seed: int,
    out_root: Optional[str],
    out_legacy: Optional[str],
    api_key: Optional[str],
    verbose: bool,
    dry_run: bool,
    force: bool,
) -> None:
    """Run the corpus generation pipeline."""
    import sys as _sys
    from confabra.pipeline import PipelineConfig, run_pipeline

    resolved_accounts, resolved_months = _resolve_profile_defaults(profile, accounts, months)

    # --out is deprecated; warn and map to --out-root.
    if out_legacy is not None:
        print(
            "WARNING: --out flag is deprecated and will be removed in a future release. "
            "Use --out-root instead.",
            file=_sys.stderr,
        )
        if out_root is None:
            out_root = out_legacy

    # --api-key deprecation warning (when the flag is explicitly passed, it will be non-None
    # only because the user typed it — envvar reads are transparent and do not need a warning).
    # Click sets envvar values before the callback; we detect explicit flag use by checking
    # whether the api_key context source was the CLI (not the env).
    # Simplest heuristic: warn whenever api_key is set AND ANTHROPIC_API_KEY env is not set,
    # meaning the value came from the flag directly.
    if api_key is not None and not os.environ.get("ANTHROPIC_API_KEY"):
        print(
            "WARNING: --api-key flag is deprecated and will be removed in a future release. "
            "Set ANTHROPIC_API_KEY environment variable instead.",
            file=_sys.stderr,
        )

    # Resolve output root directory.
    # pipeline.py appends profile.name internally → ./corpus/<profile>/
    if out_root is not None:
        output_root = Path(out_root)
    else:
        output_root = Path("corpus")

    # --dry-run overrides any API key: treat as None (no LLM calls).
    effective_api_key: Optional[str] = None if dry_run else api_key

    config = PipelineConfig(
        profile_name=profile,
        accounts=resolved_accounts,
        months=resolved_months,
        seed=seed,
        output_root=output_root,
        anthropic_api_key=effective_api_key,
        verbose=verbose,
        force=force,
    )

    mode_label = "dry-run" if effective_api_key is None else "live"
    console.print(
        f"[bold]confabra generate[/bold]  profile=[cyan]{profile}[/cyan]  "
        f"accounts=[cyan]{resolved_accounts}[/cyan]  months=[cyan]{resolved_months}[/cyan]  "
        f"seed=[cyan]{seed}[/cyan]  mode=[cyan]{mode_label}[/cyan]"
    )

    try:
        manifest = run_pipeline(config)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[bold red]Error:[/bold red] {exc}")
        sys.exit(1)

    # Print summary table.
    table = Table(title="Generation Summary", show_header=True, header_style="bold cyan")
    table.add_column("Metric", style="bold")
    table.add_column("Value", justify="right")

    table.add_row("Profile", manifest.profile_name)
    table.add_row("Seed", str(manifest.seed))
    table.add_row("Accounts", str(manifest.accounts))
    table.add_row("Months", str(manifest.months))
    table.add_row("Events", str(manifest.event_count))
    table.add_row("Conversations", str(manifest.conversation_count))
    table.add_row("Planted quality", str(manifest.planted_quality_count))
    table.add_row("Skipped", str(manifest.skipped_conversation_count))
    table.add_row("Corrections", str(manifest.corrections_count))
    table.add_row("Agents", str(manifest.agent_count))
    table.add_row("KB docs", str(manifest.knowledge_base_doc_count))
    table.add_row("KB chunks", str(manifest.knowledge_base_chunk_count))
    table.add_row("Output dir", str(output_root / manifest.profile_name))  # profile subdir

    console.print(table)
    console.print("[bold green]Done.[/bold green]")


# ---------------------------------------------------------------------------
# validate
# ---------------------------------------------------------------------------


@cli.command("validate")
@click.argument("corpus_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--profile", required=True, help="Profile name (required)")
def validate(corpus_dir: Path, profile: str) -> None:
    """Validate a generated corpus directory against its manifest."""
    profile_dir = corpus_dir / profile

    checks: list[tuple[str, bool, str]] = []  # (label, passed, detail)

    # -----------------------------------------------------------------------
    # 1. manifest.json exists and is valid JSON
    # -----------------------------------------------------------------------
    manifest_path = profile_dir / "manifest.json"
    manifest_data: Optional[dict] = None
    if not manifest_path.exists():
        checks.append(("manifest.json exists", False, "file not found"))
    else:
        try:
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            checks.append(("manifest.json exists & valid JSON", True, ""))
        except json.JSONDecodeError as exc:
            checks.append(("manifest.json exists & valid JSON", False, str(exc)))

    # -----------------------------------------------------------------------
    # 2. Each expected JSONL file exists and has the correct line count
    # -----------------------------------------------------------------------
    if manifest_data is not None:
        jsonl_specs = [
            ("events.jsonl", manifest_data.get("event_count", 0)),
            ("snapshots.jsonl", manifest_data.get("snapshot_count", 0)),
            ("conversations.jsonl", manifest_data.get("conversation_count", 0)),
            ("planted_quality.jsonl", manifest_data.get("planted_quality_count", 0)),
            ("corrections.jsonl", manifest_data.get("corrections_count", 0)),
        ]
        for filename, expected_count in jsonl_specs:
            fpath = profile_dir / filename
            if not fpath.exists():
                checks.append((f"{filename} exists", False, "file not found"))
                continue
            # Count non-empty lines.
            lines = [ln for ln in fpath.read_text(encoding="utf-8").splitlines() if ln.strip()]
            actual_count = len(lines)
            if actual_count == expected_count:
                checks.append((f"{filename} line count ({expected_count})", True, ""))
            else:
                checks.append(
                    (
                        f"{filename} line count",
                        False,
                        f"expected {expected_count}, got {actual_count}",
                    )
                )

    # -----------------------------------------------------------------------
    # 3. tenant_config.json exists and parses
    # -----------------------------------------------------------------------
    tc_path = profile_dir / "tenant_config.json"
    if not tc_path.exists():
        checks.append(("tenant_config.json exists", False, "file not found"))
    else:
        try:
            json.loads(tc_path.read_text(encoding="utf-8"))
            checks.append(("tenant_config.json exists & valid JSON", True, ""))
        except json.JSONDecodeError as exc:
            checks.append(("tenant_config.json exists & valid JSON", False, str(exc)))

    # -----------------------------------------------------------------------
    # 4. agents/ directory exists with expected agent count
    # -----------------------------------------------------------------------
    agents_dir = profile_dir / "agents"
    if not agents_dir.exists() or not agents_dir.is_dir():
        checks.append(("agents/ directory exists", False, "directory not found"))
    else:
        # Each agent lives in a subdirectory (e.g. agents/agent_001/profile.json).
        # Count subdirectories that contain a profile.json file.
        agent_subdirs = [d for d in agents_dir.iterdir() if d.is_dir() and (d / "profile.json").exists()]
        expected_agents = manifest_data.get("agent_count", 0) if manifest_data else 0
        actual_agents = len(agent_subdirs)
        if actual_agents == expected_agents:
            checks.append((f"agents/ count ({expected_agents})", True, ""))
        else:
            checks.append(
                (
                    "agents/ count",
                    False,
                    f"expected {expected_agents}, got {actual_agents}",
                )
            )

    # -----------------------------------------------------------------------
    # Render results table
    # -----------------------------------------------------------------------
    table = Table(title=f"Validation — {profile_dir}", show_header=True, header_style="bold")
    table.add_column("Check", style="bold")
    table.add_column("Result", justify="center")
    table.add_column("Detail")

    all_passed = True
    for label, passed, detail in checks:
        result_text = "[bold green]✓[/bold green]" if passed else "[bold red]✗[/bold red]"
        table.add_row(label, result_text, detail)
        if not passed:
            all_passed = False

    console.print(table)

    if all_passed:
        console.print("[bold green]PASS[/bold green] — all checks passed.")
    else:
        console.print("[bold red]FAIL[/bold red] — one or more checks failed.")
        sys.exit(1)


# ---------------------------------------------------------------------------
# inspect
# ---------------------------------------------------------------------------


@cli.command("inspect")
@click.argument("corpus_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--profile", required=True, help="Profile name (required)")
@click.option("--limit", default=5, show_default=True, type=int, help="Max records to show per artifact")
def inspect(corpus_dir: Path, profile: str, limit: int) -> None:
    """Print sample records from each corpus artifact."""
    profile_dir = corpus_dir / profile

    artifacts = [
        ("events.jsonl", "Events"),
        ("conversations.jsonl", "Conversations"),
        ("corrections.jsonl", "Corrections"),
    ]

    for filename, label in artifacts:
        fpath = profile_dir / filename
        console.rule(f"[bold cyan]{label}[/bold cyan] — {filename}")

        if not fpath.exists():
            console.print(f"  [red]File not found:[/red] {fpath}")
            continue

        raw_lines = [ln for ln in fpath.read_text(encoding="utf-8").splitlines() if ln.strip()]
        sample_lines = raw_lines[:limit]
        total = len(raw_lines)

        if not sample_lines:
            console.print("  [dim](empty)[/dim]")
            continue

        # Parse first line to discover keys for the table header.
        try:
            first_row = json.loads(sample_lines[0])
        except json.JSONDecodeError:
            console.print(f"  [red]Could not parse JSON in {filename}[/red]")
            continue

        # Limit columns to keep the table readable: show first 6 keys max.
        all_keys = list(first_row.keys())
        display_keys = all_keys[:6]
        truncated = len(all_keys) > 6

        table = Table(show_header=True, header_style="bold", expand=False)
        for key in display_keys:
            table.add_column(key, max_width=40, overflow="fold")
        if truncated:
            table.add_column("…", style="dim")

        for raw in sample_lines:
            try:
                row = json.loads(raw)
            except json.JSONDecodeError:
                continue
            cells = [str(row.get(k, "")) for k in display_keys]
            if truncated:
                cells.append(f"+{len(all_keys) - 6} more")
            table.add_row(*cells)

        console.print(table)
        console.print(f"  [dim]Showing {len(sample_lines)} of {total} records.[/dim]")

    # Agent count from manifest.
    console.rule("[bold cyan]Agents[/bold cyan]")
    manifest_path = profile_dir / "manifest.json"
    if manifest_path.exists():
        try:
            manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
            agent_count = manifest_data.get("agent_count", "unknown")
            console.print(f"  Agent count (from manifest): [cyan]{agent_count}[/cyan]")
        except json.JSONDecodeError:
            console.print("  [red]Could not parse manifest.json[/red]")
    else:
        console.print("  [red]manifest.json not found[/red]")


# ---------------------------------------------------------------------------
# stats
# ---------------------------------------------------------------------------


@cli.command("stats")
@click.argument("corpus_dir", type=click.Path(exists=True, file_okay=False, path_type=Path))
@click.option("--profile", required=True, help="Profile name (required)")
def stats(corpus_dir: Path, profile: str) -> None:
    """Print manifest statistics."""
    profile_dir = corpus_dir / profile
    manifest_path = profile_dir / "manifest.json"

    if not manifest_path.exists():
        console.print(f"[bold red]Error:[/bold red] manifest.json not found at {manifest_path}")
        sys.exit(1)

    try:
        manifest_data = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        console.print(f"[bold red]Error:[/bold red] could not parse manifest.json: {exc}")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # Build a single stats table.
    # -----------------------------------------------------------------------
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_column("Key", style="bold")
    table.add_column("Value", justify="right", style="cyan")

    def _row(key: str, value: object) -> None:
        table.add_row(key, str(value))

    def _section(title: str) -> None:
        table.add_row(f"[dim]— {title} —[/dim]", "")

    # Run metadata.
    _section("Run")
    _row("Profile", manifest_data.get("profile_name", ""))
    _row("Profile version", manifest_data.get("profile_version", ""))
    _row("Generator version", manifest_data.get("generator_version", ""))
    _row("Seed", manifest_data.get("seed", ""))
    _row("Accounts", manifest_data.get("accounts", ""))
    _row("Months", manifest_data.get("months", ""))
    _row("Generated at", manifest_data.get("generated_at", ""))

    # Conversation stats.
    _section("Conversations")
    conv_count = manifest_data.get("conversation_count", 0)
    skipped = manifest_data.get("skipped_conversation_count", 0)
    planted = manifest_data.get("planted_quality_count", 0)
    total_attempted = conv_count + skipped
    skip_rate = skipped / total_attempted if total_attempted > 0 else 0.0
    _row("Conversations generated", conv_count)
    _row("Planted quality", planted)
    _row("Skipped", skipped)
    _row("Skip rate", f"{skip_rate:.1%}")
    _row("Events", manifest_data.get("event_count", ""))
    _row("Snapshots", manifest_data.get("snapshot_count", ""))

    # Other artifact counts.
    _section("Artifacts")
    _row("Corrections", manifest_data.get("corrections_count", ""))
    _row("Agents", manifest_data.get("agent_count", ""))
    _row("KB docs", manifest_data.get("knowledge_base_doc_count", ""))
    _row("KB chunks", manifest_data.get("knowledge_base_chunk_count", ""))

    # Quality rates.
    _section("Quality rates")
    _row("Prose fact violation rate", f"{manifest_data.get('prose_fact_violation_rate', 0):.3f}")
    _row("Validator rule failure rate", f"{manifest_data.get('validator_rule_failure_rate', 0):.3f}")
    _row("Disagreement rate", f"{manifest_data.get('disagreement_rate', 0):.3f}")

    # Hashes.
    _section("Hashes")
    hash_keys = [
        ("events_hash", "Events"),
        ("snapshots_hash", "Snapshots"),
        ("conversations_hash", "Conversations"),
        ("planted_quality_hash", "Planted quality"),
        ("kb_chunks_hash", "Knowledge base chunks"),
        ("tenant_config_hash", "Tenant config"),
        ("agent_fixtures_hash", "Agent fixtures"),
        ("corrections_hash", "Corrections"),
    ]
    for field_name, label in hash_keys:
        h = manifest_data.get(field_name, "")
        # Truncate for readability — first 16 hex chars.
        _row(label, h[:16] + "…" if len(h) > 16 else h)

    console.print(Panel(table, title=f"[bold]Corpus Stats — {profile}[/bold]", border_style="cyan"))


# ---------------------------------------------------------------------------
# check-invariants
# ---------------------------------------------------------------------------


@cli.command("check-invariants")
@click.option("--profile", default="saas", show_default=True, help="Profile name to load KB chunks from.")
def check_invariants_cmd(profile: str) -> None:
    """Run the KB invariant checker against the named profile's knowledge base."""
    from confabra.kb.saas_content import get_saas_kb_chunks
    from confabra.validators.invariant_checker import run_checker

    if profile != "saas":
        console.print(f"[red]Unknown profile: {profile!r}. Only 'saas' is supported.[/red]")
        raise SystemExit(1)

    chunks = get_saas_kb_chunks()
    report = run_checker(chunks)

    if report.warnings:
        for w in report.warnings:
            console.print(f"[yellow]WARNING:[/yellow] {w}")

    if report.errors:
        for e in report.errors:
            console.print(f"[red]ERROR:[/red] {e}")
        console.print(f"\n[red]Invariant check FAILED — {len(report.errors)} error(s).[/red]")
        raise SystemExit(1)

    console.print(f"[green]Invariant check PASSED — 0 errors, {len(report.warnings)} warning(s).[/green]")
