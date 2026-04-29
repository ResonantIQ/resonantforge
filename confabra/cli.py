"""
Confabra CLI entry point.

Subcommands (generate, validate, inspect, stats) are implemented in later tasks.
This module registers the root click group so the pyproject.toml scripts entry
resolves correctly from the installed package.
"""

import click


@click.group()
def cli() -> None:
    """Confabra — deterministic synthetic customer-conversation corpus generator."""
