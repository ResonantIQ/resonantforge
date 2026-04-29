# Confabra

Deterministic synthetic customer-conversation corpus generator for the Resonant IQ intelligence test harness.

## Overview

Confabra generates a reproducible corpus of customer-support conversations with planted quality signals, KB citations, agent trajectories, and human corrections. Given the same `--seed`, it produces bit-for-bit identical output, enabling CI-stable regression tests for every intelligence algorithm built on top of it.

## Requirements

- Python 3.11+
- `pip install -e .` (installs `pydantic`, `anthropic`, `click`, `rich`, `pytest`)

## Quick start

```bash
cd harness
pip install -e .
confabra --help
```

## Structure

```
confabra/
  schemas.py          # All Pydantic v2 data models (source of truth)
  cli.py              # Click CLI entry point
  layer1/             # Account simulation engine (event stream)
  validators/         # Rule-based per-dimension signal validation
    extractors/       # Signal extraction from raw prose
  kb/                 # Knowledge base document and chunk management
  profiles/           # Tenant brand-voice and rubric profiles
    lexicons/         # Domain-specific vocabulary lists
  agents/             # Synthetic agent fixtures and trajectory logic
  corrections/        # Planted human score corrections (Section 11.3)
  tenant_config/      # Per-tenant configuration loading
tests/
  test_properties.py  # 22-assertion property test suite (Task 17)
```

## Determinism guarantee

All randomness is seeded from a single integer passed via `--seed`. The generator uses Python's `random.Random(seed)` (never the global state) so parallel runs with different seeds are independent.
