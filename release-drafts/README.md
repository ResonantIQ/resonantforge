# ResonantForge

> **Pre-release notice:** This is ResonantForge v0.2.0, currently in pre-release. APIs and output formats may change before the stable release. Feedback is welcome via [GitHub Issues](https://github.com/resonantiq/resonantforge/issues).

ResonantForge is a Python CLI tool that generates deterministic synthetic customer-support conversation corpora with planted quality signals, designed for testing and evaluating AI scoring pipelines. You point it at a profile (industry template), give it a seed, and it produces a realistic corpus of conversations — complete with quality-signal annotations, knowledge-base citations, agent trajectories, and planted human corrections — that is bit-for-bit identical every time you run it with the same `--seed`. This reproducibility makes it practical to use as a stable CI fixture for regression-testing conversation quality evaluation systems.

## Who it's for

ResonantForge is built for AI/ML engineers building conversation quality evaluation systems, QA tooling developers who need realistic synthetic data to test scoring pipelines without exposing real customer conversations, and NLP researchers studying synthetic data generation for support-domain text. If you are building a system that scores, classifies, or analyzes customer-support conversations and you need a controlled, reproducible corpus to validate it against, ResonantForge is for you.

## Requirements

- Python 3.11+
- pip

## 60-second quickstart

No API key needed to get started — `--dry-run` skips all LLM calls and generates deterministic placeholder prose.

```bash
git clone https://github.com/resonantiq/resonantforge
cd resonantforge
pip install -e .
rforge --help
rforge generate --dry-run --smoke
```

This produces a smoke corpus (roughly 100 conversations) under `./corpus/saas/` with placeholder prose. Inspect what was generated:

```bash
rforge stats corpus --profile saas
rforge inspect corpus --profile saas
```

**For real corpus generation** (requires an Anthropic API key):

```bash
export ANTHROPIC_API_KEY=your-key-here
rforge generate --smoke
```

## CLI overview

| Command | What it does |
|---|---|
| `rforge generate` | Run the corpus generation pipeline. Produces JSONL artifacts under `./corpus/<profile>/`. |
| `rforge validate <corpus_dir> --profile <name>` | Verify a generated corpus directory against its manifest (file counts, JSON validity, agent directory). |
| `rforge inspect <corpus_dir> --profile <name>` | Print sample records from each corpus artifact (events, conversations, corrections). |
| `rforge stats <corpus_dir> --profile <name>` | Print manifest statistics: conversation counts, skip rates, quality rates, determinism hashes. |
| `rforge replay extract-envelopes` | Freeze validator inputs into replay envelopes. One LLM call per conversation; run once. |
| `rforge replay run` | Run all validators against frozen envelopes at zero LLM cost. Use this for validator iteration. |

### Key `generate` flags

| Flag | Default | Description |
|---|---|---|
| `--profile` | `saas` | Profile name: `saas` or `professional_services` |
| `--seed` | `42` | Deterministic PRNG seed. Same seed → identical output. |
| `--smoke` | off | Smoke mode: ~100 conversations instead of the full default. |
| `--dry-run` | off | Skip LLM calls; use placeholder prose. No API key required. |
| `--accounts N` | profile default | Number of synthetic accounts to simulate. |
| `--months N` | profile default | Duration of simulation in months. |
| `--out-root PATH` | `./corpus` | Root directory for output. Corpus lands at `<out-root>/<profile>/`. |

## Extending ResonantForge

The built-in profiles — `saas` and `professional_services` — are opinionated configurations for those two verticals. To generate corpora for a different domain, subclass `Profile` from `resonantforge/profiles/base.py`.

A `Profile` subclass defines:
- **Lifecycle stages** — the states an account can move through (e.g. onboarding, active, at-risk, churned)
- **Signal vocabulary** — named quality signals and how they are detected (LLM, embedding, or rule)
- **Surface channels** — conversation channels (email, chat, phone) and their prose style characteristics
- **Persona archetypes** — agent persona templates used to seed synthetic agent fixtures
- **Brand voice variants** — distinct writing styles for the simulated tenant
- **Knowledge base content** — KB chunks that the corpus can cite
- **Correction patterns** — planted human score corrections with configurable bias patterns
- **Lexicons** — domain-specific vocabulary lists used by the rule-based signal extractors

The abstract base class enforces the interface: every method that must be implemented raises `NotImplementedError` until overridden, so gaps surface immediately rather than silently producing empty outputs. See `resonantforge/profiles/base.py` for the full interface and docstrings.

## Links

- [LICENSE](LICENSE)
- [CONTRIBUTING](CONTRIBUTING.md)
- [GitHub Issues](https://github.com/resonantiq/resonantforge/issues)
