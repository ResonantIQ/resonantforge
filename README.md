# ResonantForge

ResonantForge is a Python CLI that generates synthetic two-party conversation corpora with planted quality signals — built for testing and evaluating systems that score, classify, route, or analyze structured conversations. Point it at a profile, give it a seed, and it produces a realistic corpus of dialogue transcripts complete with quality-signal annotations, source-document citations, participant trajectories, and planted human review corrections. The simulation layer is fully deterministic from `--seed`; in `--dry-run` mode the entire output is bit-identical across runs, and in live mode the structure, metadata, and signal annotations are reproducible while the LLM-generated prose itself varies per call.

The shipped profiles model customer-support conversations (SaaS and professional services verticals), but the architecture is domain-agnostic. Subclassing `Profile` lets you generate corpora for any two-party conversation domain — sales calls, onboarding flows, technical interviews, advisory sessions, intake conversations, or anything else that fits a two-participant structure.

> **v0.2.0** — initial public release. Output formats and APIs may evolve in future versions; pin to a specific version if you need stability. Feedback welcome via [GitHub Issues](https://github.com/ResonantIQ/resonantforge/issues).

## Who it's for

ResonantForge is built for engineers and teams building conversation evaluation systems — anyone who needs realistic, controlled, reproducible dialogue data to test scoring pipelines, validate classifiers, or benchmark routing logic without exposing real user conversations. The deterministic core makes it practical to use as a stable CI fixture. The profile system makes it adaptable to whatever conversation domain you actually work in.

## Requirements

- Python 3.11+
- pip
- (Optional) An Anthropic API key for live prose generation

## 60-second quickstart

No API key needed to start — `--dry-run` skips all LLM calls and generates deterministic placeholder prose.

```bash
git clone https://github.com/ResonantIQ/resonantforge
cd resonantforge
pip install -e .
rforge --help
rforge generate --dry-run --smoke
```

This produces a smoke corpus (~100 conversations) under the default output directory `./corpus/saas/`. Inspect what was generated:

```bash
rforge stats corpus --profile saas
rforge inspect corpus --profile saas
```

**For live corpus generation** (calls the Anthropic API; a smoke run costs a few cents, a full run costs a few dollars at current Haiku pricing):

```bash
export ANTHROPIC_API_KEY=your-key-here
rforge generate --smoke
```

You can override the model with `RFORGE_MODEL` if you want to use something other than the current default (`claude-haiku-4-5-20251001`).

## CLI overview

| Command | What it does |
|---|---|
| `rforge generate` | Run the corpus generation pipeline. Produces JSONL artifacts under `./corpus/<profile>/`. |
| `rforge validate <corpus_dir> --profile <name>` | Verify a generated corpus against its manifest. |
| `rforge inspect <corpus_dir> --profile <name>` | Print sample records from each corpus artifact. |
| `rforge stats <corpus_dir> --profile <name>` | Print manifest statistics, skip rates, and determinism hashes. |
| `rforge replay extract-envelopes` | Freeze validator inputs into replay envelopes. One LLM call per conversation; run once. |
| `rforge replay run` | Run all validators against frozen envelopes at zero LLM cost. Use this for validator iteration. |

### Key `generate` flags

| Flag | Default | Description |
|---|---|---|
| `--profile` | `saas` | Profile name: `saas` or `professional_services` |
| `--seed` | `42` | Deterministic PRNG seed. Same seed → identical simulation structure. |
| `--smoke` | off | Smoke mode: ~100 conversations instead of the full default. |
| `--dry-run` | off | Skip LLM calls; use placeholder prose. No API key required. |
| `--accounts N` | profile default | Number of synthetic accounts to simulate. |
| `--months N` | profile default | Duration of simulation in months. |
| `--out-root PATH` | `./corpus` | Root directory for output. Corpus lands at `<out-root>/<profile>/`. |

## Extending ResonantForge

To generate corpora for a new domain, subclass `Profile` from `resonantforge/profiles/base.py`. A profile defines the brand voice variants the simulated participants use, the knowledge base chunks the corpus can cite, the lexicons that drive rule-based signal extraction, the volume of planted-quality conversations to inject, and the correction patterns used to model human reviewer noise. The abstract base class enforces the interface — unimplemented methods raise `NotImplementedError` until overridden, so gaps surface immediately rather than silently producing empty outputs.

The current pipeline assumes a two-party dialogue structure (two participants alternating turns). Adapting Forge to multi-party conversations, monologues, or non-dialogue formats would require pipeline-level changes beyond a `Profile` subclass.

See `resonantforge/profiles/base.py` for the full interface and `resonantforge/profiles/saas.py` for a complete worked example.

## Prior art

ResonantForge builds on two pieces of recent work:

**[OrgForge](https://arxiv.org/abs/2603.14997)** (Flynt, 2026) introduced the physics-cognition boundary that ResonantForge applies to two-party conversations: a deterministic engine maintains the SimEvent ground-truth bus, and LLMs operate only at designated injection points, generating surface prose from validated proposals rather than mutating state directly. This makes cross-artifact consistency an architectural guarantee rather than an empirical claim. Where OrgForge models multi-artifact organizational corpora (Slack, JIRA, email, postmortems), ResonantForge applies the same boundary to dialogue transcripts with planted quality signals.

**[BrainBench](https://github.com/braingpt-lovelab/BrainBench)** (Luo et al., 2024) introduced the contrastive evaluation pattern that ResonantForge uses for scoring: present two versions of an artifact — one ground-truth, one altered to shift a specific property while remaining coherent — and measure whether a system can identify which is which. ResonantForge adapts this from neuroscience abstracts to conversation prose with planted quality variations.

If you're working in this space, the OrgForge paper and codebase are the most direct architectural predecessor to ResonantForge and worth reading.

## Links

- [LICENSE](LICENSE)
- [CONTRIBUTING](CONTRIBUTING.md)
- [Known limitations and footguns](COMMON_MISTAKES.md)
- [GitHub Issues](https://github.com/ResonantIQ/resonantforge/issues)
