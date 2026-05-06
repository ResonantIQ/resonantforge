# resonantforge.replay

Frozen-replay harness for zero-cost validator iteration.

## Purpose

Run all 4 dimension validators (empathy, resolution, brand_voice, accuracy) against
pre-frozen conversation envelopes at **zero LLM cost**. The only paid step is
`rforge replay extract-envelopes`, which runs once to freeze `extracted_claims`
(the sole LLM-derived artifact). All subsequent replay runs are free.

## Package structure

```
resonantforge/replay/
├── __init__.py       public API exports + REPLAY_MODE guard documentation
├── schemas.py        Pydantic models: ReplayEnvelope, ReplayLabels, ReplayResult, AgreementResult
├── engine.py         load_envelope, load_labels, replay_one
├── agreement.py      compute_agreement (three-layer: outcome, rule, diverging rules)
├── fingerprint.py    compute_fingerprint, compute_corpus_fingerprint
├── diff.py           format_conv_line, format_summary
└── extractor.py      extract_envelopes (called by `rforge replay extract-envelopes`)
```

## Public API

```python
from resonantforge.replay import (
    load_envelope,    # load + validate envelope.json → ReplayEnvelope
    load_labels,      # load + validate labels.json  → ReplayLabels
    replay_one,       # run validators               → ReplayResult
    compute_fingerprint,        # fingerprint one result
    compute_corpus_fingerprint, # fingerprint a full corpus run
    format_conv_line,   # single-line progress output
    format_summary,     # multi-line summary block
    # Error types
    EnvelopeSchemaError,
    LabelSchemaError,
    ReplayModeError,
)
```

## Determinism contract

`replay_one` is fully deterministic — same envelope + same code → byte-identical output every run.

Four guarantees enforced by the engine:

1. **No LLM calls** — `extracted_claims` is injected directly into `run_kb_alignment_pipeline`;
   `extract_claims_llm` is bypassed entirely. When `RFORGE_REPLAY_MODE=1`, calling
   `extract_claims_llm` raises `ReplayModeError` (fail-loud, not silent).

2. **No `datetime.now()` / `time.time()`** — no timestamps are generated in the replay
   code path. Provenance timestamps come from `envelope.metadata.extraction_timestamp` (frozen).

3. **Stable sort** — `_normalize_text()` synonym map iteration uses `sorted()` (longest-first);
   `run_kb_alignment_pipeline()` iterates `candidate_chunks` in envelope load order.

4. **No RNG** — none of the 4 extractors use randomness.

The `test_determinism` test enforces this by computing fingerprints across two replay runs
of the same corpus and asserting they are byte-identical.

## Three-layer agreement

`compute_agreement` returns an `AgreementResult` with:

| Layer | Meaning |
|---|---|
| `outcome_match` | overall pass/fail matches `expected_outcome` (`None` for "uncertain") |
| `rule_match` | exact set of failed rules matches `expected_failures` |
| `false_positive_rules` | rules that fired but label says should pass |
| `missed_rules` | rules label says should fail that the validator didn't catch |

## Replay validates rule scoring logic, not extraction logic

Replay verifies that the 4 dimension validators score correctly given a fixed set of inputs.
It does **not** verify that `extract_claims_llm` extracts claims correctly — that step happens
during `rforge replay extract-envelopes` and its output is frozen into `envelope.json`.

The `claim_extraction` label flag (`expected_failures.claim_extraction: True`) captures
conversations where extraction failed during live generation (LLM call returned empty or
parse-failed). In replay, `claim_extraction_ok` is always `True` because frozen claims are
injected directly — the extraction code is never called. As a result, convs labeled
`claim_extraction: True` always appear as `missed_rules: ["claim_extraction"]` in the
agreement layer. This is expected, not a bug.

**Implication:** Changes to `extract_claims_llm` or `CLAIM_EXTRACTION_PROMPT` require separate
verification — either a full smoke run or extraction-specific fixtures. `rforge replay run`
cannot detect extraction regressions; it only detects rule-scoring regressions.

## Error types

| Error | When |
|---|---|
| `EnvelopeSchemaError` | envelope.json missing, invalid JSON, wrong schema_version, conv_id mismatch |
| `LabelSchemaError` | labels.json missing, invalid JSON, wrong schema_version, conv_id mismatch, unknown tag, missing expected_failures key, empty rationale, revision without notes |
| `ReplayModeError` | `extract_claims_llm` called while `RFORGE_REPLAY_MODE=1` |
