# Replay Corpus

The replay corpus contains frozen conversation envelopes and human-authored labels
used by `rforge replay run` to iterate on validators at zero LLM cost.

## Directory Layout

```
replay_corpus/
└── {profile}/           e.g. saas/, professional_services/
    └── {conv_id}/
        ├── envelope.json    frozen validator inputs (machine-generated)
        └── labels.json      ground-truth labels (human-authored)
```

Each `{conv_id}/` directory is one self-contained unit. Both files must be
present for a conversation to be included in a replay run.

## Workflow

### Step 1 — Extract envelopes (one-time, paid)

```bash
rforge replay extract-envelopes \
  --corpus corpus/saas \
  --profile saas \
  --out replay_corpus/saas
```

This reads `corpus/saas/conversations.jsonl`, calls `extract_claims_llm` once
per conversation (the only LLM call in the replay workflow), and writes:
- `replay_corpus/saas/{conv_id}/envelope.json` — frozen inputs
- `replay_corpus/saas/{conv_id}/labels.json` — empty template (if absent)

Re-running is safe: envelopes are overwritten; existing `labels.json` files
are never touched.

### Step 2 — Author labels

Open each `labels.json` and fill in:
- `expected_outcome`: `"pass"` | `"fail"` | `"uncertain"`
- `expected_failures`: which rules should fail (one boolean per rule)
- `confidence`: `"high"` | `"medium"` | `"low"`
- `tags`: one or more tags from the closed vocabulary
- `rationale`: 1–3 sentences explaining your reasoning

Leave `expected_outcome: "uncertain"` for conversations where you're unsure —
these are still run but excluded from agreement rate calculations.

See `harness/docs/replay-label-schema.md` for the full schema and field reference.

### Step 3 — Run replay (free, iteratable)

```bash
rforge replay run --corpus replay_corpus/saas
```

Prints per-conversation agreement results live and a final summary:
- **Outcome match rate** — overall pass/fail correct (excluding uncertain)
- **Rule match rate** — exact set of failing rules correct
- **False positive rules** — rules that fired but label says should pass
- **Missed rules** — rules label says should fail that didn't fire

To get the determinism fingerprint (useful for verifying stability after a code change):

```bash
rforge replay run --corpus replay_corpus/saas --fingerprint
```

### Step 4 — Iterate

Modify a validator or extractor, then re-run Step 3. No LLM calls, no cost.
The fingerprint changes when verdicts change — that's the signal that your
code change had an effect.

To re-extract after changing `extracted_claims` (e.g. accuracy prompt revision),
re-run Step 1. Existing labels survive re-extraction unchanged — see
`harness/docs/replay-envelope-schema.md § Re-extraction and Label Survival`.

## Schema Versions

| Artifact | Current version | Doc |
|---|---|---|
| `envelope.json` | v1 | `harness/docs/replay-envelope-schema.md` |
| `labels.json` | v1 | `harness/docs/replay-label-schema.md` |

Version mismatches always fail loudly (`EnvelopeSchemaError` / `LabelSchemaError`).
There are no implicit migrations.

## What lives in this directory

- **`{profile}/{conv_id}/envelope.json`** — machine-generated, safe to re-generate
- **`{profile}/{conv_id}/labels.json`** — human-authored; never auto-overwritten

Both types are committed to git. Envelopes are reproducible; labels represent
human judgment and are the only artifact that can't be regenerated automatically.
