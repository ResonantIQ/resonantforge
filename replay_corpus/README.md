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

### Step 1 — Generate corpus (envelopes written automatically)

```bash
rforge generate --profile saas --accounts 10 --months 3
```

Envelopes are written to `replay_corpus/saas/` automatically alongside the
corpus. Planted conversations use claims already extracted during accuracy
validation — **no extra LLM cost**. Organic conversations get an envelope
with empty claims (useful for replaying non-accuracy dimensions).

To write envelopes to a custom path:

```bash
rforge generate --profile saas --accounts 10 --months 3 \
  --replay-out /path/to/my/replay_corpus
```

To skip envelope writing entirely (uncommon):

```bash
rforge generate --profile saas --accounts 10 --months 3 --no-replay
```

### Step 1b — Extract envelopes from an existing corpus (post-hoc, paid)

If you have a corpus without envelopes (generated before this feature, or with
`--no-replay`), you can extract them after the fact. This path calls
`extract_claims_llm` once per conversation — the one LLM cost in the replay
workflow.

```bash
rforge replay extract-envelopes \
  --corpus corpus/saas \
  --profile saas \
  --out replay_corpus/saas
```

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
re-run Step 1b. Existing labels survive re-extraction unchanged — see
`harness/docs/replay-envelope-schema.md § Re-extraction and Label Survival`.

## Cost model

| Path | LLM calls |
|---|---|
| `rforge generate` (default) | Prose generation + claim extraction for planted convs (already paid). No extra cost for envelopes. |
| `rforge replay extract-envelopes` | One claim extraction call per conversation. |
| `rforge replay run` | Zero. Free forever. |

The key principle: **every smoke run automatically produces a replay corpus**.
Never lose the ability to replay a failing conversation.

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
