# Replay Label Schema

**Version:** 1  
**File:** `{corpus}/{conv_id}/labels.json`

A label file records human-authored ground truth for one conversation envelope. It is the basis for three-layer agreement scoring during replay:
1. **Outcome match** — did the validator's pass/fail match `expected_outcome`?
2. **Rule match** — did the set of failed rules match `expected_failures`?
3. **False positive / missed rules** — which rules fired incorrectly?

Labels are the only artifact in the corpus that a human authors by hand. Everything else (envelope, verdicts) is machine-generated.

---

## Full Schema

```json
{
  "schema_version": 1,
  "conv_id": "conv_evt_abc123",
  "label_version": 1,
  "labeled_at": "2026-05-05",
  "labeled_by": "tj",
  "expected_outcome": "pass" | "fail" | "uncertain",
  "expected_failures": {
    "accuracy": false,
    "empathy": false,
    "resolution": false,
    "brand_voice": false,
    "claim_extraction": false
  },
  "confidence": "high" | "medium" | "low",
  "tags": ["edge_case"],
  "rationale": "Agent acknowledged the frustration, followed through with a refund offer, cited the correct policy. Clear pass.",
  "revised_from": null,
  "revision_notes": null
}
```

---

## Field Reference

### `schema_version`
Integer. Must be `1` for this schema. The harness raises `LabelSchemaError` if the value is missing or does not match the expected version.

### `conv_id`
Must match the `conv_id` of its sibling `envelope.json`. The harness validates this at load time.

### `label_version`
Starts at `1`. Increment by 1 on each re-labeling. Git history preserves prior labels; `revised_from` and `revision_notes` carry forward the human-authored audit trail.

### `labeled_at`
ISO date string (`YYYY-MM-DD`). The date you completed the labeling, not a machine timestamp.

### `labeled_by`
Free-form identifier for the labeler (e.g. `"tj"`, `"team"`). Used for provenance in aggregate reports.

### `expected_outcome`

| Value | Meaning |
|---|---|
| `"pass"` | All validators should produce PASS. The conversation is a clean positive example. |
| `"fail"` | At least one validator should produce FAIL. See `expected_failures` for which ones. |
| `"uncertain"` | You are unsure of the correct outcome. The harness includes these convs in replay but excludes them from agreement rate calculations. |

### `expected_failures`

One boolean per validator rule. `true` means you expect that rule to FAIL for this conversation.

| Key | Validator rule |
|---|---|
| `accuracy` | Accuracy KB alignment fails (contradicted, partial, not_found, or overgeneralized when not intended) |
| `empathy` | Empathy signals do not meet rubric target |
| `resolution` | Resolution signals do not meet rubric target |
| `brand_voice` | Brand voice features fall outside calibrated range for the variant |
| `claim_extraction` | `extract_claims_llm()` exhausted retries or returned malformed JSON |

`expected_failures` is only meaningful when `expected_outcome` is `"fail"`. For a `"pass"` outcome, all entries should be `false`.

### `confidence`

| Value | Meaning |
|---|---|
| `"high"` | You are confident in the label. Use when the conversation clearly demonstrates the intended behavior. |
| `"medium"` | You are reasonably confident but the conversation has some ambiguity. |
| `"low"` | The conversation is genuinely ambiguous or you are unsure. Consider tagging `"borderline"` as well. |

Aggregate reports break agreement rates down by confidence bucket. Low-confidence labels that consistently disagree with the validator indicate either labeling uncertainty or a genuine validator blind spot.

### `tags`

Zero or more tags from the **closed vocabulary** below. Tags enable filtering in aggregate reports (`rforge replay run` breaks out agreement by tag).

**Tags are an enforced enum.** The harness raises `LabelSchemaError` on any value not in this list — this prevents typos (e.g. `"borderlne"`) from silently polluting aggregate tag breakdowns with phantom categories.

| Tag | When to use |
|---|---|
| `"high_confidence_fail"` | Clear, unambiguous validator failure. High signal for regression testing. |
| `"high_confidence_pass"` | Clear, unambiguous pass. Use for the golden baseline. |
| `"edge_case"` | Conversation exercises a boundary condition (e.g. zero claims, apology-as-acknowledgment, single-turn exchange). |
| `"borderline"` | The conversation is near a decision boundary and a small code change might flip the verdict. Pair with `confidence: "low"` or `confidence: "medium"`. |
| `"claim_extraction_stress"` | Prose that stresses the claim extractor (long prose, unusual formatting, no claims at all). |
| `"multi_chunk"` | Conversation requires multi-chunk KB satisfaction. |
| `"deny_condition"` | Tests a DENY_CONDITION KB chunk being active. |
| `"overgeneralization"` | Agent drops a constraint present in the KB chunk. |
| `"planted"` | Conversation was a planted-quality case (has a `quality_plan`). |
| `"organic"` | Conversation was organically generated (no `quality_plan`). |

Multiple tags may be applied. Tags are additive — a conversation can be both `"edge_case"` and `"high_confidence_fail"`.

To add a new tag to the vocabulary: update the harness validator and this document in the same PR. Do not use free-form strings — they will be rejected at load time.

### `rationale`
Required string. One to three sentences explaining why you assigned this label. Focus on what is observable in `agent_prose` and `customer_prose` — not on what the validator does. The rationale serves future re-labelers who need to understand your reasoning.

Good: `"Agent uses 'I'm sorry' (apology bridge) followed by 'I'll escalate this now' — satisfies follow_through. Resolution is directional only (no specific actor). Clear empathy:pass, resolution:fail."`

Bad: `"Validator should pass this."` (says nothing about the conversation)

### `revised_from`
`null` for the initial labeling. On re-labeling, set to the previous `label_version` integer (e.g. `1` when revising from v1 to v2). The harness uses this to detect revision chains.

### `revision_notes`
`null` for the initial labeling. On re-labeling, explain **why** the label changed (e.g. `"Re-read: apology bridge rule means apology_present counts as acknowledgment_present. Changed empathy from fail to pass."`). This is the audit trail for label drift.

---

## Re-labeling Workflow

Labels may need revision as your understanding of the validator rules deepens.

1. Open the `labels.json` for the conversation.
2. Increment `label_version` by 1.
3. Set `revised_from` to the prior `label_version`.
4. Update `labeled_at` to today's date.
5. Change `expected_outcome`, `expected_failures`, `confidence`, `tags`, or `rationale` as needed.
6. Add `revision_notes` explaining what changed and why.
7. Save. Git records the prior state — no need to copy the old file.

**Do not create new label files for revisions.** The single `labels.json` per conversation is the canonical label; git history is the audit trail.

---

## Initial Empty Label Template

When `rforge replay extract-envelopes` generates an envelope, it also writes a template `labels.json` with all fields at their zero-state defaults. Fill in `expected_outcome`, `expected_failures`, `confidence`, `tags`, and `rationale` before running `rforge replay run`.

```json
{
  "schema_version": 1,
  "conv_id": "FILL_IN",
  "label_version": 1,
  "labeled_at": "FILL_IN",
  "labeled_by": "tj",
  "expected_outcome": "uncertain",
  "expected_failures": {
    "accuracy": false,
    "empathy": false,
    "resolution": false,
    "brand_voice": false,
    "claim_extraction": false
  },
  "confidence": "low",
  "tags": [],
  "rationale": "FILL_IN",
  "revised_from": null,
  "revision_notes": null
}
```

`expected_outcome: "uncertain"` and `confidence: "low"` are intentional defaults — they prevent unlabeled convs from polluting agreement rate calculations while still running through the validator.

---

## Agreement Computation

The harness computes three agreement layers per conversation:

```json
{
  "outcome_match": true,
  "rule_match": false,
  "false_positive_rules": ["resolution"],
  "missed_rules": ["brand_voice"]
}
```

| Layer | Definition |
|---|---|
| `outcome_match` | `validator_outcome == expected_outcome` (ignoring `"uncertain"`) |
| `rule_match` | The set of failed rules matches `expected_failures` exactly |
| `false_positive_rules` | Rules the validator failed that the label says should pass |
| `missed_rules` | Rules the label says should fail that the validator did not catch |

A perfect run has `outcome_match: true` and `rule_match: true` for every labeled conv. `rule_match` can fail even when `outcome_match` passes — for example, if the validator correctly identifies a failure but fires on the wrong rule.

---

## Validation

The harness raises `LabelSchemaError` (with a clear message) on load if:
- `schema_version` is missing or not `1`
- `conv_id` does not match the sibling envelope
- `expected_outcome` is not one of the three allowed values
- `expected_failures` is missing any of the five required keys
- `confidence` is not one of the three allowed values
- `label_version` is not a positive integer
- `rationale` is an empty string or missing
- `revised_from` is non-null but `revision_notes` is null (revision audit trail is incomplete)
- `tags` contains any value not in the closed vocabulary above

---

## Schema Migration Policy

When the label schema advances to version 2, the harness will hard-error (`LabelSchemaError`) on any file where `schema_version != 2`. There are no implicit migrations — a version mismatch always fails loudly.

**Intended story for v1 → v2:**
- If the new version only adds optional fields: add the fields with defaults to every `labels.json`. Existing labels remain valid after the field is added.
- If the new version adds a required field to `expected_failures` (e.g. a new validator rule): every existing `labels.json` must gain that key. The safe default is `false`. Re-evaluate per conversation to confirm the default is correct.
- If the new version removes or renames a field: the migration guide in `replay_corpus/README.md` will document the path. The harness will not silently ignore unknown keys.
- Labels never need to be re-authored from scratch across a schema version bump — `rationale` and `expected_outcome` are conversation-level observations that survive structural changes to the schema.

**To migrate:** run `rforge replay migrate-labels --corpus PATH --to-version 2` (this command is a v2 addition; the implementation is the schema author's responsibility at that time).
