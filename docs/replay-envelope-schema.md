# Replay Envelope Schema

**Version:** 1  
**File:** `{corpus}/{conv_id}/envelope.json`

An envelope is a self-contained snapshot of every input a validator needs to reproduce its verdict without calling any LLM or external service. The extraction step (see `rforge replay extract-envelopes`) runs LLM calls **once** to populate `validator_inputs.accuracy.extracted_claims`; all subsequent replays are free.

---

## Full Schema

```json
{
  "schema_version": 1,
  "conv_id": "conv_evt_abc123",
  "agent_prose": "...",
  "customer_prose": "...",
  "quality_plan": {
    "conversation_id": "...",
    "trigger_event_id": "...",
    "rubric_targets": {
      "empathy": "high" | "low" | null,
      "resolution": "strong" | "weak" | null,
      "brand_voice_against": "bv_baseline" | null,
      "brand_voice_target": "on_brand" | "off_brand" | null,
      "accuracy": {
        "status": "supported" | "contradicted" | "insufficient_information",
        "precision": "exact" | "overgeneralized" | "missing_constraint" | "conditional_applied" | "condition_missed"
      } | null
    },
    "kb_chunks_required": ["kb_chunk_refund_policy_v3", "..."],
    "knowledge_citations": {
      "should_cite": ["..."],
      "must_not_cite": ["..."]
    },
    "cat11_gate": "..." | null,
    "multi_chunk_required": false,
    "prose_generation_directives": "..."
  },
  "kb_chunks": [
    {
      "chunk_id": "kb_chunk_refund_policy_v3",
      "document_id": "...",
      "document_path": "policies/refund_policy.md",
      "chunk_text": "...",
      "constraint_type": "allow_condition" | "deny_condition" | "informational",
      "cat11_gate": "..." | null,
      "tone_variant": "..." | null,
      "domains": ["billing"],
      "intent_tags": ["refund_request"],
      "adversarial": false,
      "sanity_probe": false,
      "claims": {},
      "metadata": {}
    }
  ],
  "lexicons": {
    "acknowledgment_phrases": ["I understand", "I hear you", "..."],
    "emotion_lexicon": ["frustrated", "upset", "..."],
    "apology_lexicon": ["I'm sorry", "I apologize", "..."],
    "action_verb_lexicon": ["resolve", "escalate", "refund", "..."],
    "hedging_lexicon": ["perhaps", "might", "..."],
    "directive_lexicon": ["ensure", "verify", "complete", "..."],
    "warm_terms": ["delighted", "pleasure", "..."],
    "clinical_terms": ["utilize", "initiate", "..."],
    "contraction_patterns": ["\\bcan't\\b", "\\bwon't\\b", "..."],
    "resolution_patterns": ["I will", "I'll", "we can", "..."],
    "deflection_patterns": ["contact support", "check the FAQ", "..."],
    "next_steps_patterns": ["next step", "following up", "..."],
    "temporal_anchor_patterns": ["within 24 hours", "by end of day", "..."],
    "specific_actor_patterns": ["I will", "our team will", "..."],
    "ownership_patterns": ["I'll personally", "I own this", "..."],
    "issue_keywords": ["billing", "charge", "refund", "..."],
    "synonym_map": {"reimburse": "refund", "charge back": "refund"}
  },
  "brand_voice": {
    "variant_id": "bv_baseline",
    "feature_profiles": {
      "bv_baseline": {
        "sentence_length_range": [8, 22],
        "question_count_range": [0, 3],
        "hedging_range": [0, 2],
        "directive_range": [1, 6],
        "aligned_term_field": "warm_terms_count",
        "aligned_terms_min": 1,
        "formality_range": [0.55, 1.0]
      }
    }
  },
  "validator_inputs": {
    "accuracy": {
      "extracted_claims": [
        {
          "claim_text": "Refunds are processed within 5 business days.",
          "claim_span": [42, 90],
          "claim_type": "policy" | "procedural" | "factual",
          "normalized_subject": "refunds",
          "normalized_predicate": "are processed",
          "normalized_object": "within 5 business days",
          "alignment": null
        }
      ]
    }
  },
  "metadata": {
    "conv_id": "conv_evt_abc123",
    "source_corpus": "corpus/saas/",
    "pipeline_version": "0.2.0",
    "kb_version": "sha256:abcdef1234...",
    "generator_version": "0.2.0",
    "extraction_timestamp": "2026-05-05T22:00:00Z"
  }
}
```

---

## Field Notes

### `agent_prose` / `customer_prose`
Split from the full conversation using the `_split_prose_turns()` helper in `pipeline.py`. Each is a newline-joined concatenation of that speaker's turns, with the speaker prefix stripped. Both are required:
- `agent_prose` — consumed by all 4 extractors.
- `customer_prose` — consumed by resolution extractor (issue keyword overlap) and accuracy extractor Step 6 (deny-condition context check against customer-stated quantities/triggers).

### `quality_plan`
A snapshot of the `QualityPlan` that drove generation. The full object is frozen because the rule engine needs `rubric_targets` (all 4 dimension targets) and the accuracy extractor needs `kb_chunks_required` (the ground-truth chunk IDs that define the candidate pool for Steps 2–8). `prose_generation_directives` is included for human reference during labeling; it is not consumed by validators.

### `kb_chunks`
**Only the chunks referenced by `kb_chunks_required`** need to be present for replay correctness. The extract-envelopes command may include additional context chunks, but the accuracy KB alignment pipeline only evaluates chunks in `kb_chunks_required`. Including extras is safe; omitting required ones will produce wrong verdicts.

### `lexicons`
All profile lexicons needed by the three fully-deterministic extractors, frozen at extraction time. This decouples replay from profile evolution — you replay against the exact lexicons that were active when the envelope was generated. If the profile's lexicons change, re-run `extract-envelopes` to refresh.

Lexicon sources by extractor:
- **Empathy**: `acknowledgment_phrases`, `emotion_lexicon`, `apology_lexicon`, `action_verb_lexicon`
- **Resolution**: `resolution_patterns`, `deflection_patterns`, `next_steps_patterns`, `temporal_anchor_patterns`, `specific_actor_patterns`, `ownership_patterns`, `issue_keywords`
- **Brand Voice**: `hedging_lexicon`, `directive_lexicon`, `warm_terms`, `clinical_terms`, `contraction_patterns`
- **Accuracy (Steps 2–8)**: `synonym_map` only — all other accuracy logic is pure text comparison

### `brand_voice`
`variant_id` is taken from `quality_plan.rubric_targets.brand_voice_against` (defaults to `"bv_baseline"`). `feature_profiles` is the full calibrated dict from `profile.brand_voice_feature_profiles()`, frozen at extraction time.

### `validator_inputs.accuracy.extracted_claims`
**This is the only LLM-derived artifact in the envelope.** It is the output of `extract_claims_llm()` (Step 1 of the accuracy pipeline). Steps 2–8 (`run_kb_alignment_pipeline()`) consume these claims and are fully deterministic. During replay, `extracted_claims` is injected directly into Steps 2–8, bypassing the LLM call entirely.

The `alignment` field on each Claim is `null` at extraction time; it is populated in-place during Steps 3–4 of KB alignment. The harness replays from the null state.

### `metadata`
- `source_corpus` — relative path to the smoke output directory from which this envelope was extracted. Used for provenance tracing.
- `pipeline_version` and `generator_version` — from `manifest.generator_version` at extraction time.
- `kb_version` — from `manifest.kb_version` (SHA-256 of sorted `chunk_id + chunk_text` pairs).
- `extraction_timestamp` — ISO-8601 UTC timestamp when `extract-envelopes` ran.

---

## What is NOT in the envelope

| Omitted field | Reason |
|---|---|
| `normalized` (prose normalization) | No pre-computed normalization step exists in the code. `_normalize_text()` runs inline per-claim during Steps 2–8. |
| Full conversation prose | Derivable from `agent_prose` + `customer_prose`; storing separately wastes space and adds a sync risk. |
| Soft-judge output | Diagnostic only, non-gating. Excluded from replay scope. |
| All 4 signal objects | These are extractor *outputs*, not inputs. The replay engine recomputes them from the frozen inputs above. |
| `ConversationRecord` fields (agent_id, surface_channel, etc.) | Not consumed by any validator. Referenced from the source JSONL if needed for labeling context. |

---

## Replay determinism guarantees

Given an unchanged envelope, a replay run MUST produce byte-identical verdicts across invocations. Sources of non-determinism that the replay engine actively suppresses:

1. **No LLM calls** — `extracted_claims` is injected; `anthropic_client` is `None` in replay mode.
2. **No `datetime.now()` / `time.time()`** — timestamps come from `metadata.extraction_timestamp` (frozen).
3. **Stable sort** — `_normalize_text()` synonym map iteration uses `sorted()` (longest-first); `run_kb_alignment_pipeline()` iterates `candidate_chunks` in JSONL load order (deterministic given a frozen envelope).
4. **No RNG** — none of the 4 extractors use randomness.

If any of these invariants are violated by a code change, the determinism test (`test_determinism`) will catch it as a fingerprint mismatch.
