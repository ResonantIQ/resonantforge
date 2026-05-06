# Common Mistakes — ResonantForge Harness

A running log of footguns discovered in production. Each entry cites the incident and the fix so future engineers know what to avoid and why it was dangerous.

---

## ResonantForge Pipeline

### Corpus data loss on mid-run interrupt

**Pre-RFORGE-17:** Conv envelopes (both passed and skipped) were only persisted at Phase 5 manifest emission. The pipeline accumulated all `ConversationRecord` and `SkippedConversationRecord` objects in memory throughout Phase 3 (prose generation) and wrote them in a single batch at Phase 5. Interrupting before Phase 5 — a `KeyboardInterrupt`, OOM kill, or network error triggering a crash — caused total data loss of all generated conversations, including the paid Anthropic API calls that produced them.

**May 5 2026 incident:** An interrupted smoke run at conv 93/159 lost ~$3 in LLM API calls and 93 generated conversations with zero recoverable artifacts. The data was only ever in terminal scrollback.

**Fixed in RFORGE-17 (PR #XXX):** Per-conv atomic JSONL append in Phase 3. Each conversation is now written to `conversations.jsonl` or `skipped_conversations.jsonl` immediately on completion via `append_jsonl_line` (open with `'a'` + `flush()` + `os.fsync()`). Interrupts now leave recoverable JSONL files containing all conversations completed up to the interrupt point. Phase 5 becomes a summary/finalization step only (writes events, snapshots, planted_quality, and manifest).

---

## Replay Harness

### Don't run a full smoke to test validator changes

**Problem:** Running `rforge generate` to regenerate a corpus every time you want to test a validator or extractor change costs real money (Anthropic API calls per conversation), takes 5–30 minutes, and produces a different corpus each run (even with the same seed, model output is non-deterministic). Using a fresh smoke run as a validator test loop is slow, expensive, and gives unreliable signal — a change that looks like an improvement might just be different LLM output.

**Correct pattern:** Use the frozen-replay harness:
1. Extract envelopes once (`rforge replay extract-envelopes`): pays LLM cost one time, freezes `extracted_claims` and all lexicon inputs into `replay_corpus/{conv_id}/envelope.json`.
2. Iterate for free (`rforge replay run`): runs all 4 validators against frozen inputs, prints per-conv agreement vs. human labels in seconds, zero LLM calls.
3. The determinism fingerprint (`--fingerprint`) confirms your code change had the intended effect.

**Rule:** Never modify a validator or extractor and then run `rforge generate` to "see what happens". Always use `rforge replay run` for the iteration loop. Reserve smoke runs for generating new corpus data.

### Calling extract_accuracy_signals instead of run_kb_alignment_pipeline in replay

**Problem:** `extract_accuracy_signals` calls `extract_claims_llm` at Step 1 — an LLM call. Calling it in the replay path silently pays LLM cost and breaks determinism (claims may differ across runs).

**Correct pattern:** Call `run_kb_alignment_pipeline` directly, injecting the frozen `extracted_claims` from `envelope.validator_inputs.accuracy.extracted_claims`. The engine (`resonantforge/replay/engine.py`) does this. Never bypass it.

**Detection:** `RFORGE_REPLAY_MODE=1` patches `extract_claims_llm` to raise `ReplayModeError` immediately. The `test_no_network.py` suite verifies this on every CI run.

### Overwriting labels.json during re-extraction

**Problem:** `rforge replay extract-envelopes` rewrites `envelope.json` on every run (safe — envelopes are machine-generated). If it also overwrote `labels.json`, human-authored ground truth would be silently destroyed.

**Rule:** `_write_label_template` in `extractor.py` checks `label_path.exists()` and returns early if the file is already present. Never remove this guard.

### Free-form strings in labels.json tags

**Problem:** Tags like `"borderlne"` (typo of `"borderline"`) silently create phantom categories in aggregate tag breakdowns.

**Rule:** `tags` is a closed enum enforced at load time — `load_labels` raises `LabelSchemaError` on any unknown value. To add a new tag: update `VALID_TAGS` in `resonantforge/replay/schemas.py` and the vocabulary table in `harness/docs/replay-label-schema.md` in the same PR.
