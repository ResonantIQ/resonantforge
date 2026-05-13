# Common Mistakes — ResonantForge

A running log of footguns discovered during development. Each entry describes the problem and the correct pattern so contributors know what to avoid and why it matters.

---

## ResonantForge Pipeline

### Corpus data loss on mid-run interrupt

**Background:** Previously, conversation envelopes (both passed and skipped) were only persisted at Phase 5 manifest emission. The pipeline accumulated all `ConversationRecord` and `SkippedConversationRecord` objects in memory throughout Phase 3 (prose generation) and wrote them in a single batch at Phase 5. Interrupting before Phase 5 — a `KeyboardInterrupt`, OOM kill, or network error triggering a crash — caused total data loss of all generated conversations, including the paid Anthropic API calls that produced them.

**Fixed:** Per-conv atomic JSONL append in Phase 3. Each conversation is now written to `conversations.jsonl` or `skipped_conversations.jsonl` immediately on completion via `append_jsonl_line` (open with `'a'` + `flush()` + `os.fsync()`). Interrupts now leave recoverable JSONL files containing all conversations completed up to the interrupt point. Phase 5 becomes a summary/finalization step only (writes events, snapshots, planted_quality, and manifest).

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

**Fix:** `_write_label_template` in `extractor.py` checks `label_path.exists()` and returns early if the file is already present. Never remove this guard. Re-running `extract-envelopes` is safe: envelopes are refreshed, existing `labels.json` files are left untouched.

### rforge generate silently overwrites frozen replay envelopes

**Symptom:** Running `rforge generate` from a directory whose CWD sibling is `replay_corpus/` silently overwrites frozen reference envelopes. Human-authored `labels.json` files are the only thing preserved (because `_write_label_template` guards them), but the machine-generated `envelope.json` files are destroyed and regenerated — making the replay harness non-deterministic and invalidating any validator iteration work done against those envelopes.

**Cause:** `rforge generate` defaulted `--replay-out` to `Path("replay_corpus")` relative to CWD, which could resolve to the directory containing the frozen reference corpus.

**Why it was dangerous:** The corpus is non-deterministic across LLM invocations even with the same seed. Re-generating envelopes means the `extracted_claims` in each envelope will differ from the human-labelled ground truth that was validated against the original claims. The result is a silent accuracy regression: `rforge replay run` agreement rates drift with no warning.

**Fixed:** `rforge generate` now detects existing envelopes at the resolved replay output path before running the pipeline. If `<replay-out>/<profile>/*/envelope.json` matches anything, the command prints a clear error and exits(1) with three safe alternatives:

```
Error: Replay output directory already contains N frozen envelope(s) for profile saas:
  /path/to/replay_corpus/saas

Overwriting these would silently destroy frozen reference envelopes.
Safe alternatives:
  --no-replay            — skip replay envelope writing entirely
  --replay-out <path>    — write envelopes to a different directory
  --force-replay-out     — override this guard and overwrite anyway
```

Use `--force-replay-out` only when intentionally regenerating a reference corpus after a profile or pipeline change that requires a full envelope refresh.

---

### Free-form strings in labels.json tags

**Problem:** Tags like `"borderlne"` (typo of `"borderline"`) silently create phantom categories in aggregate tag breakdowns.

**Rule:** `tags` is a closed enum enforced at load time — `load_labels` raises `LabelSchemaError` on any unknown value. To add a new tag: update `VALID_TAGS` in `resonantforge/replay/schemas.py` and the vocabulary table in `docs/replay-label-schema.md` in the same PR. If you believe a new tag is needed, open a GitHub issue first to discuss before adding it.
