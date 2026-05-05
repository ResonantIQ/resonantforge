# Common Mistakes — ResonantForge Harness

A running log of footguns discovered in production. Each entry cites the incident and the fix so future engineers know what to avoid and why it was dangerous.

---

## ResonantForge Pipeline

### Corpus data loss on mid-run interrupt

**Pre-RFORGE-17:** Conv envelopes (both passed and skipped) were only persisted at Phase 5 manifest emission. The pipeline accumulated all `ConversationRecord` and `SkippedConversationRecord` objects in memory throughout Phase 3 (prose generation) and wrote them in a single batch at Phase 5. Interrupting before Phase 5 — a `KeyboardInterrupt`, OOM kill, or network error triggering a crash — caused total data loss of all generated conversations, including the paid Anthropic API calls that produced them.

**May 5 2026 incident:** An interrupted smoke run at conv 93/159 lost ~$3 in LLM API calls and 93 generated conversations with zero recoverable artifacts. The data was only ever in terminal scrollback.

**Fixed in RFORGE-17 (PR #XXX):** Per-conv atomic JSONL append in Phase 3. Each conversation is now written to `conversations.jsonl` or `skipped_conversations.jsonl` immediately on completion via `append_jsonl_line` (open with `'a'` + `flush()` + `os.fsync()`). Interrupts now leave recoverable JSONL files containing all conversations completed up to the interrupt point. Phase 5 becomes a summary/finalization step only (writes events, snapshots, planted_quality, and manifest).
