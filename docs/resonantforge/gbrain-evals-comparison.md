# gbrain-evals vs ResonantForge accuracy extractor — comparison

**Date:** 2026-05-04
**Purpose:** Engineering decision doc — continue with bespoke accuracy.py fix (Option A), full
refactor (Option B), or hybrid borrow (Option C)?

---

## 1. How does gbrain-evals score retrieval quality against expected behavior?

gbrain-evals is a **retrieval benchmark** (`P@5 / R@5` against a 240-page fictional corpus). It
is not an "agent prose quality" system. That distinction matters throughout this doc.

### LLM-as-judge

Yes — used selectively, not globally.

- **Where:** Cat 5 (source attribution/provenance), Cat 8 (skill behavior compliance), Cat 9
  (end-to-end workflows). Cat 1–4 and Cat 6–7 are entirely programmatic (no LLM judge).
- **Model:** `claude-haiku-4-5-20251001`. Hard-coded as `DEFAULT_MODEL` in
  `eval/runner/judge.ts:L97`.
- **Prompt structure:** The judge receives a pre-digested `JudgeEvidence` object — NOT raw
  tool output. This is called the "structured evidence contract" (judge.ts:L15–28). The
  judge sees: probe metadata, `final_answer_text`, `evidence_refs` (citation slugs only, not
  content), `tool_call_summary` (counts, not transcripts), `ground_truth_pages` (full text),
  and a `rubric` with per-criterion weights. Prompt-injection payloads from poison fixtures
  are explicitly excluded from the judge context.
- **Output:** Forced through a `score_answer` tool call (structured tool_use, not free text)
  scoring each rubric criterion 0–5. The judge's verdict label is overridden by a computed
  weighted mean if they disagree — the aggregation rule is canonical
  (`judge.ts:L258–265`, pass ≥ 3.5, partial 2.5–3.5, fail < 2.5).
- **Retry policy:** One retry on malformed tool_use response. Second failure → `judge_failed`
  sentinel (all scores 0) so the run completes rather than errors.

### Sealed qrels

Yes — central to the design. Called out explicitly in the README ("Day 9 sealed-qrels
enforcement").

- **Format:** `eval/data/gold/qrels.json` — query relevance judgments. Each query has
  a `relevant` list of page slugs and optional `grades` (0–3 relevance scores).
- **Construction:** Generated at corpus-build time from `_facts` metadata embedded in each
  page JSON. The `_facts` field is the canonical gold source; it never crosses the adapter
  boundary. `multi-adapter.ts` calls `sanitizePage()` to strip `_facts` before passing pages
  to adapters (`types.ts:L57–66`).
- **Enforcement:** The adapter interface is structurally split — adapters call `query()` on
  sanitized `PublicQuery` objects (gold field stripped). Scorers (P@K, R@K) receive full
  gold separately. This is the "adapter boundary" pattern.
- **Cat 5 gold:** `eval/data/gold/citations.json` — 100 claims with `expected_label`
  (supported / unsupported / over-generalized) and `expected_evidence` page slugs.

### Tolerance bands

Yes — explicitly designed for reproducibility.

- The README describes "N-run tolerance bands" as an anti-gaming measure alongside
  judge-version pinning and randomized query order.
- The scorecard format (`recorder.ts:L95`) includes `ScoredMetric.tolerance` and
  `ScoredMetric.per_run` (individual run scores stored, not just the mean).
- Adapters that are deterministic are documented as "byte-match" reproducible
  (README: "deterministic adapters byte-match"). LLM-involved runs use the tolerance band
  to accept runs where the median agrees.

### Disagreement sampling

Not explicitly named as "disagreement sampling." The N-run mechanism covers the same
intent: run N times, take median/mean, accept if within band. There is no explicit
sampling of "runs that disagree" for special handling — the fallback is `judge_failed`
(zeroed verdict), not a re-sample-on-disagreement loop.

---

## 2. What patterns would be liftable into ResonantForge's accuracy extractor?

ResonantForge `accuracy.py` currently does: LLM claim extraction (Step 1) → deterministic
KB alignment pipeline (Steps 2–8). The LLM call is already isolated to Step 1. The bugs
surface in the deterministic pipeline (Steps 3–7).

### Pattern A: Sealed qrels / planted ground truth at corpus-build time

**gbrain-evals:** `_facts` fields in corpus pages serve as ground truth. Built into the corpus
generator, consumed by scorers only, never exposed to the system under test. qrels.json
stores query → expected-page mappings authored or derived at corpus-build time.

**Applies to ResonantForge?** Partially. ResonantForge already has `kb_chunks_required`
(planted ground-truth chunk IDs passed to `run_kb_alignment_pipeline`). This is
structurally equivalent to gbrain-evals' qrels. The gap is that `kb_chunks_required` is
passed by the caller at evaluation time, not sealed at corpus-build time. A stricter sealed
approach would pre-compute expected chunk coverage per conversation fixture and store it in
test fixtures — preventing test authors from accidentally passing wrong `kb_chunks_required`
values.

**Refactor cost:** Hours (not days). The pipeline signature doesn't change; the test fixture
format does. Gains: removes a class of test-authoring bugs where `kb_chunks_required` is
inconsistent with the KB content.

**Assessment:** Nice-to-have for test hygiene. Not the source of the 9 current bugs.

### Pattern B: Judge-version pinning

**gbrain-evals:** `judge_prompt_version` is committed in
`eval/data/gold/personalization-rubric.json` and recorded in the scorecard `config_card`
(`recorder.ts:L73–80`). The model ID is hard-coded as `DEFAULT_MODEL = 'claude-haiku-4-5-20251001'`
(`judge.ts:L97`) rather than read from a mutable env var. The system prompt is versioned via
`systemPromptVersion` in `JudgeConfig`.

**Applies to ResonantForge?** Yes — directly. `accuracy.py` hard-codes
`model="claude-haiku-4-5-20251001"` in `extract_claims_llm` (line 83). This is already
pinned. The `CLAIM_EXTRACTION_PROMPT` is a module-level constant, not versioned or
committed alongside test gold. If the prompt changes, existing test expectations can
silently shift.

**Refactor cost:** Hours. Add a `CLAIM_EXTRACTION_PROMPT_VERSION` constant and record it in
`AccuracySignals.meta`. No pipeline change needed.

**Assessment:** Nice-to-have for reproducibility. Not the source of the 9 current bugs (the
model is already pinned; prompt drift is a future risk, not a current one).

### Pattern C: N-run tolerance bands

**gbrain-evals:** Runner stores per-run scores in `ScoredMetric.per_run` and accepts a run
if median is within `tolerance`. Implementation detail is in the scorecard recorder, not in
the scorer itself.

**Applies to ResonantForge?** Limited. ResonantForge's Step 1 (LLM claim extraction) is
non-deterministic, so N-run median could smooth over claim-set variance. However, the 9
bugs are in the deterministic Steps 2–8 — running the same deterministic logic N times
produces the same wrong answer every time. Tolerance bands don't help there.

**Refactor cost:** 1 day (adding an outer N-run loop + median aggregation).

**Assessment:** Not applicable to the current bug pattern. Would be valuable for Step 1 if
claim extraction variance were the problem — it is not.

### Pattern D: Adapter boundary (separate "what to compare" from "how to score")

**gbrain-evals:** The adapter boundary is structural. `sanitizePage()` / `sanitizeQuery()`
strip gold data before it reaches the system under test. Scorers (P@K, R@K) operate on
adapter output using the unsanitized gold separately. The judge receives a `JudgeEvidence`
object assembled by the runner — not raw adapter output. This prevents the judge from seeing
prompt-injection payloads.

**Applies to ResonantForge?** Partially. `accuracy.py` already separates claim extraction
(Step 1, LLM) from alignment scoring (Steps 2–8, deterministic). The bug cluster, however,
is within the deterministic pipeline: `_check_chunk_relevance`, `_check_constraint_type_applies`,
and the worst-case alignment aggregation logic. These don't have a sub-boundary between
"what chunks are candidates" and "how to score alignment."

A liftable refinement: split `_check_chunk_relevance` into (a) a pure relevance gate and (b)
a pure alignment classifier. Currently they're interleaved. This would make the logic easier
to test in isolation, which is where bugs live.

**Refactor cost:** Hours (decompose one function into two). Gains: testability, not
correctness — the logic itself still needs fixing.

**Assessment:** Nice-to-have for testability. Not a root cause fix for the 9 bugs.

---

## 3. What does gbrain-evals NOT solve that ResonantForge still needs to invent?

gbrain-evals scores **retrieval** — does the right page/document surface in top-K? The
evaluation objects are page slugs, not claims or policies. The judge evaluates whether an
agent answer is grounded in source pages, but always against a fictional personal knowledge
base (biographical pages, emails, calendar events).

ResonantForge's accuracy problem is structurally different:

| Dimension | gbrain-evals | ResonantForge |
|---|---|---|
| Domain | Personal knowledge retrieval | CX policy compliance (KB chunks) |
| Ground truth | Pre-generated `_facts` + qrels (structural) | Live KB content + planted chunk IDs per conversation |
| Claim types | "Jordan founded NovaMind" (biographical facts) | "You'll receive a refund within 7 days" (policy/procedural) |
| Constraint semantics | Not modeled | DENY_CONDITION chunks, constraint preservation, overgeneralization |
| Negation detection | Not modeled | Core: chunk has `not allowed`, claim asserts positivity |
| Multi-KB-chunk requirement | Not modeled | `kb_chunks_required` multi-chunk satisfaction |
| Conversation context | Not modeled | Customer turns drive `_check_constraint_type_applies` |
| Empathy / tone | Entirely out of scope | Core ResonantForge extractor (separate file) |
| Resolution quality | Entirely out of scope | Core ResonantForge extractor (separate file) |
| Brand voice compliance | Entirely out of scope | Core ResonantForge extractor (separate file) |

Specifically bespoke to ResonantForge that gbrain-evals has no analogue for:

1. **Policy constraint semantics** — DENY_CONDITION chunks, `ConstraintType`, `_check_constraint_type_applies`. gbrain-evals has no concept of a KB document that *prohibits* something.
2. **Overgeneralization detection** — detecting when the agent drops a qualifying constraint from a policy. gbrain-evals Cat 5 has "over-generalized" as a label but applies it to biographical factual drift, not policy scope narrowing.
3. **Synonym normalization for CX domain** — the `synonym_map` approach (refund/reimburse, cancel/terminate) is domain-specific to CX policy language.
4. **Conversation-context-aware scoring** — the deny-condition check reads customer turns to determine if a deny condition is triggered. gbrain-evals scoring is query-only.
5. **The other three extractors** — empathy, resolution, brand_voice have no gbrain-evals equivalent.

---

## 4. Bottom-line recommendation

**Recommendation: Option A — ship PR1 fix as planned, continue with bespoke extractor.**

### Reasoning

The 9 bugs are **implementation issues in the deterministic pipeline**, not approach issues. Looking at the specific failure modes documented (constraint detection false positives/negatives, overgeneralization mis-classification, deny-condition triggering on wrong context):

- The alignment pipeline logic in `_check_chunk_relevance` and `_check_constraint_type_applies` has brittle heuristics (20% term overlap gate, trigger phrase lists, number comparison for deny conditions) that produce wrong results on real CX prose. These are fixable point bugs — they don't require a different architecture.
- gbrain-evals' judge.ts pattern (structured tool_use, retry-on-malformed, verdict computed from weighted mean rather than trusted from model) is genuinely good. But ResonantForge already does the LLM-isolation correctly: Step 1 is the only LLM call, Steps 2–8 are deterministic. The judge.ts patterns apply to the LLM layer, not the deterministic layer where the bugs live.
- The sealed qrels pattern is already present (conceptually) in `kb_chunks_required`. Formalizing it into fixture files is a test hygiene improvement, not a correctness fix.
- Option B (full refactor to gbrain-evals patterns) would cost multiple days and solve a different problem — retrieval scoring reliability — not the CX-domain policy alignment bugs that are actually failing.
- Option C (hybrid borrow) is attractive on paper but the specific liftable patterns (judge-version pinning via prompt version constant, adapter boundary decomposition of `_check_chunk_relevance`) are hours of work each, not blocking PR1.

**What to do alongside PR1:** After PR1 ships, add `CLAIM_EXTRACTION_PROMPT_VERSION` constant (1 hour) and decompose `_check_chunk_relevance` into separate relevance-gate and alignment-classifier functions (2 hours). These are the two gbrain-evals patterns with genuine long-term value for ResonantForge's test surface. Do them on a follow-on PR, not as a blocker.

The bugs are in the heuristics. Fix the heuristics.
