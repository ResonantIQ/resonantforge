# ResonantForge v0.2.0 — Extraction Inventory

**Jira:** [RFORGE-64](https://resonantiq.atlassian.net/browse/RFORGE-64)
**Date:** 2026-05-13
**Branch audited:** `release/v0.2.0-inventory`
**Harness root:** `harness/`
**Package version:** `0.2.0` (confirmed in `harness/pyproject.toml`)

---

## Section 1: Files That Move to Public

These files and directories under `harness/` should appear in the public `resonantiq/resonantforge` repo. Generated data directories (`corpus-live-*/`, `corpus-smoke-*/`, `corpus/`), evaluation artefacts (`eval-*.json`, `eval-*.log`, `eval-*.jsonl`), build caches, and `.DS_Store` are excluded.

| Path | File count | Notes |
|---|---|---|
| `harness/resonantforge/` | 53 source files (excl. `__pycache__`) | Full package source |
| `harness/tests/` | 40 files (excl. `__pycache__`) | Full test suite, including fixtures and replay fixtures |
| `harness/replay_corpus/` | 482 files — saas smoke corpus only | Frozen reference corpus — ground-truth envelopes and labels. full-1, full-2, full-3, smoke-5, smoke-10, smoke-13 stay in the internal harness repo. |
| `harness/replay_corpus/README.md` | 1 | Exists; public-safe (no internal refs) |
| `harness/resonantforge/replay/README.md` | 1 | Exists; public-safe (no internal refs) |
| `harness/pyproject.toml` | 1 | Build config; all deps are public packages |
| `harness/README.md` | 1 | Will be replaced by WS2 public README |
| `harness/COMMON_MISTAKES.md` | 1 | Will be replaced by `COMMON_MISTAKES.public.md` (see Section 3) |
| `harness/docs/replay-envelope-schema.md` | 1 | Public-safe; no internal refs |
| `harness/docs/replay-label-schema.md` | 1 | Public-safe; no internal refs |
| `harness/.gitignore` | 1 | Standard Python gitignore; safe to include |

**Total files moving to public (source + tests + corpus):** ~576 files (53 source + 40 tests + 482 saas corpus + 1 corpus README).

---

## Section 2: Files That Stay Internal

| Path | Reason |
|---|---|
| `harness/replay_corpus/full-1/` | Full evaluation corpus. Large (1,708 files). Ship saas smoke corpus only for v0.2.0 public release. |
| `harness/replay_corpus/full-2/` | Full evaluation corpus. Large (1,098 files). Ship saas smoke corpus only for v0.2.0 public release. |
| `harness/replay_corpus/full-3/` | Full evaluation corpus. Large (1,630 files). Ship saas smoke corpus only for v0.2.0 public release. |
| `harness/replay_corpus/smoke-5/` | Full evaluation corpus. Large (292 files). Ship saas smoke corpus only for v0.2.0 public release. |
| `harness/replay_corpus/smoke-10/` | Full evaluation corpus. Large (1,618 files). Ship saas smoke corpus only for v0.2.0 public release. |
| `harness/replay_corpus/smoke-13/` | Full evaluation corpus. Large (214 files). Ship saas smoke corpus only for v0.2.0 public release. |
| `harness/docs/resonantforge/accuracy-extractor-exception-trace.md` | Internal engineering debug trace. References internal conversation IDs (`conv_evt_00036`) and references `gbrain-evals` (a separate internal evaluation framework). No public value as-is. |
| `harness/docs/resonantforge/gbrain-evals-comparison.md` | Internal engineering decision document comparing `gbrain-evals` to the ResonantForge accuracy extractor. References internal file paths (`eval/runner/judge.ts:L97`, `judge.ts:L15–28`), internal tooling, and internal architectural decision context. Entirely internal. |
| `harness/scripts/label_first_pass.py` | First-pass LLM labeling utility script. Uses `anthropic` SDK directly against the replay corpus. This is an internal operator tool, not part of the public `rforge` CLI. Does not reference internal infrastructure but is scoped to internal workflows. Flag for TJ decision: could be included as a community utility in a future release. |
| `harness/corpus/` | Generated corpus data (conversations, run log). Not source. |
| `harness/corpus-live-smoke-4/` | Live-generated smoke corpus. Generated data, not source. |
| `harness/corpus-live-smoke-10/` | Live-generated smoke corpus. Generated data, not source. |
| `harness/corpus-smoke-7/` | Generated smoke corpus. Generated data, not source. |
| `harness/corpus-smoke-8/` | Generated smoke corpus. Generated data, not source. |
| `harness/.venv/` | Local Python virtual environment. Never committed to any repo. |
| `harness/__pycache__/` | Python bytecode cache. Never committed to any repo. |
| `harness/.pytest_cache/` | Pytest cache. Never committed to any repo. |
| `harness/.DS_Store` | macOS metadata file. Never committed to any repo. |

**Notes on `harness/docs/resonantforge/`:** This directory contains exactly two files, both internal-only. The directory itself should not exist in the public repo. The two files are engineering decision/debug docs tied to internal tooling (`gbrain-evals`) and internal conversation corpora. Neither should be published.

---

## Section 3: Files Needing Rename or Content Edits Before Extraction

| Current path | What needs changing | Who handles it |
|---|---|---|
| `harness/COMMON_MISTAKES.md` | The file itself is public-safe (no internal URLs, no internal service references checked). However, the task spec calls for a `COMMON_MISTAKES.public.md` replacement that may omit or reframe any internal-org context. The current file references `PR #XXX` placeholders (line in the corpus data loss section) which should be resolved or removed. Review for any Resonant IQ–specific organisational context before publishing. | WS1 handles `COMMON_MISTAKES.md` |
| `harness/README.md` | Will be fully replaced by WS2 public README. The current README is an internal developer README referencing the monorepo structure, internal dev commands, and internal context. Do not publish as-is. | WS2 handles public README |
| `harness/resonantforge/pipeline.py` | Hardcoded model string `claude-haiku-4-5-20251001` at line 283 (and documented at line 258). See Check 5 in Section 5 for recommendation. This is a known hygiene item requiring a TJ decision before extraction. | WS1 (source change) |
| `harness/resonantforge/pipeline.py` | Comment at line 264 references `docs/resonantforge/cache-diagnosis.md` — an internal doc that does not exist in the harness directory and is not being published. The comment should be rewritten to remove the internal path reference or replaced with inline content. | WS1 |
| `harness/resonantforge/validators/invariant_checker.py` | Module-level docstring at line 9 references `docs/resonantforge/saas-invariants.md` — an internal doc not in the harness directory. Rewrite to remove or inline the reference. | WS1 |
| `harness/resonantforge/replay/schemas.py` | Lines 5–6 and lines 118, 175 reference `harness/docs/replay-envelope-schema.md` and `harness/docs/replay-label-schema.md` using the `harness/` prefix. Once extracted to the public repo, these files live at `docs/replay-envelope-schema.md` (no `harness/` prefix). Update all four references to drop the `harness/` prefix. | WS1 |

---

## Section 4: Cross-Boundary Imports

**Command run:**
```bash
grep -r "^from|^import" harness/resonantforge/ --include="*.py" | \
  grep -v "resonantforge|pydantic|anthropic|click|rich|json|os|sys|pathlib| \
           typing|dataclasses|datetime|re|random|hashlib|itertools|collections| \
           functools|enum|abc|io|time|math|copy|uuid|string|textwrap|logging| \
           warnings|contextlib|inspect|importlib|unittest|pytest|__future__"
```

**Result: No hits.**

All imports in `harness/resonantforge/` are confined to:
- The `resonantforge` package itself
- Standard library modules (filtered above)
- Declared dependencies: `pydantic`, `anthropic`, `click`, `rich`

There are no cross-boundary imports into the parent monorepo (Next.js app, Supabase client, etc.).

---

## Section 5: Coupling Audit

### Check 1 — Internal URLs

**Command:**
```bash
grep -rn "resonantiq\.app|supabase|\.vercel\.|sentry|inngest|voyage-ai|voyageai" harness/resonantforge/
```

**Result: PASS** — zero hits in `harness/resonantforge/`.

**Tests separately:** Zero hits in `harness/tests/` either. Both directories are clean.

---

### Check 2 — Unauthorized HTTP Libraries

**Command:**
```bash
grep -rn "^import requests|^from requests|^import httpx|^from httpx|^import urllib|^import aiohttp|^from aiohttp|^import socket" harness/resonantforge/
```

**Result: PASS** — zero hits. The only network path is the Anthropic SDK (`import anthropic`), which is the expected and sole approved network dependency.

---

### Check 3 — Telemetry/Analytics SDKs

**Command:**
```bash
grep -rn "posthog|mixpanel|segment|amplitude|datadog|newrelic" harness/resonantforge/
```

**Result: PASS** — zero hits.

---

### Check 4 — Hardcoded Non-Anthropic URLs

**Command:**
```bash
grep -rn "https://" harness/resonantforge/ | grep -v "api\.anthropic\.com|#|docstring|comment"
```

**Result: PASS** — zero hits. No hardcoded URLs found in `harness/resonantforge/`.

---

### Check 5 — Hardcoded Model Name

**Status: FLAGGED — TJ decision required.**

Confirmed location:
- `harness/resonantforge/pipeline.py`, line 283: `model="claude-haiku-4-5-20251001"`
- `harness/resonantforge/pipeline.py`, line 258: inline comment documenting the model choice

This is a known hygiene item. The model string is hardcoded in the `pipeline.py` scoring function.

**Option A — `RFORGE_MODEL` env var with `claude-haiku-4-5-20251001` as default.**
Add `model = os.environ.get("RFORGE_MODEL", "claude-haiku-4-5-20251001")` before the API call. Users can override the model without touching source. When `claude-haiku-4-5-20251001` is retired, users set the env var; the default is updated in a patch release.

**Option B — Document in README that users may need to update the model string when models deprecate.**
No code change. Lower maintenance burden. Acceptable for a corpus-generation tool where reproducibility (not portability) is the primary concern.

**Recommendation: Option A.** The change is four lines of code and makes the tool usable without a fork when the model is deprecated. For a publicly released tool, hardcoded model strings are a friction point that `RFORGE_MODEL` eliminates at negligible cost. Option B is acceptable if the preference is strict reproducibility (same model string = same behavior guarantee), but the public README will need a prominent callout about the deprecation risk.

---

### Check 6 — Internal-Doc References in Source Comments

**Command:**
```bash
grep -rn "docs/resonantforge/|docs/replay|harness/docs|COMMON_MISTAKES" harness/resonantforge/
```

**Hits found:**

| File | Line | Matched string | Recommendation |
|---|---|---|---|
| `harness/resonantforge/pipeline.py` | 264 | `docs/resonantforge/cache-diagnosis.md` | **(a) Remove the reference.** `docs/resonantforge/` is an internal-only directory not being extracted. Rewrite the comment to include the diagnostic information inline, or simply drop the reference. The doc explains a cache-diagnosis finding; if the content is essential, inline the key point. |
| `harness/resonantforge/validators/invariant_checker.py` | 9 | `docs/resonantforge/saas-invariants.md` | **(a) Remove the reference.** `docs/resonantforge/saas-invariants.md` is not being extracted. Rewrite the module docstring to either stand alone or reference the public `docs/replay-label-schema.md` if the design rationale is captured there. |
| `harness/resonantforge/replay/schemas.py` | 5 | `harness/docs/replay-envelope-schema.md` | **(c) The referenced doc is public-safe and is being included.** Update the path reference to drop the `harness/` prefix (becomes `docs/replay-envelope-schema.md`). |
| `harness/resonantforge/replay/schemas.py` | 6 | `harness/docs/replay-label-schema.md` | **(c) The referenced doc is public-safe and is being included.** Update the path reference to drop the `harness/` prefix (becomes `docs/replay-label-schema.md`). |
| `harness/resonantforge/replay/schemas.py` | 118 | `harness/docs/replay-envelope-schema.md` | **(c) Same as lines 5–6.** Drop the `harness/` prefix. |
| `harness/resonantforge/replay/schemas.py` | 175 | `harness/docs/replay-label-schema.md` | **(c) Same as lines 5–6.** Drop the `harness/` prefix. |

**Summary:** 2 references to internal-only docs (pipeline.py and invariant_checker.py) require removal or inline replacement. 4 references in replay/schemas.py are to public-safe docs and only need the `harness/` path prefix stripped.

---

## Section 6: Git filter-repo Command

**Pre-flight verification:**

**1. Does `harness/` exist as a direct subdirectory of the repo root?**

```
$ ls /Users/tjbaker/Projects/resonantiq/
harness/   src/   supabase/   tests/   ...
```

Confirmed: `harness/` is a direct subdirectory of the monorepo root. The `--subdirectory-filter harness/` flag will work correctly.

**2. Does `git log --all --oneline -- harness/ | head -5` return commits?**

```
7f50387b feat(corpus): spec-driven off-brand directive with multi-variant support (RFORGE-58) (#666)
eed35552 feat(corpus): spec-driven off-brand directive with multi-variant support (RFORGE-58)
4f6dd083 fix(rforge): exclude negative-target convs from prose_fact_rate gate (RFORGE-48) (#649)
d28df8af fix(rforge): exclude negative-target convs from prose_fact_rate gate (RFORGE-48)
fa8a8ede feat(rforge): negative-quality labels and negative_fraction control (RFORGE-47) (#647)
```

Confirmed: 149 commits touch `harness/`. History is present and will be rewritten by `filter-repo`.

**3. Edge cases:**

- The `harness/` path has been stable throughout the project's git history — no renames or relocations detected. `filter-repo` should produce a clean rewrite.
- The `replay_corpus/` subdirectories contain 7,043 files. The filter-repo run will process these. Expect the operation to take several minutes on the full clone.
- `corpus-live-*/` and `corpus-smoke-*/` directories outside `harness/` (at the repo root) are not under `harness/` and will not appear in the extracted repo. Only the `harness/corpus-live-smoke-4/` and `harness/corpus-live-smoke-10/`, `harness/corpus-smoke-7/`, `harness/corpus-smoke-8/` directories (which are inside `harness/`) would be rewritten. **Add a `.gitignore` or clean these before extracting** if they should not appear in the public repo history — or accept that they will appear in history but not in HEAD if they are deleted before the tag.

**Why filter-time exclusion, not post-filter `git rm`:**

Stripping large directories at filter time removes them from every commit in history, so a fresh `git clone` of the public repo never downloads those files. A post-filter `git rm + commit` is simpler to write, but the data remains in all prior commits — every contributor who clones the repo must download gigabytes of evaluation corpora that are irrelevant to them. For a public OSS repo with large generated data, filter-time exclusion is the correct choice.

**Tradeoff summary:**

| Approach | History size | Clone size | Complexity |
|---|---|---|---|
| Filter-time exclusion (this approach) | Clean — files absent from all commits | Minimal | Two `filter-repo` invocations (recommended) |
| Post-filter `git rm` | Full corpora present in all prior commits | Large — every `git clone` downloads the data | Simpler, but wrong for public OSS |

**Verified command sequence:**

```bash
# Step 1: Clone the monorepo to a scratch location
git clone /Users/tjbaker/Projects/resonantiq /tmp/resonantforge-extract
cd /tmp/resonantforge-extract

# Step 2: Make harness/ the repo root (rewrites history)
git filter-repo --subdirectory-filter harness/

# Step 3: Strip the large corpora from history entirely
# (--force required because filter-repo already ran once)
git filter-repo --force \
  --invert-paths \
  --path replay_corpus/full-1/ \
  --path replay_corpus/full-2/ \
  --path replay_corpus/full-3/ \
  --path replay_corpus/smoke-5/ \
  --path replay_corpus/smoke-10/ \
  --path replay_corpus/smoke-13/

# Step 4: Verify
git log --oneline | head -5
du -sh replay_corpus/   # should show only saas/ content

# Step 5: Add public remote and push
# git remote add origin git@github.com:resonantiq/resonantforge.git
# git push -u origin main
```

---

## Section 7: Risks and Unknowns

**1. Coupling audit: All checks PASS.** No internal URLs, no unauthorized HTTP libraries, no telemetry SDKs, no non-Anthropic hardcoded URLs. The two issues flagged (Check 5, Check 6) are documentation hygiene items, not blocking coupling failures.

**2. Check 5 — RFORGE_MODEL hardcoded model string (TJ decision required before extraction).**
`claude-haiku-4-5-20251001` is hardcoded at `harness/resonantforge/pipeline.py:283`. This will become a user pain point when the model is deprecated. Recommend Option A (env var with default). This is a code change required before the public release — it cannot be deferred to WS2.

**3. Check 6 — 2 internal-doc references in source comments require cleanup before extraction.**
- `harness/resonantforge/pipeline.py:264` — references `docs/resonantforge/cache-diagnosis.md` (internal, not published). Remove or inline.
- `harness/resonantforge/validators/invariant_checker.py:9` — references `docs/resonantforge/saas-invariants.md` (internal, not published). Remove or rewrite module docstring.
- 4 path prefix fixes in `harness/resonantforge/replay/schemas.py` (drop `harness/` prefix from doc links).

**4. `harness/scripts/label_first_pass.py` — uncertain scope.**
This script is an internal operator utility for first-pass LLM labeling of the replay corpus. It does not reference internal infrastructure and is functionally self-contained. It is currently excluded from the "moves to public" list because it is not part of the `rforge` CLI. Flag for TJ: should this be included as a community utility or excluded as an internal operator tool?

**5. Internal corpus directories in git history.**
`harness/corpus-live-smoke-4/`, `harness/corpus-live-smoke-10/`, `harness/corpus-smoke-7/`, `harness/corpus-smoke-8/` exist inside `harness/` and will be rewritten into the public repo's history by `filter-repo`. These contain generated conversation data. Decide whether to: (a) strip them from history using additional `filter-repo --path-glob` exclusions before publishing, or (b) accept them in history since they are generated (not sensitive) data.

**6. `harness/docs/resonantforge/gbrain-evals-comparison.md` references an internal product name.**
The file references "gbrain-evals" — an internal evaluation framework. This file is correctly excluded from the public repo (Section 2), but confirm that no references to `gbrain` leaked into any source files. Audit confirmed: zero `gbrain` references in `harness/resonantforge/` or the public docs.

**7. No credentials detected in git history.**
No `.env` files were ever committed under `harness/`. The 149-commit history for `harness/` does not contain any flagged credential patterns based on file name inspection. A deeper `git log -p` scan for API key patterns is recommended as a final pre-push check before publishing (outside the scope of this inventory).

**8. `replay_corpus/` — decision made: ship saas smoke corpus only.**
Only `replay_corpus/saas/` (482 files) ships in the public repo. `full-1`, `full-2`, `full-3`, `smoke-5`, `smoke-10`, `smoke-13` are excluded at filter time (see Section 6) so they never appear in the public clone's history. This keeps the public repo lightweight and the saas smoke corpus is sufficient as a public reference. Resolved — no further action required.

**9. `harness/COMMON_MISTAKES.md` contains a `PR #XXX` placeholder.**
The corpus data loss section references `PR #XXX` (unfilled). This should be resolved before publishing — either filled in with the actual PR number or rewritten to remove the reference.
