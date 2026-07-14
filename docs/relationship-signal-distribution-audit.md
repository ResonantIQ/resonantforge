# ResonantForge — Relationship-Signal Planted Distribution Audit

**Corpus audited:** `saas` profile, `--dry-run --accounts 60 --months 8 --seed 42`
(60 accounts, 240 simulated days, generator v0.2.0)
**Signals covered:** `relationship_champion_at_risk` (engine Rule 4),
`relationship_single_threaded` (engine Rule 5)

> These two detectors track *specific customer contacts across many
> conversations*, so they could not be exercised until the corpus modeled
> customer-side contacts as first-class entities. This audit reports the planted
> positive / negative / decoy distribution for both, so the benchmark harness can
> compute precision/recall/F1 and a false-positive rate per detector.

The audit is fully reproducible in `--dry-run` (no LLM cost): contacts,
attribution, snapshot rollups, and labels are all deterministic from `--seed`.

---

## Section 0 — What was planted, and how the ground truth is guaranteed

Each account is assigned exactly one *relationship scenario* (positive, decoy, or
null). A scenario is only assigned to an account whose realized conversation
distribution can actually express it, so a planted positive genuinely fires and a
planted decoy genuinely does not. Labels are then computed from the account's
*actual* final-day contact rollup via pure Python mirrors of the engine's Rule 4
/ Rule 5 logic (`champion_at_risk_fires` / `single_threaded_fires` in
`resonantforge/layer1/contact_planner.py`). The label is therefore, by
construction, exactly what a correct detector must output — not an aspiration.

Two ground-truth labels are emitted per account (one per detector) to
`relationship_labels.jsonl`; the account's contacts go to `contacts.jsonl`; and
every `DaySnapshot` carries the pre-rolled trailing-window contact aggregates
(`distinct_active_contacts_60d`, `distinct_active_contacts_prior_180d`,
`total_contacts_all_time`, `champion_rollups`) so a scorer can evaluate the rules
without replaying events.

**Engine thresholds mirrored (verified against
`resonantiq/src/lib/intelligence/churn-detectors.ts`):**

| Detector | Fires when |
|---|---|
| Rule 4 champion-at-risk | account has ≥1 champion contact AND a champion (with known recency) is unseen ≥30d **OR** engages <1×/month (touches over trailing 3mo ÷ 3) |
| Rule 5 single-threaded | `totalContactsAllTime > 2` AND exactly **1** distinct contact active in the last 60d AND **≥3** distinct contacts active in the prior 60–240d window |

---

## Section 1 — Scenario distribution (per account, n=60)

| Scenario | Accounts | % | Plants for |
|---|---:|---:|---|
| `single_threaded_positive` | 10 | 16.7% | Rule 5 positive |
| `champion_silence_positive` | 8 | 13.3% | Rule 4 positive (silence branch) |
| `champion_lowfreq_positive` | 5 | 8.3% | Rule 4 positive (frequency branch) |
| `champion_active_decoy` | 7 | 11.7% | Rule 4 **decoy** (near-miss) |
| `single_threaded_decoy_narrowed` | 6 | 10.0% | Rule 5 **decoy** (narrowed to 2) |
| `single_threaded_decoy_guard` | 4 | 6.7% | Rule 5 **decoy** (small-account guard) |
| `null_multithreaded` | 14 | 23.3% | negative control (multi-threaded, healthy) |
| `sparse_single` | 6 | 10.0% | negative control (1-contact account) |

Both Rule 4 positive branches are represented, so the OR in the detector is
exercised on both sides.

---

## Section 2 — Per-detector label distribution

Every account contributes one label per detector (n=60 each).

### 2a. `relationship_champion_at_risk` (Rule 4)

| Label | Count | % |
|---|---:|---:|
| Positive (should fire) | 13 | 21.7% |
| Negative (should not fire) | 47 | 78.3% |
| &nbsp;&nbsp;— of which **decoy** (near-miss) | 7 | 11.7% |
| &nbsp;&nbsp;— of which plain null | 40 | 66.7% |

Positive : negative = **0.28 : 1** (positives are the minority — the correct
direction for a churn signal; most accounts should not trip it).
**20 / 60 accounts have a champion contact at all** (13 positives + 7 active
decoys), so the decoys are true FP tests: they *have* a champion, but the
champion is engaged, so a correct detector must stay silent.

### 2b. `relationship_single_threaded` (Rule 5)

| Label | Count | % |
|---|---:|---:|
| Positive (should fire) | 10 | 16.7% |
| Negative (should not fire) | 50 | 83.3% |
| &nbsp;&nbsp;— of which **decoy** (near-miss) | 10 | 16.7% |
| &nbsp;&nbsp;— of which plain null | 40 | 66.7% |

Positive : negative = **0.20 : 1**. The 10 decoys split into the two distinct
near-miss shapes below.

---

## Section 3 — Decoy inspection (false-positive traps)

Decoys mirror the RFORGE-73 paraphrase-trap discipline: cases engineered to
*look* like a fire while being genuine negatives, so the FP rate is measurable.
Representative examples from the audited corpus (as-of day 239):

### Rule 4 decoy — `champion_active_decoy` (acct_034)

> Champion last seen **2 days ago**, engagement **≈1.33/month** → clear of both
> the 30-day silence bound and the 1×/month frequency bound. **Must NOT fire.**

The account has a champion (so Rule 4 has a champion to evaluate), but the
champion is actively engaged — the single most common false positive a naive
detector would make. Frequency is planted comfortably above the boundary
(≥1.33, not exactly 1.0) so a one-touch ingestion drift can't flip the label.

### Rule 5 decoy A — `single_threaded_decoy_narrowed` (acct_024)

> `total=4, active_60d=2, prior_60_240d=3` → narrowed from 4 threads to **2**,
> not 1. **Must NOT fire.**

A genuine relationship contraction that stops one contact short of the trigger —
the hardest single-threading false positive.

### Rule 5 decoy B — `single_threaded_decoy_guard` (acct_030)

> `total=2, active_60d=1, prior_60_240d=2` → effectively single-threaded, but the
> account only ever had **2** contacts, so the engine's `totalContactsAllTime > 2`
> guard suppresses it. **Must NOT fire.**

Tests the small-account guard specifically: a detector that ignores the guard
would false-fire here.

---

## Section 4 — Positive inspection (gold examples)

### Rule 4 positive — silence branch — `champion_silence_positive` (acct_011)

> Champion last seen **238 days ago**, frequency **0.0/month** → silence fires.
> `expected_fire=true`.

### Rule 4 positive — frequency branch — `champion_lowfreq_positive` (acct_019)

> Champion last seen **1 day ago** (not silent) but only **0.33/month** → the
> frequency branch fires independently of silence. `expected_fire=true`.

### Rule 5 positive — `single_threaded_positive` (acct_001)

> `total=4, active_60d=1, prior_60_240d=3` → exactly one recent thread, three
> distinct in the prior window, guard satisfied. `expected_fire=true`. The final
> `DaySnapshot` rollup for acct_001 carries these same three values, so a scorer
> reading the pre-roll reaches the identical verdict.

---

## Section 5 — Integrity checks

All enforced by `tests/test_contact_modeling.py` (15 tests) and
`tests/test_properties.py::test_determinism`:

- **Determinism:** `contacts.jsonl`, `relationship_labels.jsonl`, and
  `snapshots.jsonl` are byte-identical across two same-seed runs
  (`--dry-run --smoke` stays reproducible).
- **Label ↔ evidence consistency:** every label's `expected_fire` equals the
  detector mirror applied to its own recorded `evidence` (0 mismatches across
  120 labels).
- **Snapshot ↔ label consistency:** the final `DaySnapshot` per account carries
  contact-rollup fields matching the single-threaded label's evidence.
- **Attribution completeness:** every emitted contact engages ≥1 conversation;
  every `CONVERSATION_STARTED` event is attributed to an existing contact;
  `total_contacts_all_time` equals the account's contact-row count.
- **Independent validator agreement:** the account-level relationship validator
  (`resonantforge/validators/relationship.py`) recomputes windowed contact
  features straight from raw attribution (not the pre-roll) and agrees with the
  planted ground truth on every account × detector.

---

## Section 6 — Consumption notes for the benchmark harness

- **Ground truth:** `relationship_labels.jsonl` — one record per (account,
  detector). Use `expected_fire` as the label; `is_decoy` marks the near-miss
  negatives for a separate FP-rate cut.
- **Contacts:** `contacts.jsonl` — first-class contact rows
  (`is_champion`, role, engagement span). Map to the engine's `contacts` table.
- **Pre-rolled features:** each `snapshots.jsonl` row carries the trailing-window
  aggregates; the last snapshot per account is the as-of-"now" state the labels
  are evaluated against (`as_of_day_index`).
- **Base rate:** positives are ~17–22% per detector by design (churn signals are
  the minority). For a balanced training cut, sample decoys and plain nulls
  separately using `is_decoy` and `scenario`.
- The `manifest.json` field `relationship_signal_distribution` carries the
  per-detector positive/negative/decoy counts and the scenario tally shown in
  Sections 1–2, so the distribution is auditable straight from the manifest.
