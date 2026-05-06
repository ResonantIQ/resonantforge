"""
Determinism test: same envelope + same code → byte-identical fingerprint across runs.

Two test classes:
  TestDeterminism       — same-process repeated runs (catches most non-determinism)
  TestDeterminismSubprocess — cross-subprocess runs with different PYTHONHASHSEED values
                             (catches hash-randomization-dependent ordering bugs)

Both are enforcement mechanisms for the determinism contract in
harness/docs/replay-envelope-schema.md § "Replay determinism guarantees".
"""
from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from resonantforge.replay import (
    compute_fingerprint,
    load_envelope,
    load_labels,
    replay_one,
)
from resonantforge.replay.fingerprint import compute_corpus_fingerprint
from tests.replay_fixtures.builder import build_fixtures


@pytest.fixture(scope="module")
def fixtures_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("det_fixtures")
    build_fixtures(d)
    return d


def _all_conv_ids() -> list[str]:
    from tests.replay_fixtures.builder import FIXTURES
    return list(FIXTURES.keys())


# ---------------------------------------------------------------------------
# Same-process determinism
# ---------------------------------------------------------------------------


class TestDeterminism:
    def test_single_conv_fingerprint_stable(self, fixtures_dir: Path) -> None:
        """replay_one on the same envelope twice → identical fingerprint."""
        conv_id = "conv_clean_pass"
        env = load_envelope(fixtures_dir / conv_id / "envelope.json")
        lbl = load_labels(fixtures_dir / conv_id / "labels.json", conv_id=conv_id)

        r1 = replay_one(env, lbl)
        r2 = replay_one(env, lbl)

        fp1 = compute_fingerprint(r1)
        fp2 = compute_fingerprint(r2)

        assert fp1 == fp2, (
            f"Fingerprint mismatch on repeated replay of {conv_id}: "
            f"{fp1!r} != {fp2!r}"
        )

    def test_corpus_fingerprint_stable(self, fixtures_dir: Path) -> None:
        """Full corpus replay twice → identical corpus fingerprint."""
        results_a = []
        results_b = []

        for conv_id in _all_conv_ids():
            env = load_envelope(fixtures_dir / conv_id / "envelope.json")
            lbl = load_labels(fixtures_dir / conv_id / "labels.json", conv_id=conv_id)
            results_a.append(replay_one(env, lbl))
            results_b.append(replay_one(env, lbl))

        fp_a = compute_corpus_fingerprint(results_a)
        fp_b = compute_corpus_fingerprint(results_b)

        assert fp_a == fp_b, (
            f"Corpus fingerprint mismatch across runs: {fp_a!r} != {fp_b!r}"
        )

    def test_different_conv_fingerprints_differ(self, fixtures_dir: Path) -> None:
        """Two different conversations produce different fingerprints."""
        def _result(conv_id: str):
            env = load_envelope(fixtures_dir / conv_id / "envelope.json")
            lbl = load_labels(fixtures_dir / conv_id / "labels.json", conv_id=conv_id)
            return replay_one(env, lbl)

        r_pass = _result("conv_clean_pass")
        r_fail = _result("conv_empathy_fail")

        fp_pass = compute_fingerprint(r_pass)
        fp_fail = compute_fingerprint(r_fail)

        assert fp_pass != fp_fail, (
            "Different conversations should produce different fingerprints"
        )

    def test_corpus_fingerprint_order_independent(self, fixtures_dir: Path) -> None:
        """corpus_fingerprint is stable regardless of input order (sorted by conv_id)."""
        all_ids = _all_conv_ids()
        results = []
        for conv_id in all_ids:
            env = load_envelope(fixtures_dir / conv_id / "envelope.json")
            lbl = load_labels(fixtures_dir / conv_id / "labels.json", conv_id=conv_id)
            results.append(replay_one(env, lbl))

        fp_forward = compute_corpus_fingerprint(results)
        fp_reverse = compute_corpus_fingerprint(list(reversed(results)))

        assert fp_forward == fp_reverse, (
            "Corpus fingerprint must be order-independent (sorted by conv_id internally)"
        )


# ---------------------------------------------------------------------------
# Cross-subprocess determinism (catches PYTHONHASHSEED-dependent ordering)
# ---------------------------------------------------------------------------

# Inline script run in each subprocess: builds fixtures, runs replay, prints fingerprint.
_SUBPROCESS_SCRIPT = textwrap.dedent("""\
    import sys, json
    from pathlib import Path

    fixtures_dir = Path(sys.argv[1])

    from tests.replay_fixtures.builder import FIXTURES
    from resonantforge.replay import load_envelope, load_labels, replay_one
    from resonantforge.replay.fingerprint import compute_corpus_fingerprint

    results = []
    for conv_id in sorted(FIXTURES.keys()):
        env = load_envelope(fixtures_dir / conv_id / "envelope.json")
        lbl = load_labels(fixtures_dir / conv_id / "labels.json", conv_id=conv_id)
        results.append(replay_one(env, lbl))

    print(compute_corpus_fingerprint(results))
""")


class TestDeterminismSubprocess:
    """
    Runs the replay engine in two separate Python subprocesses with different
    PYTHONHASHSEED values. Any hash-randomization-dependent dict or set iteration
    that feeds into the fingerprint will produce a mismatch.

    Uses PYTHONHASHSEED=0 (deterministic) and PYTHONHASHSEED=1 (alternative seed)
    to expose ordering bugs without requiring a full random seed sweep.
    """

    def _run_fingerprint(self, fixtures_dir: Path, hashseed: str) -> str:
        """Run the fingerprint script in a subprocess with a fixed PYTHONHASHSEED."""
        import os
        env = {**os.environ, "PYTHONHASHSEED": hashseed}
        result = subprocess.run(
            [sys.executable, "-c", _SUBPROCESS_SCRIPT, str(fixtures_dir)],
            capture_output=True,
            text=True,
            env=env,
            cwd=str(Path(__file__).parent.parent),  # harness/ directory
        )
        if result.returncode != 0:
            pytest.fail(
                f"Subprocess (PYTHONHASHSEED={hashseed}) failed:\n"
                f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
            )
        return result.stdout.strip()

    def test_fingerprint_stable_across_hash_seeds(self, fixtures_dir: Path) -> None:
        """Corpus fingerprint is identical regardless of PYTHONHASHSEED."""
        fp_seed0 = self._run_fingerprint(fixtures_dir, "0")
        fp_seed1 = self._run_fingerprint(fixtures_dir, "1")

        assert fp_seed0, "Subprocess with PYTHONHASHSEED=0 returned empty fingerprint"
        assert fp_seed1, "Subprocess with PYTHONHASHSEED=1 returned empty fingerprint"
        assert fp_seed0 == fp_seed1, (
            f"Fingerprint differs across PYTHONHASHSEED values — "
            f"a dict/set iteration in the fingerprint path is hash-order-dependent.\n"
            f"  PYTHONHASHSEED=0: {fp_seed0}\n"
            f"  PYTHONHASHSEED=1: {fp_seed1}"
        )

    def test_fingerprint_same_process_matches_subprocess(self, fixtures_dir: Path) -> None:
        """The in-process fingerprint matches the subprocess fingerprint."""
        results = []
        for conv_id in sorted(_all_conv_ids()):
            env = load_envelope(fixtures_dir / conv_id / "envelope.json")
            lbl = load_labels(fixtures_dir / conv_id / "labels.json", conv_id=conv_id)
            results.append(replay_one(env, lbl))

        in_process_fp = compute_corpus_fingerprint(results)
        subprocess_fp = self._run_fingerprint(fixtures_dir, "0")

        assert in_process_fp == subprocess_fp, (
            f"In-process fingerprint {in_process_fp!r} != "
            f"subprocess fingerprint {subprocess_fp!r}"
        )
