"""
Regression test: off-brand variant selection must be stable across processes.

`_pick_off_brand_variant_id` previously indexed with the builtin `hash(conv_id)`,
which Python salts per process via PYTHONHASHSEED. That made the chosen off-brand
variant — and therefore `planted_quality.jsonl` — differ across otherwise
identical `--dry-run --smoke` runs, breaking the bit-identical determinism
guarantee. It now uses a stable blake2b digest. This test pins that: the picked
variant for a fixed conv_id is identical under different PYTHONHASHSEED values.
"""
from __future__ import annotations

import os
import subprocess
import sys

from resonantforge.layer1.quality_plan_injector import _pick_off_brand_variant_id

_VARIANTS = {"aggressive": 1, "clinical_detached": 1, "robotic": 1, "verbose": 1}

# Inline script: print the picked variant for a set of conv_ids. Run under two
# different PYTHONHASHSEED values; the output must be byte-identical.
_SCRIPT = (
    "from resonantforge.layer1.quality_plan_injector import _pick_off_brand_variant_id as p;"
    "v={'aggressive':1,'clinical_detached':1,'robotic':1,'verbose':1};"
    "print(','.join(p(f'conv_evt_{i:05d}', v) for i in range(50)))"
)


def _run(hashseed: str) -> str:
    env = {**os.environ, "PYTHONHASHSEED": hashseed}
    out = subprocess.run(
        [sys.executable, "-c", _SCRIPT],
        capture_output=True, text=True, env=env, check=True,
    )
    return out.stdout.strip()


def test_offbrand_pick_is_hashseed_invariant():
    a = _run("0")
    b = _run("1")
    c = _run("12345")
    assert a == b == c, "off-brand variant selection varies with PYTHONHASHSEED"
    # And it must be non-trivial (actually distributes across variants).
    assert len(set(a.split(","))) > 1


def test_offbrand_pick_matches_stable_digest():
    # Locks the algorithm to the stable digest, so a revert to builtin hash() is caught.
    import hashlib

    keys = sorted(_VARIANTS.keys())
    for i in range(20):
        conv_id = f"conv_evt_{i:05d}"
        digest = hashlib.blake2b(conv_id.encode("utf-8"), digest_size=8).digest()
        expected = keys[int.from_bytes(digest, "big") % len(keys)]
        assert _pick_off_brand_variant_id(conv_id, _VARIANTS) == expected
