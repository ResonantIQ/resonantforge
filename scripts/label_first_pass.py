#!/usr/bin/env python3
"""
First-pass LLM labeling for replay corpus.

Reads each envelope.json, calls Claude Haiku to produce a label, and writes
labels.json. Skips any conv already labeled (labeled_at != "FILL_IN").

Usage:
    python scripts/label_first_pass.py --corpus replay_corpus/saas [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

import anthropic


_LABELED_BY = "claude-haiku-4-5-first-pass"
_LABELED_AT = date.today().isoformat()

_ALLOWED_TAGS = {
    "high_confidence_fail",
    "high_confidence_pass",
    "edge_case",
    "borderline",
    "claim_extraction_stress",
    "multi_chunk",
    "deny_condition",
    "overgeneralization",
    "planted",
    "organic",
    "skipped_during_generation",
}

_PROMPT = """\
You are a quality reviewer for customer support conversations at a SaaS company.

Label this conversation for automated validator testing.

## Conversation

CUSTOMER:
{customer_prose}

AGENT:
{agent_prose}

## Agent's extracted factual claims:
{claims_text}

## Instructions

Return a JSON object — no prose, no markdown fence, just raw JSON.

{{
  "expected_outcome": "pass" | "fail" | "uncertain",
  "expected_failures": {{
    "accuracy": <bool>,
    "empathy": <bool>,
    "resolution": <bool>,
    "brand_voice": <bool>,
    "claim_extraction": <bool>
  }},
  "confidence": "high" | "medium" | "low",
  "tags": [<strings>],
  "rationale": "<1-3 sentences>"
}}

### expected_outcome
- pass: agent responded well on all dimensions
- fail: agent made at least one clear error
- uncertain: genuinely ambiguous — use sparingly

### expected_failures (only mark true if clearly violated)
- accuracy: agent stated something incorrect, unsupported, or overgeneralized as fact
- empathy: agent failed to acknowledge the customer's emotions, frustration, or situation
- resolution: agent gave no clear next steps, ownership, or directional path to resolution
- brand_voice: agent tone is significantly off (robotic/stiff or inappropriately casual)
- claim_extraction: ONLY true if extracted_claims is empty AND the prose clearly contains factual claims that should have been extracted

### confidence
- high: unambiguous verdict
- medium: mostly clear with some nuance
- low: genuinely borderline

### tags — use all that apply from this list only
- "organic" — always include (all these convs are organically generated)
- "high_confidence_pass" — clearly good response across all dimensions
- "high_confidence_fail" — clearly deficient response
- "edge_case" — unusual pattern (zero claims, single-turn, very short prose)
- "borderline" — near a decision boundary; small change might flip verdict
- "claim_extraction_stress" — very long agent prose or no claims extracted from non-trivial prose
- "overgeneralization" — agent dropped a caveat or overstated a policy
- "skipped_during_generation" — only if already present in the existing tags

### rationale
1–3 sentences. Describe what you observed in the prose — not what the validator should do.
Example: "Agent acknowledges frustration and commits to a specific action. Claims are grounded and qualified. Clear pass on all dimensions."
"""


def _format_claims(claims: list[dict]) -> str:
    if not claims:
        return "(none extracted)"
    lines = []
    for i, c in enumerate(claims[:20], 1):  # cap at 20 to stay within token budget
        alignment = c.get("kb_alignment", "?")
        text = c.get("claim_text", "")
        lines.append(f"  {i}. [{alignment}] {text}")
    if len(claims) > 20:
        lines.append(f"  ... ({len(claims) - 20} more claims omitted)")
    return "\n".join(lines)


def _label_one(client: anthropic.Anthropic, envelope: dict) -> dict:
    agent_prose = envelope.get("agent_prose", "")
    customer_prose = envelope.get("customer_prose", "")
    claims = (
        envelope.get("validator_inputs", {})
        .get("accuracy", {})
        .get("extracted_claims", [])
    )

    prompt = _PROMPT.format(
        agent_prose=agent_prose,
        customer_prose=customer_prose,
        claims_text=_format_claims(claims),
    )

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        messages=[{"role": "user", "content": prompt}],
    )
    raw = response.content[0].text.strip()
    # Strip markdown fence if model adds one despite instructions
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[1]
        raw = raw.rsplit("```", 1)[0].strip()
    return json.loads(raw)


def _apply_label(existing: dict, result: dict, existing_tags: list[str]) -> dict:
    # Preserve skipped_during_generation tag from the template
    tags = result.get("tags", ["organic"])
    if "skipped_during_generation" in existing_tags and "skipped_during_generation" not in tags:
        tags = ["skipped_during_generation"] + tags

    # Clamp to allowed vocabulary
    tags = [t for t in tags if t in _ALLOWED_TAGS]
    if "organic" not in tags:
        tags = ["organic"] + tags

    return {
        **existing,
        "labeled_at": _LABELED_AT,
        "labeled_by": _LABELED_BY,
        "expected_outcome": result["expected_outcome"],
        "expected_failures": {
            "accuracy": bool(result["expected_failures"].get("accuracy", False)),
            "empathy": bool(result["expected_failures"].get("empathy", False)),
            "resolution": bool(result["expected_failures"].get("resolution", False)),
            "brand_voice": bool(result["expected_failures"].get("brand_voice", False)),
            "claim_extraction": bool(result["expected_failures"].get("claim_extraction", False)),
        },
        "confidence": result["confidence"],
        "tags": tags,
        "rationale": result["rationale"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path, help="Path to replay corpus dir")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be done without writing")
    args = parser.parse_args()

    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    conv_dirs = sorted(d for d in args.corpus.iterdir() if d.is_dir())
    total = len(conv_dirs)
    counts = {"labeled": 0, "skipped": 0, "error": 0}

    for idx, conv_dir in enumerate(conv_dirs, 1):
        conv_id = conv_dir.name
        label_path = conv_dir / "labels.json"
        envelope_path = conv_dir / "envelope.json"

        if not label_path.exists() or not envelope_path.exists():
            print(f"[{idx}/{total}] {conv_id} — SKIP (missing files)")
            counts["skipped"] += 1
            continue

        existing_label = json.loads(label_path.read_text())
        if existing_label.get("labeled_at") != "FILL_IN":
            print(f"[{idx}/{total}] {conv_id} — SKIP (already labeled by {existing_label.get('labeled_by')})")
            counts["skipped"] += 1
            continue

        print(f"[{idx}/{total}] {conv_id} ...", end="", flush=True)

        if args.dry_run:
            print(" DRY RUN")
            continue

        try:
            envelope = json.loads(envelope_path.read_text())
            result = _label_one(client, envelope)
            updated = _apply_label(existing_label, result, existing_label.get("tags", []))
            label_path.write_text(json.dumps(updated, indent=2))
            failures = [k for k, v in updated["expected_failures"].items() if v]
            failure_str = f" failures=[{','.join(failures)}]" if failures else ""
            print(f" {updated['expected_outcome']} ({updated['confidence']}){failure_str}")
            counts["labeled"] += 1
        except Exception as exc:
            print(f" ERROR: {exc}")
            counts["error"] += 1

        time.sleep(0.05)  # gentle pacing

    print(f"\nDone. labeled={counts['labeled']} skipped={counts['skipped']} errors={counts['error']}")


if __name__ == "__main__":
    main()
