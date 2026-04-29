# confabra/validators/disagreement_ledger.py
"""
Disagreement ledger — non-gating LLM soft-judge for corpus quality diagnostics.
Records cases where human perception would differ from the deterministic validator.
Per Section 5.1 disagreement ledger specification.
"""
from __future__ import annotations
import json
from anthropic import Anthropic
from confabra.schemas import DisagreementRecord

# Soft judge prompt — locked-down, no interpretation
SOFT_JUDGE_PROMPT = """Assess the following agent conversation for the specified quality dimension.
Report the OBSERVABLE FEATURES only — do not interpret intent.
Return JSON only:
{
  "perceived_level": "low" | "medium" | "high",
  "reasoning": "brief observable description",
  "disagreement_class": "clear" | "borderline_case" | "ambiguous_phrasing" | "context_dependent"
}
Do not explain further. Return only valid JSON."""


def run_soft_judge(
    conversation_id: str,
    agent_prose: str,
    dimension: str,
    validator_verdict: str,
    anthropic_client: Anthropic | None = None,
) -> DisagreementRecord | None:
    """
    Run the soft-judge LLM call and return a DisagreementRecord if disagreement is detected.

    Returns None if:
    - anthropic_client is None (test/dry-run mode)
    - LLM call fails
    - No disagreement detected

    Never raises — failures are swallowed (this is a diagnostic, not a gate).
    """
    if anthropic_client is None:
        return None

    try:
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-6",  # Sonnet for disagreement ledger (higher quality)
            max_tokens=256,
            temperature=0,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"{SOFT_JUDGE_PROMPT}\n\n"
                        f"Dimension: {dimension}\n\n"
                        f"Agent prose:\n{agent_prose[:2000]}"  # cap to 2K chars
                    ),
                }
            ],
        )

        content = response.content[0].text.strip()
        raw = json.loads(content)
        perceived_level = raw.get("perceived_level", "medium")
        reasoning = raw.get("reasoning", "")
        disagreement_class = raw.get("disagreement_class", "clear")

        # Compare validator verdict to perceived level
        # Validator verdicts are like "empathy:low" or "pass"/"fail" per dimension
        # Map to low/high for comparison
        validator_level = _extract_level_from_verdict(validator_verdict, dimension)

        disagrees = perceived_level != validator_level

        return DisagreementRecord(
            conversation_id=conversation_id,
            validator_verdict=validator_verdict,
            soft_judge_perception=f"{dimension}:{perceived_level}",
            disagreement=disagrees,
            disagreement_class=disagreement_class if disagrees else "clear",
            notes=reasoning if disagrees else None,
        )

    except Exception:
        # Swallow all errors — disagreement ledger never gates
        return None


def _extract_level_from_verdict(validator_verdict: str, dimension: str) -> str:
    """Map validator verdict string to a level string for comparison with soft judge."""
    # validator_verdict might be "empathy:low", "resolution:weak", "on_brand", etc.
    if "low" in validator_verdict or "weak" in validator_verdict or "off_brand" in validator_verdict:
        return "low"
    if "high" in validator_verdict or "strong" in validator_verdict or "on_brand" in validator_verdict:
        return "high"
    # Default
    return "medium"


class DisagreementLedger:
    """
    Collects disagreement records and monitors disagreement rate.
    Not a gate — all monitoring is informational.
    """

    def __init__(self):
        """Initialize the ledger with empty records and zero counters."""
        self.records: list[DisagreementRecord] = []
        self._total_checked = 0
        self._disagreements = 0

    def add_record(self, record: DisagreementRecord | None) -> None:
        """Add a record (no-op if None)."""
        if record is None:
            return
        self._total_checked += 1
        self.records.append(record)
        if record.disagreement:
            self._disagreements += 1

    @property
    def disagreement_rate(self) -> float:
        """Fraction of checked records that show disagreement between validator and soft judge."""
        return self._disagreements / max(1, self._total_checked)

    def check_health(self) -> tuple[str, str]:
        """
        Returns (status, message):
        - ("healthy", ...) if rate < 0.15
        - ("warning", ...) if 0.15 <= rate < 0.25
        - ("blocking", ...) if rate >= 0.25
        """
        rate = self.disagreement_rate
        if rate < 0.15:
            return ("healthy", f"Disagreement rate {rate:.1%} — within healthy range")
        if rate < 0.25:
            return ("warning", f"Disagreement rate {rate:.1%} — approaching threshold; investigate")
        return ("blocking", f"Disagreement rate {rate:.1%} — exceeds 25%; blocks corpus extraction")

    def to_jsonl(self) -> str:
        """Serialize all records to JSONL format."""
        import json
        lines = []
        for record in self.records:
            lines.append(record.model_dump_json())
        return "\n".join(lines)
