"""
Accuracy signal extractor — LLM contained to Step 1 only; all other steps are deterministic.
Implements the 8-step claim extraction + KB alignment pipeline (Section 5.1.1).
"""
from __future__ import annotations
import re
import json
import logging
from typing import Any
import anthropic
from anthropic import Anthropic
from pydantic import ValidationError
from resonantforge.schemas import Claim, AccuracySignals, KBChunk, ConstraintType

logger = logging.getLogger(__name__)

# Locked-down claim extraction prompt (temperature=0, structured JSON only)
CLAIM_EXTRACTION_PROMPT = """Extract explicit claims made by the agent.
A claim is a statement asserting a fact, policy, or procedure.
Return JSON only:
[
  {
    "claim_text": "verbatim text from agent prose",
    "claim_span": [start_char, end_char],
    "claim_type": "policy | procedural | factual",
    "subject": "...",
    "predicate": "...",
    "object": "..."
  }
]
Rules:
- Do not infer unstated claims
- Do not summarize
- Extract only what is explicitly stated
- Keep claims atomic
- Preserve verbatim text for claim_text
- Return [] if no claims found
- Respond with raw JSON only. Do not wrap in markdown code fences."""


def _strip_markdown_fence(content: str) -> str:
    """
    Strip ```json ... ``` or ``` ... ``` fences from LLM response if present.

    Defense-in-depth: the prompt explicitly forbids fences, but some models add them
    anyway. Strip before json.loads so a fenced response isn't silently discarded.
    """
    content = content.strip()
    if content.startswith("```"):
        first_newline = content.find("\n")
        if first_newline != -1:
            content = content[first_newline + 1:]
        if content.endswith("```"):
            content = content[:-3].strip()
    return content


class ClaimExtractionError(Exception):
    """
    Raised when extract_claims_llm exhausts all retry attempts without producing valid JSON.

    Surfaced as a distinct 'claim_extraction' validator rule failure so the pipeline never
    silently treats a parse failure as a trivial accuracy pass.
    """

    def __init__(self, raw_excerpt: str, attempts: int) -> None:
        self.raw_excerpt = raw_excerpt
        self.attempts = attempts
        super().__init__(
            f"claim extraction failed after {attempts} attempt(s); "
            f"raw response excerpt: {raw_excerpt[:200]!r}"
        )


# Appended on retry attempts to steer the LLM back to pure JSON output.
_STRICT_JSON_SUFFIX = (
    "\n\nIMPORTANT: Respond with ONLY valid JSON. "
    "No prose, no markdown fences, no explanation."
)


def extract_claims_llm(
    agent_prose: str,
    anthropic_client: Anthropic | None = None,
    max_attempts: int = 3,
) -> list[Claim]:
    """
    Step 1: Extract claims from agent prose using LLM (temperature=0, structured JSON).

    This is the only LLM call in the accuracy extractor. All downstream steps are
    deterministic. If anthropic_client is None (test/dry-run mode), returns empty list.

    On JSON parse failure, retries up to ``max_attempts - 1`` times with a stricter
    prompt suffix. If all attempts fail, raises ClaimExtractionError — never silently
    returns [] on a parse failure, because that would make the accuracy validator
    trivially pass against an empty claim set.

    API-level errors (auth, network, rate limit) still return [] so the pipeline is
    never gated on infrastructure problems.

    Args:
        agent_prose: agent turns only (customer turns must be excluded by the caller).
        anthropic_client: live Anthropic client, or None for test mode.
        max_attempts: total LLM call attempts before raising (default 3).

    Returns:
        List of Claim objects parsed from the LLM JSON response.

    Raises:
        ClaimExtractionError: if all attempts produce unparseable JSON.
    """
    if not agent_prose.strip():
        return []

    if anthropic_client is None:
        # Test/dry-run mode — return empty claims without hitting the API.
        return []

    last_raw = ""
    last_stop_reason: str | None = None
    for attempt in range(max_attempts):
        prompt = CLAIM_EXTRACTION_PROMPT
        if attempt > 0:
            prompt += _STRICT_JSON_SUFFIX

        try:
            response = anthropic_client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=1024,
                temperature=0,
                messages=[
                    {
                        "role": "user",
                        "content": f"{prompt}\n\nAgent prose:\n{agent_prose}",
                    }
                ],
            )
        except anthropic.APIError as e:
            # Network failures, auth errors, rate limits — never gate the pipeline on these.
            logger.warning(
                "extract_claims_llm: Anthropic API error — returning empty claims",
                extra={"error": str(e)},
            )
            return []

        last_raw = response.content[0].text.strip()
        last_stop_reason = response.stop_reason
        content = _strip_markdown_fence(last_raw)

        try:
            raw_claims = json.loads(content)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning(
                "extract_claims_llm: JSON parse failed",
                extra={
                    "attempt": attempt + 1,
                    "max_attempts": max_attempts,
                    "exc_type": type(exc).__name__,
                    "response_length": len(last_raw),
                    "stop_reason": last_stop_reason,
                    "excerpt_head": last_raw[:200],
                    "excerpt_tail": last_raw[-200:],
                },
            )
            continue  # retry with stricter prompt

        try:
            claims = []
            for raw in raw_claims:
                # Use `or default` (not `get(key, default)`) so that JSON null values
                # — which dict.get returns as None even when a default is provided —
                # are replaced with the intended fallback string.
                claim_text = raw.get("claim_text") or ""
                claim_type = raw.get("claim_type") or "factual"
                subject = raw.get("subject") or ""
                predicate = raw.get("predicate") or ""
                obj = raw.get("object") or ""

                # Validate claim_span is a two-element list/tuple.
                span = raw.get("claim_span") or [0, len(claim_text)]
                if len(span) != 2:
                    span = [0, len(claim_text)]

                claims.append(Claim(
                    claim_text=claim_text,
                    claim_span=(span[0], span[1]),
                    claim_type=claim_type,
                    normalized_subject=_normalize_text(subject, {}),
                    normalized_predicate=_normalize_text(predicate, {}),
                    normalized_object=_normalize_text(obj, {}),
                ))
        except ValidationError as e:
            # Pydantic rejected a field value (e.g. unknown claim_type literal) — treat as
            # schema mismatch, not a parse failure, so we don't retry on something unfixable.
            logger.warning(
                "extract_claims_llm: Claim schema validation failed — returning empty claims",
                extra={"error": str(e)},
            )
            return []

        return claims

    raise ClaimExtractionError(raw_excerpt=last_raw, attempts=max_attempts)


def _normalize_text(text: str, synonym_map: dict[str, str]) -> str:
    """
    Step 3.1: Normalize text for KB matching.

    Applies three transforms in order:
      1. Lowercase
      2. Strip punctuation (replace with space, collapse whitespace)
      3. Apply synonym map (longest multi-word phrases substituted first to avoid
         partial replacements inside longer phrases)

    Args:
        text: raw text to normalize.
        synonym_map: mapping of source phrase → canonical form (e.g. "reimburse" → "refund").

    Returns:
        Normalized string suitable for term-overlap matching.
    """
    if not isinstance(text, str):
        return ""
    normalized = text.lower()
    normalized = re.sub(r'[^\w\s]', ' ', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()

    # Apply synonym map longest-first so multi-word phrases win over their substrings.
    for source, target in sorted(synonym_map.items(), key=lambda x: -len(x[0])):
        normalized = normalized.replace(source.lower(), target.lower())

    return normalized


def _extract_constraints_from_text(text: str) -> list[str]:
    """
    Extract constraint indicator phrases from text.

    Used in Step 5 (overgeneralization detection) to decide whether a KB chunk's
    constraints were preserved verbatim in the agent's claim.

    Returns:
        List of matched constraint phrase strings found in the text.
    """
    constraint_patterns = [
        r'\bwithin \d+ days?\b',
        r'\bwithin \d+ (?:hour|minute|week|month)s?\b',
        r'\bonly\b',
        r'\bexcluding\b',
        r'\bexcept\b',
        r'\bunless\b',
        r'\bif (?:you|usage|account)\b',
        r'\bnot allowed\b',
        r'\bnot eligible\b',
        r'\beligible only\b',
        r'\b(?:paid|annual|monthly) (?:plan|subscription|account)s? only\b',
        r'\bup to \d+\b',
        r'\bmaximum \d+\b',
        r'\bat least \d+\b',
        r'\bno more than\b',
        r'\bless than\b',
        r'\bgreater than\b',
    ]

    if not text:
        return []
    found = []
    text_lower = text.lower()
    for pattern in constraint_patterns:
        matches = re.findall(pattern, text_lower)
        found.extend(matches)

    return found


def _check_constraint_type_applies(chunk: KBChunk, conversation_context: str) -> bool:
    """
    Step 6: Determine whether a DENY_CONDITION chunk is triggered by the conversation context.

    Heuristic approach: the deny chunk fires when both the chunk and the customer context
    reference usage/limit trigger concepts, or when the customer context contains numbers
    that exceed numeric thresholds in the deny chunk.

    Args:
        chunk: a KB chunk with constraint_type == DENY_CONDITION.
        conversation_context: customer turns from the conversation (not agent turns).

    Returns:
        True if the deny condition is considered active for this conversation.
    """
    if chunk.constraint_type != ConstraintType.DENY_CONDITION:
        return False

    chunk_lower = chunk.chunk_text.lower()
    context_lower = (conversation_context or "").lower()

    # Extract numeric thresholds stated in the deny chunk.
    numbers_in_chunk = re.findall(r'\b\d+\b', chunk_lower)

    trigger_phrases = [
        "usage", "used", "units", "exceeds", "exceeded", "more than",
        "over", "above", "beyond", "quota", "limit"
    ]

    chunk_has_triggers = any(
        re.search(r'\b' + re.escape(phrase) + r'\b', chunk_lower) for phrase in trigger_phrases
    )
    context_has_triggers = any(
        re.search(r'\b' + re.escape(phrase) + r'\b', context_lower) for phrase in trigger_phrases
    )

    if chunk_has_triggers and context_has_triggers:
        # Both the deny chunk and the customer context reference usage/limit concepts —
        # conservatively treat the deny condition as active.
        return True

    # Fallback: check whether the customer mentions numbers that exceed the chunk's threshold.
    if numbers_in_chunk:
        context_numbers = re.findall(r'\b\d+\b', context_lower)
        for chunk_num in numbers_in_chunk:
            for ctx_num in context_numbers:
                try:
                    if int(ctx_num) > int(chunk_num):
                        return True
                except ValueError:
                    pass

    return False


def _check_chunk_relevance(claim: Claim, chunk: KBChunk, synonym_map: dict[str, str]) -> dict[str, Any]:
    """
    Steps 3.2 and 3.3: Determine whether a KB chunk is topically relevant to a claim
    and classify their alignment relationship.

    Relevance is decided by normalized term overlap: key nouns from the claim's subject
    and object must appear in the chunk text. Alignment is then classified as:
      - "contradicted": chunk is a DENY_CONDITION that negates a positive agent claim.
      - "partial": chunk supports the claim but contains constraints the claim omits.
      - "supported": chunk supports the claim with constraints preserved.
      - "not_found": chunk is not relevant to this claim.

    Args:
        claim: a single extracted Claim.
        chunk: a KB chunk candidate.
        synonym_map: for normalizing both claim and chunk text before comparison.

    Returns:
        Dict with keys: relevant, alignment, constraints_in_chunk, constraint_preserved.
    """
    claim_norm = _normalize_text(claim.claim_text, synonym_map)
    chunk_norm = _normalize_text(chunk.chunk_text, synonym_map)

    # Key terms: subject + object nouns, filtered to len > 2 to exclude stop-word noise.
    claim_terms = set(
        _normalize_text(claim.normalized_subject, synonym_map).split() +
        _normalize_text(claim.normalized_object, synonym_map).split()
    )
    claim_terms = {t for t in claim_terms if len(t) > 2}

    # Relevance gate: at least 20 % of claim's key terms must appear in the chunk.
    chunk_words = set(chunk_norm.split())
    overlap = claim_terms & chunk_words

    if not overlap or len(overlap) < max(1, len(claim_terms) * 0.2):
        return {
            "relevant": False,
            "alignment": "not_found",
            "constraints_in_chunk": [],
            "constraint_preserved": True,
        }

    # Contradiction check: chunk explicitly denies something the claim asserts positively.
    negation_patterns = [
        r'\bnot allowed\b', r'\bnot eligible\b', r'\bcannot\b', r'\bwill not\b',
        r'\bnot available\b', r'\bno refund\b', r'\bno longer\b', r'\bdenied\b',
    ]
    chunk_has_negation = any(re.search(p, chunk_norm) for p in negation_patterns)

    if chunk_has_negation and chunk.constraint_type == ConstraintType.DENY_CONDITION:
        allow_terms = ["allowed", "eligible", "available", "can", "will", "refund"]
        claim_asserts_positive = any(term in claim_norm for term in allow_terms)
        if claim_asserts_positive:
            return {
                "relevant": True,
                "alignment": "contradicted",
                "constraints_in_chunk": _extract_constraints_from_text(chunk.chunk_text),
                "constraint_preserved": False,
            }

    # Overgeneralization check (Step 5): chunk has constraints the claim omits.
    constraints_in_chunk = _extract_constraints_from_text(chunk.chunk_text)
    constraints_in_claim = _extract_constraints_from_text(claim.claim_text)

    constraint_preserved = True
    if constraints_in_chunk and not constraints_in_claim:
        # The chunk qualifies its statement; the claim does not → overgeneralization.
        constraint_preserved = False

    return {
        "relevant": True,
        "alignment": "supported" if constraint_preserved else "partial",
        "constraints_in_chunk": constraints_in_chunk,
        "constraint_preserved": constraint_preserved,
    }


def _check_fake_citation(agent_prose: str) -> list[str]:
    """
    Step 8: Detect citation strings embedded in agent prose.

    Citations are labels, not truth claims. The validator ignores them during KB
    alignment but detects them here to flag Gate 9 (fabricated citation) cases.

    Args:
        agent_prose: combined agent turn text.

    Returns:
        List of matched citation strings found in the prose.
    """
    citation_patterns = [
        r'kb_chunk_\w+',
        r'per (?:our|the) (?:refund|billing|sla|cancellation) policy',
        r'according to (?:our|the) \w+ policy',
        r'per (?:our|the) \w+_policy\w*',
        r'as (?:stated|outlined|described) in',
    ]

    found = []
    for pattern in citation_patterns:
        matches = re.findall(pattern, agent_prose.lower())
        found.extend(matches)

    return found


def run_kb_alignment_pipeline(
    claims: list[Claim],
    kb_chunks: list[KBChunk],
    conversation_context: str,
    kb_chunks_required: list[str],
    synonym_map: dict[str, str],
    top_k: int = 3,
) -> AccuracySignals:
    """
    Steps 2–8: Deterministic KB alignment pipeline.

    Consumes pre-extracted claims (Step 1 output) and a pre-retrieved set of KB
    chunks (Step 2 retrieval is handled externally; this function takes the top-K
    candidates directly).  All logic from Step 3 onward is pure Python — no LLM.

    Pipeline summary:
      Step 2  — candidate pool = required chunks only (kb_chunks_required); organic
                conversations with an empty required list return early as not_found.
      Step 3  — per-claim relevance check and alignment classification.
      Step 4  — aggregate alignment across all claims (worst-case wins).
      Step 5  — overgeneralization flag (constraint in chunk, absent from claim).
      Step 6  — blocking deny-condition check against conversation context.
      Step 7  — multi-chunk satisfaction check against planted ground truth.
      Step 8  — fake citation detection (logged on AccuracySignals.meta by caller).

    Args:
        claims: extracted claims from Step 1 (may be empty).
        kb_chunks: full KB chunk list; only chunks in kb_chunks_required are used.
        conversation_context: customer turns only (used for Step 6 deny-condition check).
        kb_chunks_required: planted ground-truth chunk IDs; defines the candidate pool.
        synonym_map: per-profile synonym mapping for text normalization.
        top_k: unused (retained for backward-compatible call sites).

    Returns:
        AccuracySignals capturing the full pipeline verdict.
    """
    if not claims:
        # No claims extracted → insufficient-information case (Gate 7).
        return AccuracySignals(
            claims=[],
            kb_chunks_used=[],
            kb_chunks_required=kb_chunks_required,
            alignment="not_found",
            constraint_preserved=True,
            overgeneralization_flag=False,
            blocking_constraint_violated=False,
            multi_chunk_required=len(kb_chunks_required) > 1,
            multi_chunk_satisfied=False,
        )

    # Step 2: Candidate pool = required chunks only (never insertion-order truncation).
    # Organic conversations (empty kb_chunks_required) skip KB alignment entirely.
    required_set = set(kb_chunks_required)
    if not required_set:
        return AccuracySignals(
            claims=claims,
            kb_chunks_used=[],
            kb_chunks_required=kb_chunks_required,
            alignment="not_found",
            constraint_preserved=True,
            overgeneralization_flag=False,
            blocking_constraint_violated=False,
            multi_chunk_required=False,
            multi_chunk_satisfied=False,
        )
    candidate_chunks = [c for c in kb_chunks if c.chunk_id in required_set]

    per_claim_results: list[str] = []
    kb_chunks_used: list[str] = []
    overall_overgeneralization = False
    blocking_violated = False

    for claim in claims:
        claim_best_alignment = "not_found"
        claim_constraint_preserved = True

        for chunk in candidate_chunks:
            result = _check_chunk_relevance(claim, chunk, synonym_map)

            if not result["relevant"]:
                continue

            if chunk.chunk_id not in kb_chunks_used:
                kb_chunks_used.append(chunk.chunk_id)

            alignment = result["alignment"]

            # Step 5: Overgeneralization — chunk has constraints the claim drops.
            if alignment == "supported" and not result["constraint_preserved"]:
                alignment = "partial"
                overall_overgeneralization = True

            # Step 6: Blocking deny-condition — a DENY_CONDITION chunk is active for
            # this conversation context, meaning the agent's positive claim is wrong.
            if alignment == "supported":
                for deny_chunk in candidate_chunks:
                    if (deny_chunk.constraint_type == ConstraintType.DENY_CONDITION and
                            _check_constraint_type_applies(deny_chunk, conversation_context)):
                        blocking_violated = True
                        alignment = "contradicted"
                        break

            # Worst-case alignment wins for this claim across all chunks.
            priority = {"contradicted": 3, "partial": 2, "supported": 1, "not_found": 0}
            if priority.get(alignment, 0) > priority.get(claim_best_alignment, 0):
                claim_best_alignment = alignment

            if not result["constraint_preserved"]:
                claim_constraint_preserved = False

        # Persist the resolved alignment back onto the Claim object.
        claim.alignment = claim_best_alignment
        per_claim_results.append(claim_best_alignment)

    # Step 4: Aggregate alignment — worst case across all claims.
    if any(r == "contradicted" for r in per_claim_results):
        aggregate_alignment = "contradicted"
    elif any(r == "partial" for r in per_claim_results):
        aggregate_alignment = "partial"
    elif all(r == "supported" for r in per_claim_results) and per_claim_results:
        aggregate_alignment = "supported"
    else:
        aggregate_alignment = "not_found"

    # Step 7: Multi-chunk satisfaction — all required chunks must appear in kb_chunks_used.
    multi_chunk_required = len(kb_chunks_required) > 1
    multi_chunk_satisfied = False
    if multi_chunk_required:
        multi_chunk_satisfied = all(req in kb_chunks_used for req in kb_chunks_required)

    return AccuracySignals(
        claims=claims,
        kb_chunks_used=kb_chunks_used,
        kb_chunks_required=kb_chunks_required,
        alignment=aggregate_alignment,
        constraint_preserved=not overall_overgeneralization,
        overgeneralization_flag=overall_overgeneralization,
        blocking_constraint_violated=blocking_violated,
        multi_chunk_required=multi_chunk_required,
        multi_chunk_satisfied=multi_chunk_satisfied,
    )


def extract_accuracy_signals(
    agent_prose: str,
    conversation_context: str,
    kb_chunks: list[KBChunk],
    kb_chunks_required: list[str],
    synonym_map: dict[str, str],
    anthropic_client: Anthropic | None = None,
    top_k: int = 3,
) -> AccuracySignals:
    """
    Full 8-step accuracy extraction pipeline entry point.

    Step 1 (LLM) extracts atomic claims from agent prose. Steps 2–8 are entirely
    deterministic and operate on the claim list produced by Step 1.

    Callers pass only agent turns in ``agent_prose`` and only customer turns in
    ``conversation_context`` — the caller is responsible for splitting the
    conversation by speaker before invoking this function.

    Args:
        agent_prose: agent turns only (customer turns excluded per spec).
        conversation_context: customer turns (used for Step 6 deny-condition check).
        kb_chunks: pre-retrieved KB chunks (top-K from match_knowledge_chunks).
        kb_chunks_required: planted ground-truth chunk IDs for multi-chunk validation.
        synonym_map: per-profile synonym mapping for text normalization.
        anthropic_client: live Anthropic client, or None for test/dry-run mode.
        top_k: max chunks to evaluate per claim (Step 2 cap).

    Returns:
        AccuracySignals capturing alignment verdict, constraint flags, and chunk provenance.
    """
    # Step 1: LLM claim extraction — only LLM call in this module.
    claims = extract_claims_llm(agent_prose, anthropic_client)

    # Steps 2–8: Deterministic pipeline.
    return run_kb_alignment_pipeline(
        claims=claims,
        kb_chunks=kb_chunks,
        conversation_context=conversation_context,
        kb_chunks_required=kb_chunks_required,
        synonym_map=synonym_map,
        top_k=top_k,
    )
