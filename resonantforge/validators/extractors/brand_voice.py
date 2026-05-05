"""
Brand voice signal extractor — fully deterministic (structural/statistical, no LLM).
Per Section 5.1 implementation-grade signal schema.
"""
import re
import math
from resonantforge.schemas import BrandVoiceSignals


def _split_sentences(text: str) -> list[str]:
    """
    Split text into sentences using punctuation boundary detection.

    Rule: sentence ends at period, question mark, or exclamation point
    followed by whitespace or end of string.
    Handles common abbreviations by requiring the punctuation to be followed
    by a capital letter or end of string (not just any whitespace).
    """
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z])|(?<=[.!?])\s*$', text.strip())
    return [s.strip() for s in sentences if s.strip()]


def _count_tokens(text: str) -> int:
    """Approximate token count (whitespace-split)."""
    return len(text.split())


def _match_hedging_terms(text: str, hedging_lexicon: list[str]) -> list[str]:
    """
    Find all hedging terms present in text using simple substring matching.

    Hedging terms are short phrases — substring match (case-insensitive) is
    sufficient and avoids regex escape complexity. Returns a deduplicated list
    of matched terms.
    """
    text_lower = text.lower()
    matched = []
    for term in hedging_lexicon:
        if term.lower() in text_lower:
            matched.append(term)
    return matched


def _match_directive_terms(text: str, directive_lexicon: list[str]) -> list[str]:
    """
    Find all directive terms present in text using word-boundary regex.

    Directive terms are single words or short fixed phrases; word boundaries
    prevent partial matches (e.g. "do" matching inside "download"). Returns a
    deduplicated list of matched terms.
    """
    text_lower = text.lower()
    matched = []
    for term in directive_lexicon:
        # Build a word-boundary pattern; escape the term in case of regex metacharacters
        pattern = r'\b' + re.escape(term) + r'\b'
        if re.search(pattern, text_lower):
            matched.append(term)
    return matched


def _count_phrase_matches(text: str, phrases: list[str]) -> int:
    """
    Count total (non-deduplicated) substring occurrences of each phrase in text.

    Used for vocabulary density counts (warm_terms, clinical_terms) where a
    repeated use of the same term should count multiple times.
    """
    text_lower = text.lower()
    count = 0
    for phrase in phrases:
        count += text_lower.count(phrase.lower())
    return count


def _count_regex_matches(text: str, patterns: list[str]) -> int:
    """
    Count total regex pattern matches in text (case-insensitive).

    Used for contraction detection where patterns already embed word boundaries.
    """
    text_lower = text.lower()
    total = 0
    for pattern in patterns:
        total += len(re.findall(pattern, text_lower))
    return total


def _compute_formality_score(
    text: str,
    contraction_patterns: list[str],
    sentence_count: int,
    token_count: int,
) -> float:
    """
    Compute formality score [0.0, 1.0].

    Formula:
    - Base: 1.0
    - Subtract 0.03 per contraction (lower contractions = more formal)
    - Subtract 0.05 per informal filler word (gonna, wanna, etc.)
    - Add 0.01 per long sentence (>= 15 tokens) — formal writing tends longer
    - Clamp to [0.0, 1.0]

    Approximate ranges: formal academic text ~0.8–1.0; casual chat ~0.1–0.4.
    """
    if token_count == 0:
        return 0.5

    contraction_count = _count_regex_matches(text, contraction_patterns)

    informal_fillers = len(
        re.findall(
            r"\b(gonna|wanna|kinda|sorta|dunno|yeah|yep|nope|nah)\b",
            text.lower(),
        )
    )

    sentences = _split_sentences(text)
    long_sentence_count = sum(1 for s in sentences if _count_tokens(s) >= 15)

    base = 1.0
    contraction_penalty = min(0.5, contraction_count * 0.03)
    informal_penalty = informal_fillers * 0.05
    formal_bonus = min(0.2, long_sentence_count * 0.01)

    score = base - contraction_penalty - informal_penalty + formal_bonus
    return round(max(0.0, min(1.0, score)), 3)


def brand_voice_feature_profile(
    variant_id: str,
    feature_profiles: dict,
) -> dict:
    """
    Look up the calibrated feature profile for a brand voice variant.

    Falls back to "bv_baseline" when the requested variant is not registered.
    Returns a dict with keys: sentence_length_range, question_count_range,
    hedging_range, directive_range, aligned_term_field, aligned_terms_min,
    formality_range.
    """
    profile = feature_profiles.get(variant_id)
    if profile is None:
        profile = feature_profiles.get("bv_baseline", {})
    return profile


def extract_brand_voice_signals(
    agent_prose: str,
    hedging_lexicon: list[str],
    directive_lexicon: list[str],
    warm_terms: list[str],
    clinical_terms: list[str],
    contraction_patterns: list[str],
) -> BrandVoiceSignals:
    """
    Extract brand voice signals from agent-side conversation prose.

    All extraction is fully deterministic (no LLM). Hedging terms are matched
    by simple substring (phrase match); directive terms use word-boundary regex
    to avoid false positives on partial word matches. Formality is computed from
    contraction density, informal filler words, and sentence length distribution.

    Args:
        agent_prose: agent turns concatenated into a single string
        hedging_lexicon: short phrases marking hedged / exploratory voice
        directive_lexicon: imperative words marking direct / clinical voice
        warm_terms: vocabulary associated with the warm-exploratory variant
        clinical_terms: vocabulary associated with the direct-clinical variant
        contraction_patterns: regex patterns detecting informal contractions

    Returns:
        BrandVoiceSignals with all fields populated
    """
    if not agent_prose.strip():
        return BrandVoiceSignals(
            avg_sentence_length=0.0,
            sentence_count=0,
            question_count=0,
            hedging_terms_count=0,
            hedging_terms=[],
            directive_terms_count=0,
            directive_terms=[],
            formality_score=0.5,
            exclamation_count=0,
            vocabulary_match={"warm_terms_count": 0, "clinical_terms_count": 0},
        )

    # 1. Sentence analysis
    sentences = _split_sentences(agent_prose)
    sentence_count = max(1, len(sentences))
    token_counts = [_count_tokens(s) for s in sentences]
    avg_sentence_length = round(sum(token_counts) / sentence_count, 2) if token_counts else 0.0

    # 2. Question and exclamation counts — character-level scan, unambiguous
    question_count = agent_prose.count("?")
    exclamation_count = agent_prose.count("!")

    # 3. Hedging terms — simple phrase/substring match (multi-word safe, deduped)
    hedging_matches = _match_hedging_terms(agent_prose, hedging_lexicon)

    # 4. Directive terms — word-boundary regex match (deduped)
    directive_matches = _match_directive_terms(agent_prose, directive_lexicon)

    # 5. Formality score
    total_tokens = _count_tokens(agent_prose)
    formality_score = _compute_formality_score(
        agent_prose, contraction_patterns, sentence_count, total_tokens
    )

    # 6. Vocabulary density (warm vs clinical) — count all occurrences, not just unique
    warm_count = _count_phrase_matches(agent_prose, warm_terms)
    clinical_count = _count_phrase_matches(agent_prose, clinical_terms)

    return BrandVoiceSignals(
        avg_sentence_length=avg_sentence_length,
        sentence_count=sentence_count,
        question_count=question_count,
        hedging_terms_count=len(hedging_matches),
        hedging_terms=hedging_matches[:10],  # cap at 10 for readability
        directive_terms_count=len(directive_matches),
        directive_terms=directive_matches[:10],
        formality_score=formality_score,
        exclamation_count=exclamation_count,
        vocabulary_match={
            "warm_terms_count": warm_count,
            "clinical_terms_count": clinical_count,
        },
    )
