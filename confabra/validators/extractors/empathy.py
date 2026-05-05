# confabra/validators/extractors/empathy.py
"""
Empathy signal extractor — fully deterministic (regex/structural, no LLM).
Per Section 5.1 implementation-grade signal schema.
"""
import re
from confabra.schemas import EmpathySignals


def _count_tokens(text: str) -> int:
    """Approximate token count (whitespace-split words)."""
    return len(text.split())


def _find_phrase_matches(text: str, phrases: list[str]) -> list[str]:
    """Find all phrase matches in text (case-insensitive). Returns matched phrases."""
    text_lower = text.lower()
    return [p for p in phrases if p in text_lower]


def _regex_matches(text: str, patterns: list[str]) -> list[str]:
    """Find all regex pattern matches. Returns matched terms."""
    text_lower = text.lower()
    matched = []
    for pattern in patterns:
        m = re.search(pattern, text_lower)
        if m:
            matched.append(m.group())
    return matched


def _check_follow_through(agent_prose: str, ack_phrases: list[str], action_verbs: list[str], window: int = 20) -> bool:
    """
    Check if an acknowledgment phrase is followed by an action verb within N tokens.

    Algorithm:
    1. Find the position (token index) of each acknowledgment phrase in the agent prose.
    2. From that position, scan the next `window` tokens.
    3. If any action verb appears in that window, return True.
    """
    if not ack_phrases:
        return False

    tokens = agent_prose.lower().split()
    token_str = " ".join(tokens)

    for phrase in ack_phrases:
        phrase_lower = phrase.lower()
        # Find position of the phrase in token stream
        phrase_pos = token_str.find(phrase_lower)
        if phrase_pos == -1:
            continue

        # Count tokens up to this position
        before_phrase = token_str[:phrase_pos]
        phrase_token_idx = len(before_phrase.split())

        # Get tokens after the acknowledgment phrase
        phrase_token_count = len(phrase.split())
        window_start = phrase_token_idx + phrase_token_count
        window_end = window_start + window
        window_tokens = tokens[window_start:window_end]
        window_text = " ".join(window_tokens)

        # Check if any action verb appears in the window
        for verb in action_verbs:
            if verb.lower() in window_text:
                return True

    return False


def _check_customer_emotion_referenced(agent_prose: str, emotion_terms: list[str], window: int = 15) -> bool:
    """
    Check if the agent references the customer's emotion near a customer reference.

    Algorithm: emotion term within N tokens of "you", "your", or customer name.
    """
    customer_refs = [r"\byou\b", r"\byour\b", r"\bcustomer\b"]
    tokens = agent_prose.lower().split()

    # Find positions of customer references
    customer_positions = []
    for i, token in enumerate(tokens):
        for ref_pattern in customer_refs:
            if re.match(ref_pattern, token):
                customer_positions.append(i)
                break

    # Check if any emotion term is within window of a customer reference
    for emo_term in emotion_terms:
        emo_lower = emo_term.lower()
        # Find all positions of emotion term
        for i, token in enumerate(tokens):
            if emo_lower in token:
                # Check if any customer reference is within window
                for cust_pos in customer_positions:
                    if abs(i - cust_pos) <= window:
                        return True

    return False


def extract_empathy_signals(
    agent_prose: str,
    acknowledgment_phrases: list[str],
    emotion_lexicon: list[str],
    apology_lexicon: list[str],
    action_verb_lexicon: list[str],
    follow_through_window: int = 20,
) -> EmpathySignals:
    """
    Extract empathy signals from agent-side conversation prose.

    Args:
        agent_prose: only the agent's turns, concatenated
        acknowledgment_phrases: from profile lexicon
        emotion_lexicon: from profile lexicon
        apology_lexicon: from profile lexicon
        action_verb_lexicon: from profile lexicon
        follow_through_window: N tokens after acknowledgment to look for action verb

    Returns:
        EmpathySignals with all fields populated
    """
    # 1. Find acknowledgment phrases
    ack_matches = _find_phrase_matches(agent_prose, acknowledgment_phrases)

    # 2. Find emotional language
    emo_matches = _find_phrase_matches(agent_prose, emotion_lexicon)
    emotional_language_present = len(emo_matches) > 0

    # 3. Find apology terms
    apology_matches = _find_phrase_matches(agent_prose, apology_lexicon)
    apology_present = len(apology_matches) > 0

    # Apology bridge: a sincere apology satisfies acknowledgment the same way an
    # acknowledgment phrase does — agents who lead with "I'm sorry" should not
    # fail the empathy high gate solely because they didn't also hit ACKNOWLEDGMENT_PHRASES.
    acknowledgment_present = len(ack_matches) > 0 or len(apology_matches) > 0

    # 4. Customer emotion referenced (emotion term near customer reference)
    customer_emotion_referenced = _check_customer_emotion_referenced(agent_prose, emotion_lexicon)

    # 5. Response length in tokens
    response_length_tokens = _count_tokens(agent_prose)

    # 6. Follow-through: search from the union of ack + apology anchors so that
    #    an agent who leads with "I'm sorry" can also satisfy follow_through.
    follow_through_present = _check_follow_through(
        agent_prose, ack_matches + apology_matches, action_verb_lexicon, follow_through_window
    )

    # 7. Fake empathy flag (derived)
    fake_empathy_flag = acknowledgment_present and not follow_through_present

    return EmpathySignals(
        acknowledgment_present=acknowledgment_present,
        acknowledgment_phrases=ack_matches,
        emotional_language_present=emotional_language_present,
        emotional_terms=emo_matches,
        apology_present=apology_present,
        apology_terms=apology_matches,
        customer_emotion_referenced=customer_emotion_referenced,
        response_length_tokens=response_length_tokens,
        follow_through_present=follow_through_present,
        fake_empathy_flag=fake_empathy_flag,
    )
