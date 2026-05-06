"""
Synthetic replay fixture builder for RFORGE-18 tests.

Five fixtures covering the representative replay scenarios:
  1. conv_clean_pass          — all validators pass, label says pass
  2. conv_accuracy_fail       — planted ungrounded claim → accuracy FAIL
  3. conv_claim_extraction    — empty extracted_claims (truncated extraction); label marks
                                claim_extraction: True (shows as missed_rule in replay since
                                replay always reports claim_extraction_ok=True)
  4. conv_empathy_fail        — agent skips acknowledgment → empathy:high FAIL
  5. conv_brand_voice_fail    — short prose violates tight sentence-length range → brand_voice FAIL

All fixtures use minimal valid lexicons so tests run without the real profile.
"""
from __future__ import annotations

import json
from pathlib import Path


# ---------------------------------------------------------------------------
# Shared minimal lexicons
# ---------------------------------------------------------------------------

_LEXICONS = {
    # Empathy
    "acknowledgment_phrases": ["I understand", "I can see"],
    "emotion_lexicon": ["frustrated", "upset"],
    "apology_lexicon": ["I'm sorry", "I apologize"],
    "action_verb_lexicon": ["resolve", "fix"],
    # Resolution
    "resolution_patterns": [r"\bI will\b", r"\byou can\b"],
    "deflection_patterns": [r"\bcontact support\b", r"\bcheck the FAQ\b"],
    "next_steps_patterns": [r"\bnext step\b", r"\bfollowing up\b"],
    "temporal_anchor_patterns": [r"\bwithin 24 hours\b", r"\bby end of day\b"],
    "specific_actor_patterns": [r"\bI will\b", r"\bour team will\b"],
    "ownership_patterns": [r"\bI personally\b", r"\bI own this\b"],
    "issue_keywords": ["billing", "refund", "charge"],
    # Brand voice
    "hedging_lexicon": ["might", "perhaps"],
    "directive_lexicon": ["do", "click"],
    "warm_terms": ["happy to", "delighted"],
    "clinical_terms": ["utilize", "initiate"],
    "contraction_patterns": [r"\bdon'?t\b", r"\bwon'?t\b"],
    # Accuracy
    "synonym_map": {"reimburse": "refund"},
}

# Wide ranges that nearly any prose will satisfy (used for fixtures where brand_voice is not the target)
_BRAND_VOICE_WIDE = {
    "variant_id": "bv_baseline",
    "feature_profiles": {
        "bv_baseline": {
            "sentence_length_range": [1, 200],
            "question_count_range": [0, 20],
            "hedging_range": [0, 20],
            "directive_range": [0, 20],
            "aligned_term_field": "warm_terms_count",
            "aligned_terms_min": 0,
            "formality_range": [0.0, 1.0],
        }
    },
}

# Tight sentence-length range that short prose cannot satisfy
_BRAND_VOICE_STRICT = {
    "variant_id": "bv_strict",
    "feature_profiles": {
        "bv_strict": {
            "sentence_length_range": [50, 200],  # requires very long sentences
            "question_count_range": [0, 20],
            "hedging_range": [0, 20],
            "directive_range": [0, 20],
            "aligned_term_field": "warm_terms_count",
            "aligned_terms_min": 0,
            "formality_range": [0.0, 1.0],
        }
    },
}

_METADATA_BASE = {
    "source_corpus": "tests/replay_fixtures",
    "pipeline_version": "0.2.0",
    "kb_version": "sha256:test",
    "generator_version": "0.2.0",
    "extraction_timestamp": "2026-05-05T22:00:00Z",
}

_KB_CHUNK = {
    "chunk_id": "kb_refund_policy_v1",
    "document_id": "doc_refund",
    "document_path": "policies/refund_policy.md",
    "chunk_text": "Refunds are processed within 5 business days for eligible orders.",
    "constraint_type": "allow_condition",
    "cat11_gate": None,
    "tone_variant": None,
    "domains": ["billing"],
    "intent_tags": [],
    "adversarial": False,
    "sanity_probe": False,
    "claims": {},
    "metadata": {},
}

# A real claim that matches the KB chunk (used for passing accuracy tests)
_GROUNDED_CLAIM = {
    "claim_text": "Refunds are processed within 5 business days.",
    "claim_span": [0, 48],
    "claim_type": "policy",
    "normalized_subject": "refunds",
    "normalized_predicate": "are processed",
    "normalized_object": "within 5 business days",
    "alignment": None,
}

# An ungrounded claim: references a chunk ID that doesn't exist in kb_chunks,
# guaranteeing alignment="not_found" → accuracy FAIL when target is "supported"
_UNGROUNDED_CLAIM = {
    "claim_text": "Refunds are processed within 24 hours.",
    "claim_span": [0, 40],
    "claim_type": "policy",
    "normalized_subject": "refunds",
    "normalized_predicate": "are processed",
    "normalized_object": "within 24 hours",
    "alignment": None,
}


def _envelope(
    conv_id: str,
    agent_prose: str,
    customer_prose: str,
    quality_plan: dict,
    extracted_claims: list,
    brand_voice: dict | None = None,
) -> dict:
    return {
        "schema_version": 1,
        "conv_id": conv_id,
        "agent_prose": agent_prose,
        "customer_prose": customer_prose,
        "quality_plan": quality_plan,
        "kb_chunks": [_KB_CHUNK],
        "lexicons": _LEXICONS,
        "brand_voice": brand_voice or _BRAND_VOICE_WIDE,
        "validator_inputs": {
            "accuracy": {
                "extracted_claims": extracted_claims,
                "extraction_meta": {
                    "model": "claude-haiku-4-5-20251001",
                    "prompt_version": "sha256:test",
                    "extracted_at": "2026-05-05T22:00:00Z",
                    "claims_hash": "sha256:test",
                },
            },
        },
        "metadata": {**_METADATA_BASE, "conv_id": conv_id},
    }


def _quality_plan(
    conv_id: str,
    rubric_targets: dict,
    kb_chunks_required: list | None = None,
) -> dict:
    return {
        "conversation_id": conv_id,
        "trigger_event_id": f"evt_{conv_id}",
        "rubric_targets": rubric_targets,
        "knowledge_citations": {"should_cite": [], "must_not_cite": []},
        "prose_generation_directives": "Test fixture.",
        "cat11_gate": None,
        "multi_chunk_required": False,
        "kb_chunks_required": kb_chunks_required or [],
    }


def _labels(
    conv_id: str,
    expected_outcome: str,
    expected_failures: dict,
    confidence: str,
    tags: list,
    rationale: str,
) -> dict:
    return {
        "schema_version": 1,
        "conv_id": conv_id,
        "label_version": 1,
        "labeled_at": "2026-05-05",
        "labeled_by": "tj",
        "expected_outcome": expected_outcome,
        "expected_failures": {
            "accuracy": False,
            "empathy": False,
            "resolution": False,
            "brand_voice": False,
            "claim_extraction": False,
            **expected_failures,
        },
        "confidence": confidence,
        "tags": tags,
        "rationale": rationale,
        "revised_from": None,
        "revision_notes": None,
    }


# ---------------------------------------------------------------------------
# Fixture 1: conv_clean_pass
# All 4 validator dimensions target pass and do pass.
# ---------------------------------------------------------------------------

def _fixture_clean_pass() -> tuple[dict, dict]:
    conv_id = "conv_clean_pass"
    agent_prose = (
        "I understand your frustration with the billing charge. "
        "I'm sorry for the inconvenience. "
        "I will personally resolve this refund immediately. "
        "Refunds are processed within 5 business days. "
        "You can expect confirmation within 24 hours."
    )
    customer_prose = "I have a billing charge I need a refund for."

    plan = _quality_plan(
        conv_id,
        rubric_targets={
            "empathy": "high",
            "resolution": "strong",
            "brand_voice_against": "bv_baseline",
            "brand_voice_target": "on_brand",
            "accuracy": {"status": "supported", "precision": "exact"},
        },
        kb_chunks_required=["kb_refund_policy_v1"],
    )

    env = _envelope(conv_id, agent_prose, customer_prose, plan, [_GROUNDED_CLAIM])
    lbl = _labels(
        conv_id,
        expected_outcome="pass",
        expected_failures={},
        confidence="high",
        tags=["high_confidence_pass", "planted"],
        rationale=(
            "Agent acknowledges frustration, apologizes, commits personally, cites correct "
            "5-business-day refund policy. All 4 dimensions should pass."
        ),
    )
    return env, lbl


# ---------------------------------------------------------------------------
# Fixture 2: conv_accuracy_fail
# Agent made a claim that doesn't align to any required KB chunk.
# Achieved by referencing a kb_chunks_required ID that isn't in kb_chunks —
# the candidate pool is empty, all claims get alignment="not_found" → FAIL.
# ---------------------------------------------------------------------------

def _fixture_accuracy_fail() -> tuple[dict, dict]:
    conv_id = "conv_accuracy_fail"
    agent_prose = (
        "I understand your concern. "
        "I will look into this for you. "
        "Refunds are processed within 24 hours."
    )
    customer_prose = "I'm frustrated about the billing charge."

    plan = _quality_plan(
        conv_id,
        rubric_targets={
            "empathy": "high",
            "accuracy": {"status": "supported", "precision": "exact"},
        },
        # Points to a chunk ID that does not exist in kb_chunks → empty candidate pool
        # → all claims get alignment="not_found" → accuracy:supported:exact → FAIL
        kb_chunks_required=["kb_nonexistent_chunk_id"],
    )

    env = _envelope(conv_id, agent_prose, customer_prose, plan, [_UNGROUNDED_CLAIM])
    lbl = _labels(
        conv_id,
        expected_outcome="fail",
        expected_failures={"accuracy": True},
        confidence="high",
        tags=["high_confidence_fail", "planted"],
        rationale=(
            "Agent claims refunds within 24 hours but required KB chunk doesn't exist "
            "in the candidate pool → alignment=not_found → accuracy:supported:exact fails."
        ),
    )
    return env, lbl


# ---------------------------------------------------------------------------
# Fixture 3: conv_claim_extraction
# Simulates a conversation where extract_claims_llm failed during live generation
# (returned empty list). Label marks claim_extraction: True.
# In replay, claim_extraction_ok is always True (frozen empty list is injected),
# so this always surfaces as missed_rules: ["claim_extraction"] — useful for
# identifying convs where extraction failure was the root cause.
# ---------------------------------------------------------------------------

def _fixture_claim_extraction() -> tuple[dict, dict]:
    conv_id = "conv_claim_extraction"
    agent_prose = (
        "Thank you for reaching out. "
        "We will investigate your concern and follow up shortly."
    )
    customer_prose = "I need information about your refund policy for my order."

    plan = _quality_plan(
        conv_id,
        rubric_targets={
            "accuracy": {"status": "supported", "precision": "exact"},
        },
        kb_chunks_required=["kb_refund_policy_v1"],
    )

    # Empty extracted_claims — simulates truncated/failed extraction
    env = _envelope(conv_id, agent_prose, customer_prose, plan, [])
    lbl = _labels(
        conv_id,
        expected_outcome="fail",
        expected_failures={"claim_extraction": True, "accuracy": True},
        confidence="medium",
        tags=["claim_extraction_stress", "planted"],
        rationale=(
            "extract_claims_llm returned empty (retries exhausted). "
            "No claims → accuracy alignment=not_found → accuracy:supported:exact fails. "
            "claim_extraction labeled True for root-cause attribution; replay always "
            "reports claim_extraction_ok=True so this appears as missed_rule."
        ),
    )
    return env, lbl


# ---------------------------------------------------------------------------
# Fixture 4: conv_empathy_fail
# Agent skips acknowledgment → empathy:high FAIL.
# ---------------------------------------------------------------------------

def _fixture_empathy_fail() -> tuple[dict, dict]:
    conv_id = "conv_empathy_fail"
    agent_prose = (
        "Your refund has been processed. "
        "You will receive confirmation by email. "
        "I will ensure this is completed today."
    )
    customer_prose = "I'm really frustrated about this billing charge on my account."

    plan = _quality_plan(
        conv_id,
        rubric_targets={
            "empathy": "high",
        },
    )

    env = _envelope(conv_id, agent_prose, customer_prose, plan, [])
    lbl = _labels(
        conv_id,
        expected_outcome="fail",
        expected_failures={"empathy": True},
        confidence="high",
        tags=["high_confidence_fail", "planted"],
        rationale=(
            "Agent jumps straight to resolution without any acknowledgment phrase "
            "or emotional language. No 'I understand', no emotion terms. "
            "Empathy:high requires acknowledgment_present + emotional_language_present → FAIL."
        ),
    )
    return env, lbl


# ---------------------------------------------------------------------------
# Fixture 5: conv_brand_voice_fail
# Agent prose is short, violating the strict sentence_length_range [50, 200].
# avg_sentence_length will be well under 50 → brand_voice:on_brand FAIL.
# ---------------------------------------------------------------------------

def _fixture_brand_voice_fail() -> tuple[dict, dict]:
    conv_id = "conv_brand_voice_fail"
    agent_prose = "Hi! Sure. Done."  # avg sentence length ≈ 2 tokens — far below 50
    customer_prose = "Can you help with my account?"

    plan = _quality_plan(
        conv_id,
        rubric_targets={
            "brand_voice_against": "bv_strict",
            "brand_voice_target": "on_brand",
        },
    )

    env = _envelope(
        conv_id, agent_prose, customer_prose, plan, [],
        brand_voice=_BRAND_VOICE_STRICT,
    )
    lbl = _labels(
        conv_id,
        expected_outcome="fail",
        expected_failures={"brand_voice": True},
        confidence="high",
        tags=["high_confidence_fail", "planted"],
        rationale=(
            "Agent prose is three words per sentence on average. "
            "bv_strict requires sentence_length_range=[50, 200]. "
            "avg_sentence_length << 50 → avg_sentence_length feature check fails → "
            "brand_voice:on_brand FAIL."
        ),
    )
    return env, lbl


# ---------------------------------------------------------------------------
# Builder entry point
# ---------------------------------------------------------------------------

FIXTURES = {
    "conv_clean_pass": _fixture_clean_pass,
    "conv_accuracy_fail": _fixture_accuracy_fail,
    "conv_claim_extraction": _fixture_claim_extraction,
    "conv_empathy_fail": _fixture_empathy_fail,
    "conv_brand_voice_fail": _fixture_brand_voice_fail,
}


def build_fixtures(output_dir: Path) -> None:
    """Write all 5 fixtures to {output_dir}/{conv_id}/envelope.json + labels.json."""
    for conv_id, factory in FIXTURES.items():
        env, lbl = factory()
        conv_dir = output_dir / conv_id
        conv_dir.mkdir(parents=True, exist_ok=True)
        (conv_dir / "envelope.json").write_text(json.dumps(env, indent=2), encoding="utf-8")
        (conv_dir / "labels.json").write_text(json.dumps(lbl, indent=2), encoding="utf-8")
