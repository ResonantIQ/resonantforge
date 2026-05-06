"""
Envelope extraction pipeline: reads smoke corpus output, runs extract_claims_llm
once per conversation, and writes frozen envelope.json + template labels.json.

Called by `rforge replay extract-envelopes`. This is the only paid step in the
replay workflow — all subsequent `rforge replay run` iterations are free.
"""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from resonantforge.pipeline import _load_lexicons, _split_prose_turns
from resonantforge.profiles import get_profile
from resonantforge.replay.schemas import (
    AccuracyValidatorInputs,
    BrandVoiceConfig,
    EnvelopeLexicons,
    EnvelopeMetadata,
    ExtractionMeta,
    ReplayEnvelope,
    ReplayLabels,
    ValidatorInputs,
)
from resonantforge.schemas import (
    Claim,
    KBChunk,
    Manifest,
    QualityPlan,
    RubricTarget,
    KnowledgeCitations,
    ConversationRecord,
    SkippedConversationRecord,
)
from resonantforge.utils.atomic_write import read_jsonl_robust
from resonantforge.validators.extractors.accuracy import (
    CLAIM_EXTRACTION_PROMPT,
    extract_claims_llm,
)
from resonantforge.validators.extractors.brand_voice import brand_voice_feature_profile


_LABEL_TEMPLATE_LABELED_BY = "tj"
_SCHEMA_VERSION = 1

# Stable hash of the extraction prompt — changes when the prompt text changes
_PROMPT_VERSION = "sha256:" + hashlib.sha256(
    CLAIM_EXTRACTION_PROMPT.encode("utf-8")
).hexdigest()


def _read_manifest(profile_dir: Path) -> Optional[Manifest]:
    manifest_path = profile_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    return Manifest.model_validate(raw)


def _read_conversations(profile_dir: Path) -> list[ConversationRecord]:
    conv_path = profile_dir / "conversations.jsonl"
    if not conv_path.exists():
        return []
    records = []
    for line in read_jsonl_robust(conv_path):
        records.append(ConversationRecord.model_validate(line))
    return records


def _read_skipped_conversations(profile_dir: Path) -> list[SkippedConversationRecord]:
    skip_path = profile_dir / "skipped_conversations.jsonl"
    if not skip_path.exists():
        return []
    records = []
    for line in read_jsonl_robust(skip_path):
        records.append(SkippedConversationRecord.model_validate(line))
    return records


def _read_quality_plans(profile_dir: Path) -> dict[str, QualityPlan]:
    """Return a dict mapping conversation_id → QualityPlan."""
    plan_path = profile_dir / "planted_quality.jsonl"
    if not plan_path.exists():
        return {}
    plans: dict[str, QualityPlan] = {}
    for line in read_jsonl_robust(plan_path):
        plan = QualityPlan.model_validate(line)
        plans[plan.conversation_id] = plan
    return plans


def _read_kb_chunks(profile_dir: Path) -> list[KBChunk]:
    chunk_path = profile_dir / "knowledge_base" / "chunks.jsonl"
    if not chunk_path.exists():
        return []
    chunks = []
    for line in read_jsonl_robust(chunk_path):
        chunks.append(KBChunk.model_validate(line))
    return chunks


def _lexicons_from_profile(profile_name: str) -> EnvelopeLexicons:
    """Load profile lexicons and convert to the envelope-frozen form."""
    profile = get_profile(profile_name)
    bundle = _load_lexicons(profile)
    return EnvelopeLexicons(
        acknowledgment_phrases=bundle.acknowledgment_phrases,
        emotion_lexicon=bundle.emotion_lexicon,
        apology_lexicon=bundle.apology_lexicon,
        action_verb_lexicon=bundle.action_verb_lexicon,
        hedging_lexicon=bundle.hedging_lexicon,
        directive_lexicon=bundle.directive_lexicon,
        warm_terms=bundle.warm_terms,
        clinical_terms=bundle.clinical_terms,
        contraction_patterns=bundle.contraction_patterns,
        resolution_patterns=bundle.resolution_patterns,
        deflection_patterns=bundle.deflection_patterns,
        next_steps_patterns=bundle.next_steps_patterns,
        temporal_anchor_patterns=bundle.temporal_anchor_patterns,
        specific_actor_patterns=bundle.specific_actor_patterns,
        ownership_patterns=bundle.ownership_patterns,
        issue_keywords=bundle.issue_keywords,
        synonym_map=bundle.synonym_map,
    )


def _brand_voice_config(profile_name: str) -> BrandVoiceConfig:
    """Build a frozen BrandVoiceConfig from the profile's feature profiles."""
    profile = get_profile(profile_name)
    # Variant ID from profile; fall back to bv_baseline
    variant_id = getattr(profile, "brand_voice_variant_id", lambda: "bv_baseline")()
    feature_profiles = getattr(profile, "brand_voice_feature_profiles", lambda: {})()
    return BrandVoiceConfig(variant_id=variant_id, feature_profiles=feature_profiles)


def _write_label_template(
    output_dir: Path,
    conv_id: str,
    *,
    skipped_during_generation: bool = False,
) -> None:
    """Write an empty labels.json template if one doesn't already exist."""
    label_path = output_dir / "labels.json"
    if label_path.exists():
        return  # never overwrite human-authored labels

    tags = ["skipped_during_generation"] if skipped_during_generation else []
    template = {
        "schema_version": _SCHEMA_VERSION,
        "conv_id": conv_id,
        "label_version": 1,
        "labeled_at": "FILL_IN",
        "labeled_by": _LABEL_TEMPLATE_LABELED_BY,
        "expected_outcome": "uncertain",
        "expected_failures": {
            "accuracy": False,
            "empathy": False,
            "resolution": False,
            "brand_voice": False,
            "claim_extraction": False,
        },
        "confidence": "low",
        "tags": tags,
        "rationale": "FILL_IN",
        "revised_from": None,
        "revision_notes": None,
    }
    label_path.write_text(json.dumps(template, indent=2), encoding="utf-8")


def extract_envelopes(
    profile_dir: Path,
    replay_corpus_dir: Path,
    *,
    profile_name: str,
    anthropic_client: Any,
    conv_filter: Optional[list[str]] = None,
    include_skipped: bool = False,
    progress_callback: Optional[Callable[[str, int, int], None]] = None,
) -> list[str]:
    """
    Extract replay envelopes from an existing smoke corpus output directory.

    For each ConversationRecord in conversations.jsonl:
      1. Reads the ConversationRecord prose, splits into agent/customer
      2. Matches the conv_id to its QualityPlan (planted cases only; organic
         conversations without a plan still get an envelope)
      3. Calls extract_claims_llm() once — the only paid LLM step
      4. Writes {replay_corpus_dir}/{conv_id}/envelope.json
      5. Writes {replay_corpus_dir}/{conv_id}/labels.json template (if absent)

    When include_skipped=True, also processes skipped_conversations.jsonl.
    Only skipped records with final_prose (POST-GEN SKIP path) are extractable;
    API-error skips with no prose are silently skipped. labels.json templates
    for skipped convs are tagged with "skipped_during_generation".

    Args:
        profile_dir: path to the profile's output directory (e.g. corpus/saas)
        replay_corpus_dir: root of the replay corpus (e.g. replay_corpus/saas)
        profile_name: profile identifier ("saas", "ps", etc.)
        anthropic_client: live Anthropic client for extract_claims_llm
        conv_filter: if provided, only extract these conv_ids
        include_skipped: if True, also extract from skipped_conversations.jsonl
        progress_callback: called as (conv_id, index_1based, total) per conversation

    Returns:
        List of conv_ids that were successfully extracted
    """
    manifest = _read_manifest(profile_dir)
    conversations = _read_conversations(profile_dir)
    quality_plans = _read_quality_plans(profile_dir)
    kb_chunks = _read_kb_chunks(profile_dir)

    lexicons = _lexicons_from_profile(profile_name)
    bv_config = _brand_voice_config(profile_name)

    pipeline_version = manifest.generator_version if manifest else ""
    kb_version = manifest.kb_version if manifest else ""
    extraction_ts = datetime.now(timezone.utc).isoformat()

    if conv_filter:
        filter_set = set(conv_filter)
        conversations = [c for c in conversations if c.conversation_id in filter_set]

    # Load skipped convs upfront so total count is correct for progress reporting.
    # Only records with final_prose are extractable (POST-GEN SKIP path).
    skipped_to_process: list[SkippedConversationRecord] = []
    if include_skipped:
        all_skipped = _read_skipped_conversations(profile_dir)
        if conv_filter:
            filter_set_s = set(conv_filter)
            all_skipped = [s for s in all_skipped if s.conversation_id in filter_set_s]
        skipped_to_process = [s for s in all_skipped if s.final_prose is not None]

    extracted: list[str] = []
    total = len(conversations) + len(skipped_to_process)

    for idx, conv_record in enumerate(conversations, 1):
        conv_id = conv_record.conversation_id
        if progress_callback:
            progress_callback(conv_id, idx, total)

        agent_prose, customer_prose = _split_prose_turns(conv_record.prose)

        # Match to quality plan (None for organic conversations)
        quality_plan = quality_plans.get(conv_id)

        # If this conversation has no quality plan, create a stub one.
        # Organic conversations need a QualityPlan for the envelope schema but
        # have no rubric targets or kb requirements.
        if quality_plan is None:
            quality_plan = QualityPlan(
                conversation_id=conv_id,
                trigger_event_id=conv_record.trigger_event_id,
                rubric_targets=RubricTarget(),
                knowledge_citations=KnowledgeCitations(),
                prose_generation_directives="(organic — no quality plan)",
                kb_chunks_required=[],
            )

        # Extract claims (the one LLM call per conversation)
        extracted_claims: list[Claim] = extract_claims_llm(
            agent_prose=agent_prose,
            anthropic_client=anthropic_client,
        )

        claims_payload = [c.model_dump(mode="json") for c in extracted_claims]
        claims_canonical = json.dumps(
            claims_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        claims_hash = "sha256:" + hashlib.sha256(claims_canonical.encode("utf-8")).hexdigest()

        extraction_meta = ExtractionMeta(
            model="claude-haiku-4-5-20251001",
            prompt_version=_PROMPT_VERSION,
            extracted_at=extraction_ts,
            claims_hash=claims_hash,
        )

        # Build envelope
        envelope = ReplayEnvelope(
            schema_version=_SCHEMA_VERSION,
            conv_id=conv_id,
            agent_prose=agent_prose,
            customer_prose=customer_prose,
            quality_plan=quality_plan,
            kb_chunks=kb_chunks,
            lexicons=lexicons,
            brand_voice=bv_config,
            validator_inputs=ValidatorInputs(
                accuracy=AccuracyValidatorInputs(
                    extracted_claims=extracted_claims,
                    extraction_meta=extraction_meta,
                )
            ),
            metadata=EnvelopeMetadata(
                conv_id=conv_id,
                source_corpus=str(profile_dir),
                pipeline_version=pipeline_version,
                kb_version=kb_version,
                generator_version=pipeline_version,
                extraction_timestamp=extraction_ts,
            ),
        )

        # Write envelope
        output_dir = replay_corpus_dir / conv_id
        output_dir.mkdir(parents=True, exist_ok=True)
        envelope_path = output_dir / "envelope.json"
        envelope_path.write_text(
            json.dumps(envelope.model_dump(mode="json"), indent=2),
            encoding="utf-8",
        )

        # Write label template (skipped if labels.json already exists)
        _write_label_template(output_dir, conv_id)

        extracted.append(conv_id)

    # Process skipped conversations (--include-skipped path).
    # These use SkippedConversationRecord which has final_prose/event_id instead
    # of prose/trigger_event_id. labels.json is tagged "skipped_during_generation".
    for idx, skipped_record in enumerate(skipped_to_process, len(conversations) + 1):
        conv_id = skipped_record.conversation_id
        if progress_callback:
            progress_callback(conv_id, idx, total)

        agent_prose, customer_prose = _split_prose_turns(skipped_record.final_prose)  # type: ignore[arg-type]

        quality_plan = quality_plans.get(conv_id)
        if quality_plan is None:
            quality_plan = QualityPlan(
                conversation_id=conv_id,
                trigger_event_id=skipped_record.event_id or conv_id,
                rubric_targets=RubricTarget(),
                knowledge_citations=KnowledgeCitations(),
                prose_generation_directives="(skipped during generation — no quality plan resolved)",
                kb_chunks_required=[],
            )

        extracted_claims = extract_claims_llm(
            agent_prose=agent_prose,
            anthropic_client=anthropic_client,
        )

        claims_payload = [c.model_dump(mode="json") for c in extracted_claims]
        claims_canonical = json.dumps(
            claims_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        claims_hash = "sha256:" + hashlib.sha256(claims_canonical.encode("utf-8")).hexdigest()

        extraction_meta = ExtractionMeta(
            model="claude-haiku-4-5-20251001",
            prompt_version=_PROMPT_VERSION,
            extracted_at=extraction_ts,
            claims_hash=claims_hash,
        )

        envelope = ReplayEnvelope(
            schema_version=_SCHEMA_VERSION,
            conv_id=conv_id,
            agent_prose=agent_prose,
            customer_prose=customer_prose,
            quality_plan=quality_plan,
            kb_chunks=kb_chunks,
            lexicons=lexicons,
            brand_voice=bv_config,
            validator_inputs=ValidatorInputs(
                accuracy=AccuracyValidatorInputs(
                    extracted_claims=extracted_claims,
                    extraction_meta=extraction_meta,
                )
            ),
            metadata=EnvelopeMetadata(
                conv_id=conv_id,
                source_corpus=str(profile_dir),
                pipeline_version=pipeline_version,
                kb_version=kb_version,
                generator_version=pipeline_version,
                extraction_timestamp=extraction_ts,
            ),
        )

        output_dir = replay_corpus_dir / conv_id
        output_dir.mkdir(parents=True, exist_ok=True)
        envelope_path = output_dir / "envelope.json"
        envelope_path.write_text(
            json.dumps(envelope.model_dump(mode="json"), indent=2),
            encoding="utf-8",
        )

        _write_label_template(output_dir, conv_id, skipped_during_generation=True)

        extracted.append(conv_id)

    return extracted
