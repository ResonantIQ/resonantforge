"""
Tests for inline envelope writing during corpus generation (RFORGE-41).

Root cause: envelopes have always required a separate post-hoc
`rforge replay extract-envelopes` step that re-runs claim extraction, paying
LLM costs we already paid during generation. The smoke pipeline already calls
extract_claims_llm for planted conversations; claims are in AccuracySignals.claims
when the conversation passes. We should write the envelope then — zero extra cost.

Fix (initial):
  - Extract write_single_envelope() from extractor.py so both the post-hoc path
    and the inline path share the same logic.
  - PipelineConfig gains replay_corpus_dir: Path | None (default None).
  - rforge generate adds --replay-out that defaults to ./replay_corpus.
  - _generate_prose_for_chunk writes an envelope for every accepted conversation:
      - Planted conversations: claims from accuracy_signals.claims (already extracted).
      - Organic conversations: empty claims list (replay still covers non-accuracy dims).

Fix (hardening):
  - Skip-path: write envelope for POST-GEN SKIP conversations with
    skipped_during_generation=True and partial claims from the last attempt.
    EnvelopeMetadata gains skipped_during_generation field.
  - Disk-write isolation: OSError in write_single_envelope never cascades to the
    smoke run; conversation record durability is independent of envelope durability.
  - kb_version unification: use chunk-content hash (sorted chunk_id + chunk_text,
    post-contamination) for both inline envelopes and the manifest. Retire the
    JSONL-file-SHA path which is brittle to serialisation changes.

Covers:
  - write_single_envelope creates envelope.json with correct schema fields
  - write_single_envelope uses provided claims directly (no LLM call)
  - empty claims list is preserved (organic path)
  - labels.json template is created when absent
  - labels.json is never overwritten (human labels preserved)
  - PipelineConfig accepts replay_corpus_dir
  - rforge generate CLI accepts --replay-out and --no-replay
  - EnvelopeMetadata has skipped_during_generation field (defaults False)
  - write_single_envelope skipped_during_generation=True → metadata flag set
  - OSError during write_single_envelope is caught; smoke continues
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from resonantforge.pipeline import PipelineConfig
from resonantforge.replay.extractor import write_single_envelope
from resonantforge.replay.schemas import BrandVoiceConfig, EnvelopeLexicons
from resonantforge.schemas import (
    Claim,
    ConstraintType,
    KBChunk,
    KnowledgeCitations,
    QualityPlan,
    RubricTarget,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────


def _minimal_quality_plan(conv_id: str = "conv_test_001") -> QualityPlan:
    return QualityPlan(
        conversation_id=conv_id,
        trigger_event_id="evt_test_001",
        rubric_targets=RubricTarget(),
        knowledge_citations=KnowledgeCitations(),
        prose_generation_directives="(test plan)",
        kb_chunks_required=["kb_refund_allow"],
    )


def _minimal_kb_chunk() -> KBChunk:
    return KBChunk(
        chunk_id="kb_refund_allow",
        document_id="doc_refund",
        document_path="policies/refund.md",
        chunk_text="Refunds are available within 30 days of purchase.",
        constraint_type=ConstraintType.ALLOW_CONDITION,
        domains=["refund_policy"],
    )


def _minimal_lexicons() -> EnvelopeLexicons:
    return EnvelopeLexicons(
        acknowledgment_phrases=["I understand"],
        emotion_lexicon=["frustrated"],
        apology_lexicon=["sorry"],
        action_verb_lexicon=["help"],
        hedging_lexicon=["might"],
        directive_lexicon=["please"],
        warm_terms=["happy"],
        clinical_terms=["terminate"],
        contraction_patterns=["don't"],
        resolution_patterns=["resolved"],
        deflection_patterns=["contact us"],
        next_steps_patterns=["I will"],
        temporal_anchor_patterns=["today"],
        specific_actor_patterns=["our team"],
        ownership_patterns=["I'll handle"],
        issue_keywords=["issue"],
        completion_verb_patterns=["fixed"],
        completion_resolution_patterns=[r"\bi'?ve fixed\b"],
        synonym_map={"reimburse": "refund"},
    )


def _minimal_bv_config() -> BrandVoiceConfig:
    return BrandVoiceConfig(
        variant_id="bv_warm_exploratory",
        feature_profiles={
            "bv_warm_exploratory": {
                "sentence_length_range": [6, 20],
                "question_count_range": [0, 4],
                "hedging_count_range": [0, 2],
                "directive_count_range": [0, 3],
                "warm_terms_min": 0,
                "clinical_terms_max": 1,
                "contractions_allowed": True,
                "formality_score_range": [0.3, 0.8],
            }
        },
    )


def _one_claim() -> Claim:
    return Claim(
        claim_text="Refunds are available within 30 days.",
        claim_span=(0, 38),
        claim_type="policy",
        normalized_subject="refund",
        normalized_predicate="are available",
        normalized_object="refund days",
        alignment=None,
    )


# ── write_single_envelope ─────────────────────────────────────────────────────


def test_write_single_envelope_creates_envelope_json(tmp_path: Path) -> None:
    """write_single_envelope writes envelope.json with schema_version and conv_id."""
    write_single_envelope(
        output_dir=tmp_path / "conv_test_001",
        conv_id="conv_test_001",
        agent_prose="Hi, your refund will be processed within 30 days.",
        customer_prose="Can I get a refund?",
        quality_plan=_minimal_quality_plan(),
        kb_chunks=[_minimal_kb_chunk()],
        lexicons=_minimal_lexicons(),
        brand_voice=_minimal_bv_config(),
        extracted_claims=[_one_claim()],
        pipeline_version="0.2.0",
        kb_version="abc123",
        source_corpus="corpus/saas",
    )
    envelope_path = tmp_path / "conv_test_001" / "envelope.json"
    assert envelope_path.exists(), "envelope.json must be created"
    data = json.loads(envelope_path.read_text())
    assert data["schema_version"] == 1
    assert data["conv_id"] == "conv_test_001"


def test_write_single_envelope_persists_provided_claims(tmp_path: Path) -> None:
    """Claims passed to write_single_envelope are written verbatim — no LLM call."""
    claim = _one_claim()
    write_single_envelope(
        output_dir=tmp_path / "conv_test_001",
        conv_id="conv_test_001",
        agent_prose="Refunds are available within 30 days.",
        customer_prose="I want a refund.",
        quality_plan=_minimal_quality_plan(),
        kb_chunks=[_minimal_kb_chunk()],
        lexicons=_minimal_lexicons(),
        brand_voice=_minimal_bv_config(),
        extracted_claims=[claim],
        pipeline_version="0.2.0",
        kb_version="abc123",
        source_corpus="corpus/saas",
    )
    data = json.loads((tmp_path / "conv_test_001" / "envelope.json").read_text())
    claims = data["validator_inputs"]["accuracy"]["extracted_claims"]
    assert len(claims) == 1
    assert claims[0]["claim_text"] == claim.claim_text


def test_write_single_envelope_empty_claims_for_organic(tmp_path: Path) -> None:
    """Organic conversations pass extracted_claims=[] — envelope is still valid."""
    write_single_envelope(
        output_dir=tmp_path / "conv_organic",
        conv_id="conv_organic",
        agent_prose="Happy to help today.",
        customer_prose="Hello.",
        quality_plan=QualityPlan(
            conversation_id="conv_organic",
            trigger_event_id="evt_organic",
            rubric_targets=RubricTarget(),
            knowledge_citations=KnowledgeCitations(),
            prose_generation_directives="(organic — no quality plan)",
            kb_chunks_required=[],
        ),
        kb_chunks=[],
        lexicons=_minimal_lexicons(),
        brand_voice=_minimal_bv_config(),
        extracted_claims=[],
        pipeline_version="0.2.0",
        kb_version="abc123",
        source_corpus="corpus/saas",
    )
    data = json.loads((tmp_path / "conv_organic" / "envelope.json").read_text())
    assert data["validator_inputs"]["accuracy"]["extracted_claims"] == []


def test_write_single_envelope_creates_label_template(tmp_path: Path) -> None:
    """write_single_envelope creates a labels.json template when absent."""
    write_single_envelope(
        output_dir=tmp_path / "conv_test_001",
        conv_id="conv_test_001",
        agent_prose="Refund in 30 days.",
        customer_prose="Refund please.",
        quality_plan=_minimal_quality_plan(),
        kb_chunks=[_minimal_kb_chunk()],
        lexicons=_minimal_lexicons(),
        brand_voice=_minimal_bv_config(),
        extracted_claims=[],
        pipeline_version="0.2.0",
        kb_version="abc123",
        source_corpus="corpus/saas",
    )
    labels_path = tmp_path / "conv_test_001" / "labels.json"
    assert labels_path.exists(), "labels.json template must be created"
    data = json.loads(labels_path.read_text())
    assert data["conv_id"] == "conv_test_001"
    assert data["schema_version"] == 1


def test_write_single_envelope_does_not_overwrite_existing_labels(tmp_path: Path) -> None:
    """write_single_envelope never overwrites an existing labels.json."""
    output_dir = tmp_path / "conv_test_001"
    output_dir.mkdir(parents=True)
    labels_path = output_dir / "labels.json"
    human_label = {"human": "authored", "expected_outcome": "pass"}
    labels_path.write_text(json.dumps(human_label))

    write_single_envelope(
        output_dir=output_dir,
        conv_id="conv_test_001",
        agent_prose="Refund in 30 days.",
        customer_prose="Refund please.",
        quality_plan=_minimal_quality_plan(),
        kb_chunks=[_minimal_kb_chunk()],
        lexicons=_minimal_lexicons(),
        brand_voice=_minimal_bv_config(),
        extracted_claims=[],
        pipeline_version="0.2.0",
        kb_version="abc123",
        source_corpus="corpus/saas",
    )
    after = json.loads(labels_path.read_text())
    assert after == human_label, "Human-authored labels.json must never be overwritten"


def test_write_single_envelope_contains_quality_plan(tmp_path: Path) -> None:
    """Quality plan fields are preserved in the envelope for rule engine replay."""
    plan = _minimal_quality_plan("conv_test_002")
    write_single_envelope(
        output_dir=tmp_path / "conv_test_002",
        conv_id="conv_test_002",
        agent_prose="Refund in 30 days.",
        customer_prose="Refund please.",
        quality_plan=plan,
        kb_chunks=[_minimal_kb_chunk()],
        lexicons=_minimal_lexicons(),
        brand_voice=_minimal_bv_config(),
        extracted_claims=[],
        pipeline_version="0.2.0",
        kb_version="abc123",
        source_corpus="corpus/saas",
    )
    data = json.loads((tmp_path / "conv_test_002" / "envelope.json").read_text())
    assert data["quality_plan"]["conversation_id"] == "conv_test_002"
    assert data["quality_plan"]["kb_chunks_required"] == ["kb_refund_allow"]


# ── PipelineConfig ────────────────────────────────────────────────────────────


def test_pipeline_config_accepts_replay_corpus_dir(tmp_path: Path) -> None:
    """PipelineConfig has an optional replay_corpus_dir field."""
    cfg = PipelineConfig(
        profile_name="saas",
        accounts=2,
        months=1,
        seed=42,
        output_root=tmp_path / "corpus",
        replay_corpus_dir=tmp_path / "replay_corpus",
    )
    assert cfg.replay_corpus_dir == tmp_path / "replay_corpus"


def test_pipeline_config_replay_corpus_dir_defaults_to_none() -> None:
    """PipelineConfig.replay_corpus_dir defaults to None (opt-in)."""
    cfg = PipelineConfig(
        profile_name="saas",
        accounts=2,
        months=1,
        seed=42,
        output_root=Path("/tmp/corpus"),
    )
    assert cfg.replay_corpus_dir is None


# ── CLI ───────────────────────────────────────────────────────────────────────


def test_generate_cli_accepts_replay_out_flag() -> None:
    """rforge generate --replay-out is a valid CLI flag."""
    from click.testing import CliRunner
    from resonantforge.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["generate", "--help"])
    assert "--replay-out" in result.output, (
        "--replay-out flag must be present in rforge generate --help output"
    )


def test_generate_cli_no_replay_flag() -> None:
    """rforge generate --no-replay is a valid CLI flag that disables envelope writing."""
    from click.testing import CliRunner
    from resonantforge.cli import cli

    runner = CliRunner()
    result = runner.invoke(cli, ["generate", "--help"])
    assert "--no-replay" in result.output, (
        "--no-replay flag must be present in rforge generate --help output"
    )


# ── Hardening: skip-path, disk-write isolation, kb_version ───────────────────


def test_envelope_metadata_has_skipped_during_generation_field() -> None:
    """EnvelopeMetadata has skipped_during_generation: bool, defaulting to False."""
    from resonantforge.replay.schemas import EnvelopeMetadata

    meta = EnvelopeMetadata(conv_id="conv_test")
    assert hasattr(meta, "skipped_during_generation")
    assert meta.skipped_during_generation is False


def test_write_single_envelope_skipped_flag_in_metadata(tmp_path: Path) -> None:
    """skipped_during_generation=True is written into envelope.json metadata."""
    write_single_envelope(
        output_dir=tmp_path / "conv_skip",
        conv_id="conv_skip",
        agent_prose="I'm sorry, we can't help with that.",
        customer_prose="I need a refund.",
        quality_plan=_minimal_quality_plan("conv_skip"),
        kb_chunks=[_minimal_kb_chunk()],
        lexicons=_minimal_lexicons(),
        brand_voice=_minimal_bv_config(),
        extracted_claims=[],
        pipeline_version="0.2.0",
        kb_version="abc123",
        source_corpus="corpus/saas",
        skipped_during_generation=True,
    )
    data = json.loads((tmp_path / "conv_skip" / "envelope.json").read_text())
    assert data["metadata"]["skipped_during_generation"] is True, (
        "envelope.json metadata must carry skipped_during_generation=True for skip-path envelopes"
    )


def test_write_single_envelope_skipped_false_by_default(tmp_path: Path) -> None:
    """skipped_during_generation defaults to False in envelope metadata."""
    write_single_envelope(
        output_dir=tmp_path / "conv_pass",
        conv_id="conv_pass",
        agent_prose="Refund in 30 days.",
        customer_prose="Refund please.",
        quality_plan=_minimal_quality_plan("conv_pass"),
        kb_chunks=[_minimal_kb_chunk()],
        lexicons=_minimal_lexicons(),
        brand_voice=_minimal_bv_config(),
        extracted_claims=[_one_claim()],
        pipeline_version="0.2.0",
        kb_version="abc123",
        source_corpus="corpus/saas",
    )
    data = json.loads((tmp_path / "conv_pass" / "envelope.json").read_text())
    assert data["metadata"]["skipped_during_generation"] is False


def test_write_single_envelope_oserror_does_not_propagate(tmp_path: Path) -> None:
    """OSError from write_single_envelope must not propagate to callers."""
    # Make the output directory a file so mkdir inside write_single_envelope fails.
    bad_dir = tmp_path / "conv_bad"
    bad_dir.write_text("I am a file, not a directory")

    # Should not raise — caller must be able to continue after envelope write failure.
    try:
        write_single_envelope(
            output_dir=bad_dir,
            conv_id="conv_bad",
            agent_prose="Hello.",
            customer_prose="Hi.",
            quality_plan=_minimal_quality_plan("conv_bad"),
            kb_chunks=[],
            lexicons=_minimal_lexicons(),
            brand_voice=_minimal_bv_config(),
            extracted_claims=[],
            pipeline_version="0.2.0",
            kb_version="abc123",
            source_corpus="corpus/saas",
        )
        raised = False
    except OSError:
        raised = True

    assert not raised, (
        "write_single_envelope must catch OSError internally — callers must not be "
        "interrupted by transient disk failures"
    )


def test_kb_version_chunk_content_hash_is_stable(tmp_path: Path) -> None:
    """chunk-content kb_version hash is identical for same chunks regardless of dict ordering."""
    from resonantforge.pipeline import compute_kb_version

    chunks = [_minimal_kb_chunk()]
    h1 = compute_kb_version(chunks)
    h2 = compute_kb_version(list(reversed(chunks)))  # order-independent
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex digest


# ── RFORGE-6: LexiconsBundle carries completion_verb/resolution_patterns ──────


def test_lexicons_bundle_has_completion_verb_patterns():
    """LexiconsBundle must have completion_verb_patterns field."""
    from resonantforge.pipeline import LexiconsBundle
    bundle = LexiconsBundle.__dataclass_fields__ if hasattr(LexiconsBundle, '__dataclass_fields__') else LexiconsBundle.model_fields
    assert "completion_verb_patterns" in bundle, (
        "LexiconsBundle must have completion_verb_patterns field"
    )


def test_lexicons_bundle_has_completion_resolution_patterns():
    """LexiconsBundle must have completion_resolution_patterns field."""
    from resonantforge.pipeline import LexiconsBundle
    bundle = LexiconsBundle.__dataclass_fields__ if hasattr(LexiconsBundle, '__dataclass_fields__') else LexiconsBundle.model_fields
    assert "completion_resolution_patterns" in bundle, (
        "LexiconsBundle must have completion_resolution_patterns field"
    )


def test_load_lexicons_populates_completion_verb_patterns():
    """_load_lexicons must populate completion_verb_patterns from profile."""
    from resonantforge.pipeline import _load_lexicons
    from resonantforge.profiles.saas import SaaSProfile
    bundle = _load_lexicons(SaaSProfile())
    assert isinstance(bundle.completion_verb_patterns, list)
    assert len(bundle.completion_verb_patterns) > 0


def test_load_lexicons_populates_completion_resolution_patterns():
    """_load_lexicons must populate completion_resolution_patterns from profile."""
    from resonantforge.pipeline import _load_lexicons
    from resonantforge.profiles.saas import SaaSProfile
    bundle = _load_lexicons(SaaSProfile())
    assert isinstance(bundle.completion_resolution_patterns, list)
    assert len(bundle.completion_resolution_patterns) > 0


def test_envelope_lexicons_has_completion_verb_patterns():
    """EnvelopeLexicons must have completion_verb_patterns field."""
    from resonantforge.replay.schemas import EnvelopeLexicons
    lex = EnvelopeLexicons()
    assert hasattr(lex, "completion_verb_patterns"), (
        "EnvelopeLexicons must have completion_verb_patterns"
    )
    assert isinstance(lex.completion_verb_patterns, list)


def test_envelope_lexicons_has_completion_resolution_patterns():
    """EnvelopeLexicons must have completion_resolution_patterns field."""
    from resonantforge.replay.schemas import EnvelopeLexicons
    lex = EnvelopeLexicons()
    assert hasattr(lex, "completion_resolution_patterns"), (
        "EnvelopeLexicons must have completion_resolution_patterns"
    )
    assert isinstance(lex.completion_resolution_patterns, list)
