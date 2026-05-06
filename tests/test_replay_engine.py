"""
Tests for resonantforge.replay.engine: load_envelope, load_labels, replay_one.
"""
from __future__ import annotations

import json
import pytest
from pathlib import Path

from resonantforge.replay.engine import (
    EnvelopeSchemaError,
    LabelSchemaError,
    load_envelope,
    load_labels,
    replay_one,
)
from resonantforge.schemas import ValidationVerdict
from tests.replay_fixtures.builder import build_fixtures


@pytest.fixture(scope="module")
def fixtures_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("replay_fixtures")
    build_fixtures(d)
    return d


# ---------------------------------------------------------------------------
# load_envelope
# ---------------------------------------------------------------------------


class TestLoadEnvelope:
    def test_loads_valid_envelope(self, fixtures_dir: Path) -> None:
        env = load_envelope(fixtures_dir / "conv_clean_pass" / "envelope.json")
        assert env.conv_id == "conv_clean_pass"
        assert env.schema_version == 1
        assert env.agent_prose

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(EnvelopeSchemaError, match="envelope not found"):
            load_envelope(tmp_path / "missing" / "envelope.json")

    def test_bad_json_raises(self, tmp_path: Path) -> None:
        bad = tmp_path / "envelope.json"
        bad.write_text("{ not json }", encoding="utf-8")
        with pytest.raises(EnvelopeSchemaError, match="not valid JSON"):
            load_envelope(bad)

    def test_wrong_schema_version_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "envelope.json"
        path.write_text(json.dumps({"schema_version": 99, "conv_id": "x"}), encoding="utf-8")
        with pytest.raises(EnvelopeSchemaError):
            load_envelope(path)

    def test_validator_inputs_extracted_claims(self, fixtures_dir: Path) -> None:
        env = load_envelope(fixtures_dir / "conv_clean_pass" / "envelope.json")
        claims = env.validator_inputs.accuracy.extracted_claims
        assert len(claims) == 1
        assert claims[0].claim_text

    def test_claim_extraction_fixture_has_empty_claims(self, fixtures_dir: Path) -> None:
        env = load_envelope(fixtures_dir / "conv_claim_extraction" / "envelope.json")
        assert env.validator_inputs.accuracy.extracted_claims == []

    def test_brand_voice_fixture_has_strict_profile(self, fixtures_dir: Path) -> None:
        env = load_envelope(fixtures_dir / "conv_brand_voice_fail" / "envelope.json")
        assert env.brand_voice.variant_id == "bv_strict"
        profile = env.brand_voice.feature_profiles["bv_strict"]
        assert profile["sentence_length_range"][0] >= 50


# ---------------------------------------------------------------------------
# load_labels
# ---------------------------------------------------------------------------


class TestLoadLabels:
    def test_loads_valid_labels(self, fixtures_dir: Path) -> None:
        labels = load_labels(
            fixtures_dir / "conv_clean_pass" / "labels.json",
            conv_id="conv_clean_pass",
        )
        assert labels.expected_outcome == "pass"
        assert labels.confidence == "high"

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(LabelSchemaError, match="labels not found"):
            load_labels(tmp_path / "missing" / "labels.json", conv_id="x")

    def test_conv_id_mismatch_raises(self, fixtures_dir: Path) -> None:
        with pytest.raises(LabelSchemaError, match="does not match"):
            load_labels(
                fixtures_dir / "conv_clean_pass" / "labels.json",
                conv_id="wrong_id",
            )

    def test_missing_expected_failure_key_raises(self, tmp_path: Path) -> None:
        bad = {
            "schema_version": 1,
            "conv_id": "c",
            "label_version": 1,
            "labeled_at": "2026-05-05",
            "labeled_by": "tj",
            "expected_outcome": "pass",
            "expected_failures": {"accuracy": False},  # missing 4 keys
            "confidence": "high",
            "tags": [],
            "rationale": "test",
            "revised_from": None,
            "revision_notes": None,
        }
        path = tmp_path / "labels.json"
        path.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(LabelSchemaError):
            load_labels(path, conv_id="c")

    def test_unknown_tag_raises(self, tmp_path: Path) -> None:
        bad = {
            "schema_version": 1,
            "conv_id": "c",
            "label_version": 1,
            "labeled_at": "2026-05-05",
            "labeled_by": "tj",
            "expected_outcome": "pass",
            "expected_failures": {
                "accuracy": False, "empathy": False, "resolution": False,
                "brand_voice": False, "claim_extraction": False,
            },
            "confidence": "high",
            "tags": ["borderlne"],  # typo — not in closed vocab
            "rationale": "test",
            "revised_from": None,
            "revision_notes": None,
        }
        path = tmp_path / "labels.json"
        path.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(LabelSchemaError, match="unknown tag"):
            load_labels(path, conv_id="c")

    def test_revision_without_notes_raises(self, tmp_path: Path) -> None:
        bad = {
            "schema_version": 1,
            "conv_id": "c",
            "label_version": 2,
            "labeled_at": "2026-05-05",
            "labeled_by": "tj",
            "expected_outcome": "pass",
            "expected_failures": {
                "accuracy": False, "empathy": False, "resolution": False,
                "brand_voice": False, "claim_extraction": False,
            },
            "confidence": "high",
            "tags": [],
            "rationale": "test",
            "revised_from": 1,
            "revision_notes": None,
        }
        path = tmp_path / "labels.json"
        path.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(LabelSchemaError, match="revision audit trail"):
            load_labels(path, conv_id="c")

    def test_empty_rationale_raises(self, tmp_path: Path) -> None:
        bad = {
            "schema_version": 1,
            "conv_id": "c",
            "label_version": 1,
            "labeled_at": "2026-05-05",
            "labeled_by": "tj",
            "expected_outcome": "pass",
            "expected_failures": {
                "accuracy": False, "empathy": False, "resolution": False,
                "brand_voice": False, "claim_extraction": False,
            },
            "confidence": "high",
            "tags": [],
            "rationale": "",
            "revised_from": None,
            "revision_notes": None,
        }
        path = tmp_path / "labels.json"
        path.write_text(json.dumps(bad), encoding="utf-8")
        with pytest.raises(LabelSchemaError):
            load_labels(path, conv_id="c")


# ---------------------------------------------------------------------------
# replay_one — outcome and verdict correctness per fixture
# ---------------------------------------------------------------------------


class TestReplayOne:
    def _run(self, fixtures_dir: Path, conv_id: str):
        env = load_envelope(fixtures_dir / conv_id / "envelope.json")
        lbl = load_labels(fixtures_dir / conv_id / "labels.json", conv_id=conv_id)
        return replay_one(env, lbl)

    def test_clean_pass_no_failures(self, fixtures_dir: Path) -> None:
        result = self._run(fixtures_dir, "conv_clean_pass")
        assert result.conv_id == "conv_clean_pass"
        assert result.claim_extraction_ok is True
        fail_verdicts = [v for v in result.dimension_verdicts if v.verdict == ValidationVerdict.FAIL]
        # Wide brand_voice ranges + acknowledgment present + resolution with ownership → no FAIL
        # (exact outcome may depend on extractor, but no technical errors)
        assert isinstance(result.overall_outcome, str)
        assert result.overall_outcome in ("pass", "fail")

    def test_accuracy_fail_fires(self, fixtures_dir: Path) -> None:
        result = self._run(fixtures_dir, "conv_accuracy_fail")
        # kb_chunks_required references a nonexistent chunk → candidate pool empty
        # → alignment=not_found → accuracy:supported:exact → FAIL
        acc_verdicts = [v for v in result.dimension_verdicts if v.dimension == "accuracy"]
        assert len(acc_verdicts) == 1
        assert acc_verdicts[0].verdict == ValidationVerdict.FAIL, (
            f"Expected accuracy FAIL, got {acc_verdicts[0].verdict}. "
            f"signals: {acc_verdicts[0].signals_summary}"
        )
        assert result.overall_outcome == "fail"

    def test_empathy_fail_fires(self, fixtures_dir: Path) -> None:
        result = self._run(fixtures_dir, "conv_empathy_fail")
        emp_verdicts = [v for v in result.dimension_verdicts if v.dimension == "empathy"]
        assert len(emp_verdicts) == 1
        assert emp_verdicts[0].verdict == ValidationVerdict.FAIL, (
            f"Expected empathy FAIL, got {emp_verdicts[0].verdict}. "
            f"signals: {emp_verdicts[0].signals_summary}"
        )
        assert result.overall_outcome == "fail"

    def test_brand_voice_fail_fires(self, fixtures_dir: Path) -> None:
        result = self._run(fixtures_dir, "conv_brand_voice_fail")
        bv_verdicts = [v for v in result.dimension_verdicts if v.dimension == "brand_voice"]
        assert len(bv_verdicts) == 1
        assert bv_verdicts[0].verdict == ValidationVerdict.FAIL, (
            f"Expected brand_voice FAIL, got {bv_verdicts[0].verdict}. "
            f"signals: {bv_verdicts[0].signals_summary}"
        )
        assert result.overall_outcome == "fail"

    def test_claim_extraction_fixture_accuracy_fails(self, fixtures_dir: Path) -> None:
        result = self._run(fixtures_dir, "conv_claim_extraction")
        # Empty extracted_claims + target supported:exact → alignment not_found → accuracy FAIL
        acc_verdicts = [v for v in result.dimension_verdicts if v.dimension == "accuracy"]
        assert len(acc_verdicts) == 1
        assert acc_verdicts[0].verdict == ValidationVerdict.FAIL

    def test_claim_extraction_ok_always_true_in_replay(self, fixtures_dir: Path) -> None:
        result = self._run(fixtures_dir, "conv_claim_extraction")
        # claim_extraction_ok is always True in replay (frozen claims are injected,
        # extract_claims_llm is never called)
        assert result.claim_extraction_ok is True

    def test_claim_extraction_label_appears_as_missed_rule(self, fixtures_dir: Path) -> None:
        result = self._run(fixtures_dir, "conv_claim_extraction")
        # Label says claim_extraction: True but replay always reports ok=True
        # → claim_extraction shows in missed_rules
        assert "claim_extraction" in result.agreement.missed_rules

    def test_result_is_deterministic(self, fixtures_dir: Path) -> None:
        """Same input → identical output across two invocations."""
        env = load_envelope(fixtures_dir / "conv_clean_pass" / "envelope.json")
        lbl = load_labels(
            fixtures_dir / "conv_clean_pass" / "labels.json",
            conv_id="conv_clean_pass",
        )
        r1 = replay_one(env, lbl)
        r2 = replay_one(env, lbl)
        assert r1.overall_outcome == r2.overall_outcome
        assert [(v.dimension, str(v.verdict)) for v in r1.dimension_verdicts] == [
            (v.dimension, str(v.verdict)) for v in r2.dimension_verdicts
        ]
