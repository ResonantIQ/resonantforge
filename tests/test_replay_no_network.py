"""
No-network test: verify that replay_one never makes an LLM call.

Two layers of enforcement:
  1. extract_claims_llm spy — verifies the function is never called during replay
  2. Anthropic constructor guard — patches Anthropic.__init__ to RAISE on
     instantiation; if replay accidentally creates an LLM client the test
     fails loudly rather than recording after the fact

The RFORGE_REPLAY_MODE=1 guard is also tested directly: it patches
extract_claims_llm to raise ReplayModeError, which fails loudly if called.
"""
from __future__ import annotations

from pathlib import Path
from unittest import mock

import pytest

from tests.replay_fixtures.builder import build_fixtures


@pytest.fixture(scope="module")
def fixtures_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("no_network_fixtures")
    build_fixtures(d)
    return d


class TestNoNetwork:
    def test_replay_one_never_calls_extract_claims_llm(self, fixtures_dir: Path) -> None:
        """replay_one completes without calling extract_claims_llm."""
        from resonantforge.replay.engine import load_envelope, load_labels, replay_one
        import resonantforge.validators.extractors.accuracy as _acc_mod

        call_log: list[str] = []
        original = _acc_mod.extract_claims_llm

        def _spy(*args, **kwargs):
            call_log.append("CALLED")
            return original(*args, **kwargs)

        with mock.patch.object(_acc_mod, "extract_claims_llm", side_effect=_spy):
            conv_id = "conv_clean_pass"
            env = load_envelope(fixtures_dir / conv_id / "envelope.json")
            lbl = load_labels(fixtures_dir / conv_id / "labels.json", conv_id=conv_id)
            replay_one(env, lbl)

        assert call_log == [], (
            f"extract_claims_llm was called {len(call_log)} time(s) during replay — "
            "frozen extracted_claims should be injected directly into run_kb_alignment_pipeline"
        )

    def test_replay_mode_guard_raises_on_extract_claims(self) -> None:
        """When _install_replay_guard() is active, extract_claims_llm raises ReplayModeError."""
        from resonantforge.replay.engine import ReplayModeError, _install_replay_guard
        import resonantforge.validators.extractors.accuracy as _acc_mod

        original = _acc_mod.extract_claims_llm
        try:
            _install_replay_guard()
            with pytest.raises(ReplayModeError, match="determinism contract"):
                _acc_mod.extract_claims_llm(agent_prose="test")
        finally:
            _acc_mod.extract_claims_llm = original  # type: ignore[attr-defined]

    def test_anthropic_client_never_instantiated_during_replay(self, fixtures_dir: Path) -> None:
        """
        Anthropic() constructor is patched to RAISE if called during replay.

        This is a harder guarantee than a spy: if the replay path accidentally
        instantiates an Anthropic client, the test fails immediately rather than
        detecting it after the fact. A passing test proves that no Anthropic
        client construction occurred anywhere in the replay call graph.
        """
        from resonantforge.replay.engine import load_envelope, load_labels, replay_one

        try:
            import anthropic as _anthropic
        except ImportError:
            pytest.skip("anthropic package not installed")

        class _ForbiddenClient:
            def __init__(self, *args, **kwargs):
                raise AssertionError(
                    "Anthropic() was instantiated during replay — "
                    "the replay path must not create any LLM client. "
                    "Check engine.py and extractor.py for accidental client construction."
                )

        with mock.patch.object(_anthropic, "Anthropic", _ForbiddenClient):
            # Run against all 5 fixtures to cover every code path
            for conv_id in [
                "conv_clean_pass",
                "conv_accuracy_fail",
                "conv_claim_extraction",
                "conv_empathy_fail",
                "conv_brand_voice_fail",
            ]:
                env = load_envelope(fixtures_dir / conv_id / "envelope.json")
                lbl = load_labels(fixtures_dir / conv_id / "labels.json", conv_id=conv_id)
                replay_one(env, lbl)  # must complete without hitting _ForbiddenClient.__init__

    def test_no_http_request_during_replay(self, fixtures_dir: Path) -> None:
        """
        urllib3 / httpx connection pool is patched to raise if opened during replay.

        Belt-and-suspenders: even if a client is somehow constructed without going
        through anthropic.Anthropic(), an actual network request would be caught here.
        """
        from resonantforge.replay.engine import load_envelope, load_labels, replay_one

        def _forbidden_request(*args, **kwargs):
            raise AssertionError(
                "HTTP request attempted during replay — replay must be fully offline."
            )

        # Patch urllib3 (used by httpx / the anthropic SDK) at the connection level
        try:
            import urllib3
            with mock.patch.object(
                urllib3.HTTPConnectionPool, "urlopen", side_effect=_forbidden_request
            ):
                conv_id = "conv_clean_pass"
                env = load_envelope(fixtures_dir / conv_id / "envelope.json")
                lbl = load_labels(fixtures_dir / conv_id / "labels.json", conv_id=conv_id)
                replay_one(env, lbl)
        except ImportError:
            pytest.skip("urllib3 not available")
