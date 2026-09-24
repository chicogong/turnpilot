from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
from collections.abc import Mapping
from pathlib import Path

_SOURCE = Path(__file__).resolve().parents[1] / "examples" / "live_jev_probe.py"
_SPEC = importlib.util.spec_from_file_location("live_jev_probe_for_test", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
live_jev_probe = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = live_jev_probe
_SPEC.loader.exec_module(live_jev_probe)


class FakeTransport:
    async def evaluate(
        self, payload: Mapping[str, object], *, timeout_s: float
    ) -> Mapping[str, object]:
        assert timeout_s > 0
        state = payload["state"]
        assert isinstance(state, dict)
        is_final = state["transcript_is_final"]
        probability = 0.9 if is_final else 0.1
        return {
            "model": payload["model"],
            "answers": {
                key: {"type": "noul", "noul": probability if key == "turn_complete" else 0.1}
                for key in (
                    "turn_complete",
                    "needs_meaning_clarification",
                    "backchannel",
                    "response_needed",
                )
            },
        }

    async def close(self) -> None:
        pass


def test_paired_probe_keeps_authored_text_and_key_out_of_output(monkeypatch) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "private-test-key")
    monkeypatch.setattr(live_jev_probe, "HttpxJevTransport", lambda _key: FakeTransport())
    result = asyncio.run(live_jev_probe.run_paired(1000))
    assert result["requests_attempted"] == 20
    assert result["partial"]["scores"]["evaluated"] == 10
    assert result["final"]["scores"]["evaluated"] == 10
    assert result["timing_counterfactual"]["fixed_640"]["false_cutoffs"] == 3
    assert result["timing_counterfactual"]["stale_prefix_jev_budget_350_ms"]["cases"] == 10
    assert result["timing_counterfactual"]["revision_320_jev_budget_350_ms"]["cases"] == 10
    assert result["timing_counterfactual"]["guarded_stale_prefix_jev_budget_350_ms"]["cases"] == 10
    assert result["guarded_candidate"]["partial_semantic_hold_extension_ms"] == 100
    encoded = json.dumps(result, ensure_ascii=False)
    assert "private-test-key" not in encoded
    assert live_jev_probe.CASES[0].text not in encoded
    assert live_jev_probe.CASES[0].partial_text not in encoded
