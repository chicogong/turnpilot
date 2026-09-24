from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SOURCE = Path(__file__).resolve().parents[1] / "examples" / "eot_bench_adapter.py"
_SPEC = importlib.util.spec_from_file_location("eot_bench_adapter_for_test", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
adapter = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = adapter
_SPEC.loader.exec_module(adapter)


def test_prediction_rows_are_causal_and_labels_are_post_hoc() -> None:
    row = {
        "id": "test-turn",
        "silence_spans": [
            {"start": 1.0, "end": 1.8},
            {"start": 2.0, "end": 2.8},
        ],
        "messages": [{"role": "user", "content": "not-for-acoustic-detection"}],
    }
    before_future = adapter._prediction_rows(row, (1640,), inference_interval=0.1)
    after_future = adapter._prediction_rows(row, (1640, 2640), inference_interval=0.1)
    first_span = [item for item in before_future if item["span_index"] == 0]
    assert first_span == [item for item in after_future if item["span_index"] == 0]
    assert first_span[0]["p_eot"] == 0.0
    assert next(item for item in first_span if item["timestamp"] == 1.6)["p_eot"] == 0.0
    assert next(item for item in first_span if item["timestamp"] == 1.7)["p_eot"] == 1.0
    assert all(item["label"] == "hold" for item in first_span)
    second_span = [item for item in after_future if item["span_index"] == 1]
    assert second_span[0]["p_eot"] == 0.0
    assert second_span[-1]["p_eot"] == 1.0
    assert all(item["label"] == "eot" for item in second_span)


def test_short_spans_and_invalid_grid() -> None:
    row = {
        "id": "test-turn",
        "silence_spans": [
            {"start": 0.0, "end": 0.05},
            {"start": 1.0, "end": 1.25},
        ],
    }
    rows = adapter._prediction_rows(row, (1120,), inference_interval=0.1)
    assert {item["span_index"] for item in rows} == {1}
    assert rows[-1]["timestamp"] == 1.25
    with pytest.raises(ValueError, match="invalid benchmark time grid"):
        adapter._grid(0.0, 1.0, 0.0)


def test_arms_are_distinct_only_in_rearm_and_noise_gain() -> None:
    unchanged = adapter.UnchangedGateEOTAdapter.gate_config
    fixed_gate = adapter.FixedGateEOTAdapter.gate_config
    fixed = adapter.FixedRearmEOTAdapter.gate_config
    adaptive = adapter.AdaptiveRearmEOTAdapter.gate_config
    assert not unchanged.rearm_during_pause
    assert not fixed_gate.rearm_during_pause
    assert fixed_gate.probability_gain_per_noise_db == 0.0
    assert fixed.rearm_during_pause and adaptive.rearm_during_pause
    assert fixed.probability_gain_per_noise_db == 0.0
    assert adaptive.probability_gain_per_noise_db > 0.0


def test_pause_tuning_value_is_validated_and_recorded(monkeypatch, tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"test-model")
    monkeypatch.setenv("TURNPILOT_SILERO_MODEL", str(model))
    monkeypatch.setenv("TURNPILOT_EOT_PAUSE_MS", "480")
    candidate = adapter.FixedGateEOTAdapter()
    assert candidate.pause_deadline_ms == 480
    assert "480ms" in candidate.adapter_id
    assert candidate.supports_language("zh")
    assert candidate.supports_language("en")
    assert not candidate.supports_language("fr")
    monkeypatch.setenv("TURNPILOT_EOT_PAUSE_MS", "199")
    with pytest.raises(ValueError, match="between 200 and 2000"):
        adapter.FixedGateEOTAdapter()
