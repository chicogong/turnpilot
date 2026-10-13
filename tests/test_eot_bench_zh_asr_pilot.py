from __future__ import annotations

import importlib.util
import io
import json
import sys
import wave
from pathlib import Path
from typing import Any

import pytest

_EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(_EXAMPLES))
_SPEC = importlib.util.spec_from_file_location(
    "eot_bench_zh_asr_pilot_for_test", _EXAMPLES / "eot_bench_zh_asr_pilot.py"
)
assert _SPEC is not None and _SPEC.loader is not None
pilot = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = pilot
_SPEC.loader.exec_module(pilot)


def _report(*, with_final: bool) -> dict[str, Any]:
    partial = {
        "kind": "asr",
        "media_at_ms": 320,
        "available_at_ms": 332,
        "revision": 1,
        "has_text": True,
        "is_final": False,
    }
    final = {
        "kind": "asr",
        "media_at_ms": 1024,
        "available_at_ms": 1050,
        "revision": 2,
        "has_text": True,
        "is_final": True,
    }
    direct = {
        "kind": "commit_user_turn",
        "media_at_ms": 1024,
        "available_at_ms": 1050,
    }
    fixed = {
        "kind": "commit_user_turn",
        "media_at_ms": 1664,
        "available_at_ms": 1680,
    }
    return {
        "eof_forced_final": False,
        "synthetic_tail_silence_ms": 0,
        "evidence_level": "paced_file_pipeline_only",
        "window": {"start_ms": 0, "ends_at_source_eof": True},
        "duration_ms": 2000,
        "processed_media_ms": 1984,
        "unprocessed_tail_samples": 256,
        "timeline": [partial, final] if with_final else [partial],
        "asr": {"revisions": 2 if with_final else 1, "nonempty_decoder_finals": int(with_final)},
        "pipeline": {
            "backlog_frames": 0,
            "availability_lag_p95_ms": 20,
            "availability_lag_max_ms": 26,
        },
        "arms": {
            "direct_asr_final": {"recommendation": direct if with_final else None},
            "fixed_640": {"recommendation": fixed if with_final else None},
            "adaptive_640": {"recommendation": fixed if with_final else None},
            "gate": {"recommendation": None},
        },
    }


def test_actual_final_and_eof_censoring_are_separate() -> None:
    spans = [{"start": 1.0, "end": 1.8}, {"start": 1.8, "end": 2.0}]
    observed = pilot.summarize_window(_report(with_final=True), spans)
    censored = pilot.summarize_window(_report(with_final=False), spans)
    assert observed["first_final_available_ms"] == 1050
    assert observed["first_final_availability_lag_ms"] == 26
    assert observed["arms"]["direct_asr_final"]["media_position_region"] == "labeled_hold"
    assert observed["arms"]["fixed_640"]["media_position_region"] == "labeled_hold"
    assert censored["first_final_available_ms"] is None
    assert (
        censored["arms"]["direct_asr_final"]["media_position_region"] == "no_recommendation_by_eof"
    )
    aggregate = pilot.aggregate_windows([observed, censored])
    assert aggregate["windows"] == 2
    assert aggregate["asr"]["nonempty_decoder_finals"] == 1
    assert aggregate["asr"]["no_nonempty_final_by_eof_censored"] == 1
    assert aggregate["processed_frames"] == 124
    assert aggregate["max_frame_availability_lag_ms"] == 26
    assert aggregate["fixed_vs_adaptive_first_recommendation_mismatch_windows"] == 0
    assert aggregate["gate_no_recommendation_by_eof"] == {
        "with_decoder_final": 1,
        "without_decoder_final": 1,
    }
    assert aggregate["paired_against_direct_final"]["fixed_640"]["both_recommended_windows"] == 1
    assert (
        aggregate["paired_against_direct_final"]["fixed_640"]["alternative_minus_direct_ms_p50"]
        == 630
    )
    assert "text" not in json.dumps(aggregate)


def test_rejects_content_leak_and_synthetic_eof_final() -> None:
    report = _report(with_final=True)
    report["timeline"][0]["text"] = "PRIVATE WORDS"
    with pytest.raises(ValueError, match="content-bearing"):
        pilot.summarize_window(report, [{"start": 1.0, "end": 2.0}])
    report = _report(with_final=True)
    report["eof_forced_final"] = True
    with pytest.raises(ValueError, match="unmodified EOF"):
        pilot.summarize_window(report, [{"start": 1.0, "end": 2.0}])
    report = _report(with_final=True)
    report["window"]["ends_at_source_eof"] = False
    with pytest.raises(ValueError, match="full source recording"):
        pilot.summarize_window(report, [{"start": 1.0, "end": 2.0}])


def test_embedded_audio_must_be_exact_supported_wav_and_not_path() -> None:
    with pytest.raises(ValueError, match="embed WAV bytes"):
        pilot._embedded_wav({"path": "never-open-me.wav", "bytes": None})
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\0" * 1024)
    content = buffer.getvalue()
    assert pilot._embedded_wav({"bytes": content, "path": "not-used.wav"}) == content


def test_cli_refuses_overwrite_without_loading_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "existing.json"
    output.write_text("preserve", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "pilot",
            "--input",
            "unavailable.arrow",
            "--silero-model",
            "unavailable.onnx",
            "--vosk-model",
            "unavailable-model",
            "--output",
            str(output),
        ],
    )
    with pytest.raises(SystemExit, match="2"):
        pilot.main()
    assert output.read_text(encoding="utf-8") == "preserve"
