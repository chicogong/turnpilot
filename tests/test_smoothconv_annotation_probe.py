from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

_SOURCE = Path(__file__).resolve().parents[1] / "examples" / "smoothconv_annotation_probe.py"
_SPEC = importlib.util.spec_from_file_location("smoothconv_annotation_probe_for_test", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
probe = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = probe
_SPEC.loader.exec_module(probe)


def test_oracle_fixed_silence_risk_separates_clean_interrupted_and_censored() -> None:
    segment = probe.Segment
    recording = (
        segment(0, 1000, "A", "incomplete"),
        segment(1500, 2000, "B", "complete"),
        segment(3000, 3500, "A", "complete"),
        segment(4000, 5000, "A", "incomplete"),
        segment(5640, 6000, "A", "complete"),
        segment(7000, 8000, "A", "incomplete"),
        segment(9000, 9500, "A", "complete"),
        segment(10000, 11000, "A", "incomplete"),
        segment(11010, 11500, "unknown", "unknown"),
    )
    result = probe.summarize((recording,))
    assert result["turn_counts"] == {
        "complete": 4,
        "incomplete": 4,
        "backchannel": 0,
        "wait": 0,
        "unknown": 1,
    }
    assert result["incomplete_with_same_speaker_continuation"] == 3
    assert result["incomplete_censored_without_continuation"] == 1
    assert result["incomplete_interrupted_before_continuation"] == 1
    assert result["incomplete_clean_continuations"] == 2
    assert result["oracle_fixed_silence_risks_all"] == 2
    assert result["oracle_fixed_silence_risks_clean"] == 1


def test_reader_keeps_transcript_out_of_summary(tmp_path: Path) -> None:
    private_text = "not-for-reporting"
    path = tmp_path / "annotation.json"
    path.write_text(
        json.dumps(
            {
                "instances": [
                    {
                        "start": 0.1,
                        "end": 1.0,
                        "text": private_text,
                        "channelIndex": 0,
                        "attributes": {"speaker": "A", "turn": "incomplete"},
                    },
                    {
                        "start": 1.8,
                        "end": 2.0,
                        "text": private_text,
                        "attributes": {"speaker": "A", "turn": "complete"},
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    result = probe.summarize((probe.read_segments(path),))
    assert result["oracle_fixed_silence_risks_clean"] == 1
    assert probe.read_segments(path)[0].channel_index == 0
    assert private_text not in json.dumps(result)
    with pytest.raises(ValueError, match="positive"):
        probe.summarize((probe.read_segments(path),), silence_ms=0)


def test_post_hoc_windows_exclude_interruption_and_short_complete_horizon() -> None:
    segment = probe.Segment
    recording = (
        segment(0, 1000, "A", "incomplete", 0),
        segment(1800, 2000, "A", "complete", 0),
        segment(3000, 3400, "A", "incomplete", 0),
        segment(3500, 3700, "B", "complete", 1),
        segment(4000, 4300, "A", "complete", 0),
        segment(5800, 6100, "A", "incomplete", 0),
    )
    windows, excluded = probe.labeled_windows(
        recording, 0, 7000, completion_window_ms=1600, min_observed_pause_ms=1200
    )
    assert [
        (item.end_ms, item.next_speech_or_horizon_ms, item.is_complete) for item in windows
    ] == [
        (1000, 1800, False),
        (4300, 5800, True),
    ]
    assert excluded == {
        "incomplete_censored": 1,
        "incomplete_interrupted": 1,
        "complete_short_horizon": 1,
    }
