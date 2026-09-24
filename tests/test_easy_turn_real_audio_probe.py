"""The real-clip probe reports binary completion separately from other states."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from easy_turn_real_audio_probe import summarize_scores


def test_completion_summary_does_not_fold_backchannel_or_wait_into_binary_error() -> None:
    results = [
        ("complete", 0.9, 25.0, 1.0),
        ("complete", 0.7, 30.0, 2.0),
        ("incomplete", 0.85, 40.0, 1.5),
        ("incomplete", 0.2, 50.0, 1.5),
        ("backchannel", 0.99, 45.0, 0.8),
        ("wait", 0.95, 35.0, 0.8),
    ]
    speakers = {
        "complete": {"a"},
        "incomplete": {"a", "b"},
        "backchannel": {"c"},
        "wait": {"d"},
    }
    report = summarize_scores(results, speakers)
    categories = report["category"]
    assert isinstance(categories, dict)
    assert categories["complete_vs_incomplete_at_0_8"] == {
        "true_complete": 1,
        "missed_complete": 1,
        "false_complete_on_incomplete": 1,
        "correct_incomplete": 1,
    }
    assert categories["backchannel"]["score_ge_0_8"] == 1
    assert categories["wait"]["score_ge_0_8"] == 1
    assert report["complete_incomplete_speaker_overlap"] == 1


def test_percentile_nearest_rank() -> None:
    from easy_turn_real_audio_probe import percentile

    assert percentile([3.0, 1.0, 2.0], 50) == 2.0
    assert percentile([3.0, 1.0, 2.0], 95) == 3.0
