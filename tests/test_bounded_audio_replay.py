"""Causal integration of the bounded controller with acoustic frames."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from bounded_audio_replay import replay_bounded_audio_endpoints

from turnpilot.acoustic_eval import AcousticPoint
from turnpilot.local_audio_eot import LocalAudioEOTConfig


class _Scorer:
    def __init__(self, probability: float) -> None:
        self.probability = probability
        self.prefix_lengths: list[int] = []

    def score(self, samples: list[float]) -> float:
        self.prefix_lengths.append(len(samples))
        return self.probability


def _frames(*, resume_at_ms: int | None = None) -> list[AcousticPoint]:
    result = []
    for at_ms in range(32, 1601, 32):
        speech = at_ms <= 320 or (resume_at_ms is not None and at_ms >= resume_at_ms)
        result.append(AcousticPoint(at_ms, 0.9 if speech else 0.01, -20 if speech else -70))
    return result


def test_fast_high_score_uses_only_observed_prefix() -> None:
    scorer = _Scorer(0.9)
    trace = replay_bounded_audio_endpoints(
        _frames(), [0.0] * 30000, scorer, model_delay_override_ms=100
    )
    assert len(scorer.prefix_lengths) == 1
    assert scorer.prefix_lengths[0] == trace.score_events[0][0] * 16
    assert scorer.prefix_lengths[0] < 30000
    assert 800 <= trace.endpoint_times_ms[0] < 960
    assert trace.model_timeouts == 0


def test_slow_score_times_out_to_original_gate_deadline() -> None:
    trace = replay_bounded_audio_endpoints(
        _frames(), [0.0] * 30000, _Scorer(0.9), model_delay_override_ms=400
    )
    assert trace.model_timeouts == 1
    assert trace.ignored_late_results == 1
    assert 960 <= trace.endpoint_times_ms[0] < 1024


def test_resume_cancels_pending_result_before_it_can_end_turn() -> None:
    trace = replay_bounded_audio_endpoints(
        _frames(resume_at_ms=800),
        [0.0] * 30000,
        _Scorer(0.9),
        model_delay_override_ms=300,
    )
    assert not trace.endpoint_times_ms
    assert trace.cancelled_requests >= 1


def test_optional_conservative_profile_only_changes_opted_in_replay() -> None:
    default = replay_bounded_audio_endpoints(
        _frames(), [0.0] * 30000, _Scorer(0.1), model_delay_override_ms=50
    )
    conservative = replay_bounded_audio_endpoints(
        _frames(),
        [0.0] * 30000,
        _Scorer(0.1),
        model_delay_override_ms=50,
        config=LocalAudioEOTConfig(early_pause_ms=640, low_hold_pause_ms=1200),
    )
    assert default.endpoint_times_ms[0] < conservative.endpoint_times_ms[0]
    assert 1500 <= conservative.endpoint_times_ms[0] < 1600

    default_high = replay_bounded_audio_endpoints(
        _frames(), [0.0] * 30000, _Scorer(0.9), model_delay_override_ms=50
    )
    conservative_high = replay_bounded_audio_endpoints(
        _frames(),
        [0.0] * 30000,
        _Scorer(0.9),
        model_delay_override_ms=50,
        config=LocalAudioEOTConfig(early_pause_ms=640, low_hold_pause_ms=800),
    )
    assert default_high.endpoint_times_ms[0] < conservative_high.endpoint_times_ms[0]
    assert 960 <= conservative_high.endpoint_times_ms[0] < 1024


def test_optional_profile_survives_rearming_after_an_endpoint() -> None:
    frames = [
        AcousticPoint(
            at_ms,
            0.9 if at_ms <= 320 or 1200 <= at_ms <= 1504 else 0.01,
            -20 if at_ms <= 320 or 1200 <= at_ms <= 1504 else -70,
        )
        for at_ms in range(32, 3201, 32)
    ]
    trace = replay_bounded_audio_endpoints(
        frames,
        [0.0] * 60000,
        _Scorer(0.1),
        model_delay_override_ms=50,
        config=LocalAudioEOTConfig(early_pause_ms=640, low_hold_pause_ms=800),
    )
    assert len(trace.endpoint_times_ms) == 2
    assert 1100 <= trace.endpoint_times_ms[0] < 1200
    assert 2300 <= trace.endpoint_times_ms[1] < 2400
