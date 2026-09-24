"""Causality and cancellation guards for the optional local audio EOT arm."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "examples"))

from smartturn_audio_adapter import replay_hybrid_endpoints

from turnpilot.acoustic_eval import AcousticPoint


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


def test_scores_only_observed_audio_and_commits_complete() -> None:
    scorer = _Scorer(0.9)
    trace = replay_hybrid_endpoints(_frames(), [0.0] * 30000, scorer)
    assert len(scorer.prefix_lengths) == 1
    assert scorer.prefix_lengths[0] == trace.score_events[0][0] * 16
    assert scorer.prefix_lengths[0] < 30000
    assert len(trace.score_events) == 1
    assert trace.endpoint_times_ms[0] >= 800


def test_incomplete_score_waits_and_resumed_speech_cancels() -> None:
    scorer = _Scorer(0.1)
    trace = replay_hybrid_endpoints(_frames(resume_at_ms=800), [0.0] * 30000, scorer)
    assert scorer.prefix_lengths
    assert not trace.endpoint_times_ms


def test_incomplete_score_uses_later_deadline() -> None:
    trace = replay_hybrid_endpoints(_frames(), [0.0] * 30000, _Scorer(0.1))
    assert trace.endpoint_times_ms[0] >= 1280


def test_ambiguous_score_uses_original_deadline_only_in_three_way_arm() -> None:
    trace = replay_hybrid_endpoints(
        _frames(),
        [0.0] * 30000,
        _Scorer(0.5),
        low_confidence_threshold=0.2,
        uncertain_deadline_ms=640,
    )
    assert 960 <= trace.endpoint_times_ms[0] < 1280


def test_three_way_threshold_order_is_validated() -> None:
    try:
        replay_hybrid_endpoints(
            _frames(), [0.0] * 30000, _Scorer(0.5), low_confidence_threshold=0.9
        )
    except ValueError as exc:
        assert "threshold" in str(exc)
    else:
        raise AssertionError("reversed thresholds were accepted")


def test_fixed_model_ready_delay_has_deterministic_endpoint() -> None:
    class SlowScorer(_Scorer):
        def score(self, samples: list[float]) -> float:
            import time

            time.sleep(0.01)
            return super().score(samples)

    fast = replay_hybrid_endpoints(_frames(), [0.0] * 30000, _Scorer(0.9), model_ready_delay_ms=300)
    slow = replay_hybrid_endpoints(
        _frames(), [0.0] * 30000, SlowScorer(0.9), model_ready_delay_ms=300
    )
    assert fast.endpoint_times_ms == slow.endpoint_times_ms


def test_silence_without_detected_speech_never_calls_model() -> None:
    scorer = _Scorer(0.9)
    frames = [AcousticPoint(at_ms, 0.01, -70) for at_ms in range(32, 1601, 32)]
    trace = replay_hybrid_endpoints(frames, [0.0] * 30000, scorer)
    assert not scorer.prefix_lengths
    assert not trace.endpoint_times_ms


def test_invalid_score_rejected() -> None:
    scorer = _Scorer(1.1)
    try:
        replay_hybrid_endpoints(_frames(), [0.0] * 30000, scorer)
    except ValueError as exc:
        assert "probability" in str(exc)
    else:
        raise AssertionError("out-of-range score was accepted")
