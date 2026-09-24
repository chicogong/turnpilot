from __future__ import annotations

import pytest

from turnpilot.acoustic import AdaptiveGateConfig
from turnpilot.acoustic_eval import (
    AcousticEndpointTrace,
    AcousticPoint,
    LabeledPauseWindow,
    paired_endpoint_alignment,
    replay_acoustic_endpoints,
    summarize_endpoint_alignment,
)


def point(at_ms: int, probability: float, energy_dbfs: float = -30) -> AcousticPoint:
    return AcousticPoint(at_ms, probability, energy_dbfs)


def test_replay_rearms_from_its_own_endpoint_without_labels() -> None:
    frames = [point(at, 0.9) for at in (32, 64, 96)]
    frames += [point(at, 0.01, -65) for at in range(128, 768, 32)]
    frames += [point(at, 0.9) for at in (768, 800, 832)]
    frames += [point(at, 0.01, -65) for at in range(864, 1504, 32)]
    trace = replay_acoustic_endpoints(frames)
    assert trace.endpoint_times_ms == (736, 1472)
    assert 96 in trace.active_frame_times_ms
    assert 832 in trace.active_frame_times_ms


def test_replay_rejects_invalid_deadline_and_reordered_frames() -> None:
    with pytest.raises(ValueError, match="positive"):
        replay_acoustic_endpoints((point(32, 0.9),), pause_deadline_ms=0)
    with pytest.raises(ValueError, match="non-monotonic"):
        replay_acoustic_endpoints((point(32, 0.9), point(32, 0.9)))


def test_rearm_candidate_does_not_change_noiseless_continuous_speech() -> None:
    frames = [point(at, 0.9) for at in range(32, 672, 32)]
    fixed = replay_acoustic_endpoints(frames, config=AdaptiveGateConfig())
    rearmed = replay_acoustic_endpoints(frames, config=AdaptiveGateConfig(rearm_during_pause=True))
    assert fixed == rearmed


def test_rearm_candidate_can_miss_a_resume_near_the_pause_deadline() -> None:
    frames = [point(at, 0.9) for at in (32, 64, 96)]
    frames += [point(at, 0.01, -65) for at in range(128, 736, 32)]
    frames += [point(at, 0.9) for at in (736, 768, 800)]
    baseline = replay_acoustic_endpoints(frames)
    candidate = replay_acoustic_endpoints(
        frames, config=AdaptiveGateConfig(rearm_during_pause=True)
    )
    assert baseline.endpoint_times_ms == ()
    assert candidate.endpoint_times_ms == (736,)


def test_label_alignment_is_post_hoc_and_resume_at_deadline_wins() -> None:
    trace = AcousticEndpointTrace((640, 1500, 2300), ())
    windows = (
        LabeledPauseWindow(0, 640, False),
        LabeledPauseWindow(1000, 2000, False),
        LabeledPauseWindow(2000, 2600, True),
    )
    result = summarize_endpoint_alignment(windows, trace)
    assert result["incomplete_windows"] == 2
    assert result["incomplete_endpoint_before_resume"] == 1
    assert result["complete_endpoint_within_window"] == 1
    assert result["complete_match_latency_p50_ms"] == 300
    with pytest.raises(ValueError, match="invalid labeled"):
        LabeledPauseWindow(500, 500, False)


def test_label_alignment_does_not_attribute_an_event_to_unseen_speech() -> None:
    trace = AcousticEndpointTrace((700,), ())
    window = LabeledPauseWindow(500, 1000, False, start_ms=100)
    result = summarize_endpoint_alignment((window,), trace)
    assert result["incomplete_windows"] == 1
    assert result["incomplete_endpoint_before_resume"] == 0
    assert result["acoustic_unobserved_label_windows"] == 1
    with pytest.raises(ValueError, match="speech span"):
        LabeledPauseWindow(500, 1000, False, start_ms=501)


def test_paired_alignment_counts_discordance_and_excludes_unseen_speech() -> None:
    windows = (
        LabeledPauseWindow(100, 200, False, start_ms=0),
        LabeledPauseWindow(300, 400, False, start_ms=250),
        LabeledPauseWindow(500, 600, False, start_ms=450),
        LabeledPauseWindow(700, 800, False, start_ms=650),
        LabeledPauseWindow(900, 1000, False, start_ms=850),
        LabeledPauseWindow(1100, 1200, True, start_ms=1050),
    )
    reference = AcousticEndpointTrace((150, 350, 750, 1150), (50, 275, 475, 675, 875, 1075))
    candidate = AcousticEndpointTrace((150, 550, 750), (50, 275, 475, 675, 1075))
    result = paired_endpoint_alignment(windows, reference, candidate)
    assert result["incomplete"] == {
        "both_endpoint": 2,
        "reference_only_endpoint": 1,
        "candidate_only_endpoint": 1,
        "neither_endpoint": 0,
        "not_jointly_observed": 1,
    }
    assert result["complete"] == {
        "both_endpoint": 0,
        "reference_only_endpoint": 1,
        "candidate_only_endpoint": 0,
        "neither_endpoint": 0,
        "not_jointly_observed": 0,
    }
    with pytest.raises(ValueError, match="at least one"):
        paired_endpoint_alignment((), reference, candidate)
