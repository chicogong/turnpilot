from __future__ import annotations

from dataclasses import replace

import pytest

from turnpilot import AudioQuality, TurnRef
from turnpilot.acoustic import AcousticFrame, AdaptiveGateConfig, AdaptiveSpeechGate

REF = TurnRef("session", "turn", 0)


def frame(at_ms: int, probability: float, energy: float = -30, **changes: object) -> AcousticFrame:
    return replace(AcousticFrame(REF, at_ms, probability, energy), **changes)


def test_provisional_pause_and_resumed_speech() -> None:
    gate = AdaptiveSpeechGate()
    assert not gate.observe(frame(0, 0.9)).speech_active
    assert not gate.observe(frame(32, 0.9)).speech_active
    start = gate.observe(frame(64, 0.9, near_end_confirmed=True))
    assert start.speech_active and start.near_end_speech
    assert start.speech_duration_ms == 96
    pause = gate.observe(frame(96, 0.05, -65, audio_quality=AudioQuality.CLEAR))
    assert not pause.speech_active and pause.pause_duration_ms == 32
    for at in (128, 160, 192, 224, 256):
        pause = gate.observe(frame(at, 0.05, -65))
    assert pause.pause_duration_ms == 192
    resumed = gate.observe(frame(288, 0.6))
    assert resumed.speech_active and resumed.pause_duration_ms is None
    assert resumed.speech_duration_ms == 32


def test_noise_raises_start_threshold_but_not_above_cap() -> None:
    gate = AdaptiveSpeechGate(
        AdaptiveGateConfig(noise_ema_alpha=1.0, probability_gain_per_noise_db=0.01)
    )
    gate.observe(frame(0, 0.05, -30))
    assert gate.start_threshold == 0.75
    for at in (32, 64, 96):
        observation = gate.observe(frame(at, 0.6, -15))
    assert not observation.speech_active
    for at in (128, 160, 192):
        observation = gate.observe(frame(at, 0.8, -15))
    assert observation.speech_active


def test_opt_in_pause_rearm_rejects_weak_noise_and_resumes_after_three_frames() -> None:
    candidate = AdaptiveSpeechGate(AdaptiveGateConfig(rearm_during_pause=True))
    baseline = AdaptiveSpeechGate()
    for at in (0, 32, 64):
        assert (
            candidate.observe(frame(at, 0.9)).speech_active
            == baseline.observe(frame(at, 0.9)).speech_active
        )
    assert candidate.observe(frame(96, 0.05, -65)).pause_duration_ms == 32
    assert baseline.observe(frame(96, 0.05, -65)).pause_duration_ms == 32
    weak = candidate.observe(frame(128, 0.4, -30))
    assert not weak.speech_active and weak.pause_duration_ms == 64
    assert baseline.observe(frame(128, 0.4, -30)).speech_active
    for at in (160, 192):
        result = candidate.observe(frame(at, 0.9, -30))
        assert not result.speech_active
    resumed = candidate.observe(frame(224, 0.9, -30))
    assert resumed.speech_active and resumed.speech_duration_ms == 96


def test_opt_in_dynamic_rearm_uses_updated_noise_threshold() -> None:
    adaptive = AdaptiveSpeechGate(AdaptiveGateConfig(rearm_during_pause=True, noise_ema_alpha=1.0))
    fixed = AdaptiveSpeechGate(
        AdaptiveGateConfig(
            rearm_during_pause=True, noise_ema_alpha=1.0, probability_gain_per_noise_db=0.0
        )
    )
    for gate in (adaptive, fixed):
        for at in (0, 32, 64):
            gate.observe(frame(at, 0.9, -20))
        gate.observe(frame(96, 0.05, -30))
    assert adaptive.start_threshold > fixed.start_threshold
    for at in (128, 160, 192):
        assert not adaptive.observe(frame(at, 0.55, -10)).speech_active
        fixed_result = fixed.observe(frame(at, 0.55, -10))
    assert fixed_result.speech_active


def test_echo_is_not_near_end_and_new_generation_resets() -> None:
    gate = AdaptiveSpeechGate()
    for at in (0, 32, 64):
        observation = gate.observe(frame(at, 0.9, echo_likely=True))
    assert not observation.speech_active
    for at in (96, 128, 160):
        observation = gate.observe(frame(at, 0.9, near_end_confirmed=True))
    assert observation.speech_active
    new_ref = TurnRef("session", "turn", 1)
    reset = gate.observe(frame(192, 0.1, ref=new_ref))
    assert reset.ref == new_ref and reset.pause_duration_ms is None
    assert not reset.speech_active
    with pytest.raises(ValueError, match="stale acoustic generation"):
        gate.observe(frame(224, 0.9))
    with pytest.raises(ValueError, match="cannot mix sessions"):
        gate.observe(frame(224, 0.9, ref=TurnRef("other", "turn", 0)))


def test_invalid_frame_and_missing_or_reordered_frames_are_rejected() -> None:
    with pytest.raises(ValueError, match="probability"):
        frame(0, float("nan"))
    with pytest.raises(ValueError, match="energy"):
        frame(0, 0.5, float("inf"))
    gate = AdaptiveSpeechGate()
    gate.observe(frame(0, 0.1))
    with pytest.raises(ValueError, match="non-monotonic"):
        gate.observe(frame(0, 0.1))
    with pytest.raises(ValueError, match="missing"):
        gate.observe(frame(200, 0.1))
