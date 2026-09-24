"""Injected-clock guards for the optional local-audio endpoint candidate."""

from __future__ import annotations

import pytest

from turnpilot.local_audio_eot import LocalAudioEOTConfig, LocalAudioEOTController
from turnpilot.models import AcousticSignal, TurnRef


def _ref() -> TurnRef:
    return TurnRef("session", "user-turn", 0)


def _pause(ref: TurnRef, duration_ms: int, *, start_ms: int = 100) -> AcousticSignal:
    return AcousticSignal(ref, start_ms + duration_ms, False, pause_duration_ms=duration_ms)


def _speech(ref: TurnRef, at_ms: int) -> AcousticSignal:
    return AcousticSignal(ref, at_ms, True, speech_duration_ms=32)


def test_high_score_can_emit_early_endpoint_candidate() -> None:
    ref = _ref()
    controller = LocalAudioEOTController(ref)
    dispatched = controller.observe(_pause(ref, 224))
    assert dispatched.request_id == 1
    assert controller.receive_score(ref, 1, 400, 0.9).reason == "score_accepted"
    assert not controller.tick(579).endpoint_candidate
    result = controller.tick(580)
    assert result.endpoint_candidate
    assert result.reason == "endpoint_model"
    assert not controller.tick(900).endpoint_candidate


def test_low_score_hold_is_bounded_and_ambiguous_score_uses_baseline() -> None:
    ref = _ref()
    low = LocalAudioEOTController(ref)
    low.observe(_pause(ref, 224))
    low.receive_score(ref, 1, 400, 0.1)
    assert not low.tick(740).endpoint_candidate
    assert low.tick(840).endpoint_candidate

    ambiguous = LocalAudioEOTController(ref)
    ambiguous.observe(_pause(ref, 224))
    ambiguous.receive_score(ref, 1, 400, 0.5)
    assert ambiguous.tick(740).endpoint_candidate


def test_timeout_cancels_request_and_returns_to_640ms_baseline() -> None:
    ref = _ref()
    controller = LocalAudioEOTController(ref)
    controller.observe(_pause(ref, 224))
    timeout = controller.tick(624)
    assert timeout.reason == "model_timeout"
    assert timeout.cancel_request_id == 1
    assert not timeout.endpoint_candidate
    assert controller.receive_score(ref, 1, 625, 0.99).reason == "stale_request"
    endpoint = controller.tick(740)
    assert endpoint.endpoint_candidate
    assert endpoint.reason == "endpoint_baseline"


def test_score_after_baseline_cannot_retroactively_choose_model_endpoint() -> None:
    ref = _ref()
    controller = LocalAudioEOTController(ref, LocalAudioEOTConfig(request_timeout_ms=500))
    controller.observe(_pause(ref, 224))
    result = controller.receive_score(ref, 1, 750, 0.99)
    assert result.endpoint_candidate
    assert result.reason == "endpoint_baseline"
    assert result.cancel_request_id == 1


def test_stale_result_still_advances_baseline_timer() -> None:
    ref = _ref()
    controller = LocalAudioEOTController(ref)
    controller.observe(_pause(ref, 224))
    result = controller.receive_score(ref, 999, 750, 0.99)
    assert result.endpoint_candidate
    assert result.reason == "endpoint_baseline"
    assert result.cancel_request_id == 1


def test_resume_wins_at_same_time_and_old_result_cannot_affect_next_pause() -> None:
    ref = _ref()
    controller = LocalAudioEOTController(ref)
    controller.observe(_pause(ref, 224))
    resume = controller.observe(_speech(ref, 580))
    assert resume.cancel_request_id == 1
    assert not resume.endpoint_candidate
    assert controller.receive_score(ref, 1, 580, 0.99).reason == "stale_request"
    new_pause = controller.observe(_pause(ref, 224, start_ms=600))
    assert new_pause.request_id == 2


def test_wrong_turn_and_model_failure_are_safe() -> None:
    ref = _ref()
    controller = LocalAudioEOTController(ref)
    controller.observe(_pause(ref, 224))
    wrong = TurnRef(ref.session_id, ref.turn_id, 1)
    assert controller.receive_score(wrong, 1, 10000, 0.9).reason == "stale_turn"
    failed = controller.fail_request(ref, 1, 500)
    assert failed.reason == "model_error"
    assert not failed.endpoint_candidate
    assert controller.tick(740).endpoint_candidate


def test_invalid_score_falls_back_and_request_budget_is_bounded() -> None:
    ref = _ref()
    controller = LocalAudioEOTController(ref, LocalAudioEOTConfig(max_requests_per_turn=1))
    controller.observe(_pause(ref, 224))
    assert controller.receive_score(ref, 1, 400, float("nan")).reason == "invalid_score"
    controller.observe(_speech(ref, 500))
    second_pause = controller.observe(_pause(ref, 224, start_ms=600))
    assert second_pause.request_id is None
    assert controller.request_count == 1
    assert controller.tick(1240).reason == "endpoint_baseline"


def test_close_cancels_pending_and_time_must_be_monotonic() -> None:
    ref = _ref()
    controller = LocalAudioEOTController(ref)
    controller.observe(_pause(ref, 224))
    assert controller.close(400).cancel_request_id == 1
    assert controller.receive_score(ref, 1, 500, 0.99).reason == "stale_turn"
    assert not controller.tick(900).endpoint_candidate
    with pytest.raises(ValueError, match="monotonic"):
        controller.tick(800)
