"""Real background execution, bounded capacity, and controller wiring."""

from __future__ import annotations

import threading
import time

import pytest

from turnpilot.local_audio_runtime import LocalAudioEOTRuntime, LocalAudioEOTWorker
from turnpilot.models import AcousticSignal, TurnRef

REF = TurnRef("session", "turn", 0)


def _pause(duration_ms: int, *, start_ms: int = 100) -> AcousticSignal:
    return AcousticSignal(REF, start_ms + duration_ms, False, pause_duration_ms=duration_ms)


def _speech(at_ms: int) -> AcousticSignal:
    return AcousticSignal(REF, at_ms, True, speech_duration_ms=32)


def _wait_for_result(worker: LocalAudioEOTWorker, at_ms: int) -> object:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        outcome = worker.poll(at_ms)
        if outcome is not None:
            return outcome
        time.sleep(0.001)
    raise AssertionError("background scorer did not finish")


def test_worker_runs_outside_caller_and_delivers_at_poll_time() -> None:
    started = threading.Event()
    release = threading.Event()

    def scorer(audio: bytes) -> float:
        assert audio == b"causal-prefix"
        started.set()
        assert release.wait(3)
        return 0.9

    worker = LocalAudioEOTWorker(scorer)
    try:
        assert worker.submit(REF, 1, b"causal-prefix", 324)
        assert started.wait(3)
        assert worker.poll(400) is None
        release.set()
        outcome = _wait_for_result(worker, 450)
        assert outcome.status == "score"
        assert outcome.received_at_ms == 450
        assert outcome.probability == 0.9
        assert outcome.elapsed_ms == 126
        assert outcome.compute_ms is not None and outcome.compute_ms >= 0
        assert worker.stats.scores == 1
        assert worker.stats.last_compute_ms == outcome.compute_ms
    finally:
        release.set()
        worker.close(wait=True)


def test_cancelled_work_keeps_capacity_bounded_until_thread_drains() -> None:
    started = threading.Event()
    release = threading.Event()

    def scorer(audio: bytes) -> float:
        if audio == b"old":
            started.set()
            assert release.wait(3)
        return 0.9

    worker = LocalAudioEOTWorker(scorer)
    try:
        assert worker.submit(REF, 1, b"old", 324)
        assert started.wait(3)
        assert not worker.cancel(TurnRef("session", "other", 0), 1, 400)
        assert worker.cancel(REF, 1, 400)
        assert not worker.submit(REF, 2, b"new", 401)
        release.set()
        deadline = time.monotonic() + 3
        while not worker.submit(REF, 2, b"new", 402):
            assert time.monotonic() < deadline
            time.sleep(0.001)
        outcome = _wait_for_result(worker, 450)
        assert outcome.request_id == 2
        assert worker.stats.cancelled == 1
        assert worker.stats.busy_rejections >= 1
        assert worker.stats.discarded_results == 1
    finally:
        release.set()
        worker.close(wait=True)


def test_timeout_discards_late_score_without_queuing_new_work() -> None:
    started = threading.Event()
    release = threading.Event()

    def scorer(_: bytes) -> float:
        started.set()
        assert release.wait(3)
        return 0.99

    worker = LocalAudioEOTWorker(scorer, timeout_ms=300)
    try:
        assert worker.submit(REF, 1, b"audio", 324)
        assert started.wait(3)
        timeout = worker.poll(624)
        assert timeout is not None and timeout.status == "timeout"
        assert timeout.elapsed_ms == 300
        assert not worker.submit(REF, 2, b"later", 625)
        release.set()
        deadline = time.monotonic() + 3
        while worker.stats.discarded_results == 0:
            assert time.monotonic() < deadline
            assert worker.poll(625) is None
            time.sleep(0.001)
        assert worker.stats.timeouts == 1
        assert worker.stats.scores == 0
    finally:
        release.set()
        worker.close(wait=True)


@pytest.mark.parametrize("score", [float("nan"), 2.0, True])
def test_invalid_scores_are_sanitized(score: float) -> None:
    worker = LocalAudioEOTWorker(lambda _: score)
    try:
        assert worker.submit(REF, 1, b"audio", 324)
        outcome = _wait_for_result(worker, 400)
        assert outcome.status == "invalid_score"
        assert outcome.probability is None
    finally:
        worker.close(wait=True)


def test_scorer_exception_does_not_leak_its_message() -> None:
    def scorer(_: bytes) -> float:
        raise RuntimeError("private transcript must not appear")

    worker = LocalAudioEOTWorker(scorer)
    try:
        assert worker.submit(REF, 1, b"audio", 324)
        outcome = _wait_for_result(worker, 400)
        assert outcome.status == "model_error"
        assert "private transcript" not in repr(outcome)
        assert worker.stats.errors == 1
    finally:
        worker.close(wait=True)


def test_runtime_connects_real_score_to_endpoint_candidate_only() -> None:
    finished = threading.Event()

    def scorer(audio: bytes) -> float:
        assert audio == b"prefix-at-324"
        finished.set()
        return 0.9

    runtime = LocalAudioEOTRuntime(REF, scorer)
    try:
        updates = runtime.observe(_pause(224), audio_prefix=b"prefix-at-324")
        assert updates[0].request_id == 1
        assert finished.wait(3)
        deadline = time.monotonic() + 3
        while not any(item.reason == "score_accepted" for item in updates):
            assert time.monotonic() < deadline
            updates += runtime.tick(500)
            time.sleep(0.001)
        assert any(item.reason == "score_accepted" for item in updates)
        assert not any(item.endpoint_candidate for item in updates)
        updates = runtime.tick(580)
        assert any(item.endpoint_candidate for item in updates)
        assert runtime.stats.scores == 1
    finally:
        runtime.close(580, wait=True)


def test_resume_cancels_old_score_and_busy_new_pause_falls_back() -> None:
    started = threading.Event()
    release = threading.Event()

    def scorer(_: bytes) -> float:
        started.set()
        assert release.wait(3)
        return 0.99

    runtime = LocalAudioEOTRuntime(REF, scorer)
    try:
        runtime.observe(_pause(224), audio_prefix=b"old")
        assert started.wait(3)
        updates = runtime.observe(_speech(400))
        assert updates[0].cancel_request_id == 1
        updates = runtime.observe(_pause(224, start_ms=500), audio_prefix=b"new")
        assert updates[0].request_id == 2
        assert updates[1].reason == "model_error"
        release.set()
        updates = runtime.tick(1140)
        assert any(item.reason == "endpoint_baseline" for item in updates)
        assert runtime.stats.cancelled == 1
        assert runtime.stats.busy_rejections == 1
        assert runtime.stats.scores == 0
    finally:
        release.set()
        runtime.close(1140, wait=True)


def test_same_time_resume_wins_over_completed_score_poll() -> None:
    started = threading.Event()
    release = threading.Event()

    def scorer(_: bytes) -> float:
        started.set()
        assert release.wait(3)
        return 0.99

    runtime = LocalAudioEOTRuntime(REF, scorer)
    try:
        runtime.observe(_pause(224), audio_prefix=b"causal")
        assert started.wait(3)
        release.set()
        updates = runtime.observe(_speech(480))
        assert updates[0].reason == "speech_resumed"
        assert updates[0].cancel_request_id == 1
        assert not any(item.endpoint_candidate for item in runtime.tick(480))
        assert runtime.stats.scores == 0
    finally:
        release.set()
        runtime.close(480, wait=True)


def test_runtime_timeout_keeps_baseline_and_reports_worker_timeout() -> None:
    started = threading.Event()
    release = threading.Event()

    def scorer(_: bytes) -> float:
        started.set()
        assert release.wait(3)
        return 0.99

    runtime = LocalAudioEOTRuntime(REF, scorer)
    try:
        runtime.observe(_pause(224), audio_prefix=b"causal")
        assert started.wait(3)
        updates = runtime.tick(624)
        assert any(item.reason == "model_timeout" for item in updates)
        assert runtime.stats.timeouts == 1
        release.set()
        updates = runtime.tick(740)
        assert any(item.reason == "endpoint_baseline" for item in updates)
        assert runtime.stats.scores == 0
    finally:
        release.set()
        runtime.close(740, wait=True)


def test_missing_audio_uses_fallback_and_close_rejects_new_events() -> None:
    runtime = LocalAudioEOTRuntime(REF, lambda _: 0.9)
    updates = runtime.observe(_pause(224))
    assert updates[0].request_id == 1
    assert updates[1].reason == "model_error"
    assert runtime.tick(740)[0].reason == "endpoint_baseline"
    assert runtime.stats.submitted == 0
    runtime.close(740, wait=True)
    with pytest.raises(RuntimeError, match="closed"):
        runtime.tick(741)


def test_shared_worker_bounds_work_across_turns_and_survives_runtime_close() -> None:
    started = threading.Event()
    release = threading.Event()
    calls: list[bytes] = []

    def scorer(audio: bytes) -> float:
        calls.append(audio)
        if audio == b"first":
            started.set()
            assert release.wait(3)
        return 0.9

    worker = LocalAudioEOTWorker(scorer)
    first = LocalAudioEOTRuntime(REF, worker=worker)
    second_ref = TurnRef("session", "next-turn", 0)
    second = LocalAudioEOTRuntime(second_ref, worker=worker)
    try:
        first.observe(_pause(224), audio_prefix=b"first")
        assert started.wait(3)
        first.close(400)
        second_pause = AcousticSignal(second_ref, 625, False, pause_duration_ms=224)
        updates = second.observe(second_pause, audio_prefix=b"second")
        assert updates[0].request_id == 1
        assert updates[1].reason == "model_error"
        assert calls == [b"first"]
        assert worker.stats.busy_rejections == 1
        release.set()
        deadline = time.monotonic() + 3
        while worker.stats.discarded_results == 0:
            assert time.monotonic() < deadline
            second.tick(625)
            time.sleep(0.001)
        second.close(625)
        third_ref = TurnRef("session", "third-turn", 0)
        third = LocalAudioEOTRuntime(third_ref, worker=worker)
        try:
            third_pause = AcousticSignal(third_ref, 850, False, pause_duration_ms=224)
            third.observe(third_pause, audio_prefix=b"third")
            deadline = time.monotonic() + 3
            while worker.stats.scores == 0:
                assert time.monotonic() < deadline
                third.tick(900)
                time.sleep(0.001)
            assert calls == [b"first", b"third"]
        finally:
            third.close(900)
    finally:
        release.set()
        first.close(900)
        second.close(900)
        worker.close(wait=True)


def test_runtime_rejects_ambiguous_worker_ownership_and_timeout() -> None:
    worker = LocalAudioEOTWorker(lambda _: 0.9, timeout_ms=200)
    try:
        with pytest.raises(ValueError, match="exactly one"):
            LocalAudioEOTRuntime(REF)
        with pytest.raises(ValueError, match="exactly one"):
            LocalAudioEOTRuntime(REF, lambda _: 0.9, worker=worker)
        with pytest.raises(ValueError, match="timeouts must match"):
            LocalAudioEOTRuntime(REF, worker=worker)
    finally:
        worker.close(wait=True)
    with pytest.raises(ValueError, match="worker must be open"):
        LocalAudioEOTRuntime(REF, worker=worker)
