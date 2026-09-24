"""Bounded background execution for the optional local-audio EOT controller.

The scorer runs in one dedicated thread per worker, never on the caller's event loop.
Cancellation and timeout invalidate a result immediately but cannot interrupt
an already-running Python thread. While that thread drains, new requests are
rejected and the pure controller follows its acoustic fallback.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal

from turnpilot.local_audio_eot import (
    LocalAudioEOTConfig,
    LocalAudioEOTController,
    LocalAudioEOTUpdate,
)
from turnpilot.models import AcousticSignal, TurnRef


@dataclass(frozen=True, slots=True)
class AudioScoreOutcome:
    ref: TurnRef
    request_id: int
    received_at_ms: int
    status: Literal["score", "timeout", "model_error", "invalid_score"]
    probability: float | None = None
    compute_ms: float | None = None
    elapsed_ms: int | None = None


@dataclass(frozen=True, slots=True)
class AudioWorkerStats:
    submitted: int = 0
    scores: int = 0
    timeouts: int = 0
    errors: int = 0
    cancelled: int = 0
    busy_rejections: int = 0
    discarded_results: int = 0
    last_compute_ms: float | None = None
    max_compute_ms: float | None = None


class LocalAudioEOTWorker:
    """Single-slot scorer with no unbounded request queue.

    ``audio_prefix`` must be an immutable, causal snapshot ending no later
    than ``submitted_at_ms``. The caller supplies a shared monotonic clock in
    milliseconds and polls after same-time speech/resume observations.
    """

    def __init__(self, scorer: Callable[[bytes], float], *, timeout_ms: int = 300) -> None:
        if timeout_ms <= 0:
            raise ValueError("timeout_ms must be positive")
        self._scorer = scorer
        self._timeout_ms = timeout_ms
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="turnpilot-eot")
        self._future: Future[tuple[float, float]] | None = None
        self._active: tuple[TurnRef, int, int] | None = None
        self._last_at_ms = -1
        self._closed = False
        self._submitted = 0
        self._scores = 0
        self._timeouts = 0
        self._errors = 0
        self._cancelled = 0
        self._busy_rejections = 0
        self._discarded_results = 0
        self._last_compute_ms: float | None = None
        self._max_compute_ms: float | None = None

    @property
    def stats(self) -> AudioWorkerStats:
        return AudioWorkerStats(
            submitted=self._submitted,
            scores=self._scores,
            timeouts=self._timeouts,
            errors=self._errors,
            cancelled=self._cancelled,
            busy_rejections=self._busy_rejections,
            discarded_results=self._discarded_results,
            last_compute_ms=self._last_compute_ms,
            max_compute_ms=self._max_compute_ms,
        )

    @property
    def timeout_ms(self) -> int:
        return self._timeout_ms

    @property
    def closed(self) -> bool:
        return self._closed

    def _score_once(self, audio_prefix: bytes) -> tuple[float, float]:
        started_ns = time.perf_counter_ns()
        probability = self._scorer(audio_prefix)
        compute_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
        return probability, compute_ms

    def _advance_clock(self, at_ms: int) -> None:
        if at_ms < 0 or at_ms < self._last_at_ms:
            raise ValueError("worker event time must be non-negative and monotonic")
        self._last_at_ms = at_ms

    def _reap_discarded(self) -> None:
        if self._active is None and self._future is not None and self._future.done():
            self._future = None
            self._discarded_results += 1

    def submit(
        self, ref: TurnRef, request_id: int, audio_prefix: bytes, submitted_at_ms: int
    ) -> bool:
        """Start immediately or reject; never queue behind an in-flight score."""
        self._advance_clock(submitted_at_ms)
        if self._closed:
            raise RuntimeError("audio worker is closed")
        if request_id <= 0 or not isinstance(audio_prefix, bytes) or not audio_prefix:
            raise ValueError("request_id and immutable audio_prefix are required")
        self._reap_discarded()
        if self._future is not None:
            self._busy_rejections += 1
            return False
        self._future = self._executor.submit(self._score_once, audio_prefix)
        self._active = (ref, request_id, submitted_at_ms)
        self._submitted += 1
        return True

    def cancel(self, ref: TurnRef, request_id: int, at_ms: int) -> bool:
        """Invalidate a request; a running scorer may continue in the background."""
        self._advance_clock(at_ms)
        if self._active is None or self._active[:2] != (ref, request_id):
            self._reap_discarded()
            return False
        self._active = None
        assert self._future is not None
        self._future.cancel()
        self._cancelled += 1
        self._reap_discarded()
        return True

    def poll(self, now_ms: int) -> AudioScoreOutcome | None:
        """Deliver at poll time, including scheduling delay, or enforce timeout."""
        self._advance_clock(now_ms)
        if self._active is None:
            self._reap_discarded()
            return None
        ref, request_id, submitted_at_ms = self._active
        assert self._future is not None
        if now_ms >= submitted_at_ms + self._timeout_ms:
            self._active = None
            self._future.cancel()
            self._timeouts += 1
            self._reap_discarded()
            return AudioScoreOutcome(
                ref, request_id, now_ms, "timeout", elapsed_ms=now_ms - submitted_at_ms
            )
        if not self._future.done():
            return None
        future = self._future
        self._active = None
        self._future = None
        try:
            probability, compute_ms = future.result()
        except Exception:
            self._errors += 1
            return AudioScoreOutcome(
                ref, request_id, now_ms, "model_error", elapsed_ms=now_ms - submitted_at_ms
            )
        self._last_compute_ms = compute_ms
        self._max_compute_ms = max(self._max_compute_ms or 0.0, compute_ms)
        if isinstance(probability, bool) or not isinstance(probability, (int, float)):
            self._errors += 1
            return AudioScoreOutcome(
                ref,
                request_id,
                now_ms,
                "invalid_score",
                compute_ms=compute_ms,
                elapsed_ms=now_ms - submitted_at_ms,
            )
        score = float(probability)
        if not math.isfinite(score) or not 0 <= score <= 1:
            self._errors += 1
            return AudioScoreOutcome(
                ref,
                request_id,
                now_ms,
                "invalid_score",
                compute_ms=compute_ms,
                elapsed_ms=now_ms - submitted_at_ms,
            )
        self._scores += 1
        return AudioScoreOutcome(
            ref,
            request_id,
            now_ms,
            "score",
            score,
            compute_ms,
            now_ms - submitted_at_ms,
        )

    def close(self, *, wait: bool = False) -> None:
        """Stop accepting results; ``wait=True`` blocks until a running scorer exits."""
        if self._closed:
            return
        self._closed = True
        self._active = None
        if self._future is not None:
            self._future.cancel()
        self._executor.shutdown(wait=wait, cancel_futures=True)


class LocalAudioEOTRuntime:
    """Wire one pure controller to one bounded background scorer.

    Call ``observe`` before ``tick`` at the same timestamp so resumed speech
    wins. Either method may return multiple updates: the acoustic transition,
    then a completed score or bounded fallback. No update is a reply command.
    A caller-supplied worker can be reused across turns and remains caller-owned.
    """

    def __init__(
        self,
        ref: TurnRef,
        scorer: Callable[[bytes], float] | None = None,
        config: LocalAudioEOTConfig | None = None,
        *,
        worker: LocalAudioEOTWorker | None = None,
    ) -> None:
        if (scorer is None) == (worker is None):
            raise ValueError("provide exactly one of scorer or worker")
        self.controller = LocalAudioEOTController(ref, config)
        if worker is not None:
            if worker.closed:
                raise ValueError("worker must be open")
            if worker.timeout_ms != self.controller.config.request_timeout_ms:
                raise ValueError("worker and controller request timeouts must match")
            self.worker = worker
        else:
            assert scorer is not None
            self.worker = LocalAudioEOTWorker(
                scorer, timeout_ms=self.controller.config.request_timeout_ms
            )
        self._owns_worker = worker is None
        self.ref = ref
        self._closed = False

    @property
    def stats(self) -> AudioWorkerStats:
        return self.worker.stats

    def observe(
        self, acoustic: AcousticSignal, *, audio_prefix: bytes | None = None
    ) -> tuple[LocalAudioEOTUpdate, ...]:
        if self._closed:
            raise RuntimeError("audio runtime is closed")
        return self._apply(self.controller.observe(acoustic), audio_prefix)

    def tick(
        self, now_ms: int, *, audio_prefix: bytes | None = None
    ) -> tuple[LocalAudioEOTUpdate, ...]:
        if self._closed:
            raise RuntimeError("audio runtime is closed")
        # No speech observation is pending at this timestamp. Poll first so a
        # completed score or real worker timeout is accounted for before the
        # controller advances its fallback timer.
        first = self._consume_outcome(self.worker.poll(now_ms))
        return (*first, *self._apply(self.controller.tick(now_ms), audio_prefix))

    def _apply(
        self, update: LocalAudioEOTUpdate, audio_prefix: bytes | None
    ) -> tuple[LocalAudioEOTUpdate, ...]:
        updates = [update]
        if update.cancel_request_id is not None:
            self.worker.cancel(self.ref, update.cancel_request_id, update.at_ms)
        if update.request_id is not None:
            if not audio_prefix or not self.worker.submit(
                self.ref, update.request_id, audio_prefix, update.at_ms
            ):
                updates.append(
                    self.controller.fail_request(self.ref, update.request_id, update.at_ms)
                )
        updates.extend(self._consume_outcome(self.worker.poll(update.at_ms)))
        return tuple(updates)

    def _consume_outcome(
        self, outcome: AudioScoreOutcome | None
    ) -> tuple[LocalAudioEOTUpdate, ...]:
        if outcome is not None:
            if outcome.status == "score":
                assert outcome.probability is not None
                return (
                    self.controller.receive_score(
                        outcome.ref,
                        outcome.request_id,
                        outcome.received_at_ms,
                        outcome.probability,
                    ),
                )
            elif outcome.status != "timeout":
                return (
                    self.controller.fail_request(
                        outcome.ref, outcome.request_id, outcome.received_at_ms
                    ),
                )
        return ()

    def close(self, now_ms: int, *, wait: bool = False) -> LocalAudioEOTUpdate:
        if self._closed:
            return LocalAudioEOTUpdate(now_ms, "closed")
        update = self.controller.close(now_ms)
        if update.cancel_request_id is not None:
            self.worker.cancel(self.ref, update.cancel_request_id, now_ms)
        if self._owns_worker:
            self.worker.close(wait=wait)
        self._closed = True
        return update
