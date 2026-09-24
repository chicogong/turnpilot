"""Experimental bounded local-audio endpoint *candidate* controller.

This module does not run a model, identify a speaker, commit a user turn, or
control playback. A host may dispatch a local scorer for each emitted request
and feed the result back with its observed arrival time. The controller only
recommends an acoustic endpoint candidate; the conversation policy and host
remain responsible for any action.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from turnpilot.models import AcousticSignal, TurnRef


@dataclass(frozen=True, slots=True)
class LocalAudioEOTConfig:
    request_pause_ms: int = 224
    request_timeout_ms: int = 300
    early_pause_ms: int = 480
    baseline_pause_ms: int = 640
    low_hold_pause_ms: int = 740
    complete_threshold: float = 0.8
    incomplete_threshold: float = 0.2
    max_requests_per_turn: int = 4

    def __post_init__(self) -> None:
        if not (
            0
            < self.request_pause_ms
            <= self.early_pause_ms
            <= self.baseline_pause_ms
            <= self.low_hold_pause_ms
        ):
            raise ValueError("pause deadlines must be ordered")
        if self.request_timeout_ms <= 0 or self.max_requests_per_turn <= 0:
            raise ValueError("request timeout and budget must be positive")
        if not (
            math.isfinite(self.incomplete_threshold)
            and math.isfinite(self.complete_threshold)
            and 0 <= self.incomplete_threshold < self.complete_threshold <= 1
        ):
            raise ValueError("model thresholds must be finite and ordered")


@dataclass(frozen=True, slots=True)
class LocalAudioEOTUpdate:
    at_ms: int
    reason: str
    request_id: int | None = None
    cancel_request_id: int | None = None
    endpoint_candidate: bool = False


class LocalAudioEOTController:
    """Causal state machine for one immutable turn reference.

    Feed acoustic speech/resume observations before timers or model results at
    the same timestamp so a resume wins a simultaneous endpoint. Never reuse
    this instance for a new TurnRef. Dispatch and cancellation are *requests*
    to the caller, not side effects performed here.
    """

    def __init__(self, ref: TurnRef, config: LocalAudioEOTConfig | None = None) -> None:
        self.ref = ref
        self.config = config or LocalAudioEOTConfig()
        self._last_at_ms = -1
        self._pause_start_ms: int | None = None
        self._last_pause_ms: int | None = None
        self._pending_id: int | None = None
        self._request_at_ms: int | None = None
        self._score: float | None = None
        self._failed_this_pause = False
        self._request_count = 0
        self._next_request_id = 1
        self._closed = False
        self._endpoint_emitted = False

    @property
    def request_count(self) -> int:
        return self._request_count

    def observe(self, acoustic: AcousticSignal) -> LocalAudioEOTUpdate:
        """Consume an acoustic observation before any same-time score/timer."""
        if acoustic.ref != self.ref:
            raise ValueError("acoustic turn reference does not match controller")
        self._advance_clock(acoustic.observed_at_ms)
        at_ms = acoustic.observed_at_ms
        if self._closed or self._endpoint_emitted:
            return LocalAudioEOTUpdate(at_ms, "inactive")
        if acoustic.speech_active or acoustic.pause_duration_ms is None:
            cancelled = self._pending_id
            self._reset_pause()
            return LocalAudioEOTUpdate(at_ms, "speech_resumed", cancel_request_id=cancelled)

        pause_ms = acoustic.pause_duration_ms
        pause_start_ms = at_ms - pause_ms
        if pause_start_ms < 0:
            raise ValueError("pause duration cannot exceed observation time")
        cancelled = None
        if self._pause_start_ms is not None and (
            pause_start_ms != self._pause_start_ms or pause_ms < (self._last_pause_ms or 0)
        ):
            cancelled = self._pending_id
            self._reset_pause()
        self._pause_start_ms = pause_start_ms
        self._last_pause_ms = pause_ms
        return self._advance_pause(at_ms, pause_ms, cancelled)

    def receive_score(
        self, ref: TurnRef, request_id: int, received_at_ms: int, probability: float
    ) -> LocalAudioEOTUpdate:
        """Accept only the current request before its deadline and endpoint."""
        if ref != self.ref:
            return LocalAudioEOTUpdate(received_at_ms, "stale_turn")
        if received_at_ms < self._last_at_ms:
            return LocalAudioEOTUpdate(received_at_ms, "stale_time")
        self._advance_clock(received_at_ms)
        if self._closed or self._endpoint_emitted:
            return LocalAudioEOTUpdate(received_at_ms, "stale_turn")
        if self._pending_id != request_id or self._request_at_ms is None:
            return self._advance_pause_at(received_at_ms, "stale_request", allow_dispatch=False)
        if (
            self._pause_start_ms is not None
            and received_at_ms - self._pause_start_ms >= self.config.baseline_pause_ms
        ):
            return self._advance_pause_at(received_at_ms, "baseline_precedes_late_score")
        if received_at_ms >= self._request_at_ms + self.config.request_timeout_ms:
            return self._advance_pause_at(received_at_ms, "model_timeout")
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            self._pending_id = None
            self._request_at_ms = None
            self._failed_this_pause = True
            return self._advance_pause_at(received_at_ms, "invalid_score")
        self._pending_id = None
        self._request_at_ms = None
        self._score = probability
        return self._advance_pause_at(received_at_ms, "score_accepted")

    def fail_request(self, ref: TurnRef, request_id: int, failed_at_ms: int) -> LocalAudioEOTUpdate:
        """Model error is a bounded fallback, never an endpoint by itself."""
        if ref != self.ref:
            return LocalAudioEOTUpdate(failed_at_ms, "stale_turn")
        if failed_at_ms < self._last_at_ms:
            return LocalAudioEOTUpdate(failed_at_ms, "stale_time")
        self._advance_clock(failed_at_ms)
        if self._closed or self._endpoint_emitted:
            return LocalAudioEOTUpdate(failed_at_ms, "stale_turn")
        if self._pending_id != request_id:
            return self._advance_pause_at(failed_at_ms, "stale_request", allow_dispatch=False)
        self._pending_id = None
        self._request_at_ms = None
        self._failed_this_pause = True
        return self._advance_pause_at(failed_at_ms, "model_error")

    def tick(self, now_ms: int) -> LocalAudioEOTUpdate:
        """Advance a timer; caller must feed same-time speech observations first."""
        self._advance_clock(now_ms)
        if self._closed or self._endpoint_emitted or self._pause_start_ms is None:
            return LocalAudioEOTUpdate(now_ms, "inactive")
        return self._advance_pause(now_ms, now_ms - self._pause_start_ms)

    def close(self, now_ms: int) -> LocalAudioEOTUpdate:
        self._advance_clock(now_ms)
        cancelled = self._pending_id
        self._reset_pause()
        self._closed = True
        return LocalAudioEOTUpdate(now_ms, "closed", cancel_request_id=cancelled)

    def _advance_clock(self, at_ms: int) -> None:
        if at_ms < 0 or at_ms < self._last_at_ms:
            raise ValueError("event time must be non-negative and monotonic")
        self._last_at_ms = at_ms

    def _reset_pause(self) -> None:
        self._pause_start_ms = None
        self._last_pause_ms = None
        self._pending_id = None
        self._request_at_ms = None
        self._score = None
        self._failed_this_pause = False

    def _advance_pause_at(
        self, at_ms: int, reason: str, *, allow_dispatch: bool = True
    ) -> LocalAudioEOTUpdate:
        if self._pause_start_ms is None:
            return LocalAudioEOTUpdate(at_ms, reason)
        return self._advance_pause(
            at_ms, at_ms - self._pause_start_ms, reason=reason, allow_dispatch=allow_dispatch
        )

    def _advance_pause(
        self,
        at_ms: int,
        pause_ms: int,
        cancelled: int | None = None,
        *,
        reason: str = "wait",
        allow_dispatch: bool = True,
    ) -> LocalAudioEOTUpdate:
        if self._pending_id is not None and self._request_at_ms is not None:
            if at_ms >= self._request_at_ms + self.config.request_timeout_ms:
                cancelled = self._pending_id
                self._pending_id = None
                self._request_at_ms = None
                self._failed_this_pause = True
                reason = "model_timeout"

        deadline = self.config.baseline_pause_ms
        if self._score is not None:
            if self._score >= self.config.complete_threshold:
                deadline = self.config.early_pause_ms
            elif self._score <= self.config.incomplete_threshold:
                deadline = self.config.low_hold_pause_ms
        if pause_ms >= deadline:
            self._endpoint_emitted = True
            if self._pending_id is not None:
                cancelled = self._pending_id
                self._pending_id = None
                self._request_at_ms = None
            return LocalAudioEOTUpdate(
                at_ms,
                "endpoint_" + ("model" if self._score is not None else "baseline"),
                cancel_request_id=cancelled,
                endpoint_candidate=True,
            )

        if (
            allow_dispatch
            and self._pending_id is None
            and self._score is None
            and not self._failed_this_pause
            and pause_ms >= self.config.request_pause_ms
            and self._request_count < self.config.max_requests_per_turn
        ):
            request_id = self._next_request_id
            self._next_request_id += 1
            self._request_count += 1
            self._pending_id = request_id
            self._request_at_ms = at_ms
            return LocalAudioEOTUpdate(
                at_ms, "dispatch_model", request_id=request_id, cancel_request_id=cancelled
            )
        return LocalAudioEOTUpdate(at_ms, reason, cancel_request_id=cancelled)
