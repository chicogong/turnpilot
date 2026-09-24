"""Experimental probability-to-pause gate for offline acoustic comparisons.

This does not run a VAD model or establish speaker identity. It consumes a
model's probability and independently supplied near-end/echo observations.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from turnpilot.models import AcousticSignal, AudioQuality, TurnRef


@dataclass(frozen=True, slots=True)
class AcousticFrame:
    ref: TurnRef
    observed_at_ms: int
    speech_probability: float
    energy_dbfs: float
    near_end_confirmed: bool = False
    echo_likely: bool = False
    audio_quality: AudioQuality = AudioQuality.UNKNOWN

    def __post_init__(self) -> None:
        if self.observed_at_ms < 0:
            raise ValueError("frame timestamp must be non-negative")
        if not math.isfinite(self.speech_probability) or not 0 <= self.speech_probability <= 1:
            raise ValueError("speech probability must be finite and within [0, 1]")
        if not math.isfinite(self.energy_dbfs):
            raise ValueError("frame energy must be finite")


@dataclass(frozen=True, slots=True)
class AdaptiveGateConfig:
    frame_ms: int = 32
    max_frame_gap_ms: int = 128
    start_probability: float = 0.5
    end_probability: float = 0.35
    max_start_probability: float = 0.75
    noise_reference_dbfs: float = -60.0
    min_snr_db: float = 8.0
    probability_gain_per_noise_db: float = 0.005
    noise_ema_alpha: float = 0.05
    start_frames: int = 3
    rearm_during_pause: bool = False

    def __post_init__(self) -> None:
        if self.frame_ms <= 0 or self.max_frame_gap_ms < self.frame_ms or self.start_frames <= 0:
            raise ValueError("invalid frame timing")
        if (
            not 0
            <= self.end_probability
            < self.start_probability
            <= self.max_start_probability
            <= 1
        ):
            raise ValueError("probability thresholds must be ordered")
        if (
            not math.isfinite(self.noise_reference_dbfs)
            or not math.isfinite(self.min_snr_db)
            or self.min_snr_db < 0
        ):
            raise ValueError("invalid noise configuration")
        if (
            not math.isfinite(self.probability_gain_per_noise_db)
            or self.probability_gain_per_noise_db < 0
            or not 0 < self.noise_ema_alpha <= 1
        ):
            raise ValueError("invalid adaptation configuration")


class AdaptiveSpeechGate:
    """Causal, stateful acoustic candidate producer; one instance per session.

    A new turn reference resets speech state. The caller owns turn generation,
    timeout, and any decision/application side effects. Thresholds are
    experimental and must be compared with a fixed-threshold arm.
    """

    def __init__(self, config: AdaptiveGateConfig | None = None) -> None:
        self.config = config or AdaptiveGateConfig()
        self.reset()

    @property
    def noise_floor_dbfs(self) -> float:
        return self._noise_floor_dbfs

    @property
    def start_threshold(self) -> float:
        increase = max(0.0, self._noise_floor_dbfs - self.config.noise_reference_dbfs)
        return min(
            self.config.max_start_probability,
            self.config.start_probability + self.config.probability_gain_per_noise_db * increase,
        )

    def reset(self) -> None:
        self._ref: TurnRef | None = None
        self._last_at_ms: int | None = None
        self._noise_floor_dbfs = self.config.noise_reference_dbfs
        self._start_hits = 0
        self._in_turn = False
        self._continuous_speech_ms = 0
        self._pause_ms = 0

    def observe(self, frame: AcousticFrame) -> AcousticSignal:
        if self._ref != frame.ref:
            if self._ref is not None:
                if frame.ref.session_id != self._ref.session_id:
                    raise ValueError("acoustic gate cannot mix sessions")
                if (
                    frame.ref.turn_id == self._ref.turn_id
                    and frame.ref.generation < self._ref.generation
                ):
                    raise ValueError("stale acoustic generation")
            self.reset()
            self._ref = frame.ref
        if self._last_at_ms is not None:
            gap = frame.observed_at_ms - self._last_at_ms
            if gap <= 0 or gap > self.config.max_frame_gap_ms:
                raise ValueError("non-monotonic or missing acoustic frames")
        self._last_at_ms = frame.observed_at_ms

        if frame.speech_probability < self.config.end_probability and not frame.echo_likely:
            alpha = self.config.noise_ema_alpha
            self._noise_floor_dbfs = (
                1 - alpha
            ) * self._noise_floor_dbfs + alpha * frame.energy_dbfs

        possible_speech = not (frame.echo_likely and not frame.near_end_confirmed)
        if not self._in_turn:
            clears_snr = frame.energy_dbfs >= self._noise_floor_dbfs + self.config.min_snr_db
            if possible_speech and clears_snr and frame.speech_probability >= self.start_threshold:
                self._start_hits += 1
            else:
                self._start_hits = 0
            if self._start_hits >= self.config.start_frames:
                self._in_turn = True
                self._continuous_speech_ms = self._start_hits * self.config.frame_ms
                self._start_hits = 0
        elif self.config.rearm_during_pause and self._pause_ms > 0:
            clears_snr = frame.energy_dbfs >= self._noise_floor_dbfs + self.config.min_snr_db
            if possible_speech and clears_snr and frame.speech_probability >= self.start_threshold:
                self._start_hits += 1
            else:
                self._start_hits = 0
            if self._start_hits >= self.config.start_frames:
                self._continuous_speech_ms = self._start_hits * self.config.frame_ms
                self._pause_ms = 0
                self._start_hits = 0
            else:
                self._pause_ms += self.config.frame_ms
                self._continuous_speech_ms = 0
        elif possible_speech and frame.speech_probability >= self.config.end_probability:
            self._continuous_speech_ms += self.config.frame_ms
            self._pause_ms = 0
        else:
            self._pause_ms += self.config.frame_ms
            self._continuous_speech_ms = 0

        speech_active = self._in_turn and self._pause_ms == 0
        return AcousticSignal(
            ref=frame.ref,
            observed_at_ms=frame.observed_at_ms,
            speech_active=speech_active,
            near_end_speech=speech_active and frame.near_end_confirmed,
            echo_likely=frame.echo_likely,
            speech_duration_ms=self._continuous_speech_ms if speech_active else 0,
            pause_duration_ms=self._pause_ms if self._in_turn and not speech_active else None,
            audio_quality=frame.audio_quality,
        )
