"""Conservative, in-memory partial-ASR stability observations, not confidence.

An unchanged hypothesis across fresh paused audio is only an experimental
eligibility signal. It does not establish correct words or a completed turn.
"""

from __future__ import annotations

from turnpilot.models import AcousticSignal, HostState, TranscriptSignal


class PartialTranscriptStability:
    def __init__(self, min_stable_ms: int = 224, *, max_acoustic_age_ms: int = 200) -> None:
        if (
            isinstance(min_stable_ms, bool)
            or not isinstance(min_stable_ms, int)
            or min_stable_ms <= 0
            or isinstance(max_acoustic_age_ms, bool)
            or not isinstance(max_acoustic_age_ms, int)
            or max_acoustic_age_ms <= 0
        ):
            raise ValueError("invalid partial stability configuration")
        self.min_stable_ms = min_stable_ms
        self.max_acoustic_age_ms = max_acoustic_age_ms
        self.reset()

    def reset(self) -> None:
        self._transcript: TranscriptSignal | None = None
        self.started_at_ms: int | None = None
        self._first_observed_ms = self._last_observed_ms = 0
        self._first_pause_ms = self._last_pause_ms = 0
        self.ready = False

    def observe(
        self,
        host: HostState,
        acoustic: AcousticSignal,
        transcript: TranscriptSignal | None,
        *,
        input_backlogged: bool = False,
    ) -> bool:
        """Count progress only from current, advancing paused acoustic frames.

        Resume, backlog, invalid observations, final ASR or a changed revision
        reset the interval. Polling an unchanged acoustic snapshot cannot age
        a partial into eligibility. Words stay in memory and never in reports.
        """
        age = host.now_ms - acoustic.observed_at_ms
        pause = acoustic.pause_duration_ms
        if (
            not host.session_active
            or host.assistant_speaking
            or input_backlogged
            or acoustic.ref != host.ref
            or not 0 <= age <= self.max_acoustic_age_ms
            or acoustic.speech_active
            or pause is None
            or transcript is None
            or transcript.ref != host.ref
            or transcript.available_at_ms > host.now_ms
            or transcript.is_final
            or not transcript.text.strip()
        ):
            self.reset()
            return False
        changed = self._transcript != transcript
        backward = acoustic.observed_at_ms < self._last_observed_ms or pause < self._last_pause_ms
        gap = acoustic.observed_at_ms - self._last_observed_ms > self.max_acoustic_age_ms
        if changed or backward or gap:
            self.reset()
            self._transcript = transcript
            self.started_at_ms = host.now_ms
            # ASR can become available after this frame's acoustic observation.
            # Do not count that pre-ASR time toward hypothesis stability.
            self._first_observed_ms = host.now_ms
            self._first_pause_ms = pause
        # Both wall-clock observation progress and processed-media pause progress
        # must reach the interval; stalled decoders or repeated polls cannot help.
        self._last_observed_ms = acoustic.observed_at_ms
        self._last_pause_ms = pause
        self.ready = (
            acoustic.observed_at_ms - self._first_observed_ms >= self.min_stable_ms
            and pause - self._first_pause_ms >= self.min_stable_ms
        )
        return self.ready
