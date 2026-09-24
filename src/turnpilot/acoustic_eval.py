"""Label-blind acoustic endpoint replay for local corpus diagnostics.

The input contains only causal frame observations. Publisher labels are joined
afterward by a separate probe, never fed into the detector.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from turnpilot.acoustic import AcousticFrame, AdaptiveGateConfig, AdaptiveSpeechGate
from turnpilot.models import TurnRef


@dataclass(frozen=True, slots=True)
class AcousticPoint:
    observed_at_ms: int
    speech_probability: float
    energy_dbfs: float


@dataclass(frozen=True, slots=True)
class AcousticEndpointTrace:
    endpoint_times_ms: tuple[int, ...]
    active_frame_times_ms: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class LabeledPauseWindow:
    """Post-hoc annotation interval; never used by the acoustic replay."""

    end_ms: int
    next_speech_or_horizon_ms: int
    is_complete: bool
    start_ms: int | None = None

    def __post_init__(self) -> None:
        if self.end_ms < 0 or self.next_speech_or_horizon_ms <= self.end_ms:
            raise ValueError("invalid labeled pause window")
        if self.start_ms is not None and not 0 <= self.start_ms <= self.end_ms:
            raise ValueError("invalid labeled speech span")


def replay_acoustic_endpoints(
    frames: Sequence[AcousticPoint],
    *,
    config: AdaptiveGateConfig | None = None,
    pause_deadline_ms: int = 640,
) -> AcousticEndpointTrace:
    """Close each detected turn at a fixed pause deadline, then causally re-arm.

    This is an endpoint-only experiment; no user identity, ASR, semantics, or
    playback actions are inferred. A new TurnRef is created only after this
    arm's own endpoint event, never from a publisher annotation.
    """
    if pause_deadline_ms <= 0:
        raise ValueError("pause deadline must be positive")
    gate = AdaptiveSpeechGate(config)
    generation = 0
    endpoints: list[int] = []
    active_frames: list[int] = []
    for point in frames:
        ref = TurnRef("corpus-replay", "channel", generation)
        signal = gate.observe(
            AcousticFrame(
                ref,
                point.observed_at_ms,
                point.speech_probability,
                point.energy_dbfs,
            )
        )
        if signal.speech_active:
            active_frames.append(point.observed_at_ms)
        if signal.pause_duration_ms is not None and signal.pause_duration_ms >= pause_deadline_ms:
            endpoints.append(point.observed_at_ms)
            generation += 1
    return AcousticEndpointTrace(tuple(endpoints), tuple(active_frames))


def summarize_endpoint_alignment(
    windows: Sequence[LabeledPauseWindow], trace: AcousticEndpointTrace
) -> dict[str, int | None]:
    """Associate causal endpoint events with independent publisher labels.

    A speech resumption at T wins over an endpoint at T. This is a diagnostic
    alignment, not a false-cutoff rate for a deployed voice agent.
    """
    if not windows:
        raise ValueError("at least one labeled window is required")
    complete = 0
    complete_matched = 0
    incomplete = 0
    incomplete_endpoint_before_resume = 0
    unobserved = 0
    latencies: list[int] = []
    for window in windows:
        if window.is_complete:
            complete += 1
        else:
            incomplete += 1
        if not _speech_was_observed(window, trace):
            unobserved += 1
            continue
        first = _first_endpoint_in_window(window, trace)
        if window.is_complete:
            if first is not None:
                complete_matched += 1
                latencies.append(first - window.end_ms)
        else:
            incomplete_endpoint_before_resume += first is not None
    latencies.sort()

    def percentile(numerator: int, denominator: int) -> int | None:
        if not latencies:
            return None
        index = (numerator * len(latencies) + denominator - 1) // denominator - 1
        return latencies[index]

    return {
        "complete_windows": complete,
        "complete_endpoint_within_window": complete_matched,
        "incomplete_windows": incomplete,
        "incomplete_endpoint_before_resume": incomplete_endpoint_before_resume,
        "acoustic_unobserved_label_windows": unobserved,
        "complete_match_latency_p50_ms": percentile(50, 100),
        "complete_match_latency_p95_ms": percentile(95, 100),
    }


def paired_endpoint_alignment(
    windows: Sequence[LabeledPauseWindow],
    reference: AcousticEndpointTrace,
    candidate: AcousticEndpointTrace,
) -> dict[str, dict[str, int]]:
    """Count paired endpoint outcomes only where both arms observed the speech.

    Endpoint presence is adverse for incomplete windows and desirable for
    complete windows. Excluded windows remain visible in each label's count.
    """
    if not windows:
        raise ValueError("at least one labeled window is required")
    outcomes = {
        label: {
            "both_endpoint": 0,
            "reference_only_endpoint": 0,
            "candidate_only_endpoint": 0,
            "neither_endpoint": 0,
            "not_jointly_observed": 0,
        }
        for label in ("complete", "incomplete")
    }
    for window in windows:
        counts = outcomes["complete" if window.is_complete else "incomplete"]
        if not _speech_was_observed(window, reference) or not _speech_was_observed(
            window, candidate
        ):
            counts["not_jointly_observed"] += 1
            continue
        reference_endpoint = _first_endpoint_in_window(window, reference) is not None
        candidate_endpoint = _first_endpoint_in_window(window, candidate) is not None
        if reference_endpoint and candidate_endpoint:
            counts["both_endpoint"] += 1
        elif reference_endpoint:
            counts["reference_only_endpoint"] += 1
        elif candidate_endpoint:
            counts["candidate_only_endpoint"] += 1
        else:
            counts["neither_endpoint"] += 1
    return outcomes


def _speech_was_observed(window: LabeledPauseWindow, trace: AcousticEndpointTrace) -> bool:
    return window.start_ms is None or any(
        window.start_ms <= at_ms <= window.end_ms for at_ms in trace.active_frame_times_ms
    )


def _first_endpoint_in_window(
    window: LabeledPauseWindow, trace: AcousticEndpointTrace
) -> int | None:
    return next(
        (
            at_ms
            for at_ms in trace.endpoint_times_ms
            if window.end_ms <= at_ms < window.next_speech_or_horizon_ms
        ),
        None,
    )
