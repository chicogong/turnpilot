"""Causal pause-timing counterfactuals for standalone, text-only probes.

The caller supplies independently labeled completion and, for incomplete
thoughts, a scripted speech-resumption time. A Jev score is only visible after
its modeled arrival and only if it arrived within the request budget. This is
not an audio or ASR benchmark: the text-availability timestamp is an explicit
assumption, not something inferred from a final transcript.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from turnpilot.models import (
    AcousticSignal,
    AudioQuality,
    DirectiveKind,
    HostState,
    SemanticSignal,
    TranscriptSignal,
    TurnRef,
)
from turnpilot.policy import TurnPolicy
from turnpilot.replay import ReplayStep, replay

_END_TURN = frozenset(
    {
        DirectiveKind.COMMIT_USER_TURN,
        DirectiveKind.IGNORE_USER_TURN,
        DirectiveKind.CLARIFY_AUDIO,
        DirectiveKind.CLARIFY_MEANING,
    }
)


@dataclass(frozen=True, slots=True)
class PauseTimingCase:
    """One authored pause, with time zero at its first silent frame."""

    case_id: str
    is_complete: bool
    transcript_available_at_ms: int
    dispatch_at_ms: int
    resume_at_ms: int | None = None
    jev_latency_ms: int | None = None
    jev_complete_probability: float | None = None
    revised_transcript_at_ms: int | None = None
    revised_jev_latency_ms: int | None = None
    revised_jev_complete_probability: float | None = None

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("case_id must be non-empty")
        if self.transcript_available_at_ms < 0 or self.dispatch_at_ms < 0:
            raise ValueError("timestamps must be non-negative")
        if self.dispatch_at_ms < self.transcript_available_at_ms:
            raise ValueError("dispatch cannot precede transcript availability")
        if self.is_complete and self.resume_at_ms is not None:
            raise ValueError("complete thought must not have a resumption time")
        if not self.is_complete and (self.resume_at_ms is None or self.resume_at_ms <= 0):
            raise ValueError("incomplete thought requires a positive resumption time")
        if (self.jev_latency_ms is None) != (self.jev_complete_probability is None):
            raise ValueError("Jev latency and probability must be supplied together")
        if self.jev_latency_ms is not None and self.jev_latency_ms < 0:
            raise ValueError("Jev latency must be non-negative")
        if self.jev_complete_probability is not None and (
            not math.isfinite(self.jev_complete_probability)
            or not 0 <= self.jev_complete_probability <= 1
        ):
            raise ValueError("Jev probability must be finite and within [0, 1]")
        if self.revised_transcript_at_ms is not None and (
            self.revised_transcript_at_ms <= self.transcript_available_at_ms
            or self.revised_transcript_at_ms < self.dispatch_at_ms
        ):
            raise ValueError("revised transcript must follow the initial transcript and dispatch")
        if (self.revised_jev_latency_ms is None) != (self.revised_jev_complete_probability is None):
            raise ValueError("revised Jev latency and probability must be supplied together")
        if self.revised_jev_latency_ms is not None and self.revised_transcript_at_ms is None:
            raise ValueError("revised Jev response requires a revised transcript")
        if self.revised_jev_latency_ms is not None and self.revised_jev_latency_ms < 0:
            raise ValueError("revised Jev latency must be non-negative")
        if self.revised_jev_complete_probability is not None and (
            not math.isfinite(self.revised_jev_complete_probability)
            or not 0 <= self.revised_jev_complete_probability <= 1
        ):
            raise ValueError("revised Jev probability must be finite and within [0, 1]")


@dataclass(frozen=True, slots=True)
class PauseTimingOutcome:
    """First terminal recommendation, or none before speech resumes."""

    at_ms: int | None
    kind: DirectiveKind | None
    used_semantic: bool
    semantic_hold_encountered: bool


def simulate_pause(
    case: PauseTimingCase,
    *,
    use_jev: bool,
    request_budget_ms: int = 500,
    policy: TurnPolicy | None = None,
) -> PauseTimingOutcome:
    """Replay the same pause under an acoustic fallback and optional Jev.

    No response that arrives after the acoustic fallback can retrospectively
    change its action. A resume at time T wins over any decision at T.
    """
    if request_budget_ms <= 0:
        raise ValueError("request_budget_ms must be positive")
    current_policy = policy or TurnPolicy()
    config = current_policy.config
    arrival_ms: int | None = None
    revised_arrival_ms: int | None = None
    if use_jev and case.jev_latency_ms is not None and case.jev_latency_ms <= request_budget_ms:
        arrival_ms = case.dispatch_at_ms + case.jev_latency_ms
    if (
        use_jev
        and case.revised_transcript_at_ms is not None
        and case.revised_jev_latency_ms is not None
        and case.revised_jev_latency_ms <= request_budget_ms
    ):
        revised_arrival_ms = case.revised_transcript_at_ms + case.revised_jev_latency_ms

    times = {config.min_pause_ms, config.baseline_pause_ms, config.max_pause_ms}
    if config.partial_semantic_hold_extension_ms is not None:
        times.add(
            min(
                config.max_pause_ms,
                config.baseline_pause_ms + config.partial_semantic_hold_extension_ms,
            )
        )
    if arrival_ms is not None:
        times.add(arrival_ms)
    if case.revised_transcript_at_ms is not None:
        times.add(case.revised_transcript_at_ms)
    if revised_arrival_ms is not None:
        times.add(revised_arrival_ms)
    ref = TurnRef("standalone-timing", case.case_id, 0)
    steps: list[ReplayStep] = []
    for at_ms in sorted(times):
        if case.resume_at_ms is not None and at_ms >= case.resume_at_ms:
            break
        if at_ms > config.max_pause_ms:
            break
        transcript = None
        revision = 1
        transcript_available_at_ms = case.transcript_available_at_ms
        if at_ms >= case.transcript_available_at_ms:
            if case.revised_transcript_at_ms is not None and at_ms >= case.revised_transcript_at_ms:
                revision = 2
                transcript_available_at_ms = case.revised_transcript_at_ms
            transcript = TranscriptSignal(
                ref,
                transcript_available_at_ms,
                revision,
                "present",
                is_final=revision == 2,
            )
        semantic = None
        if (
            revision == 2
            and revised_arrival_ms is not None
            and at_ms >= revised_arrival_ms
            and transcript is not None
        ):
            assert case.revised_jev_complete_probability is not None
            semantic = SemanticSignal(
                ref,
                revised_arrival_ms,
                2,
                case.revised_jev_complete_probability,
                0.0,
                0.0,
                "timing-probe",
                response_probability=1.0,
            )
        elif (
            revision == 1
            and arrival_ms is not None
            and at_ms >= arrival_ms
            and transcript is not None
        ):
            assert case.jev_complete_probability is not None
            semantic = SemanticSignal(
                ref,
                arrival_ms,
                1,
                case.jev_complete_probability,
                0.0,
                0.0,
                "timing-probe",
                response_probability=1.0,
            )
        steps.append(
            ReplayStep(
                HostState(ref, at_ms, True),
                AcousticSignal(
                    ref,
                    at_ms,
                    False,
                    pause_duration_ms=at_ms,
                    audio_quality=AudioQuality.CLEAR,
                ),
                transcript,
                semantic,
            )
        )

    semantic_hold_encountered = False
    for decision in replay(current_policy, steps):
        semantic_hold_encountered |= decision.reason == "semantic_hold"
        if decision.kind in _END_TURN:
            return PauseTimingOutcome(
                decision.decided_at_ms,
                decision.kind,
                decision.used_semantic,
                semantic_hold_encountered,
            )
    return PauseTimingOutcome(None, None, False, semantic_hold_encountered)


def summarize_pauses(
    cases: tuple[PauseTimingCase, ...],
    *,
    use_jev: bool,
    request_budget_ms: int = 500,
    policy: TurnPolicy | None = None,
) -> dict[str, int | float | None]:
    """Aggregate-only comparison; never emits case IDs or transcript text."""
    if not cases:
        raise ValueError("at least one case is required")
    if len({case.case_id for case in cases}) != len(cases):
        raise ValueError("case IDs must be unique")
    outcomes = tuple(
        simulate_pause(case, use_jev=use_jev, request_budget_ms=request_budget_ms, policy=policy)
        for case in cases
    )
    complete = sum(case.is_complete for case in cases)
    incomplete = len(cases) - complete
    false_cutoffs = sum(
        not case.is_complete and outcome.at_ms is not None
        for case, outcome in zip(cases, outcomes, strict=True)
    )
    missed_completions = sum(
        case.is_complete and outcome.at_ms is None
        for case, outcome in zip(cases, outcomes, strict=True)
    )
    latencies = sorted(
        outcome.at_ms
        for case, outcome in zip(cases, outcomes, strict=True)
        if case.is_complete and outcome.at_ms is not None
    )

    def percentile(value: int) -> int | None:
        if not latencies:
            return None
        return latencies[math.ceil(len(latencies) * value / 100) - 1]

    return {
        "cases": len(cases),
        "complete": complete,
        "incomplete": incomplete,
        "false_cutoffs": false_cutoffs,
        "false_cutoff_rate": false_cutoffs / incomplete if incomplete else None,
        "missed_completions": missed_completions,
        "endpoint_p50_ms": percentile(50),
        "endpoint_p95_ms": percentile(95),
        "terminal_semantic_present": sum(outcome.used_semantic for outcome in outcomes),
        "semantic_holds": sum(outcome.semantic_hold_encountered for outcome in outcomes),
    }
