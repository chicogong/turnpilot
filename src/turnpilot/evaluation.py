"""Small, label-driven offline scorecard; labels must be human adjudicated."""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

from turnpilot.models import DirectiveKind, TurnRef
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
class LabeledCase:
    """One candidate decision with an independent expected action.

    `is_complete` refers to a human-labeled user thought, not a VAD/ASR/model
    output. `true_eot_ms` is required for complete thoughts only.
    """

    case_id: str
    step: ReplayStep
    is_complete: bool
    expected_action: DirectiveKind
    true_eot_ms: int | None = None
    speaker_id: str | None = None
    device_id: str | None = None

    def __post_init__(self) -> None:
        if not self.case_id:
            raise ValueError("case_id must be non-empty")
        if self.is_complete and self.true_eot_ms is None:
            raise ValueError("complete cases need a human-labeled true_eot_ms")
        if self.true_eot_ms is not None and self.true_eot_ms < 0:
            raise ValueError("true_eot_ms must be non-negative")
        if self.speaker_id == "" or self.device_id == "":
            raise ValueError("speaker_id and device_id must be non-empty when provided")


@dataclass(frozen=True, slots=True)
class EvaluationSummary:
    event_count: int
    candidate_count: int
    incomplete_count: int
    false_cutoff_count: int
    action_error_count: int
    complete_count: int
    complete_without_endpoint_count: int
    endpoint_latency_ms: tuple[int, ...]
    action_confusion: tuple[tuple[DirectiveKind, DirectiveKind, int], ...]

    @property
    def false_cutoff_rate(self) -> float | None:
        if not self.incomplete_count:
            return None
        return self.false_cutoff_count / self.incomplete_count

    @property
    def action_error_rate(self) -> float | None:
        return self.action_error_count / self.event_count if self.event_count else None

    @property
    def premature_endpoint_count(self) -> int:
        return sum(latency < 0 for latency in self.endpoint_latency_ms)

    def endpoint_latency_percentile_ms(self, percentile: float) -> int | None:
        """Nearest-rank percentile over non-premature endpoints only."""
        if not 0 < percentile <= 100:
            raise ValueError("percentile must be within (0, 100]")
        values = sorted(value for value in self.endpoint_latency_ms if value >= 0)
        if not values:
            return None
        rank = max(1, math.ceil(len(values) * percentile / 100))
        return values[rank - 1]


def evaluate_labeled(policy: TurnPolicy, cases: Iterable[LabeledCase]) -> EvaluationSummary:
    """Score a causal event stream without exposing transcript or audio data.

    This is per-candidate accounting, not a session-level or user-perceived
    quality claim. Split, clustering, and uncertainty analysis belong above it.
    """
    items = tuple(cases)
    ids = [case.case_id for case in items]
    if len(ids) != len(set(ids)):
        raise ValueError("case IDs must be unique")
    decisions = replay(policy, (case.step for case in items))
    incomplete_count = 0
    false_cutoff_count = 0
    action_error_count = 0
    complete_count = 0
    complete_without_endpoint_count = 0
    latencies: list[int] = []
    confusion: Counter[tuple[DirectiveKind, DirectiveKind]] = Counter()
    for case, decision in zip(items, decisions, strict=True):
        confusion[case.expected_action, decision.kind] += 1
        terminal = decision.kind in _END_TURN
        is_pause_candidate = (
            case.step.acoustic.pause_duration_ms is not None
            and not case.step.acoustic.speech_active
        )
        if decision.kind != case.expected_action:
            action_error_count += 1
        if not is_pause_candidate:
            continue
        if case.is_complete:
            complete_count += 1
            if terminal:
                assert case.true_eot_ms is not None
                latencies.append(decision.decided_at_ms - case.true_eot_ms)
            else:
                complete_without_endpoint_count += 1
        else:
            incomplete_count += 1
            if terminal:
                false_cutoff_count += 1
    return EvaluationSummary(
        event_count=len(items),
        candidate_count=complete_count + incomplete_count,
        incomplete_count=incomplete_count,
        false_cutoff_count=false_cutoff_count,
        action_error_count=action_error_count,
        complete_count=complete_count,
        complete_without_endpoint_count=complete_without_endpoint_count,
        endpoint_latency_ms=tuple(latencies),
        action_confusion=tuple(
            (expected, actual, count)
            for (expected, actual), count in sorted(
                confusion.items(), key=lambda item: (item[0][0].value, item[0][1].value)
            )
        ),
    )


@dataclass(frozen=True, slots=True)
class PairedRateInterval:
    """Speaker-clustered candidate-minus-reference false-cutoff rate interval."""

    delta: float
    lower: float
    upper: float
    speaker_count: int
    incomplete_candidate_count: int
    bootstrap_iterations: int


def paired_false_cutoff_interval(
    reference_policy: TurnPolicy,
    reference_cases: Iterable[LabeledCase],
    candidate_policy: TurnPolicy,
    candidate_cases: Iterable[LabeledCase],
    *,
    min_speakers: int = 10,
    bootstrap_iterations: int = 2000,
    seed: int = 0,
) -> PairedRateInterval | None:
    """Resample speakers, retaining all paired incomplete pauses per speaker.

    Returns None for missing speaker IDs or too few clusters. The interval
    describes sampling uncertainty in the supplied corpus only; it does not
    establish speaker/device independence or corpus representativeness.
    """
    if min_speakers < 2 or bootstrap_iterations < 100:
        raise ValueError("min_speakers >= 2 and bootstrap_iterations >= 100 required")
    reference = tuple(reference_cases)
    candidate = tuple(candidate_cases)
    if len(reference) != len(candidate):
        raise ValueError("paired arms must contain the same cases")
    for left, right in zip(reference, candidate, strict=True):
        if (
            left.case_id,
            left.is_complete,
            left.expected_action,
            left.true_eot_ms,
            left.speaker_id,
            left.device_id,
            left.step.host.ref,
            left.step.host.now_ms,
        ) != (
            right.case_id,
            right.is_complete,
            right.expected_action,
            right.true_eot_ms,
            right.speaker_id,
            right.device_id,
            right.step.host.ref,
            right.step.host.now_ms,
        ):
            raise ValueError("paired arms must use the same labels and candidate times")
    left_decisions = replay(reference_policy, (case.step for case in reference))
    right_decisions = replay(candidate_policy, (case.step for case in candidate))
    clusters: dict[str, list[int]] = {}
    for left, right, left_decision, right_decision in zip(
        reference, candidate, left_decisions, right_decisions, strict=True
    ):
        left_is_pause = (
            not left.step.acoustic.speech_active
            and left.step.acoustic.pause_duration_ms is not None
        )
        right_is_pause = (
            not right.step.acoustic.speech_active
            and right.step.acoustic.pause_duration_ms is not None
        )
        if left_is_pause != right_is_pause:
            raise ValueError("paired arms must use the same candidate pause eligibility")
        if left.is_complete or not left_is_pause:
            continue
        if left.speaker_id is None:
            return None
        counts = clusters.setdefault(left.speaker_id, [0, 0, 0])
        counts[0] += 1
        counts[1] += int(left_decision.kind in _END_TURN)
        counts[2] += int(right_decision.kind in _END_TURN)
    if len(clusters) < min_speakers:
        return None
    values = tuple(clusters.values())
    total = sum(item[0] for item in values)
    delta = sum(item[2] - item[1] for item in values) / total
    rng = random.Random(seed)
    draws = []
    for _ in range(bootstrap_iterations):
        sample = rng.choices(values, k=len(values))
        denominator = sum(item[0] for item in sample)
        draws.append(sum(item[2] - item[1] for item in sample) / denominator)
    draws.sort()
    return PairedRateInterval(
        delta=delta,
        lower=draws[math.floor(0.025 * bootstrap_iterations)],
        upper=draws[math.ceil(0.975 * bootstrap_iterations) - 1],
        speaker_count=len(clusters),
        incomplete_candidate_count=total,
        bootstrap_iterations=bootstrap_iterations,
    )


def compare_labeled_arms(
    arms: Mapping[str, tuple[TurnPolicy, Iterable[LabeledCase]]],
) -> dict[str, EvaluationSummary]:
    """Compare policy arms on the same labeled, clocked candidate events.

    This checks pairing, not corpus representativeness or statistical
    uncertainty. The unchanged production reference A0 remains external.
    """
    if not arms:
        raise ValueError("at least one arm is required")
    reference: (
        tuple[
            tuple[str, bool, DirectiveKind, int | None, str | None, str | None, TurnRef, int, bool],
            ...,
        ]
        | None
    ) = None
    prepared: dict[str, tuple[TurnPolicy, tuple[LabeledCase, ...]]] = {}
    for name, (policy, source) in arms.items():
        if not name:
            raise ValueError("arm name must be non-empty")
        cases = tuple(source)
        identity = tuple(
            (
                case.case_id,
                case.is_complete,
                case.expected_action,
                case.true_eot_ms,
                case.speaker_id,
                case.device_id,
                case.step.host.ref,
                case.step.host.now_ms,
                (
                    case.step.acoustic.pause_duration_ms is not None
                    and not case.step.acoustic.speech_active
                ),
            )
            for case in cases
        )
        if reference is None:
            reference = identity
        elif identity != reference:
            raise ValueError("arms must use the same labels, turns, and candidate times")
        prepared[name] = (policy, cases)
    return {name: evaluate_labeled(policy, cases) for name, (policy, cases) in prepared.items()}
