from __future__ import annotations

from dataclasses import replace

import pytest

from turnpilot import (
    AcousticSignal,
    AudioQuality,
    DirectiveKind,
    HostState,
    SemanticSignal,
    TranscriptSignal,
    TurnPolicy,
    TurnRef,
)
from turnpilot.evaluation import (
    LabeledCase,
    compare_labeled_arms,
    evaluate_labeled,
    paired_false_cutoff_interval,
)
from turnpilot.replay import ReplayStep


def step(now_ms: int, pause_ms: int, *, semantic_complete: float | None) -> ReplayStep:
    ref = TurnRef("session", f"turn-{now_ms}", 0)
    host = HostState(ref, now_ms, True)
    acoustic = AcousticSignal(
        ref, now_ms, False, pause_duration_ms=pause_ms, audio_quality=AudioQuality.CLEAR
    )
    transcript = TranscriptSignal(ref, now_ms - 100, 1, "测试文本")
    semantic = None
    if semantic_complete is not None:
        semantic = SemanticSignal(ref, now_ms - 20, 1, semantic_complete, 0.0, 0.0, "fake")
    return ReplayStep(host, acoustic, transcript, semantic)


def test_label_driven_scorecard_counts_false_cutoffs_and_latency() -> None:
    cases = [
        LabeledCase(
            "continues", step(1000, 320, semantic_complete=0.95), False, DirectiveKind.WAIT
        ),
        LabeledCase(
            "finished",
            step(1100, 640, semantic_complete=None),
            True,
            DirectiveKind.COMMIT_USER_TURN,
            1020,
        ),
        LabeledCase(
            "early", step(1200, 96, semantic_complete=None), True, DirectiveKind.WAIT, 1190
        ),
    ]
    summary = evaluate_labeled(TurnPolicy(), cases)
    assert summary.candidate_count == 3
    assert summary.incomplete_count == 1
    assert summary.false_cutoff_count == 1
    assert summary.false_cutoff_rate == 1.0
    assert summary.action_error_count == 1
    assert summary.action_error_rate == 1 / 3
    assert (DirectiveKind.WAIT, DirectiveKind.COMMIT_USER_TURN, 1) in summary.action_confusion
    assert summary.complete_count == 2
    assert summary.complete_without_endpoint_count == 1
    assert summary.endpoint_latency_ms == (80,)


def test_evaluation_requires_independent_label_and_unique_cases() -> None:
    first = LabeledCase("one", step(1000, 320, semantic_complete=None), False, DirectiveKind.WAIT)
    with pytest.raises(ValueError, match="true_eot_ms"):
        replace(first, is_complete=True)
    with pytest.raises(ValueError, match="unique"):
        evaluate_labeled(TurnPolicy(), [first, first])
    empty = evaluate_labeled(TurnPolicy(), [])
    assert empty.false_cutoff_rate is None
    assert empty.action_error_rate is None


def test_speaker_clustered_paired_interval_is_deterministic() -> None:
    baseline = tuple(
        LabeledCase(
            f"case-{index}",
            step(1000 + index, 640, semantic_complete=None),
            False,
            DirectiveKind.WAIT,
            speaker_id=f"speaker-{index}",
            device_id=f"device-{index}",
        )
        for index in range(10)
    )
    candidate = tuple(
        replace(case, step=step(1000 + index, 320, semantic_complete=None))
        for index, case in enumerate(baseline)
    )
    interval = paired_false_cutoff_interval(TurnPolicy(), baseline, TurnPolicy(), candidate)
    assert interval is not None
    assert (interval.delta, interval.lower, interval.upper) == (-1.0, -1.0, -1.0)
    assert interval.speaker_count == 10
    assert interval.incomplete_candidate_count == 10
    assert (
        paired_false_cutoff_interval(TurnPolicy(), baseline[:9], TurnPolicy(), candidate[:9])
        is None
    )
    assert (
        paired_false_cutoff_interval(
            TurnPolicy(),
            (replace(baseline[0], speaker_id=None), *baseline[1:]),
            TurnPolicy(),
            (replace(candidate[0], speaker_id=None), *candidate[1:]),
        )
        is None
    )


def test_paired_interval_rejects_unmatched_label_and_pause_eligibility() -> None:
    baseline = (
        LabeledCase(
            "one",
            step(1000, 640, semantic_complete=None),
            False,
            DirectiveKind.WAIT,
            speaker_id="speaker",
        ),
    )
    with pytest.raises(ValueError, match="same labels"):
        paired_false_cutoff_interval(
            TurnPolicy(), baseline, TurnPolicy(), (replace(baseline[0], speaker_id="other"),)
        )
    active = replace(baseline[0].step.acoustic, speech_active=True, pause_duration_ms=None)
    with pytest.raises(ValueError, match="same candidate pause eligibility"):
        paired_false_cutoff_interval(
            TurnPolicy(),
            baseline,
            TurnPolicy(),
            (replace(baseline[0], step=replace(baseline[0].step, acoustic=active)),),
        )


def test_evaluation_rejects_future_event_order() -> None:
    later = LabeledCase("later", step(1100, 320, semantic_complete=None), False, DirectiveKind.WAIT)
    earlier = LabeledCase(
        "earlier", step(1000, 320, semantic_complete=None), False, DirectiveKind.WAIT
    )
    with pytest.raises(ValueError, match="non-monotonic"):
        evaluate_labeled(TurnPolicy(), [later, earlier])


def test_four_arm_pairing_uses_identical_labels_and_candidate_time() -> None:
    def cases(pause_ms: int, semantic_score: float | None) -> tuple[LabeledCase, ...]:
        return (
            LabeledCase(
                "pause-1",
                step(1000, pause_ms, semantic_complete=semantic_score),
                False,
                DirectiveKind.WAIT,
            ),
        )

    arms = {
        "A": (TurnPolicy(), cases(320, None)),
        "B": (TurnPolicy(), cases(640, None)),
        "C": (TurnPolicy(), cases(320, 0.95)),
        "D": (TurnPolicy(), cases(96, 0.95)),
    }
    reports = compare_labeled_arms(arms)
    assert {name: item.false_cutoff_count for name, item in reports.items()} == {
        "A": 0,
        "B": 1,
        "C": 1,
        "D": 0,
    }
    bad = dict(arms)
    bad["D"] = (
        TurnPolicy(),
        (replace(cases(96, 0.95)[0], true_eot_ms=800),),
    )
    with pytest.raises(ValueError, match="same labels"):
        compare_labeled_arms(bad)

    active = replace(
        cases(96, 0.95)[0].step.acoustic,
        speech_active=True,
        pause_duration_ms=None,
    )
    bad["D"] = (
        TurnPolicy(),
        (replace(cases(96, 0.95)[0], step=replace(cases(96, 0.95)[0].step, acoustic=active)),),
    )
    with pytest.raises(ValueError, match="same labels"):
        compare_labeled_arms(bad)


def test_premature_latency_is_kept_separate_from_positive_percentiles() -> None:
    cases = (
        LabeledCase(
            "premature",
            step(1000, 640, semantic_complete=None),
            True,
            DirectiveKind.COMMIT_USER_TURN,
            1100,
        ),
        LabeledCase(
            "late",
            step(1200, 640, semantic_complete=None),
            True,
            DirectiveKind.COMMIT_USER_TURN,
            1050,
        ),
    )
    report = evaluate_labeled(TurnPolicy(), cases)
    assert report.endpoint_latency_ms == (-100, 150)
    assert report.premature_endpoint_count == 1
    assert report.endpoint_latency_percentile_ms(95) == 150


def test_barge_in_event_does_not_dilute_false_cutoff_denominator() -> None:
    pause = LabeledCase("pause", step(1000, 640, semantic_complete=None), False, DirectiveKind.WAIT)
    overlap = step(1100, 0, semantic_complete=None)
    overlap = replace(
        overlap,
        host=replace(overlap.host, assistant_speaking=True),
        acoustic=replace(
            overlap.acoustic,
            speech_active=True,
            pause_duration_ms=None,
            near_end_speech=True,
            speech_duration_ms=352,
        ),
    )
    barge_in = LabeledCase("barge-in", overlap, False, DirectiveKind.YIELD_ASSISTANT)
    report = evaluate_labeled(TurnPolicy(), [pause, barge_in])
    assert report.event_count == 2
    assert report.candidate_count == 1
    assert report.incomplete_count == 1
    assert report.false_cutoff_rate == 1.0
