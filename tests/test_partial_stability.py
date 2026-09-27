from __future__ import annotations

from dataclasses import replace

import pytest

from turnpilot.action_gate import ProvisionalActionGate
from turnpilot.models import (
    AcousticSignal,
    DirectiveKind,
    HostState,
    SemanticSignal,
    TranscriptSignal,
    TurnRef,
)
from turnpilot.partial_stability import PartialTranscriptStability
from turnpilot.policy import PolicyConfig, TurnPolicy

REF = TurnRef("fixture", "turn", 0)
PARTIAL = TranscriptSignal(REF, 100, 1, "not logged")


def host(at: int) -> HostState:
    return HostState(REF, at, True)


def audio(at: int, pause: int) -> AcousticSignal:
    return AcousticSignal(REF, at, False, pause_duration_ms=pause)


def mature(tracker: PartialTranscriptStability) -> None:
    assert not tracker.observe(host(200), audio(200, 32), PARTIAL)
    assert not tracker.observe(host(312), audio(312, 144), PARTIAL)
    assert tracker.observe(host(424), audio(424, 256), PARTIAL)


def score(at: int = 424, **changes: object) -> SemanticSignal:
    return replace(SemanticSignal(REF, at, 1, 0.9, 0.0, 0.0, "fixture"), **changes)


@pytest.mark.parametrize("value", [0, -1, True, 1.5])
def test_invalid_interval(value: int) -> None:
    with pytest.raises(ValueError):
        PartialTranscriptStability(value)
    with pytest.raises(ValueError):
        PartialTranscriptStability(max_acoustic_age_ms=value)


def test_stability_requires_fresh_audio_and_media_progress() -> None:
    tracker = PartialTranscriptStability()
    assert not tracker.observe(host(200), audio(200, 32), PARTIAL)
    assert not tracker.observe(host(360), audio(200, 32), PARTIAL)
    assert not tracker.observe(host(360), audio(360, 32), PARTIAL)
    mature(tracker)
    assert tracker.started_at_ms == 200
    assert tracker.observe(host(440), audio(424, 256), PARTIAL)
    tracker.reset()
    assert not tracker.ready
    assert tracker.started_at_ms is None


def test_acoustic_time_before_partial_availability_does_not_count_as_stability() -> None:
    tracker = PartialTranscriptStability()
    text = replace(PARTIAL, available_at_ms=300)
    assert not tracker.observe(host(300), audio(200, 32), text)
    assert not tracker.observe(host(312), audio(312, 144), text)
    assert not tracker.observe(host(424), audio(424, 256), text)
    assert tracker.observe(host(536), audio(536, 368), text)


@pytest.mark.parametrize(
    "change",
    [
        "resume",
        "backlog",
        "inactive",
        "playback",
        "stale_audio",
        "future_audio",
        "wrong_audio_turn",
        "wrong_text_turn",
        "future_text",
        "empty",
        "final",
        "none",
        "missing_pause",
        "revision",
        "text",
        "pause_backward",
        "audio_backward",
        "gap",
    ],
)
def test_invalid_or_changed_evidence_restarts_interval(change: str) -> None:
    tracker = PartialTranscriptStability()
    mature(tracker)
    h, a, t = host(456), audio(456, 288), PARTIAL
    backlogged = change == "backlog"
    other = TurnRef("fixture", "next", 1)
    if change == "resume":
        a = replace(a, speech_active=True, pause_duration_ms=None)
    elif change == "inactive":
        h = replace(h, session_active=False)
    elif change == "playback":
        h = replace(h, assistant_speaking=True)
    elif change == "stale_audio":
        a = replace(a, observed_at_ms=100)
    elif change == "future_audio":
        a = replace(a, observed_at_ms=457)
    elif change == "wrong_audio_turn":
        a = replace(a, ref=other)
    elif change == "wrong_text_turn":
        t = replace(t, ref=other)
    elif change == "future_text":
        t = replace(t, available_at_ms=457)
    elif change == "empty":
        t = replace(t, text=" ")
    elif change == "final":
        t = replace(t, is_final=True)
    elif change == "none":
        t = None  # type: ignore[assignment]
    elif change == "missing_pause":
        a = replace(a, pause_duration_ms=None)
    elif change == "revision":
        t = replace(t, revision=2)
    elif change == "text":
        t = replace(t, text="changed without revision")
    elif change == "pause_backward":
        a = replace(a, pause_duration_ms=32)
    elif change == "audio_backward":
        a = replace(a, observed_at_ms=400)
    elif change == "gap":
        h, a = host(700), audio(700, 532)
    assert not tracker.observe(h, a, t, input_backlogged=backlogged)
    assert not tracker.ready


def ready_gate() -> ProvisionalActionGate:
    gate = ProvisionalActionGate(stable_partial_ms=224)
    assert gate.arm(host(200), audio(200, 224))
    for at, pause in ((200, 224), (312, 336)):
        assert gate.decide(host(at), audio(at, pause), PARTIAL).kind is DirectiveKind.WAIT
    return gate


def test_only_experiment_can_release_partial_with_current_request_provenance() -> None:
    strict = ProvisionalActionGate()
    assert strict.arm(host(424), audio(424, 448))
    assert strict.decide(host(424), audio(424, 448), PARTIAL, score()).kind is DirectiveKind.WAIT
    gate = ready_gate()
    decision = gate.decide(
        host(424), audio(424, 448), PARTIAL, score(), semantic_requested_at_ms=400
    )
    assert decision.kind is DirectiveKind.COMMIT_USER_TURN
    assert decision.reason == "semantic_complete_stable_partial"
    assert not PARTIAL.is_final
    assert gate.decide(host(456), audio(456, 480), PARTIAL, score()).reason == "already_emitted"


@pytest.mark.parametrize("requested", [None, 199, 425, 999])
def test_partial_requires_request_start_in_current_stability_epoch(requested: int | None) -> None:
    decision = ready_gate().decide(
        host(424), audio(424, 448), PARTIAL, score(), semantic_requested_at_ms=requested
    )
    assert decision.kind is DirectiveKind.WAIT


@pytest.mark.parametrize(
    "signal",
    [None, score(transcript_revision=2), score(complete_probability=0.5), score(at=100)],
)
def test_no_aligned_high_score_means_no_partial_release(signal: SemanticSignal | None) -> None:
    decision = ready_gate().decide(
        host(424), audio(424, 448), PARTIAL, signal, semantic_requested_at_ms=400
    )
    assert decision.kind is DirectiveKind.WAIT


def test_resume_and_backlog_reset_stability_and_reject_old_pause_score() -> None:
    gate = ready_gate()
    resumed = replace(audio(424, 0), speech_active=True, pause_duration_ms=None)
    assert gate.decide(host(424), resumed, PARTIAL).kind is DirectiveKind.WAIT
    assert gate.canceled_candidates == 1
    assert gate.arm(host(456), audio(456, 224))
    for at, pause in ((456, 224), (568, 336), (680, 448)):
        assert (
            gate.decide(
                host(at), audio(at, pause), PARTIAL, score(at=680), semantic_requested_at_ms=400
            ).kind
            is DirectiveKind.WAIT
        )
    assert (
        gate.decide(
            host(712), audio(712, 480), PARTIAL, score(at=712), input_backlogged=True
        ).reason
        == "input_backlogged"
    )
    assert (
        gate.decide(host(744), audio(744, 512), PARTIAL, score(at=744)).kind is DirectiveKind.WAIT
    )
    assert (
        gate.decide(host(1432), audio(1432, 1200), PARTIAL).kind is DirectiveKind.COMMIT_USER_TURN
    )


@pytest.mark.parametrize(
    "changes,kind",
    [
        ({"response_probability": 0.05}, DirectiveKind.IGNORE_USER_TURN),
        ({"clarification_probability": 0.95}, DirectiveKind.CLARIFY_MEANING),
    ],
)
def test_partial_experiment_preserves_policy_action_kind(
    changes: dict[str, object], kind: DirectiveKind
) -> None:
    assert (
        ready_gate()
        .decide(host(424), audio(424, 448), PARTIAL, score(**changes), semantic_requested_at_ms=400)
        .kind
        is kind
    )


def test_partial_experiment_does_not_override_host_policy_disabling_partial_commit() -> None:
    gate = ProvisionalActionGate(
        TurnPolicy(PolicyConfig(allow_partial_semantic_commit=False)), stable_partial_ms=224
    )
    assert gate.arm(host(200), audio(200, 224))
    for at, pause in ((200, 224), (312, 336), (424, 448)):
        assert (
            gate.decide(
                host(at), audio(at, pause), PARTIAL, score(at), semantic_requested_at_ms=200
            ).kind
            is DirectiveKind.WAIT
        )


def test_experiment_keeps_final_fastpath_and_resets_on_stop_turn_and_stale_audio() -> None:
    gate = ready_gate()
    assert gate.decide(host(424), audio(100, 448), PARTIAL).kind is DirectiveKind.NO_ACTION
    assert (
        gate.decide(host(424), audio(424, 448), replace(PARTIAL, is_final=True), score()).kind
        is DirectiveKind.COMMIT_USER_TURN
    )
    assert gate.decide(replace(host(456), session_active=False), audio(456, 480)).kind is (
        DirectiveKind.NO_ACTION
    )
    assert gate.decide(host(488), audio(488, 512), PARTIAL).reason == "no_endpoint_candidate"
    next_ref = TurnRef("fixture", "next", 1)
    next_host = replace(host(520), ref=next_ref)
    assert gate.arm(next_host, replace(audio(520, 544), ref=next_ref))
    assert (
        gate.decide(
            next_host, replace(audio(520, 544), ref=next_ref), replace(PARTIAL, ref=next_ref)
        ).kind
        is DirectiveKind.WAIT
    )
