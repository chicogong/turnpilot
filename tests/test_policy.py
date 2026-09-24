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
from turnpilot.policy import PolicyConfig, decision_is_current
from turnpilot.replay import ReplayStep, replay

REF = TurnRef("session-1", "turn-1", 0)


def host(**changes: object) -> HostState:
    return replace(HostState(ref=REF, now_ms=1000, session_active=True), **changes)


def acoustic(**changes: object) -> AcousticSignal:
    return replace(
        AcousticSignal(
            ref=REF,
            observed_at_ms=900,
            speech_active=False,
            pause_duration_ms=320,
            audio_quality=AudioQuality.CLEAR,
        ),
        **changes,
    )


def transcript(**changes: object) -> TranscriptSignal:
    return replace(TranscriptSignal(REF, 850, 2, "明天下午三点", False), **changes)


def semantic(**changes: object) -> SemanticSignal:
    return replace(SemanticSignal(REF, 950, 2, 0.9, 0.1, 0.0, "jev-1.13.0"), **changes)


def test_short_pause_waits_even_with_high_semantic_score() -> None:
    decision = TurnPolicy().decide(host(), acoustic(pause_duration_ms=96), transcript(), semantic())
    assert decision.kind is DirectiveKind.WAIT
    assert decision.reason == "minimum_pause"


def test_complete_semantic_can_commit_before_acoustic_baseline() -> None:
    decision = TurnPolicy().decide(host(), acoustic(), transcript(), semantic())
    assert decision.kind is DirectiveKind.COMMIT_USER_TURN
    assert decision.used_semantic
    assert decision.ref == REF


def test_incomplete_semantic_holds_until_maximum_deadline() -> None:
    policy = TurnPolicy()
    holding = policy.decide(
        host(), acoustic(pause_duration_ms=800), transcript(), semantic(complete_probability=0.1)
    )
    deadline = policy.decide(
        host(now_ms=1500),
        acoustic(observed_at_ms=1450, pause_duration_ms=1200),
        transcript(),
        semantic(received_at_ms=1460, complete_probability=0.1),
    )
    assert holding.kind is DirectiveKind.WAIT
    assert deadline.kind is DirectiveKind.COMMIT_USER_TURN
    assert deadline.reason == "max_pause"


def test_no_semantic_uses_acoustic_fallback() -> None:
    policy = TurnPolicy()
    waiting = policy.decide(host(), acoustic(pause_duration_ms=320), transcript())
    committed = policy.decide(host(), acoustic(pause_duration_ms=640), transcript())
    assert waiting.kind is DirectiveKind.WAIT
    assert committed.kind is DirectiveKind.COMMIT_USER_TURN
    assert not committed.used_semantic


def test_bad_audio_and_ambiguous_meaning_are_different_actions() -> None:
    policy = TurnPolicy()
    bad_audio = policy.decide(
        host(), acoustic(audio_quality=AudioQuality.POOR), transcript(), semantic()
    )
    ambiguous = policy.decide(
        host(), acoustic(), transcript(), semantic(clarification_probability=0.9)
    )
    assert bad_audio.kind is DirectiveKind.CLARIFY_AUDIO
    assert ambiguous.kind is DirectiveKind.CLARIFY_MEANING


def test_future_transcript_and_old_semantic_cannot_force_early_commit() -> None:
    policy = TurnPolicy()
    future_text = policy.decide(host(), acoustic(), transcript(available_at_ms=1100), semantic())
    old_generation = policy.decide(
        host(ref=TurnRef("session-1", "turn-1", 1)),
        acoustic(ref=TurnRef("session-1", "turn-1", 1)),
        transcript(ref=TurnRef("session-1", "turn-1", 1)),
        semantic(),
    )
    assert future_text.kind is DirectiveKind.WAIT
    assert old_generation.kind is DirectiveKind.WAIT


def test_stale_acoustic_or_stopped_session_does_nothing() -> None:
    policy = TurnPolicy()
    stale = policy.decide(host(), acoustic(ref=TurnRef("other", "turn-1", 0)))
    stopped = policy.decide(host(session_active=False), acoustic(), transcript(), semantic())
    assert stale.kind is DirectiveKind.NO_ACTION
    assert stopped.kind is DirectiveKind.NO_ACTION


def test_aged_acoustic_and_semantic_are_not_used() -> None:
    policy = TurnPolicy()
    old_acoustic = policy.decide(host(), acoustic(observed_at_ms=799), transcript(), semantic())
    old_semantic = policy.decide(
        host(), acoustic(), transcript(available_at_ms=400), semantic(received_at_ms=499)
    )
    assert old_acoustic.kind is DirectiveKind.NO_ACTION
    assert old_semantic.kind is DirectiveKind.WAIT
    assert not old_semantic.used_semantic


def test_host_rejects_late_or_wrong_generation_directive() -> None:
    decision = TurnPolicy().decide(host(), acoustic(), transcript(), semantic())
    assert decision_is_current(decision, host(now_ms=1100))
    assert not decision_is_current(decision, host(now_ms=1151))
    assert not decision_is_current(decision, host(session_active=False))
    assert not decision_is_current(decision, host(ref=TurnRef("session-1", "turn-1", 1)))


def test_fast_barge_in_never_needs_jev() -> None:
    decision = TurnPolicy().decide(
        host(assistant_speaking=True),
        acoustic(
            speech_active=True,
            pause_duration_ms=None,
            near_end_speech=True,
            speech_duration_ms=352,
        ),
    )
    assert decision.kind is DirectiveKind.YIELD_ASSISTANT
    assert not decision.used_semantic


def test_echo_and_backchannel_do_not_yield_assistant() -> None:
    policy = TurnPolicy()
    talking = acoustic(
        speech_active=True,
        pause_duration_ms=None,
        near_end_speech=True,
        speech_duration_ms=352,
    )
    echo = policy.decide(host(assistant_speaking=True), replace(talking, echo_likely=True))
    backchannel = policy.decide(
        host(assistant_speaking=True),
        talking,
        transcript(),
        semantic(backchannel_probability=0.95),
    )
    assert echo.kind is DirectiveKind.NO_ACTION
    assert backchannel.kind is DirectiveKind.NO_ACTION


def test_nudge_is_opt_in_once_and_never_during_tool_or_assistant_speech() -> None:
    policy = TurnPolicy()
    waiting = acoustic(pause_duration_ms=None)
    eligible = host(asked_question=True, allow_nudge=True, idle_duration_ms=12000)
    assert policy.decide(eligible, waiting).kind is DirectiveKind.NUDGE_USER
    assert policy.decide(replace(eligible, nudge_count=1), waiting).kind is DirectiveKind.NO_ACTION
    assert (
        policy.decide(replace(eligible, tool_running=True), waiting).kind is DirectiveKind.NO_ACTION
    )
    assert (
        policy.decide(replace(eligible, assistant_speaking=True), waiting).kind
        is DirectiveKind.NO_ACTION
    )


def test_replay_rejects_time_reversal_and_preserves_order() -> None:
    policy = TurnPolicy()
    steps = [
        ReplayStep(host(), acoustic(), transcript()),
        ReplayStep(host(now_ms=1100), acoustic(observed_at_ms=1050), transcript(), semantic()),
    ]
    assert [item.kind for item in replay(policy, steps)] == [
        DirectiveKind.WAIT,
        DirectiveKind.COMMIT_USER_TURN,
    ]
    with pytest.raises(ValueError, match="non-monotonic"):
        replay(policy, reversed(steps))


def test_invalid_config_rejected() -> None:
    with pytest.raises(ValueError, match="ordered"):
        PolicyConfig(min_pause_ms=800, baseline_pause_ms=640)


def test_guarded_candidate_does_not_early_commit_from_partial_score() -> None:
    policy = TurnPolicy(
        PolicyConfig(allow_partial_semantic_commit=False, partial_semantic_hold_extension_ms=100)
    )
    partial = policy.decide(host(), acoustic(), transcript(), semantic())
    final = policy.decide(host(), acoustic(), transcript(is_final=True), semantic())
    assert partial.kind is DirectiveKind.WAIT
    assert partial.reason == "acoustic_fallback_wait"
    assert final.kind is DirectiveKind.COMMIT_USER_TURN
    assert final.reason == "semantic_complete"


def test_guarded_partial_hold_has_a_bounded_extension() -> None:
    policy = TurnPolicy(
        PolicyConfig(allow_partial_semantic_commit=False, partial_semantic_hold_extension_ms=100)
    )
    low = semantic(complete_probability=0.05)
    before_cap = policy.decide(host(), acoustic(pause_duration_ms=739), transcript(), low)
    at_cap = policy.decide(host(), acoustic(pause_duration_ms=740), transcript(), low)
    final_at_cap = policy.decide(
        host(), acoustic(pause_duration_ms=740), transcript(is_final=True), low
    )
    assert before_cap.kind is DirectiveKind.WAIT
    assert before_cap.reason == "semantic_hold"
    assert at_cap.kind is DirectiveKind.COMMIT_USER_TURN
    assert at_cap.reason == "acoustic_fallback"
    assert final_at_cap.kind is DirectiveKind.WAIT
    assert final_at_cap.reason == "semantic_hold"


def test_guarded_partial_hold_zero_extension_uses_baseline() -> None:
    policy = TurnPolicy(PolicyConfig(partial_semantic_hold_extension_ms=0))
    low = semantic(complete_probability=0.05)
    assert (
        policy.decide(host(), acoustic(pause_duration_ms=640), transcript(), low).kind
        is DirectiveKind.COMMIT_USER_TURN
    )
    with pytest.raises(ValueError, match="non-negative"):
        PolicyConfig(partial_semantic_hold_extension_ms=-1)
