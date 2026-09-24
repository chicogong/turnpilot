from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from turnpilot.action_gate import ProvisionalActionGate
from turnpilot.action_trace import (
    TraceEvent,
    TraceRecorder,
    load_trace_jsonl,
    main,
    replay_action_trace,
    replay_direct_asr_final,
)
from turnpilot.models import (
    AcousticSignal,
    AudioQuality,
    DirectiveKind,
    HostState,
    SemanticSignal,
    TranscriptSignal,
    TurnRef,
)

REF = TurnRef("session-1", "turn-1", 1)


def host(at_ms: int = 1000, **changes: object) -> HostState:
    return replace(HostState(REF, at_ms, True), **changes)


def acoustic(at_ms: int = 1000, pause: int | None = 320, **changes: object) -> AcousticSignal:
    return replace(
        AcousticSignal(
            REF, at_ms, False, pause_duration_ms=pause, audio_quality=AudioQuality.CLEAR
        ),
        **changes,
    )


def transcript(at_ms: int = 950, **changes: object) -> TranscriptSignal:
    return replace(TranscriptSignal(REF, at_ms, 1, "synthetic", True), **changes)


def semantic(at_ms: int = 980, **changes: object) -> SemanticSignal:
    return replace(SemanticSignal(REF, at_ms, 1, 0.9, 0.1, 0.0, "synthetic"), **changes)


def event(
    kind: str, at_ms: int, data: dict[str, object] | None = None, ref: TurnRef = REF
) -> TraceEvent:
    return TraceEvent(kind, at_ms, ref, data or {})


def audio_data(
    *,
    speech: bool = False,
    pause: int | None = 320,
    quality: str = "clear",
    duration: int = 0,
    near_end: bool = False,
    echo: bool = False,
) -> dict[str, object]:
    return {
        "speech_active": speech,
        "near_end_speech": near_end,
        "echo_likely": echo,
        "speech_duration_ms": duration,
        "pause_duration_ms": pause,
        "audio_quality": quality,
    }


def score_data(revision: int = 1, complete: float = 0.9) -> dict[str, object]:
    return {
        "transcript_revision": revision,
        "complete_probability": complete,
        "clarification_probability": 0.1,
        "backchannel_probability": 0.0,
        "response_probability": None,
    }


def test_gate_requires_candidate_and_final_aligned_score_before_cap() -> None:
    gate = ProvisionalActionGate()
    assert (
        gate.decide(host(), acoustic(), transcript(), semantic()).reason == "no_endpoint_candidate"
    )
    assert gate.arm(host(), acoustic())
    partial = gate.decide(host(), acoustic(), transcript(is_final=False), semantic())
    assert partial.kind is DirectiveKind.WAIT
    assert partial.reason == "await_final_or_max_pause"
    final = gate.decide(host(), acoustic(), transcript(), semantic())
    assert final.kind is DirectiveKind.COMMIT_USER_TURN
    assert final.used_semantic
    assert gate.decide(host(), acoustic(), transcript(), semantic()).reason == "already_emitted"


def test_gate_holds_acoustic_fallback_but_resolves_at_hard_cap() -> None:
    gate = ProvisionalActionGate()
    assert gate.arm(host(), acoustic())
    at_baseline = gate.decide(host(1320), acoustic(1320, 640), transcript(), None)
    at_cap = gate.decide(host(1880), acoustic(1880, 1200), transcript(), None)
    assert at_baseline.kind is DirectiveKind.WAIT
    assert at_cap.kind is DirectiveKind.COMMIT_USER_TURN
    assert at_cap.reason == "max_pause"


def test_gate_rejects_stale_semantic_and_cancels_on_continuation() -> None:
    gate = ProvisionalActionGate()
    assert gate.arm(host(), acoustic())
    outdated = gate.decide(host(), acoustic(), transcript(revision=2), semantic())
    assert outdated.kind is DirectiveKind.WAIT
    assert not outdated.used_semantic
    resumed = gate.decide(
        host(1050), acoustic(1050, None, speech_active=True, speech_duration_ms=500)
    )
    assert resumed.reason == "user_speaking"
    assert gate.canceled_candidates == 1
    assert gate.decide(host(1100), acoustic(1100, 420), transcript(), semantic()).reason == (
        "no_endpoint_candidate"
    )


def test_gate_preserves_audio_and_meaning_actions_and_fast_barge_in() -> None:
    poor = ProvisionalActionGate()
    assert poor.arm(host(), acoustic())
    unclear = poor.decide(
        host(), acoustic(audio_quality=AudioQuality.POOR), transcript(), semantic()
    )
    assert unclear.kind is DirectiveKind.CLARIFY_AUDIO
    empty = ProvisionalActionGate()
    assert empty.arm(host(), acoustic())
    assert (
        empty.decide(host(1880), acoustic(1880, 1200), transcript(text=""), None).kind
        is DirectiveKind.CLARIFY_AUDIO
    )
    meaning = ProvisionalActionGate()
    assert meaning.arm(host(), acoustic())
    assert (
        meaning.decide(
            host(), acoustic(), transcript(), semantic(clarification_probability=0.95)
        ).kind
        is DirectiveKind.CLARIFY_MEANING
    )
    barge = ProvisionalActionGate().decide(
        host(assistant_speaking=True),
        acoustic(
            speech_active=True, pause_duration_ms=None, near_end_speech=True, speech_duration_ms=400
        ),
    )
    assert barge.kind is DirectiveKind.YIELD_ASSISTANT
    ignored = ProvisionalActionGate()
    assert ignored.arm(host(), acoustic())
    assert (
        ignored.decide(host(), acoustic(), transcript(), semantic(response_probability=0.05)).kind
        is DirectiveKind.IGNORE_USER_TURN
    )
    echoed = ProvisionalActionGate().decide(
        host(assistant_speaking=True),
        acoustic(
            speech_active=True,
            pause_duration_ms=None,
            near_end_speech=True,
            echo_likely=True,
            speech_duration_ms=400,
        ),
    )
    assert echoed.kind is DirectiveKind.NO_ACTION


def test_gate_stops_and_rejects_stale_audio() -> None:
    gate = ProvisionalActionGate()
    assert not gate.arm(host(), acoustic(700))
    assert gate.decide(host(), acoustic(700)).kind is DirectiveKind.NO_ACTION
    assert gate.arm(host(), acoustic())
    assert gate.decide(host(session_active=False), acoustic()).kind is DirectiveKind.NO_ACTION
    assert gate.decide(host(), acoustic()).reason == "no_endpoint_candidate"


def test_content_free_roundtrip_and_causal_replay() -> None:
    recorder = TraceRecorder()
    for item in (
        event("turn_start", 0),
        event("acoustic", 1000, audio_data()),
        event("candidate", 1000),
        event("asr", 1050, {"revision": 1, "has_text": True, "is_final": False}),
        event("semantic", 1060, score_data()),
        event("acoustic", 1320, audio_data(pause=640)),
        event("asr", 1350, {"revision": 1, "has_text": True, "is_final": True}),
        event("semantic", 1360, score_data()),
        event("acoustic", 1380, audio_data(pause=700)),
    ):
        recorder.append(item)
    raw = recorder.to_jsonl()
    assert "synthetic" not in raw
    result = replay_action_trace(load_trace_jsonl(raw))
    assert [item.kind for item in result.decisions].count(DirectiveKind.COMMIT_USER_TURN) == 1
    assert result.summary()["action_counts"] == {"commit_user_turn": 1}
    assert result.candidate_count == 1


def test_recorder_freezes_payload_after_validation() -> None:
    payload = audio_data()
    item = event("acoustic", 1000, payload)
    payload["text"] = "should not enter trace"
    assert "text" not in item.to_dict()["data"]
    with pytest.raises(TypeError):
        item.data["text"] = "blocked"  # type: ignore[index]


def test_same_time_resume_wins_and_old_generation_is_ignored() -> None:
    newer = TurnRef("session-1", "turn-2", 2)
    trace = (
        event("turn_start", 0),
        event("acoustic", 1000, audio_data()),
        event("candidate", 1000),
        event("acoustic", 1300, audio_data(pause=620)),
        event("acoustic", 1300, audio_data(speech=True, pause=None, duration=400)),
        event("candidate", 1300),
        event("turn_start", 1400, ref=newer),
        event("semantic", 1500, score_data(), ref=REF),
        event("asr", 1500, {"revision": 1, "has_text": True, "is_final": True}, ref=REF),
        event("acoustic", 1600, audio_data(pause=1200), ref=newer),
        event("candidate", 1600, ref=newer),
    )
    result = replay_action_trace(trace)
    assert result.canceled_candidates == 1
    assert result.stale_events == 2
    assert result.summary()["action_counts"] == {"clarify_audio": 1}


def test_replay_playback_barge_in_and_stop_wins() -> None:
    trace = (
        event("turn_start", 0),
        event("playback", 100, {"speaking": True}),
        event("acoustic", 400, audio_data(speech=True, pause=None, duration=400, near_end=True)),
        event("stop", 500),
        event("acoustic", 500, audio_data(pause=1200)),
        event("candidate", 500),
    )
    result = replay_action_trace(trace)
    assert result.summary()["action_counts"] == {"yield_assistant": 1}
    assert result.candidate_count == 0


@pytest.mark.parametrize(
    "data",
    [
        {**audio_data(), "text": "private words"},
        {**audio_data(), "audio_path": "/private/file.wav"},
        {**audio_data(), "speech_active": 1},
        {**audio_data(), "audio_quality": []},
        {**score_data(), "complete_probability": float("nan")},
    ],
)
def test_trace_rejects_private_or_invalid_payload(data: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="invalid content-free"):
        TraceEvent("acoustic" if "speech_active" in data else "semantic", 1000, REF, data)


def test_trace_rejects_clock_reversal_duplicate_keys_and_oversize() -> None:
    recorder = TraceRecorder()
    recorder.append(event("turn_start", 100))
    with pytest.raises(ValueError, match="monotonic"):
        recorder.append(event("tick", 99))
    with pytest.raises(ValueError, match="invalid trace JSON"):
        load_trace_jsonl('{"schema":1,"schema":1}\n')
    with pytest.raises(ValueError, match="size limit"):
        load_trace_jsonl("x" * 2_000_001)
    with pytest.raises(ValueError, match="empty"):
        load_trace_jsonl(" \n")
    raw = event("turn_start", 0).to_dict()
    raw["transcript"] = "secret"
    with pytest.raises(ValueError, match="fields"):
        load_trace_jsonl(json.dumps(raw))


def test_replay_stale_revision_cannot_authorize_early_action() -> None:
    trace = (
        event("turn_start", 0),
        event("acoustic", 1000, audio_data()),
        event("candidate", 1000),
        event("asr", 1020, {"revision": 2, "has_text": True, "is_final": True}),
        event("semantic", 1030, score_data(revision=1)),
        event("acoustic", 1100, audio_data(pause=420)),
    )
    result = replay_action_trace(trace)
    assert result.stale_events == 1
    assert not result.summary()["action_counts"]


def test_invalid_cli_does_not_echo_trace_contents(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "invalid.jsonl"
    path.write_text('{"text":"private transcript"}\n', encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["turnpilot-action-replay", str(path)])
    with pytest.raises(SystemExit) as exit_info:
        main()
    assert exit_info.value.code == 2
    captured = capsys.readouterr()
    assert "private transcript" not in captured.out + captured.err
    assert "Invalid or unreadable action trace" in captured.err


def test_new_turn_id_can_share_generation_but_old_start_cannot_reactivate() -> None:
    newer = TurnRef("session-1", "turn-2", 1)
    result = replay_action_trace(
        (
            event("turn_start", 0),
            event("turn_start", 100, ref=newer),
            event("turn_start", 200, ref=REF),
            event("acoustic", 300, audio_data(pause=1200), ref=newer),
            event("candidate", 300, ref=newer),
        )
    )
    assert result.stale_events == 1
    assert result.summary()["action_counts"] == {"clarify_audio": 1}


def test_direct_asr_baseline_uses_first_nonempty_final_per_turn() -> None:
    newer = TurnRef("session-1", "turn-2", 2)
    result = replay_direct_asr_final(
        (
            event("turn_start", 0),
            event("asr", 100, {"revision": 1, "has_text": True, "is_final": False}),
            event("asr", 110, {"revision": 1, "has_text": False, "is_final": True}),
            event("asr", 120, {"revision": 2, "has_text": True, "is_final": True}),
            event("asr", 130, {"revision": 3, "has_text": True, "is_final": True}),
            event("turn_start", 200, ref=newer),
            event("acoustic", 210, audio_data(speech=True, pause=None), ref=newer),
            event(
                "asr",
                210,
                {"revision": 1, "has_text": True, "is_final": True},
                ref=newer,
            ),
            event("asr", 220, {"revision": 4, "has_text": True, "is_final": True}),
            event("stop", 300, ref=newer),
            event(
                "asr",
                300,
                {"revision": 2, "has_text": True, "is_final": True},
                ref=newer,
            ),
        )
    )
    assert [(item.at_ms, item.while_speech_active) for item in result.actions] == [
        (120, False),
        (210, True),
    ]
    assert result.summary()["action_count"] == 2
    assert result.summary()["while_speech_active"] == 1
    assert result.stale_events == 1


def test_direct_asr_same_time_stop_and_stale_revision_are_ignored() -> None:
    stopped = replay_direct_asr_final(
        (
            event("turn_start", 0),
            event("stop", 100),
            event("asr", 100, {"revision": 1, "has_text": True, "is_final": True}),
        )
    )
    assert not stopped.actions
    stale = replay_direct_asr_final(
        (
            event("turn_start", 0),
            event("asr", 100, {"revision": 2, "has_text": True, "is_final": False}),
            event("asr", 110, {"revision": 1, "has_text": True, "is_final": True}),
        )
    )
    assert not stale.actions
    assert stale.stale_events == 1


def test_comparison_cli_is_aggregate_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "trace.jsonl"
    recorder = TraceRecorder()
    recorder.append(event("turn_start", 0))
    recorder.append(event("asr", 100, {"revision": 1, "has_text": True, "is_final": True}))
    path.write_text(recorder.to_jsonl(), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["turnpilot-action-replay", str(path), "--compare-direct-asr"])
    main()
    output = capsys.readouterr().out
    assert "session-1" not in output
    assert "turn-1" not in output
    summary = json.loads(output)
    assert summary["direct_asr_final"]["action_count"] == 1
    assert summary["turnpilot"]["action_counts"] == {}
