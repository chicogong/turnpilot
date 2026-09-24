"""Content-free, same-clock event trace and causal provisional-action replay.

This is an explicit in-memory interface, not a microphone or ASR recorder. It
never accepts audio bytes, transcript text, paths, or arbitrary metadata.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field
from itertools import groupby
from pathlib import Path
from types import MappingProxyType
from typing import cast

from turnpilot.action_gate import ProvisionalActionGate
from turnpilot.models import (
    AcousticSignal,
    AudioQuality,
    Decision,
    DirectiveKind,
    HostState,
    SemanticSignal,
    TranscriptSignal,
    TurnRef,
)

_KINDS = frozenset(
    {"turn_start", "stop", "acoustic", "asr", "semantic", "playback", "candidate", "tick"}
)
_EMPTY = frozenset({"turn_start", "stop", "candidate", "tick"})
_ACTIONS = frozenset(
    {
        DirectiveKind.COMMIT_USER_TURN,
        DirectiveKind.IGNORE_USER_TURN,
        DirectiveKind.CLARIFY_AUDIO,
        DirectiveKind.CLARIFY_MEANING,
        DirectiveKind.YIELD_ASSISTANT,
    }
)
_ID = re.compile(r"[A-Za-z0-9_-]{1,64}\Z")
_MAX_BYTES = 2_000_000
_MAX_EVENTS = 10_000
_MAX_LINE = 4096


def _uint(value: object) -> bool:
    return type(value) is int and value >= 0


def _probability(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and 0.0 <= value <= 1.0


def _valid_data(kind: str, data: Mapping[str, object]) -> bool:
    if kind in _EMPTY:
        return not data
    if kind == "playback":
        return set(data) == {"speaking"} and type(data["speaking"]) is bool
    if kind == "asr":
        return (
            set(data) == {"revision", "has_text", "is_final"}
            and _uint(data["revision"])
            and type(data["has_text"]) is bool
            and type(data["is_final"]) is bool
        )
    if kind == "acoustic":
        return (
            set(data)
            == {
                "speech_active",
                "near_end_speech",
                "echo_likely",
                "speech_duration_ms",
                "pause_duration_ms",
                "audio_quality",
            }
            and all(
                type(data[key]) is bool
                for key in ("speech_active", "near_end_speech", "echo_likely")
            )
            and _uint(data["speech_duration_ms"])
            and (data["pause_duration_ms"] is None or _uint(data["pause_duration_ms"]))
            and type(data["audio_quality"]) is str
            and data["audio_quality"] in {quality.value for quality in AudioQuality}
            and not (data["speech_active"] and data["pause_duration_ms"] is not None)
        )
    if kind == "semantic":
        return (
            set(data)
            == {
                "transcript_revision",
                "complete_probability",
                "clarification_probability",
                "backchannel_probability",
                "response_probability",
            }
            and _uint(data["transcript_revision"])
            and all(
                _probability(data[key])
                for key in (
                    "complete_probability",
                    "clarification_probability",
                    "backchannel_probability",
                )
            )
            and (data["response_probability"] is None or _probability(data["response_probability"]))
        )
    return False


@dataclass(frozen=True, slots=True)
class TraceEvent:
    kind: str
    at_ms: int
    ref: TurnRef
    data: Mapping[str, object] = field(default_factory=dict, repr=False)

    def __post_init__(self) -> None:
        if (
            self.kind not in _KINDS
            or not _uint(self.at_ms)
            or not _ID.fullmatch(self.ref.session_id)
            or not _ID.fullmatch(self.ref.turn_id)
            or not _valid_data(self.kind, self.data)
        ):
            raise ValueError("invalid content-free trace event")
        object.__setattr__(self, "data", MappingProxyType(dict(self.data)))

    def to_dict(self) -> dict[str, object]:
        return {
            "schema": 1,
            "kind": self.kind,
            "at_ms": self.at_ms,
            "session_id": self.ref.session_id,
            "turn_id": self.ref.turn_id,
            "generation": self.ref.generation,
            "data": dict(self.data),
        }

    @classmethod
    def from_dict(cls, raw: object) -> TraceEvent:
        if not isinstance(raw, dict) or set(raw) != {
            "schema",
            "kind",
            "at_ms",
            "session_id",
            "turn_id",
            "generation",
            "data",
        }:
            raise ValueError("invalid trace event fields")
        if (
            type(raw["schema"]) is not int
            or raw["schema"] != 1
            or type(raw["kind"]) is not str
            or not _uint(raw["at_ms"])
            or type(raw["session_id"]) is not str
            or type(raw["turn_id"]) is not str
            or not _uint(raw["generation"])
            or type(raw["data"]) is not dict
        ):
            raise ValueError("invalid trace event types")
        return cls(
            kind=raw["kind"],
            at_ms=cast(int, raw["at_ms"]),
            ref=TurnRef(
                raw["session_id"],
                raw["turn_id"],
                cast(int, raw["generation"]),
            ),
            data=cast(dict[str, object], raw["data"]),
        )


class TraceRecorder:
    """Explicit, bounded in-memory collector; export only on caller request."""

    def __init__(self) -> None:
        self._events: list[TraceEvent] = []

    def append(self, event: TraceEvent) -> None:
        if len(self._events) >= _MAX_EVENTS:
            raise ValueError("trace event limit exceeded")
        if self._events and event.at_ms < self._events[-1].at_ms:
            raise ValueError("trace clock must be globally monotonic")
        self._events.append(event)

    @property
    def events(self) -> tuple[TraceEvent, ...]:
        return tuple(self._events)

    def to_jsonl(self) -> str:
        content = "".join(
            json.dumps(item.to_dict(), sort_keys=True) + "\n" for item in self._events
        )
        if len(content.encode("utf-8")) > _MAX_BYTES:
            raise ValueError("trace size limit exceeded")
        return content


def _no_duplicate_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate trace field")
        result[key] = value
    return result


def _no_constant(_value: str) -> object:
    raise ValueError("non-finite trace number")


def load_trace_jsonl(content: str) -> tuple[TraceEvent, ...]:
    if len(content.encode("utf-8")) > _MAX_BYTES:
        raise ValueError("trace size limit exceeded")
    recorder = TraceRecorder()
    for line in content.splitlines():
        if not line.strip():
            continue
        if len(line.encode("utf-8")) > _MAX_LINE:
            raise ValueError("trace line limit exceeded")
        try:
            raw = json.loads(
                line, object_pairs_hook=_no_duplicate_keys, parse_constant=_no_constant
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError("invalid trace JSON") from exc
        recorder.append(TraceEvent.from_dict(raw))
    if not recorder.events:
        raise ValueError("empty trace")
    return recorder.events


@dataclass(slots=True)
class _ReplayState:
    ref: TurnRef
    gate: ProvisionalActionGate = field(default_factory=ProvisionalActionGate)
    acoustic: AcousticSignal | None = None
    transcript: TranscriptSignal | None = None
    semantic: SemanticSignal | None = None
    assistant_speaking: bool = False


@dataclass(frozen=True, slots=True)
class TraceReplayResult:
    decisions: tuple[Decision, ...]
    candidate_count: int
    canceled_candidates: int
    stale_events: int

    def summary(self) -> dict[str, object]:
        counts = Counter(item.kind.value for item in self.decisions if item.kind in _ACTIONS)
        return {
            "evidence_level": "synthetic_or_declared_trace_only",
            "action_counts": dict(sorted(counts.items())),
            "candidate_count": self.candidate_count,
            "canceled_candidates": self.canceled_candidates,
            "stale_events": self.stale_events,
            "wait_count": sum(item.kind is DirectiveKind.WAIT for item in self.decisions),
        }


@dataclass(slots=True)
class _DirectAsrState:
    ref: TurnRef
    revision: int = -1
    is_final: bool = False
    speech_active: bool = False
    emitted: bool = False


@dataclass(frozen=True, slots=True)
class DirectAsrAction:
    ref: TurnRef
    at_ms: int
    while_speech_active: bool


@dataclass(frozen=True, slots=True)
class DirectAsrReplayResult:
    actions: tuple[DirectAsrAction, ...]
    stale_events: int

    def summary(self) -> dict[str, object]:
        return {
            "evidence_level": "synthetic_or_declared_trace_only",
            "baseline": "first_nonempty_final_asr_per_turn",
            "action_count": len(self.actions),
            "while_speech_active": sum(item.while_speech_active for item in self.actions),
            "stale_events": self.stale_events,
        }


def replay_direct_asr_final(events: tuple[TraceEvent, ...]) -> DirectAsrReplayResult:
    """Replay a deliberately simple final-ASR trigger on the same event clock.

    The first nonempty final ASR revision triggers one response recommendation
    per active turn. This baseline has no VAD or semantic checks. It is *not*
    a stand-in for every real ASR integration or a measured quality result.
    """
    recorder = TraceRecorder()
    for event in events:
        recorder.append(event)
    states: dict[str, _DirectAsrState] = {}
    seen_refs: dict[str, set[TurnRef]] = {}
    stopped: set[str] = set()
    actions: list[DirectAsrAction] = []
    stale = 0
    for at_ms, batch_iter in groupby(recorder.events, key=lambda event: event.at_ms):
        batch = tuple(batch_iter)
        for session_id in sorted({item.ref.session_id for item in batch}):
            local = tuple(item for item in batch if item.ref.session_id == session_id)
            starts = [item for item in local if item.kind == "turn_start"]
            if starts and session_id not in stopped:
                if len({item.ref for item in starts}) > 1:
                    raise ValueError("ambiguous simultaneous turn starts")
                latest = max(starts, key=lambda item: item.ref.generation)
                prior = states.get(session_id)
                seen = seen_refs.setdefault(session_id, set())
                if latest.ref not in seen and (
                    prior is None or latest.ref.generation >= prior.ref.generation
                ):
                    states[session_id] = _DirectAsrState(latest.ref)
                    seen.add(latest.ref)
                elif prior is None or latest.ref != prior.ref:
                    stale += len(starts)
            state = states.get(session_id)
            if state is not None and any(
                item.kind == "stop" and item.ref == state.ref for item in local
            ):
                states.pop(session_id)
                stopped.add(session_id)
                continue
            for item in local:
                if item.kind != "turn_start" and (state is None or item.ref != state.ref):
                    stale += 1
            if state is None:
                continue
            current = [
                item for item in local if item.ref == state.ref and item.kind != "turn_start"
            ]
            acoustics = [item for item in current if item.kind == "acoustic"]
            if acoustics:
                # Observe resumed speech before considering a same-time ASR final.
                acoustic = next(
                    (item for item in acoustics if item.data["speech_active"]), acoustics[-1]
                )
                state.speech_active = cast(bool, acoustic.data["speech_active"])
            revisions = [item for item in current if item.kind == "asr"]
            if not revisions:
                continue
            latest_asr = max(
                revisions,
                key=lambda item: (
                    cast(int, item.data["revision"]),
                    cast(bool, item.data["is_final"]),
                ),
            )
            revision = cast(int, latest_asr.data["revision"])
            is_final = cast(bool, latest_asr.data["is_final"])
            if revision < state.revision or (revision == state.revision and state.is_final):
                stale += len(revisions)
                continue
            state.revision = revision
            state.is_final = is_final
            if is_final and latest_asr.data["has_text"] and not state.emitted:
                actions.append(DirectAsrAction(state.ref, at_ms, state.speech_active))
                state.emitted = True
    return DirectAsrReplayResult(tuple(actions), stale)


def replay_action_trace(events: tuple[TraceEvent, ...]) -> TraceReplayResult:
    """Replay events by availability time, with speech/stop winning ties.

    ASR text is represented only by a synthetic nonempty placeholder. This
    validates causal wiring, not transcription, semantics, or action quality.
    """
    recorder = TraceRecorder()
    for event in events:
        recorder.append(event)
    states: dict[str, _ReplayState] = {}
    seen_refs: dict[str, set[TurnRef]] = {}
    stopped: set[str] = set()
    decisions: list[Decision] = []
    candidates = 0
    stale = 0
    canceled = 0
    for at_ms, batch_iter in groupby(recorder.events, key=lambda event: event.at_ms):
        batch = tuple(batch_iter)
        for session_id in sorted({item.ref.session_id for item in batch}):
            local = tuple(item for item in batch if item.ref.session_id == session_id)
            starts = [item for item in local if item.kind == "turn_start"]
            if starts and session_id not in stopped:
                if len({item.ref for item in starts}) > 1:
                    raise ValueError("ambiguous simultaneous turn starts")
                latest = max(starts, key=lambda item: item.ref.generation)
                prior = states.get(session_id)
                seen = seen_refs.setdefault(session_id, set())
                if latest.ref not in seen and (
                    prior is None or latest.ref.generation >= prior.ref.generation
                ):
                    if prior is not None:
                        canceled += prior.gate.canceled_candidates
                    states[session_id] = _ReplayState(
                        ref=latest.ref,
                        assistant_speaking=prior.assistant_speaking if prior else False,
                    )
                    seen.add(latest.ref)
                elif prior is None or latest.ref != prior.ref:
                    stale += len(starts)
            state = states.get(session_id)
            if state is not None and any(
                item.kind == "stop" and item.ref == state.ref for item in local
            ):
                canceled += state.gate.canceled_candidates
                states.pop(session_id)
                stopped.add(session_id)
                continue
            for item in local:
                if item.kind == "turn_start":
                    continue
                if state is None or item.ref != state.ref:
                    stale += 1
                    continue
            if state is None:
                continue
            current = [
                item for item in local if item.ref == state.ref and item.kind != "turn_start"
            ]
            if not current:
                continue
            acoustics = [item for item in current if item.kind == "acoustic"]
            if acoustics:
                # A same-time continuation always beats a pause/candidate.
                item = next(
                    (event for event in acoustics if event.data["speech_active"]), acoustics[-1]
                )
                data = item.data
                state.acoustic = AcousticSignal(
                    ref=state.ref,
                    observed_at_ms=at_ms,
                    speech_active=cast(bool, data["speech_active"]),
                    near_end_speech=cast(bool, data["near_end_speech"]),
                    echo_likely=cast(bool, data["echo_likely"]),
                    speech_duration_ms=cast(int, data["speech_duration_ms"]),
                    pause_duration_ms=cast(int | None, data["pause_duration_ms"]),
                    audio_quality=AudioQuality(cast(str, data["audio_quality"])),
                )
            playback = [item for item in current if item.kind == "playback"]
            if playback:
                state.assistant_speaking = any(
                    cast(bool, item.data["speaking"]) for item in playback
                )
            revisions = [item for item in current if item.kind == "asr"]
            if revisions:
                item = max(
                    revisions,
                    key=lambda event: (
                        cast(int, event.data["revision"]),
                        cast(bool, event.data["is_final"]),
                    ),
                )
                data = item.data
                revision = cast(int, data["revision"])
                if (
                    state.transcript is None
                    or revision > state.transcript.revision
                    or (revision == state.transcript.revision and cast(bool, data["is_final"]))
                ):
                    state.transcript = TranscriptSignal(
                        state.ref,
                        at_ms,
                        revision,
                        "present" if data["has_text"] else "",
                        cast(bool, data["is_final"]),
                    )
                    state.semantic = None
                else:
                    stale += len(revisions)
            scores = [item for item in current if item.kind == "semantic"]
            if scores:
                item = scores[-1]
                data = item.data
                if (
                    state.transcript is not None
                    and data["transcript_revision"] == state.transcript.revision
                ):
                    state.semantic = SemanticSignal(
                        state.ref,
                        at_ms,
                        data["transcript_revision"],
                        cast(float, data["complete_probability"]),
                        cast(float, data["clarification_probability"]),
                        cast(float, data["backchannel_probability"]),
                        "trace-score",
                        cast(float | None, data["response_probability"]),
                    )
                else:
                    stale += len(scores)
            if state.acoustic is None:
                continue
            host = HostState(
                ref=state.ref,
                now_ms=at_ms,
                session_active=True,
                assistant_speaking=state.assistant_speaking,
            )
            if any(item.kind == "candidate" for item in current) and state.gate.arm(
                host, state.acoustic
            ):
                candidates += 1
            decisions.append(
                state.gate.decide(host, state.acoustic, state.transcript, state.semantic)
            )
    canceled += sum(state.gate.canceled_candidates for state in states.values())
    return TraceReplayResult(tuple(decisions), candidates, canceled, stale)


def main() -> None:
    parser = argparse.ArgumentParser(description="Replay a content-free TurnPilot action trace")
    parser.add_argument("trace", type=Path)
    parser.add_argument("--compare-direct-asr", action="store_true")
    args = parser.parse_args()
    try:
        content = args.trace.read_text(encoding="utf-8")
        events = load_trace_jsonl(content)
        result = replay_action_trace(events)
        summary: dict[str, object] = result.summary()
        if args.compare_direct_asr:
            summary = {
                "turnpilot": summary,
                "direct_asr_final": replay_direct_asr_final(events).summary(),
            }
    except (OSError, UnicodeError, ValueError):
        parser.exit(2, "Invalid or unreadable action trace.\n")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
