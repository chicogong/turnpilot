"""Immutable, timestamped observations and side-effect-free directives."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class AudioQuality(str, Enum):
    UNKNOWN = "unknown"
    CLEAR = "clear"
    POOR = "poor"


class DirectiveKind(str, Enum):
    NO_ACTION = "no_action"
    WAIT = "wait"
    COMMIT_USER_TURN = "commit_user_turn"
    IGNORE_USER_TURN = "ignore_user_turn"
    CLARIFY_AUDIO = "clarify_audio"
    CLARIFY_MEANING = "clarify_meaning"
    YIELD_ASSISTANT = "yield_assistant"
    NUDGE_USER = "nudge_user"


@dataclass(frozen=True, slots=True)
class TurnRef:
    session_id: str
    turn_id: str
    generation: int

    def __post_init__(self) -> None:
        if not self.session_id or not self.turn_id:
            raise ValueError("session_id and turn_id must be non-empty")
        if self.generation < 0:
            raise ValueError("generation must be non-negative")


@dataclass(frozen=True, slots=True)
class AcousticSignal:
    ref: TurnRef
    observed_at_ms: int
    speech_active: bool
    near_end_speech: bool = False
    echo_likely: bool = False
    speech_duration_ms: int = 0
    pause_duration_ms: int | None = None
    audio_quality: AudioQuality = AudioQuality.UNKNOWN

    def __post_init__(self) -> None:
        if self.observed_at_ms < 0 or self.speech_duration_ms < 0:
            raise ValueError("timestamps and durations must be non-negative")
        if self.pause_duration_ms is not None and self.pause_duration_ms < 0:
            raise ValueError("pause_duration_ms must be non-negative")
        if self.speech_active and self.pause_duration_ms is not None:
            raise ValueError("active speech cannot also be a pause candidate")


@dataclass(frozen=True, slots=True)
class TranscriptSignal:
    ref: TurnRef
    available_at_ms: int
    revision: int
    text: str = field(repr=False)
    is_final: bool = False

    def __post_init__(self) -> None:
        if self.available_at_ms < 0 or self.revision < 0:
            raise ValueError("timestamp and revision must be non-negative")


@dataclass(frozen=True, slots=True)
class SemanticSignal:
    ref: TurnRef
    received_at_ms: int
    transcript_revision: int
    complete_probability: float
    clarification_probability: float
    backchannel_probability: float
    model: str
    response_probability: float | None = None

    def __post_init__(self) -> None:
        if self.received_at_ms < 0 or self.transcript_revision < 0:
            raise ValueError("timestamp and revision must be non-negative")
        for name in (
            "complete_probability",
            "clarification_probability",
            "backchannel_probability",
        ):
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")
        if self.response_probability is not None and not 0.0 <= self.response_probability <= 1.0:
            raise ValueError("response_probability must be within [0, 1]")
        if not self.model:
            raise ValueError("model must be non-empty")


@dataclass(frozen=True, slots=True)
class HostState:
    ref: TurnRef
    now_ms: int
    session_active: bool
    assistant_speaking: bool = False
    asked_question: bool = False
    tool_running: bool = False
    idle_duration_ms: int = 0
    nudge_count: int = 0
    allow_nudge: bool = False

    def __post_init__(self) -> None:
        if self.now_ms < 0 or self.idle_duration_ms < 0 or self.nudge_count < 0:
            raise ValueError("timestamps and counters must be non-negative")


@dataclass(frozen=True, slots=True)
class Decision:
    ref: TurnRef
    kind: DirectiveKind
    reason: str
    decided_at_ms: int
    expires_at_ms: int
    used_semantic: bool = False

    def __post_init__(self) -> None:
        if self.decided_at_ms < 0 or self.expires_at_ms < self.decided_at_ms:
            raise ValueError("invalid decision time")
