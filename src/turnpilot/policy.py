"""Experimental pure policy; the host validates and applies directives."""

from __future__ import annotations

from dataclasses import dataclass

from turnpilot.models import (
    AcousticSignal,
    AudioQuality,
    Decision,
    DirectiveKind,
    HostState,
    SemanticSignal,
    TranscriptSignal,
)


@dataclass(frozen=True, slots=True)
class PolicyConfig:
    """Prototype thresholds, not calibrated production defaults."""

    min_pause_ms: int = 192
    baseline_pause_ms: int = 640
    max_pause_ms: int = 1200
    min_barge_in_ms: int = 320
    idle_check_ms: int = 10000
    decision_ttl_ms: int = 150
    max_acoustic_age_ms: int = 200
    max_semantic_age_ms: int = 500
    semantic_commit_probability: float = 0.8
    semantic_hold_probability: float = 0.2
    semantic_clarify_probability: float = 0.8
    semantic_ignore_probability: float = 0.1
    backchannel_probability: float = 0.8
    allow_partial_semantic_commit: bool = True
    partial_semantic_hold_extension_ms: int | None = None

    def __post_init__(self) -> None:
        if not 0 <= self.min_pause_ms <= self.baseline_pause_ms <= self.max_pause_ms:
            raise ValueError("pause thresholds must be ordered")
        if (
            min(
                self.min_barge_in_ms,
                self.idle_check_ms,
                self.decision_ttl_ms,
                self.max_acoustic_age_ms,
                self.max_semantic_age_ms,
            )
            <= 0
        ):
            raise ValueError("timing thresholds must be positive")
        for name in (
            "semantic_commit_probability",
            "semantic_hold_probability",
            "semantic_clarify_probability",
            "semantic_ignore_probability",
            "backchannel_probability",
        ):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} must be within [0, 1]")
        if self.semantic_hold_probability >= self.semantic_commit_probability:
            raise ValueError("hold probability must be below commit probability")
        if self.partial_semantic_hold_extension_ms is not None and (
            self.partial_semantic_hold_extension_ms < 0
        ):
            raise ValueError("partial semantic hold extension must be non-negative")


class TurnPolicy:
    """Return a recommendation from causal observations, without side effects."""

    def __init__(self, config: PolicyConfig | None = None) -> None:
        self.config = config or PolicyConfig()

    def decide(
        self,
        host: HostState,
        acoustic: AcousticSignal,
        transcript: TranscriptSignal | None = None,
        semantic: SemanticSignal | None = None,
    ) -> Decision:
        def result(kind: DirectiveKind, reason: str, *, used_semantic: bool = False) -> Decision:
            return self._result(host, kind, reason, used_semantic=used_semantic)

        if not host.session_active:
            return result(DirectiveKind.NO_ACTION, "session_inactive")
        if (
            acoustic.ref != host.ref
            or acoustic.observed_at_ms > host.now_ms
            or host.now_ms - acoustic.observed_at_ms > self.config.max_acoustic_age_ms
        ):
            return result(DirectiveKind.NO_ACTION, "invalid_acoustic_observation")
        if transcript is not None and (
            transcript.ref != host.ref or transcript.available_at_ms > host.now_ms
        ):
            transcript = None
        if semantic is not None and (
            semantic.ref != host.ref
            or semantic.received_at_ms > host.now_ms
            or host.now_ms - semantic.received_at_ms > self.config.max_semantic_age_ms
            or transcript is None
            or semantic.transcript_revision != transcript.revision
            or semantic.received_at_ms < transcript.available_at_ms
        ):
            semantic = None

        if host.assistant_speaking:
            if acoustic.echo_likely or not acoustic.speech_active or not acoustic.near_end_speech:
                return result(DirectiveKind.NO_ACTION, "no_verified_barge_in")
            if acoustic.speech_duration_ms < self.config.min_barge_in_ms:
                return result(DirectiveKind.WAIT, "short_overlap")
            if semantic is not None and (
                semantic.backchannel_probability >= self.config.backchannel_probability
            ):
                return result(DirectiveKind.NO_ACTION, "backchannel", used_semantic=True)
            return result(DirectiveKind.YIELD_ASSISTANT, "near_end_barge_in")

        if acoustic.speech_active:
            return result(DirectiveKind.WAIT, "user_speaking")

        pause = acoustic.pause_duration_ms
        if pause is None:
            if (
                host.allow_nudge
                and host.asked_question
                and not host.tool_running
                and host.nudge_count == 0
                and host.idle_duration_ms >= self.config.idle_check_ms
            ):
                return result(DirectiveKind.NUDGE_USER, "question_wait_timeout")
            return result(DirectiveKind.NO_ACTION, "no_pending_user_turn")

        if pause < self.config.min_pause_ms:
            return result(DirectiveKind.WAIT, "minimum_pause")

        used_semantic = semantic is not None
        if semantic is not None:
            partial_text = transcript is not None and not transcript.is_final
            if semantic.complete_probability >= self.config.semantic_commit_probability and (
                self.config.allow_partial_semantic_commit or not partial_text
            ):
                return self._complete(host, acoustic, transcript, semantic)
            hold_deadline = self.config.max_pause_ms
            if partial_text and self.config.partial_semantic_hold_extension_ms is not None:
                hold_deadline = min(
                    hold_deadline,
                    self.config.baseline_pause_ms + self.config.partial_semantic_hold_extension_ms,
                )
            if (
                semantic.complete_probability <= self.config.semantic_hold_probability
                and pause < hold_deadline
            ):
                return result(DirectiveKind.WAIT, "semantic_hold", used_semantic=True)

        if pause < self.config.baseline_pause_ms:
            return result(DirectiveKind.WAIT, "acoustic_fallback_wait", used_semantic=used_semantic)
        if pause >= self.config.max_pause_ms:
            return self._complete(host, acoustic, transcript, semantic, reason="max_pause")
        return self._complete(host, acoustic, transcript, semantic, reason="acoustic_fallback")

    def _result(
        self,
        host: HostState,
        kind: DirectiveKind,
        reason: str,
        *,
        used_semantic: bool = False,
    ) -> Decision:
        return Decision(
            ref=host.ref,
            kind=kind,
            reason=reason,
            decided_at_ms=host.now_ms,
            expires_at_ms=host.now_ms + self.config.decision_ttl_ms,
            used_semantic=used_semantic,
        )

    def _complete(
        self,
        host: HostState,
        acoustic: AcousticSignal,
        transcript: TranscriptSignal | None,
        semantic: SemanticSignal | None,
        *,
        reason: str = "semantic_complete",
    ) -> Decision:
        used_semantic = semantic is not None
        if acoustic.audio_quality is AudioQuality.POOR or not (
            transcript is not None and transcript.text.strip()
        ):
            return self._result(
                host,
                DirectiveKind.CLARIFY_AUDIO,
                f"{reason}:audio_or_asr_unclear",
                used_semantic=used_semantic,
            )
        if semantic is not None and semantic.response_probability is not None:
            if semantic.response_probability <= self.config.semantic_ignore_probability:
                return self._result(
                    host,
                    DirectiveKind.IGNORE_USER_TURN,
                    f"{reason}:not_addressed",
                    used_semantic=True,
                )
        if semantic is not None and (
            semantic.clarification_probability >= self.config.semantic_clarify_probability
        ):
            return self._result(
                host,
                DirectiveKind.CLARIFY_MEANING,
                f"{reason}:meaning_ambiguous",
                used_semantic=True,
            )
        return self._result(
            host, DirectiveKind.COMMIT_USER_TURN, reason, used_semantic=used_semantic
        )


def decision_is_current(decision: Decision, host: HostState) -> bool:
    """Minimum host-side guard before applying a recommendation.

    Hosts must also recheck action-specific state and playback ownership.
    """
    return (
        host.session_active
        and decision.ref == host.ref
        and decision.decided_at_ms <= host.now_ms <= decision.expires_at_ms
    )
