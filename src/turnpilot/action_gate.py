"""Optional provisional endpoint gate around the existing pure turn policy.

This gate recommends actions only. A host must still validate the current turn,
playback ownership, and directive expiry before doing anything externally.
"""

from __future__ import annotations

from dataclasses import replace

from turnpilot.models import (
    AcousticSignal,
    Decision,
    DirectiveKind,
    HostState,
    SemanticSignal,
    TranscriptSignal,
    TurnRef,
)
from turnpilot.partial_stability import PartialTranscriptStability
from turnpilot.policy import TurnPolicy

_TERMINAL = {
    DirectiveKind.COMMIT_USER_TURN,
    DirectiveKind.IGNORE_USER_TURN,
    DirectiveKind.CLARIFY_AUDIO,
    DirectiveKind.CLARIFY_MEANING,
}


class ProvisionalActionGate:
    """Hold a candidate endpoint until aligned evidence or the hard pause cap.

    A candidate is explicitly armed by the caller; this does not replace a VAD
    or own a session. The original TurnPolicy remains the source of action kind.
    ``stable_partial_ms`` is a disabled-by-default, uncalibrated experiment;
    early partial release also requires the semantic request's actual start.
    """

    def __init__(
        self, policy: TurnPolicy | None = None, *, stable_partial_ms: int | None = None
    ) -> None:
        self.policy = policy or TurnPolicy()
        # Experimental only. None preserves the original final-ASR requirement.
        self._stability = (
            PartialTranscriptStability(
                stable_partial_ms, max_acoustic_age_ms=self.policy.config.max_acoustic_age_ms
            )
            if stable_partial_ms is not None
            else None
        )
        self._ref: TurnRef | None = None
        self._armed = False
        self._emitted = False
        self.canceled_candidates = 0

    def reset(self) -> None:
        self._ref = None
        self._armed = False
        self._emitted = False
        if self._stability is not None:
            self._stability.reset()

    def arm(self, host: HostState, acoustic: AcousticSignal) -> bool:
        """Accept a current, non-speaking acoustic endpoint candidate."""
        if not self._valid_acoustic(host, acoustic) or host.assistant_speaking:
            return False
        self._select_ref(host.ref)
        if self._emitted or acoustic.speech_active or acoustic.pause_duration_ms is None:
            return False
        self._armed = True
        return True

    def decide(
        self,
        host: HostState,
        acoustic: AcousticSignal,
        transcript: TranscriptSignal | None = None,
        semantic: SemanticSignal | None = None,
        *,
        input_backlogged: bool = False,
        semantic_requested_at_ms: int | None = None,
    ) -> Decision:
        """Evaluate a snapshot without executing a directive."""
        if not host.session_active:
            self.reset()
            return self._result(host, DirectiveKind.NO_ACTION, "session_inactive")
        self._select_ref(host.ref)
        stable_partial = False
        if self._stability is not None:
            stable_partial = self._stability.observe(
                host, acoustic, transcript, input_backlogged=input_backlogged
            )
        if not self._valid_acoustic(host, acoustic):
            return self._result(host, DirectiveKind.NO_ACTION, "invalid_acoustic_observation")
        if acoustic.speech_active and self._armed:
            self._armed = False
            self.canceled_candidates += 1
        if input_backlogged:
            return self._result(host, DirectiveKind.WAIT, "input_backlogged")
        if host.assistant_speaking:
            self._armed = False
            # Barge-in must not wait for text or an endpoint candidate.
            if self._emitted:
                return self._result(host, DirectiveKind.NO_ACTION, "already_emitted")
            decision = self.policy.decide(host, acoustic, transcript, semantic)
            if decision.kind is DirectiveKind.YIELD_ASSISTANT:
                self._emitted = True
            return decision
        if self._emitted:
            return self._result(host, DirectiveKind.NO_ACTION, "already_emitted")
        if acoustic.speech_active:
            return self._result(host, DirectiveKind.WAIT, "user_speaking")
        if not self._armed:
            return self._result(host, DirectiveKind.WAIT, "no_endpoint_candidate")

        decision = self.policy.decide(host, acoustic, transcript, semantic)
        pause = acoustic.pause_duration_ms
        if pause is None:
            self._armed = False
            return self._result(host, DirectiveKind.WAIT, "candidate_lost")
        if decision.kind not in _TERMINAL:
            return decision
        # By default, early release needs final ASR plus an aligned high score.
        # The policy's semantic_complete reason is only produced
        # after its own turn/revision/age validation; fallback is not enough.
        aligned_complete = decision.used_semantic and decision.reason.startswith(
            "semantic_complete"
        )
        early_final = (
            transcript is not None
            and transcript.ref == host.ref
            and transcript.available_at_ms <= host.now_ms
            and transcript.is_final
            and aligned_complete
        )
        # A score from a previous pause cannot authorize the partial experiment.
        # Result-arrival time alone is insufficient: the caller must also supply
        # the actual start time of this exact revision's semantic request.
        early_partial = (
            stable_partial
            and aligned_complete
            and self._stability is not None
            and self._stability.started_at_ms is not None
            and semantic is not None
            and semantic_requested_at_ms is not None
            and self._stability.started_at_ms
            <= semantic_requested_at_ms
            <= semantic.received_at_ms
            <= host.now_ms
        )
        if pause < self.policy.config.max_pause_ms:
            if not (early_final or early_partial):
                reason = (
                    "await_stable_partial_or_final_or_max_pause"
                    if self._stability is not None
                    else "await_final_or_max_pause"
                )
                return self._result(host, DirectiveKind.WAIT, reason)
            if early_partial:
                decision = replace(decision, reason="semantic_complete_stable_partial")
        self._emitted = True
        self._armed = False
        return decision

    def _select_ref(self, ref: TurnRef) -> None:
        if ref != self._ref:
            self._ref = ref
            self._armed = False
            self._emitted = False
            if self._stability is not None:
                self._stability.reset()

    def _valid_acoustic(self, host: HostState, acoustic: AcousticSignal) -> bool:
        age = host.now_ms - acoustic.observed_at_ms
        return (
            host.session_active
            and acoustic.ref == host.ref
            and 0 <= age <= self.policy.config.max_acoustic_age_ms
        )

    def _result(self, host: HostState, kind: DirectiveKind, reason: str) -> Decision:
        return Decision(
            ref=host.ref,
            kind=kind,
            reason=reason,
            decided_at_ms=host.now_ms,
            expires_at_ms=host.now_ms + self.policy.config.decision_ttl_ms,
        )
