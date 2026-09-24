"""Optional provisional endpoint gate around the existing pure turn policy.

This gate recommends actions only. A host must still validate the current turn,
playback ownership, and directive expiry before doing anything externally.
"""

from __future__ import annotations

from turnpilot.models import (
    AcousticSignal,
    Decision,
    DirectiveKind,
    HostState,
    SemanticSignal,
    TranscriptSignal,
    TurnRef,
)
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
    """

    def __init__(self, policy: TurnPolicy | None = None) -> None:
        self.policy = policy or TurnPolicy()
        self._ref: TurnRef | None = None
        self._armed = False
        self._emitted = False
        self.canceled_candidates = 0

    def reset(self) -> None:
        self._ref = None
        self._armed = False
        self._emitted = False

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
    ) -> Decision:
        """Evaluate a snapshot without executing a directive."""
        if not host.session_active:
            self.reset()
            return self._result(host, DirectiveKind.NO_ACTION, "session_inactive")
        self._select_ref(host.ref)
        if not self._valid_acoustic(host, acoustic):
            return self._result(host, DirectiveKind.NO_ACTION, "invalid_acoustic_observation")
        if acoustic.speech_active and self._armed:
            self._armed = False
            self.canceled_candidates += 1
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
        # Before the hard cap, final ASR plus a *causally aligned* high score
        # is needed. The policy's semantic_complete reason is only produced
        # after its own turn/revision/age validation; fallback is not enough.
        early = (
            transcript is not None
            and transcript.ref == host.ref
            and transcript.available_at_ms <= host.now_ms
            and transcript.is_final
            and decision.used_semantic
            and decision.reason.startswith("semantic_complete")
        )
        if pause < self.policy.config.max_pause_ms and not early:
            return self._result(host, DirectiveKind.WAIT, "await_final_or_max_pause")
        self._emitted = True
        self._armed = False
        return decision

    def _select_ref(self, ref: TurnRef) -> None:
        if ref != self._ref:
            self._ref = ref
            self._armed = False
            self._emitted = False

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
