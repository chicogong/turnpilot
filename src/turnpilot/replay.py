"""Deterministic causal replay of timestamped observations."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from turnpilot.models import AcousticSignal, Decision, HostState, SemanticSignal, TranscriptSignal
from turnpilot.policy import TurnPolicy


@dataclass(frozen=True, slots=True)
class ReplayStep:
    host: HostState
    acoustic: AcousticSignal
    transcript: TranscriptSignal | None = None
    semantic: SemanticSignal | None = None


def replay(policy: TurnPolicy, steps: Iterable[ReplayStep]) -> tuple[Decision, ...]:
    """Evaluate observations in order; future revisions are filtered by the policy.

    This is an event-level runner, not a claimed EOT accuracy benchmark. Human
    pause labels and action metrics are required before quality can be reported.
    """
    last_time_by_session: dict[str, int] = {}
    decisions: list[Decision] = []
    for step in steps:
        session_id = step.host.ref.session_id
        prior = last_time_by_session.get(session_id)
        if prior is not None and step.host.now_ms < prior:
            raise ValueError(f"non-monotonic replay clock for session {session_id!r}")
        last_time_by_session[session_id] = step.host.now_ms
        decisions.append(policy.decide(step.host, step.acoustic, step.transcript, step.semantic))
    return tuple(decisions)
