"""Synthetic P0 contract checks, not a measured conversation-quality result."""

from pathlib import Path

from turnpilot.action_trace import load_trace_jsonl, replay_action_trace, replay_direct_asr_final
from turnpilot.models import DirectiveKind


def test_missing_late_and_revised_final_on_one_causal_trace() -> None:
    path = Path(__file__).resolve().parents[1] / "examples/synthetic-asr-timing-trace.jsonl"
    events = load_trace_jsonl(path.read_text(encoding="utf-8"))
    gate = replay_action_trace(events)
    direct = replay_direct_asr_final(events)

    # Independent, author-labeled synthetic thought states. These labels are
    # not inferred from ASR, Jev, or model output, and prove no field accuracy.
    expected_action = {
        "missing": DirectiveKind.COMMIT_USER_TURN,
        "late": DirectiveKind.COMMIT_USER_TURN,
        "revised": DirectiveKind.COMMIT_USER_TURN,
        "resumed": DirectiveKind.WAIT,
    }
    gate_actions = {
        decision.ref.session_id: (decision.kind, decision.decided_at_ms, decision.reason)
        for decision in gate.decisions
        if decision.kind not in {DirectiveKind.WAIT, DirectiveKind.NO_ACTION}
    }
    direct_actions = {
        action.ref.session_id: (action.at_ms, action.while_speech_active)
        for action in direct.actions
    }
    observed_gate = {
        case: gate_actions[case][0] if case in gate_actions else DirectiveKind.WAIT
        for case in expected_action
    }
    observed_direct = {
        case: DirectiveKind.COMMIT_USER_TURN if case in direct_actions else DirectiveKind.WAIT
        for case in expected_action
    }

    assert set(expected_action) == {
        item.ref.session_id for item in events if item.kind == "turn_start"
    }
    assert observed_gate == expected_action
    assert {case for case in expected_action if observed_direct[case] != expected_action[case]} == {
        "missing",
        "resumed",
    }
    assert gate.candidate_count == 4
    assert gate.canceled_candidates == 1
    assert gate.stale_events == 1  # prior partial's semantic score after revision 2

    # No final: the direct-final comparator never responds; the optional gate
    # needs a fresh acoustic observation at the 1200 ms cap.
    assert "missing" not in direct_actions
    assert gate_actions["missing"] == (DirectiveKind.COMMIT_USER_TURN, 1880, "max_pause")

    # A final arriving after that cap cannot cause a duplicate gate action.
    assert gate_actions["late"] == (DirectiveKind.COMMIT_USER_TURN, 2880, "max_pause")
    assert direct_actions["late"] == (3100, False)

    # An old score cannot authorize early release after a final revision.
    assert direct_actions["revised"] == (4030, False)
    assert gate_actions["revised"] == (DirectiveKind.COMMIT_USER_TURN, 4200, "semantic_complete")

    # A same-time continuation beats the late final for the gate, while the
    # deliberately simple direct-final comparator still fires during speech.
    assert "resumed" not in gate_actions
    assert direct_actions["resumed"] == (5320, True)
    assert len(gate_actions) == 3
    assert len(direct_actions) == 3
