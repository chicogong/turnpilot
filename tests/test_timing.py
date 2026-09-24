from __future__ import annotations

from dataclasses import replace

import pytest

from turnpilot.models import DirectiveKind
from turnpilot.policy import PolicyConfig, TurnPolicy
from turnpilot.timing import PauseTimingCase, simulate_pause, summarize_pauses


def case(
    case_id: str,
    *,
    complete: bool,
    resume_at_ms: int | None = None,
    latency_ms: int | None = None,
    probability: float | None = None,
) -> PauseTimingCase:
    return PauseTimingCase(
        case_id,
        complete,
        transcript_available_at_ms=192,
        dispatch_at_ms=192,
        resume_at_ms=resume_at_ms,
        jev_latency_ms=latency_ms,
        jev_complete_probability=probability,
    )


def test_fast_semantic_result_can_commit_before_fixed_endpoint() -> None:
    item = case("complete", complete=True, latency_ms=200, probability=0.95)
    baseline = simulate_pause(item, use_jev=False)
    assisted = simulate_pause(item, use_jev=True)
    assert baseline.at_ms == 640
    assert not baseline.used_semantic
    assert assisted.at_ms == 392
    assert assisted.kind is DirectiveKind.COMMIT_USER_TURN
    assert assisted.used_semantic


def test_late_result_cannot_retroactively_change_fallback() -> None:
    item = case("late", complete=True, latency_ms=500, probability=0.95)
    assert simulate_pause(item, use_jev=True, request_budget_ms=800).at_ms == 640
    assert not simulate_pause(item, use_jev=True, request_budget_ms=800).used_semantic
    assert simulate_pause(item, use_jev=True, request_budget_ms=350).at_ms == 640


def test_incomplete_pause_can_be_saved_or_falsely_cut_off() -> None:
    low = case("hold", complete=False, resume_at_ms=800, latency_ms=200, probability=0.1)
    high = replace(low, case_id="false-cutoff", jev_complete_probability=0.95)
    assert simulate_pause(low, use_jev=False).at_ms == 640
    assert simulate_pause(low, use_jev=True).at_ms is None
    assert simulate_pause(high, use_jev=True).at_ms == 392


def test_resumed_speech_discards_response_at_same_time_or_later() -> None:
    same_time = case("resume", complete=False, resume_at_ms=320, latency_ms=128, probability=0.99)
    assert simulate_pause(same_time, use_jev=True).at_ms is None
    early = replace(same_time, jev_latency_ms=100)
    assert simulate_pause(early, use_jev=True).at_ms == 292


def test_final_text_cannot_be_speculatively_dispatched() -> None:
    item = case("final", complete=True, latency_ms=200, probability=0.95)
    with pytest.raises(ValueError, match="availability"):
        replace(item, transcript_available_at_ms=640)
    final_only = replace(item, transcript_available_at_ms=640, dispatch_at_ms=640)
    assert simulate_pause(final_only, use_jev=True).at_ms == 640


def test_revision_invalidates_old_low_score_and_new_result_can_commit() -> None:
    item = case("revised", complete=True, latency_ms=200, probability=0.05)
    item = replace(
        item,
        revised_transcript_at_ms=500,
        revised_jev_latency_ms=100,
        revised_jev_complete_probability=0.95,
    )
    assert simulate_pause(item, use_jev=True).at_ms == 600
    assert simulate_pause(item, use_jev=True).used_semantic
    assert simulate_pause(replace(item, revised_jev_latency_ms=500), use_jev=True).at_ms == 640


def test_revision_cannot_undo_early_commit_from_old_partial() -> None:
    item = case("too-early", complete=False, resume_at_ms=800, latency_ms=100, probability=0.9)
    item = replace(
        item,
        revised_transcript_at_ms=500,
        revised_jev_latency_ms=100,
        revised_jev_complete_probability=0.05,
    )
    assert simulate_pause(item, use_jev=True).at_ms == 292


def test_old_score_arriving_after_revision_is_not_applied() -> None:
    item = case("stale", complete=False, resume_at_ms=800, latency_ms=350, probability=0.99)
    item = replace(item, revised_transcript_at_ms=320)
    assert simulate_pause(item, use_jev=True, request_budget_ms=500).at_ms == 640
    assert not simulate_pause(item, use_jev=True, request_budget_ms=500).used_semantic


def test_guarded_candidate_caps_stale_prefix_delay_and_does_not_early_commit() -> None:
    guarded = TurnPolicy(
        PolicyConfig(allow_partial_semantic_commit=False, partial_semantic_hold_extension_ms=100)
    )
    stale_low = case("stale-low", complete=True, latency_ms=200, probability=0.05)
    early_high = case(
        "early-high", complete=False, resume_at_ms=800, latency_ms=100, probability=0.95
    )
    assert simulate_pause(stale_low, use_jev=True).at_ms == 1200
    assert simulate_pause(stale_low, use_jev=True, policy=guarded).at_ms == 740
    assert simulate_pause(early_high, use_jev=True).at_ms == 292
    assert simulate_pause(early_high, use_jev=True, policy=guarded).at_ms == 640


def test_guarded_candidate_uses_revised_final_score() -> None:
    guarded = TurnPolicy(
        PolicyConfig(allow_partial_semantic_commit=False, partial_semantic_hold_extension_ms=100)
    )
    item = case("revised-final", complete=True, latency_ms=100, probability=0.95)
    item = replace(
        item,
        revised_transcript_at_ms=320,
        revised_jev_latency_ms=200,
        revised_jev_complete_probability=0.95,
    )
    assert simulate_pause(item, use_jev=True).at_ms == 292
    assert simulate_pause(item, use_jev=True, policy=guarded).at_ms == 520


def test_invalid_timing_and_probability_are_rejected() -> None:
    item = case("valid", complete=True)
    with pytest.raises(ValueError, match="together"):
        replace(item, jev_latency_ms=100)
    with pytest.raises(ValueError, match="positive resumption"):
        replace(item, is_complete=False)
    with pytest.raises(ValueError, match="finite"):
        replace(item, jev_latency_ms=100, jev_complete_probability=float("nan"))
    with pytest.raises(ValueError, match="revised transcript"):
        replace(item, revised_jev_latency_ms=100, revised_jev_complete_probability=0.9)
    with pytest.raises(ValueError, match="follow"):
        replace(item, revised_transcript_at_ms=192)
    with pytest.raises(ValueError, match="positive"):
        simulate_pause(item, use_jev=False, request_budget_ms=0)


def test_aggregate_is_text_free_and_uses_independent_labels() -> None:
    items = (
        case("complete-private-id", complete=True, latency_ms=200, probability=0.95),
        case(
            "unfinished-private-id",
            complete=False,
            resume_at_ms=800,
            latency_ms=200,
            probability=0.1,
        ),
    )
    baseline = summarize_pauses(items, use_jev=False)
    assisted = summarize_pauses(items, use_jev=True)
    assert baseline["false_cutoffs"] == 1
    assert assisted["false_cutoffs"] == 0
    assert baseline["endpoint_p50_ms"] == 640
    assert assisted["endpoint_p50_ms"] == 392
    assert "private-id" not in str(assisted)
    with pytest.raises(ValueError, match="unique"):
        summarize_pauses((items[0], items[0]), use_jev=True)
