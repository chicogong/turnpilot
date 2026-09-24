"""Aggregate-only A/B/C/D evaluation of precomputed, text-free observations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, cast

from turnpilot.evaluation import LabeledCase, compare_labeled_arms, paired_false_cutoff_interval
from turnpilot.models import (
    AcousticSignal,
    AudioQuality,
    DirectiveKind,
    HostState,
    SemanticSignal,
    TranscriptSignal,
    TurnRef,
)
from turnpilot.policy import TurnPolicy
from turnpilot.replay import ReplayStep

ARMS = ("A", "B", "C", "D")


def _object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return cast(int, value)


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number")
    return float(value)


def _optional_integer(value: Any, name: str) -> int | None:
    return None if value is None else _integer(value, name)


def _optional_string(value: Any, name: str) -> str | None:
    return None if value is None else _string(value, name)


def _case_for_arm(raw: dict[str, Any], arm: str) -> LabeledCase:
    ref = TurnRef(
        _string(raw.get("session_id"), "session_id"),
        _string(raw.get("turn_id"), "turn_id"),
        _integer(raw.get("generation"), "generation"),
    )
    now_ms = _integer(raw.get("now_ms"), "now_ms")
    host_data = _object(raw.get("host"), "host")
    host = HostState(
        ref=ref,
        now_ms=now_ms,
        session_active=_boolean(host_data.get("session_active"), "session_active"),
        assistant_speaking=_boolean(
            host_data.get("assistant_speaking", False), "assistant_speaking"
        ),
        asked_question=_boolean(host_data.get("asked_question", False), "asked_question"),
        tool_running=_boolean(host_data.get("tool_running", False), "tool_running"),
        idle_duration_ms=_integer(host_data.get("idle_duration_ms", 0), "idle_duration_ms"),
        nudge_count=_integer(host_data.get("nudge_count", 0), "nudge_count"),
        allow_nudge=_boolean(host_data.get("allow_nudge", False), "allow_nudge"),
    )
    transcript_data = raw.get("transcript")
    transcript = None
    if transcript_data is not None:
        item = _object(transcript_data, "transcript")
        if "text" in item:
            raise ValueError("manifest must not contain transcript text")
        transcript = TranscriptSignal(
            ref=ref,
            available_at_ms=_integer(item.get("available_at_ms"), "available_at_ms"),
            revision=_integer(item.get("revision"), "revision"),
            text="present" if _boolean(item.get("has_text"), "has_text") else "",
            is_final=_boolean(item.get("is_final", False), "is_final"),
        )
    arm_data = _object(_object(raw.get("arms"), "arms").get(arm), f"arm {arm}")
    acoustic_data = _object(arm_data.get("acoustic"), "acoustic")
    acoustic = AcousticSignal(
        ref=ref,
        observed_at_ms=_integer(acoustic_data.get("observed_at_ms"), "observed_at_ms"),
        speech_active=_boolean(acoustic_data.get("speech_active"), "speech_active"),
        near_end_speech=_boolean(acoustic_data.get("near_end_speech", False), "near_end_speech"),
        echo_likely=_boolean(acoustic_data.get("echo_likely", False), "echo_likely"),
        speech_duration_ms=_integer(
            acoustic_data.get("speech_duration_ms", 0), "speech_duration_ms"
        ),
        pause_duration_ms=_optional_integer(
            acoustic_data.get("pause_duration_ms"), "pause_duration_ms"
        ),
        audio_quality=AudioQuality(acoustic_data.get("audio_quality", "unknown")),
    )
    semantic_data = arm_data.get("semantic")
    semantic = None
    if semantic_data is not None:
        item = _object(semantic_data, "semantic")
        response_probability = item.get("response_probability")
        semantic = SemanticSignal(
            ref=ref,
            received_at_ms=_integer(item.get("received_at_ms"), "received_at_ms"),
            transcript_revision=_integer(item.get("transcript_revision"), "transcript_revision"),
            complete_probability=_number(item.get("complete_probability"), "complete_probability"),
            clarification_probability=_number(
                item.get("clarification_probability"), "clarification_probability"
            ),
            backchannel_probability=_number(
                item.get("backchannel_probability"), "backchannel_probability"
            ),
            response_probability=(
                None
                if response_probability is None
                else _number(response_probability, "response_probability")
            ),
            model=_string(item.get("model"), "model"),
        )
    label = _object(raw.get("label"), "label")
    return LabeledCase(
        case_id=_string(raw.get("case_id"), "case_id"),
        step=ReplayStep(host, acoustic, transcript, semantic),
        is_complete=_boolean(label.get("is_complete"), "is_complete"),
        expected_action=DirectiveKind(label.get("expected_action")),
        true_eot_ms=_optional_integer(label.get("true_eot_ms"), "true_eot_ms"),
        speaker_id=_optional_string(label.get("speaker_id"), "speaker_id"),
        device_id=_optional_string(label.get("device_id"), "device_id"),
    )


def evaluate_manifest(document: Any) -> dict[str, Any]:
    """Return aggregate metrics only; the input declaration is not verified consent."""
    root = _object(document, "manifest")
    if _integer(root.get("schema_version"), "schema_version") != 1:
        raise ValueError("unsupported manifest schema")
    evidence_level = _string(root.get("evidence_level"), "evidence_level")
    if evidence_level not in {"synthetic", "human_reviewed"}:
        raise ValueError("evidence_level must be synthetic or human_reviewed")
    rows = root.get("cases")
    if not isinstance(rows, list):
        raise ValueError("cases must be an array")
    if not rows:
        raise ValueError("cases must not be empty")
    for row in rows:
        item = _object(row, "case")
        if set(_object(item.get("arms"), "arms")) != set(ARMS):
            raise ValueError("each case must contain exactly A/B/C/D arms")
    arms = {
        arm: (TurnPolicy(), tuple(_case_for_arm(_object(row, "case"), arm) for row in rows))
        for arm in ARMS
    }
    summaries = compare_labeled_arms(arms)
    paired_intervals = {
        arm: paired_false_cutoff_interval(arms["A"][0], arms["A"][1], arms[arm][0], arms[arm][1])
        for arm in ARMS
        if arm != "A"
    }
    return {
        "schema_version": 1,
        "declared_evidence_level": evidence_level,
        "note": "offline per-candidate metrics; no device or user-perceived quality claim",
        "arms": {
            arm: {
                "events": summary.event_count,
                "candidates": summary.candidate_count,
                "incomplete": summary.incomplete_count,
                "false_cutoffs": summary.false_cutoff_count,
                "false_cutoff_rate": summary.false_cutoff_rate,
                "action_errors": summary.action_error_count,
                "action_error_rate": summary.action_error_rate,
                "action_confusion": [
                    {"expected": expected.value, "actual": actual.value, "count": count}
                    for expected, actual, count in summary.action_confusion
                ],
                "complete_without_endpoint": summary.complete_without_endpoint_count,
                "premature_endpoints": summary.premature_endpoint_count,
                "endpoint_p50_ms": summary.endpoint_latency_percentile_ms(50),
                "endpoint_p95_ms": summary.endpoint_latency_percentile_ms(95),
                "endpoint_p99_ms": summary.endpoint_latency_percentile_ms(99),
            }
            for arm, summary in summaries.items()
        },
        "paired_false_cutoff_vs_A": {
            arm: (
                None
                if interval is None
                else {
                    "rate_delta": interval.delta,
                    "ci95_lower": interval.lower,
                    "ci95_upper": interval.upper,
                    "speaker_clusters": interval.speaker_count,
                    "incomplete_candidates": interval.incomplete_candidate_count,
                    "bootstrap_iterations": interval.bootstrap_iterations,
                }
            )
            for arm, interval in paired_intervals.items()
        },
        "paired_interval_note": (
            "null means fewer than 10 speaker clusters or missing speaker IDs; "
            "intervals do not establish device independence or product quality"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate text-free TurnPilot A/B/C/D replay")
    parser.add_argument("manifest", type=Path, help="JSON manifest outside Git for private corpora")
    args = parser.parse_args(argv)
    try:
        if args.manifest.stat().st_size > 20_000_000:
            raise ValueError("manifest exceeds 20 MB")
        document = json.loads(args.manifest.read_text(encoding="utf-8"))
        result = evaluate_manifest(document)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, TypeError):
        print(
            "Invalid or unavailable evaluation manifest; no input content printed.", file=sys.stderr
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
