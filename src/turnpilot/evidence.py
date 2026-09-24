"""Privacy-preserving structural audit of continuous-conversation evidence.

This validates declared metadata and clock order; it cannot authenticate a
recording, consent, human annotation, or the source of an ASR/playback trace.
No transcript, audio, file path, or per-session identifier is emitted.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any, cast

from turnpilot.models import DirectiveKind

SPLITS = ("development", "tuning", "heldout")
ORIGINS = ("near_end_user", "other_person", "echo", "noise", "unknown")
QUALITIES = ("clear", "poor", "not_speech", "unknown")
SOURCE_KINDS = ("consented_device", "licensed_public", "synthetic")
MAX_MANIFEST_BYTES = 20_000_000


def _object(value: Any, name: str, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"invalid {name} object")
    return cast(dict[str, Any], value)


def _array(value: Any, name: str) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"{name} must be an array")
    return value


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an integer")
    return cast(int, value)


def _boolean(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _optional_integer(value: Any, name: str) -> int | None:
    return None if value is None else _integer(value, name)


def _optional_action(value: Any) -> str | None:
    if value is None:
        return None
    action = _string(value, "expected_action")
    if action not in {item.value for item in DirectiveKind}:
        raise ValueError("invalid expected_action")
    return action


def _within(at_ms: int, duration_ms: int, name: str) -> None:
    if not 0 <= at_ms <= duration_ms:
        raise ValueError(f"{name} must lie inside the session")


def _audit_candidate(raw: Any, duration_ms: int) -> dict[str, Any]:
    candidate = _object(raw, "candidate", {"candidate_id", "at_ms", "label"})
    _string(candidate["candidate_id"], "candidate_id")
    at_ms = _integer(candidate["at_ms"], "candidate.at_ms")
    _within(at_ms, duration_ms, "candidate.at_ms")
    label = _object(
        candidate["label"],
        "candidate label",
        {
            "speech_start_ms",
            "speech_end_ms",
            "origin",
            "quality",
            "is_complete",
            "continuation_at_ms",
            "true_eot_ms",
            "expected_action",
            "reviewer_count",
            "adjudicated",
            "ambiguous",
        },
    )
    origin = _string(label["origin"], "origin")
    quality = _string(label["quality"], "quality")
    if origin not in ORIGINS or quality not in QUALITIES:
        raise ValueError("invalid origin or quality")
    start = _optional_integer(label["speech_start_ms"], "speech_start_ms")
    end = _optional_integer(label["speech_end_ms"], "speech_end_ms")
    if (start is None) != (end is None):
        raise ValueError("speech interval must have both endpoints")
    if start is not None and end is not None:
        _within(start, duration_ms, "speech_start_ms")
        if not start < end <= at_ms:
            raise ValueError("speech interval must end by candidate time")
    if origin == "near_end_user" and start is None:
        raise ValueError("near-end user needs a speech interval")
    if origin == "near_end_user" and quality == "not_speech":
        raise ValueError("near-end user cannot have not_speech quality")
    complete = label["is_complete"]
    if complete is not None:
        complete = _boolean(complete, "is_complete")
    continuation = _optional_integer(label["continuation_at_ms"], "continuation_at_ms")
    true_eot = _optional_integer(label["true_eot_ms"], "true_eot_ms")
    if continuation is not None:
        _within(continuation, duration_ms, "continuation_at_ms")
        if continuation <= at_ms or complete is not False:
            raise ValueError("continuation requires a later incomplete candidate")
    if true_eot is not None:
        _within(true_eot, duration_ms, "true_eot_ms")
        if true_eot > at_ms or complete is not True or end is not None and true_eot < end:
            raise ValueError("true_eot requires a completed candidate")
    if complete is True and true_eot is None:
        raise ValueError("completed candidate needs true_eot_ms")
    if origin != "near_end_user" and (
        complete is not None or continuation is not None or true_eot is not None
    ):
        raise ValueError("non-user candidate cannot have a user turn completion label")
    action = _optional_action(label["expected_action"])
    reviewers = _integer(label["reviewer_count"], "reviewer_count")
    adjudicated = _boolean(label["adjudicated"], "adjudicated")
    ambiguous = _boolean(label["ambiguous"], "ambiguous")
    if (
        reviewers < 0
        or (adjudicated and reviewers < 1)
        or (ambiguous and adjudicated and reviewers < 2)
    ):
        raise ValueError("invalid review count or adjudication")
    reviewed = adjudicated and origin != "unknown" and quality != "unknown"
    if origin == "near_end_user":
        reviewed = reviewed and complete is not None
    return {
        "origin": origin,
        "action": action,
        "adjudicated": adjudicated,
        "reviewed": reviewed,
        "action_ready": reviewed and action is not None,
        "cutoff_ready": reviewed
        and origin == "near_end_user"
        and (true_eot is not None or continuation is not None),
        "ambiguous": ambiguous,
    }


def _audit_session(raw: Any) -> dict[str, Any]:
    session = _object(
        raw,
        "session",
        {
            "session_id",
            "split",
            "speaker_id",
            "device_id",
            "source_kind",
            "permission_ref",
            "audio_sha256",
            "duration_ms",
            "clock",
            "capture",
            "asr_revisions",
            "playback_intervals",
            "candidates",
        },
    )
    session_id = _string(session["session_id"], "session_id")
    split = _string(session["split"], "split")
    speaker = _string(session["speaker_id"], "speaker_id")
    device = _string(session["device_id"], "device_id")
    source = _string(session["source_kind"], "source_kind")
    _string(session["permission_ref"], "permission_ref")
    digest = _string(session["audio_sha256"], "audio_sha256")
    if split not in SPLITS or source not in SOURCE_KINDS:
        raise ValueError("invalid split or source_kind")
    if re.fullmatch(r"[0-9a-f]{64}", digest) is None:
        raise ValueError("audio_sha256 must be a lowercase SHA-256 digest")
    duration = _integer(session["duration_ms"], "duration_ms")
    if duration <= 0:
        raise ValueError("duration_ms must be positive")
    clock = _string(session["clock"], "clock")
    if clock not in {"recorded_monotonic", "derived_offline"}:
        raise ValueError("invalid clock")
    capture = _object(session["capture"], "capture", {"asr_revisions", "playback", "near_end_echo"})
    captured = {key: _boolean(value, key) for key, value in capture.items()}
    revisions = _array(session["asr_revisions"], "asr_revisions")
    if revisions and not captured["asr_revisions"]:
        raise ValueError("ASR events require declared ASR capture")
    previous_time = -1
    previous_revision = -1
    for raw_revision in revisions:
        revision = _object(
            raw_revision, "ASR revision", {"available_at_ms", "revision", "has_text", "is_final"}
        )
        at_ms = _integer(revision["available_at_ms"], "available_at_ms")
        number = _integer(revision["revision"], "revision")
        _within(at_ms, duration, "available_at_ms")
        if at_ms < previous_time or number <= previous_revision:
            raise ValueError("ASR revisions must be causal and strictly increasing")
        _boolean(revision["has_text"], "has_text")
        _boolean(revision["is_final"], "is_final")
        previous_time = at_ms
        previous_revision = number
    intervals = _array(session["playback_intervals"], "playback_intervals")
    if intervals and not captured["playback"]:
        raise ValueError("playback intervals require declared playback capture")
    previous_end = -1
    for raw_interval in intervals:
        interval = _object(raw_interval, "playback interval", {"start_ms", "end_ms"})
        start = _integer(interval["start_ms"], "playback.start_ms")
        end = _integer(interval["end_ms"], "playback.end_ms")
        _within(start, duration, "playback.start_ms")
        _within(end, duration, "playback.end_ms")
        if start < previous_end or not start < end:
            raise ValueError("playback intervals must be ordered and non-overlapping")
        previous_end = end
    candidates = _array(session["candidates"], "candidates")
    if not candidates:
        raise ValueError("session needs at least one candidate")
    candidate_ids: set[str] = set()
    previous_candidate = -1
    audited = []
    for raw_candidate in candidates:
        item = _object(raw_candidate, "candidate", {"candidate_id", "at_ms", "label"})
        candidate_id = _string(item["candidate_id"], "candidate_id")
        at_ms = _integer(item["at_ms"], "candidate.at_ms")
        if candidate_id in candidate_ids or at_ms < previous_candidate:
            raise ValueError("candidate IDs must be unique and times ordered")
        candidate_ids.add(candidate_id)
        previous_candidate = at_ms
        audited.append(_audit_candidate(item, duration))
    return {
        "session_id": session_id,
        "split": split,
        "speaker": speaker,
        "device": device,
        "source": source,
        "recorded_clock": clock == "recorded_monotonic",
        "capture": captured,
        "capture_complete": all(captured.values()),
        "asr_events": len(revisions),
        "playback_intervals": len(intervals),
        "candidates": audited,
    }


def audit_evidence_manifest(document: Any) -> dict[str, Any]:
    """Summarize declared evidence readiness without echoing private input."""
    root = _object(document, "manifest", {"schema_version", "evidence_level", "sessions"})
    if _integer(root["schema_version"], "schema_version") != 1:
        raise ValueError("unsupported evidence schema")
    level = _string(root["evidence_level"], "evidence_level")
    if level not in {"synthetic", "human_reviewed"}:
        raise ValueError("invalid evidence_level")
    sessions = [_audit_session(item) for item in _array(root["sessions"], "sessions")]
    if not sessions:
        raise ValueError("manifest needs at least one session")
    if level == "human_reviewed" and any(item["source"] == "synthetic" for item in sessions):
        raise ValueError("human_reviewed evidence cannot use a synthetic source")
    if len({item["session_id"] for item in sessions}) != len(sessions):
        raise ValueError("session IDs must be unique")
    speakers_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    devices_by_split: dict[str, set[str]] = {split: set() for split in SPLITS}
    for item in sessions:
        speakers_by_split[item["split"]].add(item["speaker"])
        devices_by_split[item["split"]].add(item["device"])
    heldout_present = bool(speakers_by_split["heldout"])
    nonheldout_speakers = speakers_by_split["development"] | speakers_by_split["tuning"]
    nonheldout_devices = devices_by_split["development"] | devices_by_split["tuning"]
    speaker_overlap = len(speakers_by_split["heldout"] & nonheldout_speakers)
    device_overlap = len(devices_by_split["heldout"] & nonheldout_devices)
    cross_split_speakers = sum(
        sum(identifier in speakers_by_split[split] for split in SPLITS) > 1
        for identifier in set().union(*speakers_by_split.values())
    )
    cross_split_devices = sum(
        sum(identifier in devices_by_split[split] for split in SPLITS) > 1
        for identifier in set().union(*devices_by_split.values())
    )
    candidates = [candidate for session in sessions for candidate in session["candidates"]]
    return {
        "schema_version": 1,
        "declared_evidence_level": level,
        "sessions": len(sessions),
        "candidates": len(candidates),
        "split_sessions": dict(Counter(item["split"] for item in sessions)),
        "source_sessions": dict(Counter(item["source"] for item in sessions)),
        "unique_speakers": len({item["speaker"] for item in sessions}),
        "unique_devices": len({item["device"] for item in sessions}),
        "cross_split_speaker_overlap": cross_split_speakers,
        "cross_split_device_overlap": cross_split_devices,
        "origin_candidates": dict(Counter(item["origin"] for item in candidates)),
        "expected_actions": dict(
            Counter(item["action"] for item in candidates if item["action_ready"])
        ),
        "adjudicated_candidates": sum(item["adjudicated"] for item in candidates),
        "structurally_reviewed_candidates": sum(item["reviewed"] for item in candidates),
        "action_ready_candidates": sum(item["action_ready"] for item in candidates),
        "cutoff_ready_candidates": sum(item["cutoff_ready"] for item in candidates),
        "ambiguous_candidates": sum(item["ambiguous"] for item in candidates),
        "sessions_with_recorded_clock": sum(item["recorded_clock"] for item in sessions),
        "sessions_with_asr_capture": sum(item["capture"]["asr_revisions"] for item in sessions),
        "sessions_with_playback_capture": sum(item["capture"]["playback"] for item in sessions),
        "sessions_with_near_end_echo_capture": sum(
            item["capture"]["near_end_echo"] for item in sessions
        ),
        "sessions_with_recorded_clock_and_all_capture": sum(
            item["recorded_clock"] and item["capture_complete"] for item in sessions
        ),
        "asr_revision_events": sum(item["asr_events"] for item in sessions),
        "playback_intervals": sum(item["playback_intervals"] for item in sessions),
        "heldout_speaker_overlap": speaker_overlap if heldout_present else None,
        "heldout_device_overlap": device_overlap if heldout_present else None,
        "heldout_isolated_by_declared_ids": (
            speaker_overlap == 0 and device_overlap == 0 if heldout_present else None
        ),
        "note": (
            "Structural audit of declarations only; no consent, annotation, clock, "
            "device, or outcome authenticity claim. No private IDs or text emitted."
        ),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit text-free continuous-conversation evidence")
    parser.add_argument("manifest", type=Path, help="JSON evidence manifest outside Git")
    args = parser.parse_args(argv)
    try:
        if args.manifest.stat().st_size > MAX_MANIFEST_BYTES:
            raise ValueError("manifest exceeds size limit")
        result = audit_evidence_manifest(json.loads(args.manifest.read_text(encoding="utf-8")))
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, TypeError):
        print(
            "Invalid or unavailable evidence manifest; no input content printed.", file=sys.stderr
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
