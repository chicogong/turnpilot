"""Offline, text-free audit of local SmoothConv annotation JSON files.

This uses human segment boundaries as an oracle. It does not run audio VAD,
streaming ASR, Jev, or TurnPilot's acoustic gate. Data must be obtained from
the dataset publisher and kept outside Git (for example under ``corpus/``).
"""

from __future__ import annotations

import argparse
import json
import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from turnpilot.acoustic_eval import LabeledPauseWindow


@dataclass(frozen=True, slots=True)
class Segment:
    start_ms: int
    end_ms: int
    speaker: str
    turn: str
    channel_index: int | None = None


def _millis(value: object) -> int:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError("segment timestamp must be numeric")
    if not math.isfinite(value) or value < 0:
        raise ValueError("segment timestamp must be finite and non-negative")
    return round(value * 1000)


def read_segments(path: Path) -> tuple[Segment, ...]:
    """Read only timing and turn labels; never emit the annotation text."""
    data: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("instances"), list):
        raise ValueError("annotation must contain an instances list")
    segments: list[Segment] = []
    for item in data["instances"]:
        if not isinstance(item, dict) or not isinstance(item.get("attributes"), dict):
            raise ValueError("invalid annotation instance")
        start_ms = _millis(item.get("start"))
        end_ms = _millis(item.get("end"))
        if end_ms < start_ms:
            raise ValueError("segment end precedes start")
        attributes = item["attributes"]
        speaker = attributes.get("speaker")
        turn = attributes.get("turn")
        channel_index = item.get("channelIndex")
        segments.append(
            Segment(
                start_ms,
                end_ms,
                speaker if isinstance(speaker, str) else "unknown",
                turn if isinstance(turn, str) else "unknown",
                channel_index
                if isinstance(channel_index, int)
                and not isinstance(channel_index, bool)
                and channel_index >= 0
                else None,
            )
        )
    return tuple(sorted(segments, key=lambda segment: (segment.start_ms, segment.end_ms)))


def audio_duration_ms(path: Path) -> int:
    """Check a publisher-provided uncompressed WAV without decoding its content."""
    with wave.open(str(path), "rb") as audio:
        return round(audio.getnframes() * 1000 / audio.getframerate())


def labeled_windows(
    segments: tuple[Segment, ...],
    channel: int,
    duration_ms: int,
    *,
    completion_window_ms: int,
    min_observed_pause_ms: int,
) -> tuple[tuple[LabeledPauseWindow, ...], dict[str, int]]:
    """Select post-hoc windows; none of these labels enter acoustic detection."""
    channel_segments = tuple(
        item for item in segments if item.channel_index == channel and item.speaker != "unknown"
    )
    windows: list[LabeledPauseWindow] = []
    excluded = {
        "incomplete_censored": 0,
        "incomplete_interrupted": 0,
        "complete_short_horizon": 0,
    }
    for index, segment in enumerate(channel_segments):
        if segment.turn not in ("complete", "incomplete"):
            continue
        next_same = next(
            (
                item
                for item in channel_segments[index + 1 :]
                if item.speaker == segment.speaker and item.start_ms >= segment.end_ms
            ),
            None,
        )
        if segment.turn == "incomplete":
            if next_same is None or next_same.start_ms <= segment.end_ms:
                excluded["incomplete_censored"] += 1
                continue
            other_speaker_between = any(
                other.speaker not in ("unknown", segment.speaker)
                and other.start_ms < next_same.start_ms
                and other.end_ms > segment.end_ms
                for other in segments
            )
            if other_speaker_between:
                excluded["incomplete_interrupted"] += 1
                continue
            windows.append(
                LabeledPauseWindow(
                    segment.end_ms, next_same.start_ms, False, start_ms=segment.start_ms
                )
            )
        else:
            horizon = min(duration_ms, segment.end_ms + completion_window_ms)
            if next_same is not None:
                horizon = min(horizon, next_same.start_ms)
            if horizon - segment.end_ms < min_observed_pause_ms:
                excluded["complete_short_horizon"] += 1
                continue
            windows.append(
                LabeledPauseWindow(segment.end_ms, horizon, True, start_ms=segment.start_ms)
            )
    return tuple(windows), excluded


def summarize(
    recordings: tuple[tuple[Segment, ...], ...], *, silence_ms: int = 640
) -> dict[str, object]:
    """Count oracle-boundary risks, not measured TurnPilot error rates."""
    if silence_ms <= 0:
        raise ValueError("silence threshold must be positive")
    if not recordings:
        raise ValueError("at least one recording is required")
    turn_counts = {
        label: 0 for label in ("complete", "incomplete", "backchannel", "wait", "unknown")
    }
    continuations = 0
    censored = 0
    interrupted = 0
    clean = 0
    all_risks = 0
    clean_risks = 0
    incomplete_unknown_speaker = 0
    for segments in recordings:
        for index, segment in enumerate(segments):
            label = segment.turn if segment.turn in turn_counts else "unknown"
            turn_counts[label] += 1
            if label != "incomplete":
                continue
            if segment.speaker == "unknown":
                incomplete_unknown_speaker += 1
                continue
            later = segments[index + 1 :]
            next_same = next(
                (
                    candidate
                    for candidate in later
                    if candidate.speaker == segment.speaker
                    and candidate.start_ms > segment.start_ms
                ),
                None,
            )
            if next_same is None:
                censored += 1
                continue
            continuations += 1
            gap_ms = max(0, next_same.start_ms - segment.end_ms)
            at_risk = gap_ms > silence_ms  # a resume exactly at the deadline wins
            all_risks += at_risk
            has_other_speaker = any(
                other.speaker not in ("unknown", segment.speaker)
                and other.start_ms < next_same.start_ms
                and other.end_ms > segment.end_ms
                for other in segments
            )
            if has_other_speaker:
                interrupted += 1
            else:
                clean += 1
                clean_risks += at_risk
    return {
        "recordings": len(recordings),
        "segments": sum(len(segments) for segments in recordings),
        "turn_counts": turn_counts,
        "silence_threshold_ms": silence_ms,
        "incomplete_with_same_speaker_continuation": continuations,
        "incomplete_censored_without_continuation": censored,
        "incomplete_unknown_speaker": incomplete_unknown_speaker,
        "incomplete_interrupted_before_continuation": interrupted,
        "incomplete_clean_continuations": clean,
        "oracle_fixed_silence_risks_all": all_risks,
        "oracle_fixed_silence_risks_clean": clean_risks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", nargs="+", type=Path)
    parser.add_argument("--silence-ms", type=int, default=640)
    parser.add_argument("--audio", type=Path, help="optional WAV matching the first annotation")
    args = parser.parse_args()
    recordings = tuple(read_segments(path) for path in args.annotations)
    result = summarize(recordings, silence_ms=args.silence_ms)
    if args.audio is not None:
        duration = audio_duration_ms(args.audio)
        if max((segment.end_ms for segment in recordings[0]), default=0) > duration + 1:
            raise ValueError("annotation extends past the WAV duration")
        result["first_audio_duration_ms"] = duration
    result["evidence_level"] = "publisher_annotations_oracle_boundaries_only"
    result["warning"] = (
        "No audio VAD, streaming ASR, Jev, or TurnPilot acoustic policy was evaluated; "
        "sample selection and segment labels are not independent product ground truth."
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
