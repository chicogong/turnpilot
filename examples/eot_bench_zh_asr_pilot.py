"""Small, paced Chinese public-audio check with *actual* local ASR arrivals.

Use the first N rows in publisher order (default four) of a separately obtained
LiveKit eot-bench zh Arrow file. Each row is a fresh, <=60-second file-run
window. Silero and Vosk are local; Jev and host playback are deliberately absent.
The output contains only aggregate timing/count data and hashes, never words,
audio, row IDs, paths, or per-window timelines. EOF never creates a Vosk final.

Example (install TurnPilot's ``file`` extra and pyarrow separately):

    uv run --no-sync --with pyarrow python examples/eot_bench_zh_asr_pilot.py \
      --input corpus/eot-bench/zh-validation.arrow \
      --silero-model corpus/models/silero_vad_v6.2.2.onnx \
      --vosk-model corpus/models/vosk-model-small-cn-0.22 \
      --output reports/zh-asr-pilot.json

This is an exploratory first-recommendation comparison, not an action-quality
benchmark. The publisher silence spans are joined only after causal replay.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import tempfile
import wave
from collections import Counter
from pathlib import Path
from typing import Any

from eot_bench_zh_replay import _load_table, eligible_spans

from turnpilot.file_adapters import SileroOnnxVad, VoskStreamingAsr, model_fingerprint
from turnpilot.file_run import MAX_DURATION_MS, run_file

ARMS = ("direct_asr_final", "fixed_640", "adaptive_640", "gate")
MAX_PILOT_TURNS = 5
MAX_PILOT_DURATION_MS = 120_000
CODE_FILES = (
    "src/turnpilot/acoustic.py",
    "src/turnpilot/action_gate.py",
    "src/turnpilot/action_trace.py",
    "src/turnpilot/file_adapters.py",
    "src/turnpilot/file_run.py",
    "src/turnpilot/models.py",
    "src/turnpilot/policy.py",
    "examples/eot_bench_zh_replay.py",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _embedded_wav(audio: dict[str, Any]) -> bytes:
    """Accept embedded, unmodified 16 kHz PCM16 WAV; never resolve row paths."""
    import io

    content = audio.get("bytes")
    if not isinstance(content, bytes) or not content:
        raise ValueError("each selected row must embed WAV bytes")
    with wave.open(io.BytesIO(content), "rb") as source:
        if (
            source.getframerate(),
            source.getnchannels(),
            source.getsampwidth(),
            source.getcomptype(),
        ) != (16000, 1, 2, "NONE"):
            raise ValueError("embedded WAV must be 16 kHz mono PCM16")
        if not 512 <= source.getnframes() <= MAX_DURATION_MS * 16:
            raise ValueError("selected row is outside file-run's 32 ms to 60 s window")
    return content


def _region(media_at_ms: int | None, spans: tuple[Any, ...]) -> str:
    if media_at_ms is None:
        return "no_recommendation_by_eof"
    for span in spans:
        if span.start_ms <= media_at_ms < span.end_ms and span.is_hold:
            return "labeled_hold"
        if span.start_ms <= media_at_ms <= span.end_ms and not span.is_hold:
            return "labeled_eot"
    return "outside_labeled_silence"


def summarize_window(report: dict[str, Any], raw_spans: list[dict[str, Any]]) -> dict[str, Any]:
    """Reduce an existing content-free file-run report; do not persist its trace."""
    if report["eof_forced_final"] or report["synthetic_tail_silence_ms"]:
        raise ValueError("pilot requires unmodified EOF behavior")
    if report["window"]["start_ms"] != 0 or not report["window"]["ends_at_source_eof"]:
        raise ValueError("pilot requires each full source recording from start to EOF")
    if report["evidence_level"] != "paced_file_pipeline_only":
        raise ValueError("pilot requires actual paced file evidence")
    timeline = report["timeline"]
    if any("text" in event for event in timeline):
        raise ValueError("content-bearing timeline is not allowed")
    events = [event for event in timeline if event["kind"] == "asr"]
    finals = [event for event in events if event["is_final"] and event["has_text"]]
    partials = [event for event in events if not event["is_final"] and event["has_text"]]
    if len(finals) != report["asr"]["nonempty_decoder_finals"]:
        raise ValueError("decoder-final count disagrees with timeline")
    spans = eligible_spans(raw_spans, audio_duration_ms=round(report["duration_ms"]))
    arms: dict[str, dict[str, Any]] = {}
    for name in ARMS:
        recommendation = report["arms"][name]["recommendation"]
        arms[name] = {
            "kind": None if recommendation is None else recommendation["kind"],
            "media_at_ms": None if recommendation is None else recommendation["media_at_ms"],
            "available_at_ms": None
            if recommendation is None
            else recommendation["available_at_ms"],
            "unprocessed_backlog": False
            if recommendation is None
            else recommendation.get("unprocessed_backlog", False),
            "media_position_region": _region(
                None if recommendation is None else recommendation["media_at_ms"], spans
            ),
        }
    return {
        "duration_ms": report["duration_ms"],
        "processed_media_ms": report["processed_media_ms"],
        "unprocessed_tail_samples": report["unprocessed_tail_samples"],
        "backlog_frames": report["pipeline"]["backlog_frames"],
        "frame_availability_lag_p95_ms": report["pipeline"]["availability_lag_p95_ms"],
        "frame_availability_lag_max_ms": report["pipeline"]["availability_lag_max_ms"],
        "asr_revisions": report["asr"]["revisions"],
        "nonempty_partial_revisions": len(partials),
        "nonempty_decoder_finals": len(finals),
        "first_partial_available_ms": None if not partials else partials[0]["available_at_ms"],
        "first_final_available_ms": None if not finals else finals[0]["available_at_ms"],
        "first_final_availability_lag_ms": None
        if not finals
        else finals[0]["available_at_ms"] - finals[0]["media_at_ms"],
        "arms": arms,
    }


def _nearest_rank(values: list[int], fraction: float) -> int | None:
    if not values:
        return None
    return sorted(values)[math.ceil(fraction * len(values)) - 1]


def aggregate_windows(windows: list[dict[str, Any]]) -> dict[str, Any]:
    """Only cohort aggregates leave this function; row details stay in memory."""
    if not windows:
        raise ValueError("at least one completed window is required")
    first_finals = [w["first_final_available_ms"] for w in windows]
    first_partials = [w["first_partial_available_ms"] for w in windows]
    final_lags = [w["first_final_availability_lag_ms"] for w in windows]
    arm_counts: dict[str, Any] = {}
    for name in ARMS:
        outcomes = [w["arms"][name] for w in windows]
        arm_counts[name] = {
            "recommended_windows": sum(o["kind"] is not None for o in outcomes),
            "no_recommendation_by_eof": sum(o["kind"] is None for o in outcomes),
            "recommended_while_backlogged": sum(o["unprocessed_backlog"] for o in outcomes),
            "first_recommendation_media_position_region": dict(
                Counter(o["media_position_region"] for o in outcomes)
            ),
            "recommendation_kind": dict(Counter(o["kind"] for o in outcomes if o["kind"])),
            "availability_ms_p50": _nearest_rank(
                [o["available_at_ms"] for o in outcomes if o["available_at_ms"] is not None],
                0.5,
            ),
        }
    paired: dict[str, Any] = {}
    for name in ("fixed_640", "adaptive_640", "gate"):
        both = [
            (w["arms"]["direct_asr_final"]["available_at_ms"], w["arms"][name]["available_at_ms"])
            for w in windows
            if w["arms"]["direct_asr_final"]["available_at_ms"] is not None
            and w["arms"][name]["available_at_ms"] is not None
        ]
        deltas = [alternative - direct for direct, alternative in both]
        paired[name] = {
            "both_recommended_windows": len(both),
            "direct_earlier": sum(delta > 0 for delta in deltas),
            "same_availability": sum(delta == 0 for delta in deltas),
            "alternative_earlier": sum(delta < 0 for delta in deltas),
            "alternative_minus_direct_ms_p50": _nearest_rank(deltas, 0.5),
        }
    fixed_adaptive_mismatches = sum(
        (
            w["arms"]["fixed_640"]["kind"],
            w["arms"]["fixed_640"]["media_at_ms"],
            w["arms"]["fixed_640"]["available_at_ms"],
        )
        != (
            w["arms"]["adaptive_640"]["kind"],
            w["arms"]["adaptive_640"]["media_at_ms"],
            w["arms"]["adaptive_640"]["available_at_ms"],
        )
        for w in windows
    )
    processed_frames = sum(w["processed_media_ms"] for w in windows) // 32
    backlog_frames = sum(w["backlog_frames"] for w in windows)
    return {
        "windows": len(windows),
        "source_audio_duration_ms": round(sum(w["duration_ms"] for w in windows)),
        "processed_media_ms": sum(w["processed_media_ms"] for w in windows),
        "unprocessed_tail_samples": sum(w["unprocessed_tail_samples"] for w in windows),
        "processed_frames": processed_frames,
        "backlog_frames": backlog_frames,
        "backlog_frame_fraction": round(backlog_frames / processed_frames, 4),
        "per_window_frame_availability_lag_p95_ms_range": [
            min(w["frame_availability_lag_p95_ms"] for w in windows),
            max(w["frame_availability_lag_p95_ms"] for w in windows),
        ],
        "max_frame_availability_lag_ms": max(w["frame_availability_lag_max_ms"] for w in windows),
        "asr": {
            "revision_events": sum(w["asr_revisions"] for w in windows),
            "nonempty_partial_revisions": sum(w["nonempty_partial_revisions"] for w in windows),
            "nonempty_decoder_finals": sum(w["nonempty_decoder_finals"] for w in windows),
            "windows_with_nonempty_final": sum(value is not None for value in first_finals),
            "no_nonempty_final_by_eof_censored": sum(value is None for value in first_finals),
            "first_partial_available_ms_p50": _nearest_rank(
                [value for value in first_partials if value is not None], 0.5
            ),
            "first_final_available_ms_p50": _nearest_rank(
                [value for value in first_finals if value is not None], 0.5
            ),
            "first_final_availability_lag_ms_p95": _nearest_rank(
                [value for value in final_lags if value is not None], 0.95
            ),
        },
        "arms": arm_counts,
        "paired_against_direct_final": paired,
        "fixed_vs_adaptive_first_recommendation_mismatch_windows": fixed_adaptive_mismatches,
        "gate_no_recommendation_by_eof": {
            "with_decoder_final": sum(
                w["arms"]["gate"]["kind"] is None and w["nonempty_decoder_finals"] > 0
                for w in windows
            ),
            "without_decoder_final": sum(
                w["arms"]["gate"]["kind"] is None and w["nonempty_decoder_finals"] == 0
                for w in windows
            ),
        },
    }


async def replay(
    input_path: Path, silero_path: Path, vosk_path: Path, max_turns: int
) -> dict[str, Any]:
    if not 1 <= max_turns <= MAX_PILOT_TURNS:
        raise ValueError("--max-turns must be between 1 and 5")
    table = _load_table(input_path)
    if table.num_rows < max_turns:
        raise ValueError("input has fewer rows than requested")
    selected_duration_ms = round(
        sum(float(table["duration"][index].as_py()) * 1000 for index in range(max_turns))
    )
    if not 0 < selected_duration_ms <= MAX_PILOT_DURATION_MS:
        raise ValueError("first N rows exceed the 120 s pilot cap")
    windows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="turnpilot-public-asr-") as temp_dir:
        wav_path = Path(temp_dir) / "current.wav"
        for index in range(max_turns):
            if table["language"][index].as_py() != "zh":
                raise ValueError("selected rows must be Chinese")
            audio = _embedded_wav(table["audio"][index].as_py())
            wav_path.write_bytes(audio)
            try:
                # Fresh model state and a fresh monotonic clock per publisher row.
                result = await run_file(
                    wav_path, SileroOnnxVad(silero_path), VoskStreamingAsr(vosk_path)
                )
                # Timing labels are read only after all causal recommendations exist.
                raw_spans = table["silence_spans"][index].as_py()
                if not raw_spans or raw_spans[0]["start"] <= 0:
                    raise ValueError("selected row must include speech before its first silence")
                windows.append(summarize_window(result.report, raw_spans))
            finally:
                wav_path.unlink(missing_ok=True)
    return {
        "schema": 1,
        "evidence_level": "public_zh_paced_file_pipeline_no_human_action_labels",
        "selection": {
            "rule": "first_N_rows_in_publisher_order_no_cherry_picking",
            "start_row_index": 0,
            "source_rows": table.num_rows,
            "selected_rows": max_turns,
            "maximum_total_source_audio_ms": MAX_PILOT_DURATION_MS,
        },
        "input_sha256": _sha256(input_path),
        "silero_model_fingerprint": model_fingerprint(silero_path),
        "vosk_model_fingerprint": model_fingerprint(vosk_path),
        "script_sha256": _sha256(Path(__file__)),
        "code_sha256": {
            relative_path: _sha256(Path(__file__).resolve().parents[1] / relative_path)
            for relative_path in CODE_FILES
        },
        "runtime": {
            "python": platform.python_version(),
            "system": platform.system(),
            "architecture": platform.machine(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("turnpilot", "numpy", "onnxruntime", "vosk", "pyarrow")
            },
        },
        "results": aggregate_windows(windows),
        "limits": [
            "First recommendation per file only; not a multi-turn host or "
            "official eot-bench score.",
            "Silence-span regions use media position, not real-time action arrival; "
            "backlog is reported separately.",
            "Publisher timing labels are not expected conversation actions.",
            "Span regions use the acoustic replay's 100 ms minimum and <=5 s hold filter; "
            "outside includes filtered-out spans.",
            "No speaker/device split, WER, playback, tools, Jev, or audible-response timing.",
            "No ASR final by EOF is censored, not a proven missed user response.",
            "ASR availability and frame backlog reflect this local machine and model, "
            "not a streaming service or device SLA.",
            "Latency medians use only windows with a recommendation; paired comparisons "
            "have their own shared-window denominator.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="local zh validation .arrow")
    parser.add_argument("--silero-model", type=Path, required=True, help="local Silero ONNX")
    parser.add_argument("--vosk-model", type=Path, required=True, help="local Vosk model directory")
    parser.add_argument("--output", type=Path, required=True, help="new aggregate-only JSON")
    parser.add_argument("--max-turns", type=int, default=4, help="first N rows, from 1 to 5")
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        parser.error("output must not already exist")
    if args.input.suffix != ".arrow" or not args.input.is_file():
        parser.error("input must be an existing local .arrow file")
    if not args.silero_model.is_file() or not args.vosk_model.is_dir():
        parser.error("local Silero file and Vosk directory are required")
    if not 1 <= args.max_turns <= MAX_PILOT_TURNS:
        parser.error("--max-turns must be between 1 and 5")
    report = asyncio.run(replay(args.input, args.silero_model, args.vosk_model, args.max_turns))
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, sort_keys=True, indent=2)
        output.write("\n")
    print(json.dumps(report["results"], ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
