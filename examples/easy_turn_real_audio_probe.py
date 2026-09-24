"""Score the human-recorded Easy Turn test clips with pinned local Smart Turn.

This is an utterance-end audio classification diagnostic, not continuous
conversation endpointing or a full-duplex action benchmark. Public metadata is
read only to verify clip IDs and count speakers; no transcript is printed or
sent to a remote service. Keep downloaded audio under the Git-ignored corpus/.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
import wave
from dataclasses import dataclass
from pathlib import Path

from smartturn_audio_adapter import SMARTTURN_V3_2_CPU_SHA256, LocalSmartTurnScorer

DATASET_REVISION = "5812651dbab429b9a4fab293de7127bfb9a56650"
CATEGORIES = ("complete", "incomplete", "backchannel", "wait")
THRESHOLD = 0.8


@dataclass(frozen=True, slots=True)
class Clip:
    category: str
    path: Path
    speaker: str


def load_clips(root: Path) -> list[Clip]:
    """Require all 400 published real clips and matching metadata entries."""
    clips: list[Clip] = []
    expected_counts = {"complete": 150, "incomplete": 150, "backchannel": 50, "wait": 50}
    for category in CATEGORIES:
        category_dir = root / "testset" / category
        list_name = (
            "incomplete_real_test.list" if category == "incomplete" else f"{category}_test.list"
        )
        rows = [
            json.loads(line)
            for line in (category_dir / list_name).read_text(encoding="utf-8").splitlines()
        ]
        real_rows = [row for row in rows if "_real_" in row["key"]]
        expected_files = set()
        for row in real_rows:
            key = row["key"]
            if not isinstance(key, str) or not key.startswith(f"{category}_real_"):
                raise ValueError(f"unexpected key in {category} metadata")
            path = category_dir / "real" / f"{key}.wav"
            expected_files.add(path.name)
            clips.append(Clip(category, path, str(row["speaker"])))
        actual_files = {path.name for path in (category_dir / "real").glob("*.wav")}
        if len(real_rows) != expected_counts[category] or expected_files != actual_files:
            raise ValueError(f"missing, extra, or duplicate {category} real clips")
    return clips


def percentile(values: list[float], percent: int) -> float:
    if not values:
        raise ValueError("cannot calculate percentile of empty values")
    ordered = sorted(values)
    return ordered[math.ceil(percent * len(ordered) / 100) - 1]


def summarize_scores(
    results: list[tuple[str, float, float, float]], speakers: dict[str, set[str]]
) -> dict[str, object]:
    """Keep four states separate; binary classification uses complete/incomplete only."""
    by_category = {
        category: [row for row in results if row[0] == category] for category in CATEGORIES
    }
    summary = {}
    for category, rows in by_category.items():
        scores = [score for _, score, _, _ in rows]
        summary[category] = {
            "clips": len(rows),
            "speakers": len(speakers[category]),
            "score_p50": round(percentile(scores, 50), 4),
            "score_p95": round(percentile(scores, 95), 4),
            "score_ge_0_8": sum(score >= THRESHOLD for score in scores),
        }
    complete = by_category["complete"]
    incomplete = by_category["incomplete"]
    durations = [duration for _, _, _, duration in results]
    compute = [runtime for _, _, runtime, _ in results]
    summary["complete_vs_incomplete_at_0_8"] = {
        "true_complete": sum(score >= THRESHOLD for _, score, _, _ in complete),
        "missed_complete": sum(score < THRESHOLD for _, score, _, _ in complete),
        "false_complete_on_incomplete": sum(score >= THRESHOLD for _, score, _, _ in incomplete),
        "correct_incomplete": sum(score < THRESHOLD for _, score, _, _ in incomplete),
    }
    return {
        "dataset": "ASLP-lab/Easy-Turn-Testset real clips",
        "dataset_revision": DATASET_REVISION,
        "test_type": "utterance_end_local_audio_score_only",
        "threshold": THRESHOLD,
        "model": "pipecat-ai/smart-turn-v3.2-cpu",
        "model_sha256": SMARTTURN_V3_2_CPU_SHA256,
        "category": summary,
        "unique_speakers": len(set().union(*speakers.values())),
        "complete_incomplete_speaker_overlap": len(speakers["complete"] & speakers["incomplete"]),
        "clips_longer_than_8s_model_context": sum(duration > 8 for duration in durations),
        "max_duration_s": round(max(durations), 3),
        "compute_ms_p50": round(percentile(compute, 50), 1),
        "compute_ms_p95": round(percentile(compute, 95), 1),
        "compute_ms_p99": round(percentile(compute, 99), 1),
        "compute_over_300ms": sum(runtime > 300 for runtime in compute),
        "warning": (
            "No continuous audio, response timing, ASR, Jev, OVA, or device-disjoint acceptance."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    import numpy as np

    clips = load_clips(args.dataset_root)
    scorer = LocalSmartTurnScorer(args.model)
    scorer.score(
        np.zeros(16000, dtype=np.float32)
    )  # separate cold initialization from steady-state
    results: list[tuple[str, float, float, float]] = []
    speakers: dict[str, set[str]] = {category: set() for category in CATEGORIES}
    converted = 0
    for clip in clips:
        with wave.open(str(clip.path), "rb") as source:
            if source.getsampwidth() != 2:
                raise ValueError(f"unsupported audio precision: {clip.path}")
            duration = source.getnframes() / source.getframerate()
            if source.getnchannels() == 1 and source.getframerate() == 16000:
                audio = np.frombuffer(source.readframes(source.getnframes()), dtype="<i2")
            else:
                converted += 1
                rendered = subprocess.run(
                    [
                        "sox",
                        "-D",  # disable automatic random dither for reproducible resampling
                        str(clip.path),
                        "-r",
                        "16000",
                        "-c",
                        "1",
                        "-b",
                        "16",
                        "-e",
                        "signed-integer",
                        "-t",
                        "raw",
                        "-",
                    ],
                    check=True,
                    capture_output=True,
                )
                audio = np.frombuffer(rendered.stdout, dtype="<i2")
        samples = audio.astype(np.float32) / 32768.0
        started = time.perf_counter()
        score = scorer.score(samples)
        runtime_ms = (time.perf_counter() - started) * 1000
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("model returned invalid probability")
        results.append((clip.category, score, runtime_ms, duration))
        speakers[clip.category].add(clip.speaker)
    report = summarize_scores(results, speakers)
    report["resampled_or_downmixed_clips"] = converted
    report["sox_random_dither_disabled"] = True
    print(json.dumps(report, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
