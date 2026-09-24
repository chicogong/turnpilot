"""Bounded Jev test on public Easy Turn complete/incomplete text labels.

Pass dataset JSONL lists stored outside Git. This is a text classification
probe, not streaming ASR, audio clarity, speaker verification, or device QA.
Only aggregate metrics are printed; no transcript or credential is logged.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any

from turnpilot import TranscriptSignal, TurnRef
from turnpilot.jev import HttpxJevTransport, JevError, JevJudge


def _load(path: Path, *, label: str) -> list[dict[str, str]]:
    suffix = f"<{label.upper()}>"
    rows: list[dict[str, str]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        item = json.loads(line)
        text = item["txt"]
        source = item["extra"]["dataset"]
        if item["lang"] != "<CN>" or not text.endswith(suffix):
            raise ValueError("unexpected Easy Turn label or language")
        if source.endswith("_real"):
            origin = "real"
        elif source.endswith("_synthetic"):
            origin = "synthetic"
        else:
            raise ValueError("unexpected Easy Turn source")
        rows.append(
            {
                "key": item["key"],
                "text": text[: -len(suffix)].strip(),
                "speaker": str(item["speaker"]),
                "label": label,
                "origin": origin,
            }
        )
    return rows


def _sample(
    rows: list[dict[str, str]], per_stratum: int, origins: tuple[str, ...] = ("real", "synthetic")
) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    for label in ("complete", "incomplete"):
        for origin in origins:
            pool = sorted(
                (row for row in rows if row["label"] == label and row["origin"] == origin),
                key=lambda row: hashlib.sha256(row["key"].encode()).digest(),
            )
            distinct: list[dict[str, str]] = []
            seen_speakers: set[str] = set()
            for row in pool:
                if row["speaker"] not in seen_speakers:
                    distinct.append(row)
                    seen_speakers.add(row["speaker"])
            chosen = distinct[:per_stratum]
            if len(chosen) < per_stratum:
                chosen_keys = {row["key"] for row in chosen}
                chosen.extend(row for row in pool if row["key"] not in chosen_keys)
                chosen = chosen[:per_stratum]
            if len(chosen) != per_stratum:
                raise ValueError("not enough rows in one Easy Turn stratum")
            selected.extend(chosen)
    return selected


def _percentile(values: list[int], pct: int) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[(len(ordered) * pct + 99) // 100 - 1]


async def _run(selected: list[dict[str, str]], timeout_ms: int) -> dict[str, Any]:
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        raise RuntimeError("TYPESAFE_API_KEY is required")
    transport = HttpxJevTransport(key)
    judge = JevJudge(transport, timeout_ms=timeout_ms, max_calls_per_turn=1)
    results: list[dict[str, Any]] = []
    try:
        for row in selected:
            now_ms = time.monotonic_ns() // 1_000_000
            transcript = TranscriptSignal(
                TurnRef("public-easyturn-probe", row["key"], 0),
                now_ms,
                1,
                row["text"],
                True,
            )
            started = time.perf_counter()
            result: dict[str, Any] = {
                "label": row["label"],
                "origin": row["origin"],
                "nonempty_dataset_text": bool(row["text"]),
                "punctuation_complete": row["text"].endswith(("。", "？", "?", "！", "!")),
            }
            try:
                signal = await judge.judge(transcript, now_ms=now_ms, allow_remote_text=True)
                result["complete_probability"] = signal.complete_probability
                result["model"] = signal.model
            except JevError:
                result["error"] = True
            result["latency_ms"] = round((time.perf_counter() - started) * 1000)
            results.append(result)
    finally:
        await transport.close()

    strata: list[dict[str, Any]] = []
    for label in ("complete", "incomplete"):
        for origin in ("real", "synthetic"):
            group = [row for row in results if row["label"] == label and row["origin"] == origin]
            if not group:
                continue
            usable = [row for row in group if "complete_probability" in row]
            latencies = [int(row["latency_ms"]) for row in usable]
            strata.append(
                {
                    "label": label,
                    "origin": origin,
                    "attempted": len(group),
                    "successful": len(usable),
                    "errors": len(group) - len(usable),
                    "nonempty_dataset_text": sum(row["nonempty_dataset_text"] for row in group),
                    "jev_complete_at_0_8": sum(
                        row["complete_probability"] >= 0.8 for row in usable
                    ),
                    "punctuation_complete": sum(row["punctuation_complete"] for row in group),
                    "latency_p50_ms": _percentile(latencies, 50),
                    "latency_p95_ms": _percentile(latencies, 95),
                    "within_350_ms": sum(value <= 350 for value in latencies),
                    "within_640_ms": sum(value <= 640 for value in latencies),
                }
            )
    selection_hash = hashlib.sha256("\n".join(row["key"] for row in selected).encode()).hexdigest()
    return {
        "evidence_level": "public_clip_text_labels_live_provider",
        "dataset": "ASLP-lab/Easy-Turn-Testset",
        "selected_keys_sha256": selection_hash,
        "requests_attempted": len(selected),
        "request_timeout_ms": timeout_ms,
        "strata": strata,
        "warning": (
            "Static dataset transcript only; no audio, ASR revision timing, conversation context, "
            "device split, or full TurnPilot action policy."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--complete-list", type=Path, required=True)
    parser.add_argument("--incomplete-list", type=Path, required=True)
    parser.add_argument("--per-stratum", type=int, default=10)
    parser.add_argument("--origin", choices=("both", "real", "synthetic"), default="both")
    parser.add_argument("--timeout-ms", type=int, default=1500)
    parser.add_argument("--allow-remote-text", action="store_true")
    args = parser.parse_args()
    if not args.allow_remote_text:
        parser.error("public transcript upload requires --allow-remote-text")
    if not 1 <= args.per_stratum <= 20:
        parser.error("--per-stratum must be between 1 and 20")
    rows = _load(args.complete_list, label="complete") + _load(
        args.incomplete_list, label="incomplete"
    )
    origins = ("real", "synthetic") if args.origin == "both" else (args.origin,)
    selected = _sample(rows, args.per_stratum, origins)
    print(json.dumps(asyncio.run(_run(selected, args.timeout_ms)), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
