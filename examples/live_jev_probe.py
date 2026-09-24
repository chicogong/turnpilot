"""Opt-in live Jev probe on predeclared, synthetic Mandarin examples.

This measures provider behavior on ten authored examples, not real-device turn
taking or a validated accuracy benchmark. No private transcript or API key is
printed or written by this script.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass

from turnpilot import TranscriptSignal, TurnRef
from turnpilot.jev import HttpxJevTransport, JevError, JevJudge
from turnpilot.policy import PolicyConfig, TurnPolicy
from turnpilot.timing import PauseTimingCase, summarize_pauses


@dataclass(frozen=True)
class Case:
    case_id: str
    text: str
    complete: bool
    pause_ms: int
    partial_text: str = ""


CASES = (
    Case("complete-reminder-short", "明天下午三点提醒我开会", True, 320, "明天下午三点提醒我"),
    Case("complete-weather-short", "北京今天天气怎么样？", True, 320, "北京今天天气"),
    Case("complete-volume-long", "请把音量调低一点。", True, 800, "请把音量"),
    Case("complete-cancel-long", "取消刚才那个提醒", True, 800, "取消刚才"),
    Case("complete-train-short", "明天上午北京到上海的高铁有哪些", True, 320, "明天上午北京到上海"),
    Case("unfinished-origin-long", "我想订一张从北京到", False, 800, "我想订一张"),
    Case(
        "unfinished-list-long",
        "这个方案有三个问题，第一是",
        False,
        800,
        "这个方案有三个问题，第一",
    ),
    Case("unfinished-if-short", "如果明天下雨的话，我们就", False, 320, "如果明天下雨"),
    Case(
        "unfinished-compare-long",
        "请帮我比较两个方案，先看",
        False,
        800,
        "请帮我比较两个方案，先",
    ),
    Case("unfinished-reminder-short", "明天下午三点提醒我……", False, 320, "明天下午三点提醒我"),
)


def _score(rows: list[dict[str, object]], key: str) -> dict[str, int]:
    usable = [row for row in rows if isinstance(row.get(key), bool)]
    return {
        "evaluated": len(usable),
        "false_cutoffs": sum(
            row[key] is True and row["expected_complete"] is False for row in usable
        ),
        "missed_completions": sum(
            row[key] is False and row["expected_complete"] is True for row in usable
        ),
    }


def _latency_summary(rows: list[dict[str, object]]) -> dict[str, int | None]:
    values = sorted(
        value
        for row in rows
        if isinstance((value := row.get("latency_ms")), int) and "error" not in row
    )
    if not values:
        return {"successful": 0, "p50_ms": None, "p95_ms": None}
    return {
        "successful": len(values),
        "p50_ms": values[(len(values) * 50 + 99) // 100 - 1],
        "p95_ms": values[(len(values) * 95 + 99) // 100 - 1],
    }


def _timing_case(
    case: Case,
    row: dict[str, object],
    final_row: dict[str, object],
    dispatch_at_ms: int,
    revised_at_ms: int | None,
) -> PauseTimingCase:
    latency = row.get("latency_ms")
    probability = row.get("jev_complete_probability")
    usable = isinstance(latency, int) and isinstance(probability, (int, float))
    final_latency = final_row.get("latency_ms")
    final_probability = final_row.get("jev_complete_probability")
    final_usable = isinstance(final_latency, int) and isinstance(final_probability, (int, float))
    return PauseTimingCase(
        case.case_id,
        case.complete,
        transcript_available_at_ms=dispatch_at_ms,
        dispatch_at_ms=dispatch_at_ms,
        resume_at_ms=None if case.complete else case.pause_ms,
        jev_latency_ms=latency if usable and isinstance(latency, int) else None,
        jev_complete_probability=(
            float(probability) if usable and isinstance(probability, (int, float)) else None
        ),
        revised_transcript_at_ms=revised_at_ms,
        revised_jev_latency_ms=(
            final_latency
            if revised_at_ms is not None and final_usable and isinstance(final_latency, int)
            else None
        ),
        revised_jev_complete_probability=(
            float(final_probability)
            if revised_at_ms is not None
            and final_usable
            and isinstance(final_probability, (int, float))
            else None
        ),
    )


async def run(timeout_ms: int) -> dict[str, object]:
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        raise RuntimeError("TYPESAFE_API_KEY is required")
    transport = HttpxJevTransport(key)
    judge = JevJudge(transport, timeout_ms=timeout_ms, max_calls_per_turn=1)
    rows: list[dict[str, object]] = []
    try:
        for case in CASES:
            now_ms = time.monotonic_ns() // 1_000_000
            transcript = TranscriptSignal(
                TurnRef("synthetic-probe", case.case_id, 0), now_ms, 1, case.text, True
            )
            row: dict[str, object] = {
                "case_id": case.case_id,
                "expected_complete": case.complete,
                "scripted_pause_ms": case.pause_ms,
                "fixed_640_complete": case.pause_ms >= 640,
                "punctuation_complete": case.text.rstrip().endswith(("。", "？", "?", "！", "!")),
            }
            started = time.perf_counter()
            try:
                signal = await judge.judge(transcript, now_ms=now_ms, allow_remote_text=True)
                latency_ms = round((time.perf_counter() - started) * 1000)
                row.update(
                    {
                        "jev_complete_probability": signal.complete_probability,
                        "jev_complete_at_0_8": signal.complete_probability >= 0.8,
                        "jev_clarify_meaning_probability": signal.clarification_probability,
                        "jev_response_needed_probability": signal.response_probability,
                        "latency_ms": latency_ms,
                        "within_350_ms": latency_ms <= 350,
                        "bounded_jev_complete": (
                            signal.complete_probability >= 0.8
                            if latency_ms <= 350
                            else case.pause_ms >= 640
                        ),
                        "model": signal.model,
                    }
                )
            except JevError as error:
                row["error"] = str(error)
                row["latency_ms"] = round((time.perf_counter() - started) * 1000)
                row["within_350_ms"] = False
                row["bounded_jev_complete"] = case.pause_ms >= 640
            rows.append(row)
    finally:
        await transport.close()
    return {
        "evidence_level": "synthetic_text_live_provider",
        "warning": (
            "Authored expectations and scripted pauses are not a validated corpus or real audio."
        ),
        "requests_attempted": len(rows),
        "request_timeout_ms": timeout_ms,
        "realtime_decision_budget_ms": 350,
        "cases": rows,
        "scores": {
            "fixed_640": _score(rows, "fixed_640_complete"),
            "punctuation": _score(rows, "punctuation_complete"),
            "jev_0_8": _score(rows, "jev_complete_at_0_8"),
            "jev_with_350_ms_fallback": _score(rows, "bounded_jev_complete"),
        },
    }


async def run_paired(timeout_ms: int, *, dispatch_at_ms: int = 192) -> dict[str, object]:
    """Probe authored prefixes and final text; simulate an optimistic pause-time dispatch.

    The assumed prefix availability at `dispatch_at_ms` is not measured ASR.
    Jev calls are sequential for rate safety; latency is measured per call.
    """
    if dispatch_at_ms < 0:
        raise ValueError("dispatch_at_ms must be non-negative")
    key = os.environ.get("TYPESAFE_API_KEY")
    if not key:
        raise RuntimeError("TYPESAFE_API_KEY is required")
    transport = HttpxJevTransport(key)
    judge = JevJudge(
        transport,
        timeout_ms=timeout_ms,
        max_calls_per_turn=2,
        min_call_interval_ms=0,
    )
    rows: list[dict[str, object]] = []
    try:
        for case in CASES:
            if not case.partial_text:
                raise ValueError("paired probe requires authored partial text")
            for phase, text, revision in (
                ("partial", case.partial_text, 1),
                ("final", case.text, 2),
            ):
                now_ms = time.monotonic_ns() // 1_000_000
                transcript = TranscriptSignal(
                    TurnRef("synthetic-paired-probe", case.case_id, 0),
                    now_ms,
                    revision,
                    text,
                    phase == "final",
                )
                row: dict[str, object] = {
                    "case_id": case.case_id,
                    "phase": phase,
                    "expected_complete": case.complete if phase == "final" else False,
                }
                started = time.perf_counter()
                try:
                    signal = await judge.judge(transcript, now_ms=now_ms, allow_remote_text=True)
                    latency_ms = round((time.perf_counter() - started) * 1000)
                    row.update(
                        {
                            "jev_complete_probability": signal.complete_probability,
                            "jev_complete_at_0_8": signal.complete_probability >= 0.8,
                            "latency_ms": latency_ms,
                            "model": signal.model,
                        }
                    )
                except JevError as error:
                    row["error"] = str(error)
                    row["latency_ms"] = round((time.perf_counter() - started) * 1000)
                rows.append(row)
    finally:
        await transport.close()

    partial = [row for row in rows if row["phase"] == "partial"]
    final = [row for row in rows if row["phase"] == "final"]
    partial_by_id = {str(row["case_id"]): row for row in partial}
    final_by_id = {str(row["case_id"]): row for row in final}
    budgets = (350, 500, 800)
    guarded_policy = TurnPolicy(
        PolicyConfig(
            allow_partial_semantic_commit=False,
            partial_semantic_hold_extension_ms=100,
        )
    )
    timing_counterfactual = {
        "fixed_640": summarize_pauses(
            tuple(
                _timing_case(
                    case,
                    partial_by_id[case.case_id],
                    final_by_id[case.case_id],
                    dispatch_at_ms,
                    None,
                )
                for case in CASES
            ),
            use_jev=False,
        ),
    }
    for revision_label, revised_at_ms in (
        ("stale_prefix", None),
        ("revision_320", 320),
        ("revision_500", 500),
    ):
        timing_cases = tuple(
            _timing_case(
                case,
                partial_by_id[case.case_id],
                final_by_id[case.case_id],
                dispatch_at_ms,
                revised_at_ms,
            )
            for case in CASES
        )
        for budget in budgets:
            timing_counterfactual[f"{revision_label}_jev_budget_{budget}_ms"] = summarize_pauses(
                timing_cases, use_jev=True, request_budget_ms=budget
            )
            timing_counterfactual[f"guarded_{revision_label}_jev_budget_{budget}_ms"] = (
                summarize_pauses(
                    timing_cases,
                    use_jev=True,
                    request_budget_ms=budget,
                    policy=guarded_policy,
                )
            )
    return {
        "evidence_level": "authored_text_live_provider_with_hypothetical_timing",
        "warning": (
            "Prefixes, completion labels, resume times, initial text availability, and "
            "revision times are authored assumptions; no real ASR, audio, or device timing "
            "was measured. Paired requests ran sequentially, not at simulated timestamps."
        ),
        "requests_attempted": len(rows),
        "request_timeout_ms": timeout_ms,
        "assumed_dispatch_at_pause_ms": dispatch_at_ms,
        "guarded_candidate": {
            "allow_partial_semantic_commit": False,
            "partial_semantic_hold_extension_ms": 100,
        },
        "model": judge.model,
        "partial": {
            "scores": _score(partial, "jev_complete_at_0_8"),
            "latency": _latency_summary(partial),
        },
        "final": {
            "scores": _score(final, "jev_complete_at_0_8"),
            "latency": _latency_summary(final),
        },
        "timing_counterfactual": timing_counterfactual,
        "cases": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--allow-remote-text", action="store_true", required=True)
    parser.add_argument("--timeout-ms", type=int, default=5000)
    parser.add_argument(
        "--paired",
        action="store_true",
        help="probe authored ASR prefixes and final text, then simulate pause-time deadlines",
    )
    args = parser.parse_args()
    if not args.allow_remote_text:
        parser.error("explicit remote-text opt-in is required")
    if args.timeout_ms <= 0:
        parser.error("timeout must be positive")
    result = asyncio.run(run_paired(args.timeout_ms) if args.paired else run(args.timeout_ms))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
