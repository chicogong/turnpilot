"""Select an acoustic timing candidate on one eot-bench slice, audit or transfer it.

This is an exploratory row-disjoint diagnostic, not a speaker/device-disjoint
validation. It reads prediction artifacts only and never prints transcripts or
audio. The public benchmark has already been inspected, so the audit slice is
not a truly untouched test set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
from eot_harness.metrics import compute_metrics_from_predictions

AUDIT_START_ID = 2496
LATENCY_BUDGET_S = 0.6
ACTION_DELAYS_S = [round(value / 10, 1) for value in range(2, 11)]
TIMEOUTS_S = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]


def is_audit_id(turn_id: str) -> bool:
    """Hold out the final contiguous ID range, not alternating adjacent turns."""
    prefix, separator, suffix = turn_id.rpartition("__")
    if not separator or prefix != "zh" or not suffix.isdecimal():
        raise ValueError("expected a Chinese eot-bench turn ID")
    return int(suffix) >= AUDIT_START_ID


def _load_candidates(items: list[str]) -> dict[str, pd.DataFrame]:
    candidates: dict[str, pd.DataFrame] = {}
    for item in items:
        name, separator, path = item.partition("=")
        if not separator or not name or not path or name in candidates:
            raise ValueError("each --candidate must be unique NAME=PATH")
        frame = pd.read_parquet(path)
        required = {"id", "span_index", "timestamp", "silence_dur", "p_eot", "label"}
        if required - set(frame.columns):
            raise ValueError("candidate lacks benchmark prediction columns")
        candidates[name] = frame
    key_sets = [
        set(
            zip(
                frame["id"],
                frame["span_index"],
                frame["timestamp"],
                frame["label"],
                strict=True,
            )
        )
        for frame in candidates.values()
    ]
    if any(keys != key_sets[0] for keys in key_sets[1:]):
        raise ValueError("candidate prediction grids or labels differ")
    return candidates


def _partition(frame: pd.DataFrame, *, audit: bool) -> pd.DataFrame:
    mask = frame["id"].map(is_audit_id)
    out = frame[mask == audit].copy()
    if out.empty:
        raise ValueError("requested split contains no prediction rows")
    return out


def _best_under_budget(sweep: pd.DataFrame, policy_type: str) -> dict[str, float] | None:
    feasible = sweep[
        (sweep["policy_type"] == policy_type) & (sweep["mean_latency"] <= LATENCY_BUDGET_S + 1e-9)
    ].sort_values(["cutoff_rate", "mean_latency", "action_delay", "timeout"], kind="stable")
    if feasible.empty:
        return None
    row = feasible.iloc[0]
    return {
        "cutoff_rate": float(row["cutoff_rate"]),
        "mean_latency_s": float(row["mean_latency"]),
        "action_delay_s": float(row["action_delay"]),
        "timeout_s": float(row["timeout"]),
    }


def select(candidates: dict[str, pd.DataFrame], selection_path: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    baseline: dict[str, float] | None = None
    counts: dict[str, int] | None = None
    for name, frame in candidates.items():
        development = _partition(frame, audit=False)
        sweep, summary = compute_metrics_from_predictions(
            development,
            thresholds=[0.5],
            action_delays=ACTION_DELAYS_S,
            timeouts=TIMEOUTS_S,
        )
        next_counts = {
            "turns": int(development["id"].nunique()),
            "hold_spans": int(summary["n_hold_spans"]),
            "eot_spans": int(summary["n_eot_spans"]),
        }
        if counts is not None and next_counts != counts:
            raise ValueError("candidate scoring denominators differ")
        counts = next_counts
        current_baseline = _best_under_budget(sweep, "vad")
        if baseline is not None and current_baseline != baseline:
            raise ValueError("fixed-silence baseline differs across candidates")
        baseline = current_baseline
        rows.append({"candidate": name, "best_at_600ms": _best_under_budget(sweep, "model")})
    eligible = [row for row in rows if row["best_at_600ms"] is not None]
    eligible.sort(
        key=lambda row: (
            row["best_at_600ms"]["cutoff_rate"],
            row["best_at_600ms"]["mean_latency_s"],
            row["candidate"],
        )
    )
    chosen = eligible[0] if eligible else None
    result = {
        "phase": "development_selection",
        "split": f"zh ID < {AUDIT_START_ID}",
        "counts": counts,
        "latency_budget_s": LATENCY_BUDGET_S,
        "model_threshold": 0.5,
        "baseline_at_600ms": baseline,
        "candidates": rows,
        "selected": chosen,
        "warning": (
            "Public, previously viewed data; row-disjoint only, not speaker/device-disjoint."
        ),
    }
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    return result


def audit(candidates: dict[str, pd.DataFrame], selection_path: Path) -> dict[str, Any]:
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    chosen = selection.get("selected")
    baseline = selection.get("baseline_at_600ms")
    if not isinstance(chosen, dict) or not isinstance(baseline, dict):
        raise ValueError("development selection has no feasible candidate or baseline")
    name = chosen["candidate"]
    if name not in candidates:
        raise ValueError("selected candidate predictions were not supplied")
    model = chosen["best_at_600ms"]
    held_out = _partition(candidates[name], audit=True)
    sweep, summary = compute_metrics_from_predictions(
        held_out,
        thresholds=[0.5],
        action_delays=[model["action_delay_s"], baseline["action_delay_s"]],
        timeouts=[model["timeout_s"], baseline["action_delay_s"]],
    )
    model_row = sweep[
        (sweep["policy_type"] == "model")
        & (sweep["action_delay"] == model["action_delay_s"])
        & (sweep["timeout"] == model["timeout_s"])
    ].iloc[0]
    baseline_row = sweep[
        (sweep["policy_type"] == "vad") & (sweep["action_delay"] == baseline["action_delay_s"])
    ].iloc[0]
    return {
        "phase": "fixed_config_audit",
        "split": f"zh ID >= {AUDIT_START_ID}",
        "candidate": name,
        "counts": {
            "turns": int(held_out["id"].nunique()),
            "hold_spans": int(summary["n_hold_spans"]),
            "eot_spans": int(summary["n_eot_spans"]),
        },
        "model": {
            "action_delay_s": model["action_delay_s"],
            "timeout_s": model["timeout_s"],
            "cutoff_rate": float(model_row["cutoff_rate"]),
            "mean_latency_s": float(model_row["mean_latency"]),
        },
        "fixed_silence": {
            "delay_s": baseline["action_delay_s"],
            "cutoff_rate": float(baseline_row["cutoff_rate"]),
            "mean_latency_s": float(baseline_row["mean_latency"]),
        },
        "warning": "Audit is row-disjoint only and public benchmark results were already viewed.",
    }


def transfer(candidates: dict[str, pd.DataFrame], selection_path: Path) -> dict[str, Any]:
    """Apply the Chinese-selected operating point unchanged to another corpus."""
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    chosen = selection.get("selected")
    baseline = selection.get("baseline_at_600ms")
    if not isinstance(chosen, dict) or not isinstance(baseline, dict):
        raise ValueError("development selection has no feasible candidate or baseline")
    model = chosen["best_at_600ms"]
    expected_counts: dict[str, int] | None = None
    baseline_result: dict[str, float] | None = None
    results: list[dict[str, Any]] = []
    for name, frame in candidates.items():
        sweep, summary = compute_metrics_from_predictions(
            frame,
            thresholds=[0.5],
            action_delays=sorted({model["action_delay_s"], baseline["action_delay_s"]}),
            timeouts=[model["timeout_s"]],
        )
        counts = {
            "turns": int(frame["id"].nunique()),
            "hold_spans": int(summary["n_hold_spans"]),
            "eot_spans": int(summary["n_eot_spans"]),
        }
        if expected_counts is not None and counts != expected_counts:
            raise ValueError("candidate scoring denominators differ")
        expected_counts = counts
        model_row = sweep[
            (sweep["policy_type"] == "model")
            & (sweep["threshold"] == 0.5)
            & (sweep["action_delay"] == model["action_delay_s"])
            & (sweep["timeout"] == model["timeout_s"])
        ].iloc[0]
        baseline_row = sweep[
            (sweep["policy_type"] == "vad") & (sweep["action_delay"] == baseline["action_delay_s"])
        ].iloc[0]
        next_baseline = {
            "delay_s": baseline["action_delay_s"],
            "cutoff_rate": float(baseline_row["cutoff_rate"]),
            "mean_latency_s": float(baseline_row["mean_latency"]),
        }
        if baseline_result is not None and next_baseline != baseline_result:
            raise ValueError("fixed-silence baseline differs across candidates")
        baseline_result = next_baseline
        results.append(
            {
                "candidate": name,
                "cutoff_rate": float(model_row["cutoff_rate"]),
                "mean_latency_s": float(model_row["mean_latency"]),
                "detect_rate": float(model_row["detect_rate"]),
                "timeout_rate": float(model_row["timeout_rate"]),
            }
        )
    return {
        "phase": "fixed_config_cross_corpus_transfer",
        "counts": expected_counts,
        "model_threshold": 0.5,
        "action_delay_s": model["action_delay_s"],
        "timeout_s": model["timeout_s"],
        "fixed_silence": baseline_result,
        "candidates": results,
        "warning": (
            "External corpus transfer, not speaker/device-disjoint or a sealed product test."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("select", "audit", "transfer"))
    parser.add_argument(
        "--candidate", action="append", required=True, help="NAME=predictions.parquet"
    )
    parser.add_argument("--selection-file", required=True, type=Path)
    args = parser.parse_args()
    candidates = _load_candidates(args.candidate)
    if args.phase == "select":
        result = select(candidates, args.selection_file)
    elif args.phase == "audit":
        result = audit(candidates, args.selection_file)
    else:
        result = transfer(candidates, args.selection_file)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
