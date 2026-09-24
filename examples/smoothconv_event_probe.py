"""Compare acoustic endpoint candidates on publisher-provided per-speaker channels.

No annotation or transcript enters detection. The labels are joined only after
all causal arms have replayed the same Silero probability stream. Requires
the local, Git-ignored SmoothConv WAV/JSON pairs, Silero ONNX, sox, numpy, and
onnxruntime. This is not a deployed-agent quality measurement.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import wave
from pathlib import Path

from bounded_audio_replay import replay_bounded_audio_endpoints
from smartturn_audio_adapter import LocalSmartTurnScorer, replay_hybrid_endpoints
from smoothconv_annotation_probe import labeled_windows, read_segments
from smoothconv_audio_probe import read_pcm, silero_probabilities

from turnpilot.acoustic import AdaptiveGateConfig
from turnpilot.acoustic_eval import (
    AcousticEndpointTrace,
    AcousticPoint,
    paired_endpoint_alignment,
    replay_acoustic_endpoints,
    summarize_endpoint_alignment,
)
from turnpilot.local_audio_eot import LocalAudioEOTConfig


def add_paired_counts(
    totals: dict[str, dict[str, int]] | None,
    current: dict[str, dict[str, int]],
) -> dict[str, dict[str, int]]:
    if totals is None:
        return current
    for label, outcomes in current.items():
        for outcome, count in outcomes.items():
            totals[label][outcome] += count
    return totals


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("annotations", nargs="+", type=Path, help="JSON with same-stem WAV")
    parser.add_argument("--model", type=Path, required=True, help="local Silero v6.2.2 ONNX")
    parser.add_argument("--pause-ms", type=int, default=640)
    parser.add_argument("--completion-window-ms", type=int, default=1600)
    parser.add_argument("--min-observed-pause-ms", type=int, default=1200)
    parser.add_argument(
        "--smartturn-model", type=Path, help="optional pinned Smart Turn v3.2 CPU ONNX"
    )
    parser.add_argument(
        "--model-ready-delay-ms",
        type=int,
        help="fixed model-ready delay for deterministic counterfactuals",
    )
    parser.add_argument("--candidate-early-ms", type=int, help="opt-in research profile only")
    parser.add_argument("--candidate-low-hold-ms", type=int, help="opt-in research profile only")
    args = parser.parse_args()
    if (
        args.pause_ms <= 0
        or args.min_observed_pause_ms < args.pause_ms
        or args.completion_window_ms <= args.min_observed_pause_ms
    ):
        parser.error("require 0 < pause <= minimum observed pause < completion window")
    if args.model_ready_delay_ms is not None and args.smartturn_model is None:
        parser.error("--model-ready-delay-ms requires --smartturn-model")
    if (args.candidate_early_ms is None) != (args.candidate_low_hold_ms is None):
        parser.error("research candidate requires both pause deadlines")
    if args.candidate_early_ms is not None and args.smartturn_model is None:
        parser.error("research candidate requires --smartturn-model")
    candidate_config = None
    if args.candidate_early_ms is not None and args.candidate_low_hold_ms is not None:
        try:
            candidate_config = LocalAudioEOTConfig(
                early_pause_ms=args.candidate_early_ms,
                low_hold_pause_ms=args.candidate_low_hold_ms,
            )
        except ValueError as error:
            parser.error(str(error))
    configs = {
        "unchanged_gate": AdaptiveGateConfig(),
        "fixed_rearm_candidate": AdaptiveGateConfig(
            rearm_during_pause=True, probability_gain_per_noise_db=0.0
        ),
        "adaptive_rearm_candidate": AdaptiveGateConfig(rearm_during_pause=True),
    }
    scorer = LocalSmartTurnScorer(args.smartturn_model) if args.smartturn_model else None
    arm_names = list(configs) + (
        ["smartturn_hybrid", "smartturn_three_way", "bounded_audio_candidate"] if scorer else []
    )
    if candidate_config is not None:
        arm_names.append("research_candidate")
    totals: dict[str, dict[str, int]] = {
        name: {
            "complete_windows": 0,
            "complete_endpoint_within_window": 0,
            "incomplete_windows": 0,
            "incomplete_endpoint_before_resume": 0,
            "acoustic_unobserved_label_windows": 0,
            "all_endpoint_events": 0,
        }
        for name in arm_names
    }
    latencies: dict[str, list[int]] = {name: [] for name in arm_names}
    model_times_ms: dict[str, list[float]] = {
        "smartturn_hybrid": [],
        "smartturn_three_way": [],
        "bounded_audio_candidate": [],
    }
    if candidate_config is not None:
        model_times_ms["research_candidate"] = []
    bounded_model_timeouts = 0
    bounded_cancelled_requests = 0
    paired_unchanged_vs_bounded: dict[str, dict[str, int]] | None = None
    paired_unchanged_vs_research: dict[str, dict[str, int]] | None = None
    research_model_timeouts = 0
    research_cancelled_requests = 0
    excluded = {
        "incomplete_censored": 0,
        "incomplete_interrupted": 0,
        "complete_short_horizon": 0,
        "multi_speaker_channels": 0,
    }
    channels = 0
    duration_ms_total = 0
    with tempfile.TemporaryDirectory(prefix="turnpilot-corpus-") as temporary:
        scratch = Path(temporary)
        for annotation in args.annotations:
            source_wav = annotation.with_suffix(".wav")
            segments = read_segments(annotation)
            with wave.open(str(source_wav), "rb") as source:
                channel_count = source.getnchannels()
                duration_ms = round(source.getnframes() * 1000 / source.getframerate())
            if max((item.end_ms for item in segments), default=0) > duration_ms + 1:
                raise ValueError("annotation extends past matching WAV")
            for channel in sorted(
                {item.channel_index for item in segments if item.channel_index is not None}
            ):
                if channel >= channel_count:
                    raise ValueError("annotation channel exceeds WAV channel count")
                speakers = {
                    item.speaker
                    for item in segments
                    if item.channel_index == channel and item.speaker != "unknown"
                }
                if len(speakers) != 1:
                    excluded["multi_speaker_channels"] += 1
                    continue
                windows, dropped = labeled_windows(
                    segments,
                    channel,
                    duration_ms,
                    completion_window_ms=args.completion_window_ms,
                    min_observed_pause_ms=args.min_observed_pause_ms,
                )
                for name, count in dropped.items():
                    excluded[name] += count
                if not windows:
                    continue
                converted = scratch / f"channel-{channels}.wav"
                subprocess.run(
                    [
                        "sox",
                        "-v",
                        "0.5",
                        str(source_wav),
                        "-r",
                        "16000",
                        "-c",
                        "1",
                        str(converted),
                        "remix",
                        str(channel + 1),
                    ],
                    check=True,
                    capture_output=True,
                )
                samples = read_pcm(converted)
                whole_frames = samples[: (len(samples) // 512) * 512]
                frames = [
                    AcousticPoint(at_ms, probability, energy_dbfs)
                    for at_ms, probability, energy_dbfs in silero_probabilities(
                        whole_frames, args.model
                    )
                ]
                channels += 1
                duration_ms_total += duration_ms
                traces = {
                    name: replay_acoustic_endpoints(
                        frames, config=config, pause_deadline_ms=args.pause_ms
                    )
                    for name, config in configs.items()
                }
                if scorer is not None:
                    hybrid = replay_hybrid_endpoints(
                        frames,
                        whole_frames,
                        scorer,
                        model_ready_delay_ms=args.model_ready_delay_ms,
                    )
                    traces["smartturn_hybrid"] = AcousticEndpointTrace(
                        hybrid.endpoint_times_ms, hybrid.active_frame_times_ms
                    )
                    model_times_ms["smartturn_hybrid"].extend(
                        event[2] for event in hybrid.score_events
                    )
                    three_way = replay_hybrid_endpoints(
                        frames,
                        whole_frames,
                        scorer,
                        low_confidence_threshold=0.2,
                        uncertain_deadline_ms=640,
                        model_ready_delay_ms=args.model_ready_delay_ms,
                    )
                    traces["smartturn_three_way"] = AcousticEndpointTrace(
                        three_way.endpoint_times_ms, three_way.active_frame_times_ms
                    )
                    model_times_ms["smartturn_three_way"].extend(
                        event[2] for event in three_way.score_events
                    )
                    bounded = replay_bounded_audio_endpoints(
                        frames,
                        whole_frames,
                        scorer,
                        model_delay_override_ms=args.model_ready_delay_ms,
                    )
                    traces["bounded_audio_candidate"] = AcousticEndpointTrace(
                        bounded.endpoint_times_ms, bounded.active_frame_times_ms
                    )
                    model_times_ms["bounded_audio_candidate"].extend(
                        event[2] for event in bounded.score_events
                    )
                    bounded_model_timeouts += bounded.model_timeouts
                    bounded_cancelled_requests += bounded.cancelled_requests
                    paired_unchanged_vs_bounded = add_paired_counts(
                        paired_unchanged_vs_bounded,
                        paired_endpoint_alignment(
                            windows, traces["unchanged_gate"], traces["bounded_audio_candidate"]
                        ),
                    )
                    if candidate_config is not None:
                        research = replay_bounded_audio_endpoints(
                            frames,
                            whole_frames,
                            scorer,
                            model_delay_override_ms=args.model_ready_delay_ms,
                            config=candidate_config,
                        )
                        traces["research_candidate"] = AcousticEndpointTrace(
                            research.endpoint_times_ms, research.active_frame_times_ms
                        )
                        model_times_ms["research_candidate"].extend(
                            event[2] for event in research.score_events
                        )
                        research_model_timeouts += research.model_timeouts
                        research_cancelled_requests += research.cancelled_requests
                        paired_unchanged_vs_research = add_paired_counts(
                            paired_unchanged_vs_research,
                            paired_endpoint_alignment(
                                windows, traces["unchanged_gate"], traces["research_candidate"]
                            ),
                        )
                for name, trace in traces.items():
                    score = summarize_endpoint_alignment(windows, trace)
                    for key in (
                        "complete_windows",
                        "complete_endpoint_within_window",
                        "incomplete_windows",
                        "incomplete_endpoint_before_resume",
                        "acoustic_unobserved_label_windows",
                    ):
                        value = score[key]
                        assert isinstance(value, int)
                        totals[name][key] += value
                    totals[name]["all_endpoint_events"] += len(trace.endpoint_times_ms)
                    for window in windows:
                        if not window.is_complete:
                            continue
                        if window.start_ms is not None and not any(
                            window.start_ms <= at_ms <= window.end_ms
                            for at_ms in trace.active_frame_times_ms
                        ):
                            continue
                        matched = next(
                            (
                                at_ms
                                for at_ms in trace.endpoint_times_ms
                                if window.end_ms <= at_ms < window.next_speech_or_horizon_ms
                            ),
                            None,
                        )
                        if matched is not None:
                            latencies[name].append(matched - window.end_ms)
    latency_summary: dict[str, dict[str, int | None]] = {}
    for name, values in latencies.items():
        values.sort()
        latency_summary[name] = {}
        for percentile in (50, 95):
            index = (percentile * len(values) + 99) // 100 - 1
            latency_summary[name][f"p{percentile}_ms"] = values[index] if values else None
    print(
        json.dumps(
            {
                "recordings": len(args.annotations),
                "analyzed_channels": channels,
                "channel_audio_duration_ms": duration_ms_total,
                "pause_deadline_ms": args.pause_ms,
                "model_ready_delay_ms": args.model_ready_delay_ms,
                "completion_window_ms": args.completion_window_ms,
                "min_observed_pause_ms": args.min_observed_pause_ms,
                "excluded_label_windows": excluded,
                "arms": totals,
                "matched_complete_latency": latency_summary,
                "smartturn_model_runtime": {
                    name: {
                        "calls": len(values),
                        "over_ready_delay_calls": (
                            sum(value > args.model_ready_delay_ms for value in values)
                            if args.model_ready_delay_ms is not None
                            else None
                        ),
                        "p50_ms": sorted(values)[(len(values) - 1) // 2] if values else None,
                        "p95_ms": (
                            sorted(values)[(95 * len(values) + 99) // 100 - 1] if values else None
                        ),
                    }
                    for name, values in model_times_ms.items()
                },
                "bounded_model_timeouts": bounded_model_timeouts,
                "bounded_cancelled_requests": bounded_cancelled_requests,
                "paired_unchanged_vs_bounded": paired_unchanged_vs_bounded,
                "research_candidate_config": (
                    {
                        "early_pause_ms": candidate_config.early_pause_ms,
                        "low_hold_pause_ms": candidate_config.low_hold_pause_ms,
                    }
                    if candidate_config is not None
                    else None
                ),
                "research_model_timeouts": research_model_timeouts,
                "research_cancelled_requests": research_cancelled_requests,
                "paired_unchanged_vs_research": paired_unchanged_vs_research,
                "evidence_level": "small_publisher_corpus_acoustic_endpoint_alignment",
                "warning": (
                    "No ASR/Jev, speaker verification, real-device route, human adjudication, "
                    "or OVA actions. Labels only join after causal acoustic replay. "
                    "Counts are not deployed false-cutoff or response-quality rates."
                ),
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
