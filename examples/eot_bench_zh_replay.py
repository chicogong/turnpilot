"""Replay local Chinese eot-bench audio through fixed and dynamic acoustic gates.

This standalone diagnostic requires separately obtained, licensed Arrow/Parquet
data and a Silero ONNX model. It neither reads transcript columns nor sends data
to a provider. Labels are used only after the causal audio replay. Output is
aggregate timing/count data and input/code hashes, never audio, text, or row IDs.

Example (after installing TurnPilot's ``file`` extra plus pyarrow and soundfile):

    python examples/eot_bench_zh_replay.py \
      --input /path/to/zh-validation.arrow \
      --silero-model /path/to/silero_vad_v6.2.2.onnx \
      --output reports/eot-bench-zh.json

These are exploratory span-alignment counts, not official eot-harness metrics,
speaker/device-disjoint validation, or a conversation-action benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from bisect import bisect_left, bisect_right
from dataclasses import dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import Any

from turnpilot.acoustic import AcousticFrame, AdaptiveGateConfig, AdaptiveSpeechGate
from turnpilot.acoustic_eval import AcousticEndpointTrace, AcousticPoint, replay_acoustic_endpoints
from turnpilot.models import AcousticSignal, TurnRef

PAUSE_DEADLINE_MS = 640
MIN_SPAN_MS = 100
MAX_HOLD_MS = 5000


@dataclass(frozen=True, slots=True)
class LabeledSpan:
    start_ms: int
    end_ms: int
    is_hold: bool
    preceding_speech_start_ms: int = 0


@dataclass(slots=True)
class ArmCounts:
    hold_cutoffs: int = 0
    hold_cutoffs_in_short_holds: int = 0
    eot_detected_within_recording: int = 0
    eot_detected_in_short_eot: int = 0
    eot_latencies_ms: list[int] = field(default_factory=list)
    hold_long_observed: int = 0
    hold_long_observed_cutoffs: int = 0
    eot_long_observed: int = 0
    eot_long_observed_detected: int = 0


@dataclass(slots=True)
class StartOpportunityCounts:
    """Causal start-check opportunities; independent of publisher labels."""

    start_checks_both_arms: int = 0
    start_checks_with_snr_both_arms: int = 0
    threshold_band_frames: int = 0
    fixed_qualifying_windows_with_band: int = 0
    fixed_only_start_transitions: int = 0
    noise_floor_dbfs: list[float] = field(default_factory=list)
    dynamic_start_threshold: list[float] = field(default_factory=list)

    def add(self, other: StartOpportunityCounts) -> None:
        self.start_checks_both_arms += other.start_checks_both_arms
        self.start_checks_with_snr_both_arms += other.start_checks_with_snr_both_arms
        self.threshold_band_frames += other.threshold_band_frames
        self.fixed_qualifying_windows_with_band += other.fixed_qualifying_windows_with_band
        self.fixed_only_start_transitions += other.fixed_only_start_transitions
        self.noise_floor_dbfs.extend(other.noise_floor_dbfs)
        self.dynamic_start_threshold.extend(other.dynamic_start_threshold)

    def summary(self) -> dict[str, Any]:
        return {
            "start_checks_both_arms": self.start_checks_both_arms,
            "start_checks_with_snr_both_arms": self.start_checks_with_snr_both_arms,
            "threshold_band_frames": self.threshold_band_frames,
            "fixed_qualifying_windows_with_band": self.fixed_qualifying_windows_with_band,
            "fixed_only_start_transitions": self.fixed_only_start_transitions,
            "noise_floor_dbfs_at_start_checks": _float_quantiles(self.noise_floor_dbfs),
            "dynamic_start_threshold_at_start_checks": _float_quantiles(
                self.dynamic_start_threshold
            ),
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def eligible_spans(
    raw_spans: list[dict[str, Any]], *, audio_duration_ms: int
) -> tuple[LabeledSpan, ...]:
    """Validate publisher spans, then retain 100 ms+ spans and <=5 s holds.

    The final *publisher* span is EOT even if earlier spans are filtered out.
    No label is passed into the acoustic gate.
    """
    if not isinstance(raw_spans, list) or not raw_spans:
        raise ValueError("each row must have at least one silence span")
    result: list[LabeledSpan] = []
    previous_end = 0
    for index, raw in enumerate(raw_spans):
        try:
            start_seconds, end_seconds = float(raw["start"]), float(raw["end"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("silence spans require numeric start and end") from exc
        if not math.isfinite(start_seconds) or not math.isfinite(end_seconds):
            raise ValueError("silence span timestamps must be finite")
        start_ms, end_ms = round(start_seconds * 1000), round(end_seconds * 1000)
        if start_ms < previous_end or end_ms <= start_ms or end_ms > audio_duration_ms + 1:
            raise ValueError("silence spans must be ordered and within the audio")
        preceding_speech_start_ms = previous_end
        previous_end = end_ms
        is_hold = index < len(raw_spans) - 1
        duration_ms = end_ms - start_ms
        if duration_ms >= MIN_SPAN_MS and (not is_hold or duration_ms <= MAX_HOLD_MS):
            result.append(LabeledSpan(start_ms, end_ms, is_hold, preceding_speech_start_ms))
    return tuple(result)


def endpoint_in_span(endpoints_ms: tuple[int, ...], span: LabeledSpan) -> int | None:
    """A resume at the hold boundary wins; EOF closes the final EOT window."""
    index = bisect_left(endpoints_ms, span.start_ms)
    if index == len(endpoints_ms):
        return None
    at_ms = endpoints_ms[index]
    if span.is_hold:
        return at_ms if at_ms < span.end_ms else None
    return at_ms if at_ms <= span.end_ms else None


def add_span_outcome(counts: ArmCounts, span: LabeledSpan, endpoint_ms: int | None) -> None:
    """Pure label-alignment aggregate; no transcripts or per-row data persist."""
    is_short = span.end_ms - span.start_ms < PAUSE_DEADLINE_MS
    if span.is_hold:
        counts.hold_cutoffs += endpoint_ms is not None
        if is_short:
            counts.hold_cutoffs_in_short_holds += endpoint_ms is not None
    elif endpoint_ms is not None:
        counts.eot_detected_within_recording += 1
        counts.eot_latencies_ms.append(endpoint_ms - span.start_ms)
        if is_short:
            counts.eot_detected_in_short_eot += 1


def preceding_speech_observed(span: LabeledSpan, trace: AcousticEndpointTrace) -> bool:
    """Only active frames strictly inside the preceding labeled speech interval count."""
    if span.start_ms <= span.preceding_speech_start_ms:
        return False
    index = bisect_right(trace.active_frame_times_ms, span.preceding_speech_start_ms)
    return (
        index < len(trace.active_frame_times_ms)
        and trace.active_frame_times_ms[index] < span.start_ms
    )


def frame_covers_label_deadline(span: LabeledSpan, frame_times_ms: tuple[int, ...]) -> bool:
    """A complete 32 ms frame reaches the label-relative timeout before resume/EOF."""
    index = bisect_left(frame_times_ms, span.start_ms + PAUSE_DEADLINE_MS)
    if index == len(frame_times_ms):
        return False
    at_ms = frame_times_ms[index]
    return at_ms < span.end_ms if span.is_hold else at_ms <= span.end_ms


def add_evaluable_outcome(
    counts: ArmCounts, span: LabeledSpan, trace: AcousticEndpointTrace
) -> None:
    """Report an arm's long, speech-observed spans without changing legacy counts."""
    if span.end_ms - span.start_ms < PAUSE_DEADLINE_MS or not preceding_speech_observed(
        span, trace
    ):
        return
    detected = endpoint_in_span(trace.endpoint_times_ms, span) is not None
    if span.is_hold:
        counts.hold_long_observed += 1
        counts.hold_long_observed_cutoffs += detected
    else:
        counts.eot_long_observed += 1
        counts.eot_long_observed_detected += detected


def _empty_paired_evaluable() -> dict[str, dict[str, int]]:
    return {
        label: {
            "jointly_evaluable": 0,
            "not_jointly_observed_preceding_speech": 0,
            "not_observed_at_label_deadline": 0,
            "both_endpoint": 0,
            "fixed_only_endpoint": 0,
            "dynamic_only_endpoint": 0,
            "neither_endpoint": 0,
        }
        for label in ("hold_at_least_640ms", "eot_at_least_640ms")
    }


def add_paired_evaluable(
    counts: dict[str, dict[str, int]],
    span: LabeledSpan,
    fixed: AcousticEndpointTrace,
    dynamic: AcousticEndpointTrace,
    frame_times_ms: tuple[int, ...],
) -> None:
    """Require both arms' preceding speech and a complete frame at the label deadline."""
    if span.end_ms - span.start_ms < PAUSE_DEADLINE_MS:
        return
    label = "hold_at_least_640ms" if span.is_hold else "eot_at_least_640ms"
    group = counts[label]
    if not (preceding_speech_observed(span, fixed) and preceding_speech_observed(span, dynamic)):
        group["not_jointly_observed_preceding_speech"] += 1
        return
    if not frame_covers_label_deadline(span, frame_times_ms):
        group["not_observed_at_label_deadline"] += 1
        return
    group["jointly_evaluable"] += 1
    fixed_endpoint = endpoint_in_span(fixed.endpoint_times_ms, span) is not None
    dynamic_endpoint = endpoint_in_span(dynamic.endpoint_times_ms, span) is not None
    if fixed_endpoint and dynamic_endpoint:
        group["both_endpoint"] += 1
    elif fixed_endpoint:
        group["fixed_only_endpoint"] += 1
    elif dynamic_endpoint:
        group["dynamic_only_endpoint"] += 1
    else:
        group["neither_endpoint"] += 1


def _percentile(values: list[int], numerator: int) -> int | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (numerator * len(ordered) + 99) // 100 - 1
    return ordered[index]


def _float_quantiles(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "p10": None, "p50": None, "p90": None, "min": None, "max": None}
    ordered = sorted(values)

    def point(numerator: int) -> float:
        index = (numerator * len(ordered) + 99) // 100 - 1
        return round(ordered[index], 4)

    return {
        "count": len(ordered),
        "p10": point(10),
        "p50": point(50),
        "p90": point(90),
        "min": round(ordered[0], 4),
        "max": round(ordered[-1], 4),
    }


def _start_check(signal: AcousticSignal | None, config: AdaptiveGateConfig) -> bool:
    if signal is None or (not signal.speech_active and signal.pause_duration_ms is None):
        return True
    return config.rearm_during_pause and (signal.pause_duration_ms or 0) > 0


def diagnose_start_opportunities(
    frames: list[AcousticPoint],
    fixed_config: AdaptiveGateConfig,
    dynamic_config: AdaptiveGateConfig,
) -> tuple[StartOpportunityCounts, AcousticEndpointTrace, AcousticEndpointTrace]:
    """Mirror both label-blind gates to inspect when the noise gain could matter.

    A band frame passes fixed but misses dynamic after both SNR checks. A
    qualifying window needs ``start_frames`` consecutive fixed passes and at
    least one band frame. Actual fixed-only transitions are counted separately.
    Endpoint/active traces are returned so the caller can check this probe
    against the unchanged replay implementation.
    """
    configs = (fixed_config, dynamic_config)
    gates = (AdaptiveSpeechGate(fixed_config), AdaptiveSpeechGate(dynamic_config))
    generations = [0, 0]
    previous: list[AcousticSignal | None] = [None, None]
    endpoints: list[list[int]] = [[], []]
    active: list[list[int]] = [[], []]
    counts = StartOpportunityCounts()
    fixed_pass_run: list[bool] = []
    for point in frames:
        checks = [_start_check(previous[i], configs[i]) for i in range(2)]
        both_check = all(checks)
        fixed_threshold = gates[0].start_threshold
        dynamic_threshold = gates[1].start_threshold
        if both_check:
            counts.start_checks_both_arms += 1
            counts.noise_floor_dbfs.append(gates[1].noise_floor_dbfs)
            counts.dynamic_start_threshold.append(dynamic_threshold)
            both_snr = all(
                point.energy_dbfs >= gate.noise_floor_dbfs + config.min_snr_db
                for gate, config in zip(gates, configs, strict=True)
            )
            if both_snr:
                counts.start_checks_with_snr_both_arms += 1
            fixed_pass = both_snr and point.speech_probability >= fixed_threshold
            in_band = fixed_pass and point.speech_probability < dynamic_threshold
            counts.threshold_band_frames += in_band
            if fixed_pass:
                fixed_pass_run.append(in_band)
                fixed_pass_run = fixed_pass_run[-fixed_config.start_frames :]
                if len(fixed_pass_run) == fixed_config.start_frames and any(fixed_pass_run):
                    counts.fixed_qualifying_windows_with_band += 1
            else:
                fixed_pass_run.clear()
        else:
            fixed_pass_run.clear()
        current: list[AcousticSignal] = []
        for i, gate in enumerate(gates):
            signal = gate.observe(
                AcousticFrame(
                    TurnRef("corpus-replay", "channel", generations[i]),
                    point.observed_at_ms,
                    point.speech_probability,
                    point.energy_dbfs,
                )
            )
            current.append(signal)
            if signal.speech_active:
                active[i].append(point.observed_at_ms)
            if (
                signal.pause_duration_ms is not None
                and signal.pause_duration_ms >= PAUSE_DEADLINE_MS
            ):
                endpoints[i].append(point.observed_at_ms)
                generations[i] += 1
                previous[i] = None
                gate.reset()  # Match the new-generation state before the next diagnostic check.
            else:
                previous[i] = signal
        if both_check and current[0].speech_active and not current[1].speech_active:
            counts.fixed_only_start_transitions += 1
    return (
        counts,
        AcousticEndpointTrace(tuple(endpoints[0]), tuple(active[0])),
        AcousticEndpointTrace(tuple(endpoints[1]), tuple(active[1])),
    )


def _configs() -> dict[str, AdaptiveGateConfig]:
    return {
        "fixed": AdaptiveGateConfig(probability_gain_per_noise_db=0.0),
        "dynamic": AdaptiveGateConfig(),
        "fixed_rearm": AdaptiveGateConfig(
            rearm_during_pause=True, probability_gain_per_noise_db=0.0
        ),
        "dynamic_rearm": AdaptiveGateConfig(rearm_during_pause=True),
    }


def _load_table(path: Path) -> Any:
    try:
        import pyarrow.ipc as ipc
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("install pyarrow separately to read Arrow/Parquet input") from exc

    required = ("audio", "language", "duration", "silence_spans")
    if path.suffix == ".arrow":
        with path.open("rb") as stream:
            table = ipc.open_stream(stream).read_all()
        missing = set(required) - set(table.column_names)
        if missing:
            raise ValueError(f"input is missing required columns: {sorted(missing)}")
        return table.select(required)
    if path.suffix == ".parquet":
        schema = parquet.read_schema(path)
        missing = set(required) - set(schema.names)
        if missing:
            raise ValueError(f"input is missing required columns: {sorted(missing)}")
        return parquet.read_table(path, columns=list(required))
    raise ValueError("input must be a local .arrow or .parquet file")


def _silero_probabilities(samples: Any, session: Any) -> list[AcousticPoint]:
    """Reset Silero context/state per turn while reusing one ONNX session.

    This matches the 512-sample, 64-context scorer in smoothconv_audio_probe,
    but avoids creating hundreds of native thread pools in one full replay.
    The input is truncated to whole 512-sample frames before this call.
    """
    import numpy as np

    if len(samples) % 512:
        raise ValueError("Silero scorer requires whole 512-sample frames")
    state = np.zeros((2, 1, 128), dtype=np.float32)
    context = np.zeros((1, 64), dtype=np.float32)
    frames: list[AcousticPoint] = []
    for offset in range(0, len(samples), 512):
        chunk = samples[offset : offset + 512]
        framed = np.concatenate((context, chunk[np.newaxis, :]), axis=1)
        output, state = session.run(
            None,
            {"input": framed, "state": state, "sr": np.array(16000, dtype=np.int64)},
        )
        context = framed[:, -64:]
        probability = float(output[0, 0])
        rms = float(np.sqrt(np.mean(chunk * chunk)))
        energy_dbfs = 20 * math.log10(max(rms, 1e-6))
        frames.append(AcousticPoint((offset + 512) // 16, probability, energy_dbfs))
    return frames


def replay(input_path: Path, model_path: Path, *, limit: int | None = None) -> dict[str, Any]:
    """Use one label-blind Silero frame stream for every arm of each turn."""
    try:
        import numpy as np
        import onnxruntime as ort
        import soundfile as sf
    except ImportError as exc:
        raise RuntimeError(
            "install TurnPilot's file extra plus soundfile for local Silero replay"
        ) from exc

    options = ort.SessionOptions()
    options.inter_op_num_threads = 1
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(
        str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
    )

    table = _load_table(input_path)
    if table.num_rows == 0:
        raise ValueError("input contains no turns")
    if limit is not None and not 1 <= limit <= table.num_rows:
        raise ValueError("--max-turns must be between 1 and the input row count")
    selected_rows = table.num_rows if limit is None else limit
    configs = _configs()
    counts = {name: ArmCounts() for name in configs}
    paired = {
        name: {
            "hold_fixed_only": 0,
            "hold_dynamic_only": 0,
            "eot_fixed_only": 0,
            "eot_dynamic_only": 0,
            "turns_with_endpoint_trace_difference": 0,
            "active_frame_symmetric_difference": 0,
        }
        for name in ("ordinary", "rearm")
    }
    pair_names = {
        "ordinary": ("fixed", "dynamic"),
        "rearm": ("fixed_rearm", "dynamic_rearm"),
    }
    paired_evaluable = {name: _empty_paired_evaluable() for name in pair_names}
    start_opportunities = {name: StartOpportunityCounts() for name in pair_names}
    total_duration_s = 0.0
    hold_total = 0
    hold_long_enough = 0
    hold_long_with_observed_frame_before_resume = 0
    eot_total = 0
    eot_long_enough = 0
    eot_short_eof_censored = 0
    eot_long_with_observed_frame_at_deadline = 0
    for index in range(selected_rows):
        if table["language"][index].as_py() != "zh":
            raise ValueError("every row must be Chinese (language=zh)")
        audio_cell = table["audio"][index].as_py()
        if not isinstance(audio_cell, dict) or not audio_cell.get("bytes"):
            raise ValueError("audio must contain local encoded bytes; paths are not followed")
        samples, sample_rate = sf.read(io.BytesIO(audio_cell["bytes"]), dtype="float32")
        if sample_rate != 16000 or samples.ndim != 1 or len(samples) < 512:
            raise ValueError("audio must be at least 32 ms of 16 kHz mono")
        try:
            declared_duration_s = float(table["duration"][index].as_py())
        except (TypeError, ValueError) as exc:
            raise ValueError("declared duration must be numeric") from exc
        if (
            not math.isfinite(declared_duration_s)
            or abs(declared_duration_s - len(samples) / 16000) > 0.04
        ):
            raise ValueError("declared duration does not match decoded audio")
        spans = eligible_spans(
            table["silence_spans"][index].as_py(),
            audio_duration_ms=round(len(samples) / 16),
        )
        whole_frames = samples[: (len(samples) // 512) * 512]
        frames = _silero_probabilities(whole_frames, session)
        traces = {
            name: replay_acoustic_endpoints(
                frames, config=config, pause_deadline_ms=PAUSE_DEADLINE_MS
            )
            for name, config in configs.items()
        }
        for pair, (fixed, dynamic) in pair_names.items():
            opportunity, diagnostic_fixed, diagnostic_dynamic = diagnose_start_opportunities(
                frames, configs[fixed], configs[dynamic]
            )
            if diagnostic_fixed != traces[fixed] or diagnostic_dynamic != traces[dynamic]:
                raise RuntimeError("start-opportunity probe diverged from endpoint replay")
            start_opportunities[pair].add(opportunity)
            paired[pair]["turns_with_endpoint_trace_difference"] += (
                traces[fixed].endpoint_times_ms != traces[dynamic].endpoint_times_ms
            )
            paired[pair]["active_frame_symmetric_difference"] += len(
                set(traces[fixed].active_frame_times_ms)
                ^ set(traces[dynamic].active_frame_times_ms)
            )
        total_duration_s += declared_duration_s
        frame_times_ms = tuple(point.observed_at_ms for point in frames)
        for span in spans:
            if span.is_hold:
                hold_total += 1
                is_long = span.end_ms - span.start_ms >= PAUSE_DEADLINE_MS
                hold_long_enough += is_long
                if is_long:
                    hold_long_with_observed_frame_before_resume += frame_covers_label_deadline(
                        span, frame_times_ms
                    )
            else:
                eot_total += 1
                is_long = span.end_ms - span.start_ms >= PAUSE_DEADLINE_MS
                eot_long_enough += is_long
                eot_short_eof_censored += not is_long
                if is_long:
                    eot_long_with_observed_frame_at_deadline += frame_covers_label_deadline(
                        span, frame_times_ms
                    )
            outcomes: dict[str, bool] = {}
            for name, trace in traces.items():
                endpoint = endpoint_in_span(trace.endpoint_times_ms, span)
                outcomes[name] = endpoint is not None
                add_span_outcome(counts[name], span, endpoint)
                add_evaluable_outcome(counts[name], span, trace)
            for pair, (fixed, dynamic) in pair_names.items():
                add_paired_evaluable(
                    paired_evaluable[pair], span, traces[fixed], traces[dynamic], frame_times_ms
                )
                if outcomes[fixed] != outcomes[dynamic]:
                    label = "hold" if span.is_hold else "eot"
                    winner = "fixed" if outcomes[fixed] else "dynamic"
                    paired[pair][f"{label}_{winner}_only"] += 1

    results = {
        name: {
            "hold_cutoffs": value.hold_cutoffs,
            "hold_cutoffs_in_short_holds": value.hold_cutoffs_in_short_holds,
            "eot_detected_within_recording": value.eot_detected_within_recording,
            "eot_detected_in_short_eot": value.eot_detected_in_short_eot,
            "detected_eot_latency_p50_ms": _percentile(value.eot_latencies_ms, 50),
            "detected_eot_latency_p95_ms": _percentile(value.eot_latencies_ms, 95),
            "hold_at_least_640ms_with_observed_preceding_speech": value.hold_long_observed,
            "hold_cutoffs_in_observed_long_holds": value.hold_long_observed_cutoffs,
            "eot_at_least_640ms_with_observed_preceding_speech": value.eot_long_observed,
            "eot_detected_in_observed_long_eot": value.eot_long_observed_detected,
        }
        for name, value in counts.items()
    }
    return {
        "evidence_level": "public_chinese_audio_acoustic_diagnostic_only",
        "input_sha256": _sha256(input_path),
        "silero_model_sha256": _sha256(model_path),
        "replay_script_sha256": _sha256(Path(__file__)),
        "acoustic_core_sha256": _sha256(
            Path(__file__).resolve().parents[1] / "src/turnpilot/acoustic.py"
        ),
        "acoustic_eval_sha256": _sha256(
            Path(__file__).resolve().parents[1] / "src/turnpilot/acoustic_eval.py"
        ),
        "turns": selected_rows,
        "audio_duration_s": round(total_duration_s, 3),
        "hold_spans_100ms_to_5s": hold_total,
        "hold_spans_at_least_640ms": hold_long_enough,
        "hold_long_spans_with_observed_frame_before_resume": (
            hold_long_with_observed_frame_before_resume
        ),
        "eot_spans_100ms_plus": eot_total,
        "eot_spans_at_least_640ms": eot_long_enough,
        "eot_spans_shorter_than_640ms_eof_censored": eot_short_eof_censored,
        "eot_long_spans_with_observed_frame_at_deadline": (
            eot_long_with_observed_frame_at_deadline
        ),
        "pause_deadline_ms": PAUSE_DEADLINE_MS,
        "fixed_vs_dynamic_only_difference": "probability_gain_per_noise_db: 0.0 vs 0.005",
        "runtime_versions": {
            "numpy": np.__version__,
            "onnxruntime": ort.__version__,
            "pyarrow": version("pyarrow"),
            "soundfile": sf.__version__,
        },
        "results": results,
        "paired_disagreement": paired,
        "paired_evaluable": paired_evaluable,
        "start_opportunities": {
            name: value.summary() for name, value in start_opportunities.items()
        },
        "limitations": (
            "Span-window counts, not official eot-harness metrics or endpoint-model accuracy. "
            "Long-span denominators are nominal 640 ms windows; jointly evaluable paired "
            "counts require both arms to activate in the preceding labeled speech interval "
            "and a complete frame at the label-relative deadline. "
            "Short final spans are nominally EOF-censored for a 640 ms timer starting "
            "at the label boundary, not model misses; an endpoint can still occur if "
            "the gate's own pause clock began earlier. Frame-at-deadline count shows "
            "whether complete 32 ms frames cover the nominal horizon. "
            "Start-opportunity diagnostics are causal gate-state counts, not outcome labels. "
            "No speaker/device IDs, "
            "streaming ASR, Jev, playback, or action labels are evaluated."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="local zh Arrow/Parquet dataset")
    parser.add_argument("--silero-model", type=Path, required=True, help="local Silero ONNX model")
    parser.add_argument("--output", type=Path, required=True, help="new aggregate JSON file")
    parser.add_argument("--max-turns", type=int, help="first N rows, for a quick diagnostic")
    args = parser.parse_args()
    if not args.input.is_file() or args.input.suffix not in {".arrow", ".parquet"}:
        parser.error("--input must be an existing local .arrow or .parquet file")
    if not args.silero_model.is_file() or args.silero_model.suffix != ".onnx":
        parser.error("--silero-model must be an existing local .onnx file")
    if args.output.exists():
        parser.error("--output already exists; choose a new path")
    if not args.output.parent.is_dir():
        parser.error("--output parent directory must already exist")
    try:
        result = replay(args.input, args.silero_model, limit=args.max_turns)
    except (RuntimeError, ValueError) as exc:
        parser.error(str(exc))
    encoded = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(encoded)
    print(encoded, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
