"""Causal, paced WAV diagnostic with shared VAD/streaming-ASR observations.

One file is one analysis window, not a host session. No microphone, synthetic
tail silence, EOF finalization, playback, or model/data auto-downloads.
"""

from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import math
import os
import platform
import struct
import sys
import time
import wave
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import NoReturn, Protocol

from turnpilot.acoustic import AcousticFrame, AdaptiveGateConfig, AdaptiveSpeechGate
from turnpilot.action_gate import ProvisionalActionGate
from turnpilot.action_trace import TraceEvent, TraceRecorder
from turnpilot.file_adapters import (
    AsrUpdate,
    SileroOnnxVad,
    VoskStreamingAsr,
    model_fingerprint,
)
from turnpilot.jev import HttpxJevTransport, JevError, JevJudge
from turnpilot.models import (
    AcousticSignal,
    Decision,
    DirectiveKind,
    HostState,
    SemanticSignal,
    TranscriptSignal,
    TurnRef,
)
from turnpilot.partial_stability import PartialTranscriptStability
from turnpilot.policy import PolicyConfig, TurnPolicy

SAMPLE_RATE = 16000
FRAME_SAMPLES = 512
FRAME_MS = 32
MAX_DURATION_MS = 60000
_TERMINAL = {
    DirectiveKind.COMMIT_USER_TURN,
    DirectiveKind.IGNORE_USER_TURN,
    DirectiveKind.CLARIFY_AUDIO,
    DirectiveKind.CLARIFY_MEANING,
}


class Vad(Protocol):
    def probability(self, pcm: bytes) -> float: ...


class StreamingAsr(Protocol):
    def feed(self, pcm: bytes) -> AsrUpdate | None: ...


class Clock(Protocol):
    def now_ms(self) -> int: ...

    async def wait_until(self, at_ms: int) -> None: ...


class MonotonicClock:
    def __init__(self) -> None:
        self._origin_ns = time.monotonic_ns()

    def now_ms(self) -> int:
        return (time.monotonic_ns() - self._origin_ns) // 1_000_000

    async def wait_until(self, at_ms: int) -> None:
        await asyncio.sleep(max(0, at_ms - self.now_ms()) / 1000)


class _SafeParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:
        raise ValueError("invalid file diagnostic arguments")


@dataclass(frozen=True, slots=True)
class _WavWindow:
    source_samples: int
    start_sample: int
    samples: int


def _inspect_window(path: Path, start_ms: int, duration_ms: int | None) -> _WavWindow:
    if (
        isinstance(start_ms, bool)
        or not isinstance(start_ms, int)
        or start_ms < 0
        or (
            duration_ms is not None
            and (
                isinstance(duration_ms, bool)
                or not isinstance(duration_ms, int)
                or not FRAME_MS <= duration_ms <= MAX_DURATION_MS
            )
        )
    ):
        raise ValueError("invalid file window bounds")
    with wave.open(str(path), "rb") as source:
        if (
            source.getnchannels() != 1
            or source.getsampwidth() != 2
            or source.getframerate() != SAMPLE_RATE
            or source.getcomptype() != "NONE"
        ):
            raise ValueError("expected 16 kHz mono PCM16 WAV")
        total = source.getnframes()
        start = start_ms * 16
        samples = total - start if duration_ms is None else duration_ms * 16
        if not FRAME_SAMPLES <= samples <= MAX_DURATION_MS * 16 or start + samples > total:
            raise ValueError("expected an explicit in-source window of 32 ms to 60 s")
        return _WavWindow(total, start, samples)


def inspect_wav(path: Path, *, start_ms: int = 0, duration_ms: int | None = None) -> int:
    """Reject invalid windows before model loading; never implicitly crop."""
    return _inspect_window(path, start_ms, duration_ms).samples


def _p95(values: list[int]) -> int:
    return sorted(values)[math.ceil(len(values) * 0.95) - 1] if values else 0


def _compute_summary(durations_ns: list[int]) -> dict[str, object]:
    return {
        "frames": len(durations_ns),
        "total_ms": round(sum(durations_ns) / 1_000_000, 3),
        "mean_ms": round(sum(durations_ns) / len(durations_ns) / 1_000_000, 3),
        "p95_ms": round(_p95(durations_ns) / 1_000_000, 3),
        "max_ms": round(max(durations_ns) / 1_000_000, 3),
        "over_frame_budget": sum(value >= FRAME_MS * 1_000_000 for value in durations_ns),
    }


@dataclass(slots=True)
class _Arm:
    recommendation: dict[str, object] | None = None
    resumed_after_recommendation: bool = False
    _media_at_ms: int | None = None

    def observe(self, signal: AcousticSignal, media_ms: int) -> None:
        if (
            self.recommendation is not None
            and signal.speech_active
            and self._media_at_ms is not None
            and media_ms > self._media_at_ms
        ):
            self.resumed_after_recommendation = True

    def record(self, decision: Decision, media_ms: int, backlog: bool) -> None:
        if self.recommendation is None and decision.kind in _TERMINAL:
            self._media_at_ms = media_ms
            self.recommendation = {
                "kind": decision.kind.value,
                "reason": decision.reason,
                "media_at_ms": media_ms,
                "available_at_ms": decision.decided_at_ms,
                "used_semantic": decision.used_semantic,
                "unprocessed_backlog": backlog,
            }

    def summary(self) -> dict[str, object]:
        return {
            "recommendation": self.recommendation,
            "no_recommendation_by_eof": self.recommendation is None,
            "vad_resumed_later": self.resumed_after_recommendation,
        }


class _SemanticProbe:
    """Single in-flight opt-in judgment; resume/revision invalidates its result."""

    def __init__(self, judge: JevJudge | None, clock: Clock) -> None:
        self.judge = judge
        self.clock = clock
        self.task: asyncio.Task[SemanticSignal] | None = None
        self.signal: SemanticSignal | None = None
        self.signal_requested_at_ms: int | None = None
        self.attempts = 0
        self.successes = 0
        self.errors = 0
        self.canceled = 0
        self.discarded = 0
        self._last_revision = -1
        self._last_call_at = -100
        self._task_requested_at_ms: int | None = None
        self._success_latency_ms: list[int] = []

    async def observe(
        self,
        acoustic: AcousticSignal,
        transcript: TranscriptSignal | None,
        *,
        backlog: bool,
        partial_ready: bool | None = None,
    ) -> None:
        now_ms = self.clock.now_ms()
        if acoustic.speech_active:
            self.signal = None
            self.signal_requested_at_ms = None
            await self.cancel()
        if self.task is not None and self.task.done():
            task, self.task = self.task, None
            try:
                signal = task.result()
            except Exception:
                self.errors += 1  # Do not expose provider exceptions or request text.
            else:
                self.successes += 1
                if self._task_requested_at_ms is not None:
                    self._success_latency_ms.append(
                        signal.received_at_ms - self._task_requested_at_ms
                    )
                if (
                    transcript is not None
                    and signal.ref == transcript.ref
                    and signal.transcript_revision == transcript.revision
                    and transcript.available_at_ms <= signal.received_at_ms <= now_ms
                    and now_ms - signal.received_at_ms <= 500
                    and not acoustic.speech_active
                ):
                    self.signal = signal
                    self.signal_requested_at_ms = self._task_requested_at_ms
                else:
                    self.discarded += 1
            self._task_requested_at_ms = None
        if self.signal is not None and (
            transcript is None
            or self.signal.transcript_revision != transcript.revision
            or now_ms - self.signal.received_at_ms > 500
        ):
            self.signal = None
            self.signal_requested_at_ms = None
        if (
            self.judge is not None
            and self.task is None
            and not backlog
            and not acoustic.speech_active
            and acoustic.pause_duration_ms is not None
            and acoustic.pause_duration_ms >= 224
            and transcript is not None
            and (partial_ready is None or partial_ready or transcript.is_final)
            and transcript.text.strip()
            and len(transcript.text) <= self.judge.max_transcript_chars
            and transcript.revision > self._last_revision
            and self.attempts < self.judge.max_calls_per_turn
            and now_ms - self._last_call_at >= self.judge.min_call_interval_ms
        ):
            self.attempts += 1
            self._last_revision = transcript.revision
            self._last_call_at = now_ms
            self._task_requested_at_ms = now_ms
            self.task = asyncio.create_task(
                self.judge.judge(transcript, now_ms=now_ms, allow_remote_text=True)
            )

    async def cancel(self) -> None:
        if self.task is not None:
            self.task.cancel()
            await asyncio.gather(self.task, return_exceptions=True)
            self.task = None
            self._task_requested_at_ms = None
            self.canceled += 1

    def summary(self) -> dict[str, object]:
        return {
            "enabled": self.judge is not None,
            "attempts": self.attempts,
            "successes": self.successes,
            "errors": self.errors,
            "canceled": self.canceled,
            "discarded": self.discarded,
            "model": self.judge.model if self.judge else None,
            "timeout_ms": self.judge.timeout_ms if self.judge else None,
            "max_calls_per_window": self.judge.max_calls_per_turn if self.judge else 0,
            "successful_request_latency_ms": {
                "samples": len(self._success_latency_ms),
                "p95": _p95(self._success_latency_ms) if self._success_latency_ms else None,
                "max": max(self._success_latency_ms) if self._success_latency_ms else None,
                "basis": "request_scheduled_to_result_arrival_not_server_inference",
            },
        }


@dataclass(frozen=True, slots=True)
class FileRunResult:
    report: dict[str, object]
    trace: TraceRecorder = field(repr=False)


async def run_file(
    path: Path,
    vad: Vad,
    asr: StreamingAsr,
    *,
    clock: Clock | None = None,
    judge: JevJudge | None = None,
    allow_remote_text: bool = False,
    experimental_stable_partial: bool = False,
    start_ms: int = 0,
    duration_ms: int | None = None,
) -> FileRunResult:
    """Run fresh adapters on one window. Injected clocks are fixture evidence.

    VAD gate durations remain media durations; timestamps are availability on
    the monotonic wall clock. Backlogged frames cannot authorize gated actions.
    """
    if judge is not None and not allow_remote_text:
        raise ValueError("file transcript upload requires explicit opt-in")
    window = _inspect_window(path, start_ms, duration_ms)
    samples = window.samples
    injected_clock = clock is not None and not isinstance(clock, MonotonicClock)
    active_clock = clock or MonotonicClock()
    if active_clock.now_ms() > FRAME_MS:
        raise ValueError("file clock must start after model initialization")
    ref = TurnRef("file-diagnostic", "window", 0)
    recorder = TraceRecorder()
    recorder.append(TraceEvent("turn_start", 0, ref))
    configs = {
        "fixed": AdaptiveGateConfig(probability_gain_per_noise_db=0.0),
        "adaptive": AdaptiveGateConfig(),
    }
    gates = {name: AdaptiveSpeechGate(config) for name, config in configs.items()}
    action_gate = ProvisionalActionGate()
    jev_gate = ProvisionalActionGate()
    stable_gate = ProvisionalActionGate(stable_partial_ms=224)
    stability = PartialTranscriptStability() if experimental_stable_partial else None
    policy = TurnPolicy()
    arms = {name: _Arm() for name in ("direct_asr_final", "fixed_640", "adaptive_640", "gate")}
    if judge is not None:
        arms["gate_jev"] = _Arm()
        if experimental_stable_partial:
            arms["gate_stable_partial_jev"] = _Arm()
    probe = _SemanticProbe(judge, active_clock)
    transcript: TranscriptSignal | None = None
    asr_revisions = asr_finals = 0
    timeline: list[dict[str, object]] = []
    lag: list[int] = []
    processing: list[int] = []
    vad_compute: list[int] = []
    asr_compute: list[int] = []
    stable_frames = 0
    first_stable_media_ms: int | None = None
    thresholds: list[float] = []
    backlog_frames = 0
    previous_states: dict[str, bool] = {}
    candidate_armed = False
    last_semantic: SemanticSignal | None = None
    media_ms = 0

    try:
        with wave.open(str(path), "rb") as source:
            source.setpos(window.start_sample)
            for frame_index in range(samples // FRAME_SAMPLES):
                pcm = source.readframes(FRAME_SAMPLES)
                if len(pcm) != FRAME_SAMPLES * 2:
                    raise ValueError("truncated PCM data")
                media_ms = (frame_index + 1) * FRAME_MS
                await active_clock.wait_until(media_ms)
                # Give optional async inference a chance even with an injected clock.
                await asyncio.sleep(0)
                started_ms = active_clock.now_ms()
                compute_start_ns = time.perf_counter_ns()
                probability = vad.probability(pcm)
                vad_compute.append(time.perf_counter_ns() - compute_start_ns)
                acoustic_at_ms = active_clock.now_ms()
                rms = math.sqrt(sum(v * v for (v,) in struct.iter_unpack("<h", pcm)) / 512)
                energy = 20 * math.log10(max(rms / 32768, 1e-6))
                frame = AcousticFrame(ref, media_ms, probability, energy)
                signals = {
                    name: replace(gate.observe(frame), observed_at_ms=acoustic_at_ms)
                    for name, gate in gates.items()
                }
                acoustic = signals["adaptive"]
                thresholds.append(gates["adaptive"].start_threshold)
                recorder.append(
                    TraceEvent("acoustic", acoustic_at_ms, ref, _acoustic_data(acoustic))
                )
                for name, signal in signals.items():
                    if previous_states.get(name) != signal.speech_active:
                        timeline.append(
                            {
                                "kind": "vad_state",
                                "arm": name,
                                "media_at_ms": media_ms,
                                "available_at_ms": acoustic_at_ms,
                                "speech_active": signal.speech_active,
                            }
                        )
                    previous_states[name] = signal.speech_active
                compute_start_ns = time.perf_counter_ns()
                update = asr.feed(pcm)
                asr_compute.append(time.perf_counter_ns() - compute_start_ns)
                now_ms = active_clock.now_ms()
                if now_ms < acoustic_at_ms or acoustic_at_ms < started_ms or started_ms < media_ms:
                    raise ValueError("file clock moved backward or input arrived early")
                backlog = now_ms - media_ms >= FRAME_MS
                if update is not None:
                    asr_revisions += 1
                    asr_finals += bool(update.is_final and update.text.strip())
                    transcript = TranscriptSignal(
                        ref, now_ms, asr_revisions, update.text, update.is_final
                    )
                    data: dict[str, object] = {
                        "revision": asr_revisions,
                        "has_text": bool(update.text.strip()),
                        "is_final": update.is_final,
                    }
                    recorder.append(TraceEvent("asr", now_ms, ref, data))
                    timeline.append(
                        {"kind": "asr", "media_at_ms": media_ms, "available_at_ms": now_ms, **data}
                    )
                    if update.is_final and update.text.strip():
                        arms["direct_asr_final"].record(
                            Decision(
                                ref,
                                DirectiveKind.COMMIT_USER_TURN,
                                "first_final_asr",
                                now_ms,
                                now_ms + policy.config.decision_ttl_ms,
                            ),
                            media_ms,
                            backlog,
                        )

                partial_ready = None
                if stability is not None:
                    partial_ready = stability.observe(
                        HostState(ref, now_ms, session_active=True),
                        acoustic,
                        transcript,
                        input_backlogged=backlog,
                    )
                    stable_frames += partial_ready
                    if partial_ready and first_stable_media_ms is None:
                        first_stable_media_ms = media_ms
                await probe.observe(
                    acoustic, transcript, backlog=backlog, partial_ready=partial_ready
                )
                now_ms = active_clock.now_ms()
                backlog = now_ms - media_ms >= FRAME_MS
                backlog_frames += backlog
                lag.append(now_ms - media_ms)
                processing.append(now_ms - started_ms)
                if probe.signal is not None and probe.signal != last_semantic:
                    last_semantic = probe.signal
                    # The trace uses poll time; the report also retains actual result arrival.
                    data = _semantic_data(probe.signal)
                    recorder.append(TraceEvent("semantic", now_ms, ref, data))
                    timeline.append(
                        {
                            "kind": "semantic",
                            "media_at_ms": media_ms,
                            "available_at_ms": probe.signal.received_at_ms,
                            "polled_at_ms": now_ms,
                            "requested_at_ms": probe.signal_requested_at_ms,
                            **data,
                        }
                    )
                host = HostState(ref, now_ms, session_active=True)
                if acoustic.speech_active:
                    candidate_armed = False
                elif (
                    not candidate_armed
                    and acoustic.pause_duration_ms is not None
                    and acoustic.pause_duration_ms >= 224
                    and arms["gate"].recommendation is None
                ):
                    if action_gate.arm(host, acoustic):
                        candidate_armed = True
                        recorder.append(TraceEvent("candidate", now_ms, ref))
                if acoustic.pause_duration_ms is not None and acoustic.pause_duration_ms >= 224:
                    jev_gate.arm(host, acoustic)
                    if experimental_stable_partial:
                        stable_gate.arm(host, acoustic)
                # Resume always cancels candidates, including while catching up.
                if not backlog or acoustic.speech_active:
                    arms["gate"].record(
                        action_gate.decide(host, acoustic, transcript), media_ms, backlog
                    )
                    if judge is not None:
                        arms["gate_jev"].record(
                            jev_gate.decide(host, acoustic, transcript, probe.signal),
                            media_ms,
                            backlog,
                        )
                if judge is not None and experimental_stable_partial:
                    # Always observe backlog/resume to reset the partial interval;
                    # the experiment itself suppresses recommendations while behind.
                    arms["gate_stable_partial_jev"].record(
                        stable_gate.decide(
                            host,
                            acoustic,
                            transcript,
                            probe.signal,
                            input_backlogged=backlog,
                            semantic_requested_at_ms=probe.signal_requested_at_ms,
                        ),
                        media_ms,
                        backlog,
                    )
                if not backlog:
                    for name in ("fixed", "adaptive"):
                        arms[name + "_640"].record(
                            policy.decide(host, signals[name], transcript), media_ms, backlog
                        )
                for name, arm in arms.items():
                    arm.observe(signals["fixed"] if name == "fixed_640" else acoustic, media_ms)
                recorder.append(TraceEvent("tick", now_ms, ref))
    finally:
        await probe.cancel()
    recorder.append(TraceEvent("stop", active_clock.now_ms(), ref))
    report: dict[str, object] = {
        "schema": 1,
        "evidence_level": "injected_clock_fixture"
        if injected_clock
        else "paced_file_pipeline_only",
        "duration_ms": samples / 16,
        "window": {
            "start_ms": window.start_sample / 16,
            "source_duration_ms": window.source_samples / 16,
            "ends_at_source_eof": window.start_sample + samples == window.source_samples,
        },
        "processed_media_ms": media_ms,
        "unprocessed_tail_samples": samples % FRAME_SAMPLES,
        "frame_ms": FRAME_MS,
        "eof_forced_final": False,
        "synthetic_tail_silence_ms": 0,
        "asr": {"revisions": asr_revisions, "nonempty_decoder_finals": asr_finals},
        "pipeline": {
            "frame_processing_p95_ms": _p95(processing),
            "availability_lag_p95_ms": _p95(lag),
            "availability_lag_max_ms": max(lag),
            "backlog_frames": backlog_frames,
            "component_compute": {
                "clock": "perf_counter_ns",
                "includes_first_decode": True,
                "excludes_model_initialization": True,
                "vad": _compute_summary(vad_compute),
                "asr": _compute_summary(asr_compute),
            },
        },
        "config": {
            "policy": asdict(PolicyConfig()),
            "acoustic": {name: asdict(config) for name, config in configs.items()},
            "candidate_pause_ms": 224,
            "adaptive_start_threshold_range": [min(thresholds), max(thresholds)],
            "experimental_stable_partial_ms": 224 if experimental_stable_partial else None,
            "jev_partial_request_schedule": "stable_paused_partial"
            if experimental_stable_partial
            else "nonempty_paused_partial",
        },
        "arms": {name: arm.summary() for name, arm in arms.items()},
        "canceled_candidates": {
            "gate": action_gate.canceled_candidates,
            "gate_jev": jev_gate.canceled_candidates if judge else None,
            "gate_stable_partial_jev": stable_gate.canceled_candidates
            if judge and experimental_stable_partial
            else None,
        },
        "partial_stability": {
            "enabled": experimental_stable_partial,
            "eligible_frames": stable_frames,
            "first_eligible_media_ms": first_stable_media_ms,
            "semantic_release_tested": judge is not None and experimental_stable_partial,
        },
        "jev": probe.summary(),
        "timeline": timeline,
        "quality_rates": None,
        "limits": [
            "One window, at most one recommendation per arm; no multi-turn host or playback.",
            "VAD resumption is a model observation, not a human continuation/action label.",
            "No WER, speaker/echo truth, device acceptance, or end-to-end quality claim.",
            "EOF censors unfinished observations; the final sub-32-ms fragment is not decoded.",
        ],
    }
    return FileRunResult(report, recorder)


def _acoustic_data(signal: AcousticSignal) -> dict[str, object]:
    return {
        "speech_active": signal.speech_active,
        "near_end_speech": signal.near_end_speech,
        "echo_likely": signal.echo_likely,
        "speech_duration_ms": signal.speech_duration_ms,
        "pause_duration_ms": signal.pause_duration_ms,
        "audio_quality": signal.audio_quality.value,
    }


def _semantic_data(signal: SemanticSignal) -> dict[str, object]:
    return {
        "transcript_revision": signal.transcript_revision,
        "complete_probability": signal.complete_probability,
        "clarification_probability": signal.clarification_probability,
        "backchannel_probability": signal.backchannel_probability,
        "response_probability": signal.response_probability,
    }


def _write_new(path: Path, content: str) -> None:
    # Never overwrite recordings, models, existing reports, or symlink targets.
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
        output.write(content)


async def _run_cli(args: argparse.Namespace) -> FileRunResult:
    if args.preconnect_jev and not args.allow_remote_text:
        raise ValueError("Jev preconnect requires explicit remote opt-in")
    inspect_wav(args.wav, start_ms=args.start_ms, duration_ms=args.duration_ms)
    if not args.vad_model.is_file() or not args.asr_model.is_dir():
        raise ValueError("local VAD file and ASR directory are required")
    fingerprints = {
        "vad": model_fingerprint(args.vad_model),
        "asr": model_fingerprint(args.asr_model),
    }
    vad = SileroOnnxVad(args.vad_model)
    asr = VoskStreamingAsr(args.asr_model)
    transport: HttpxJevTransport | None = None
    clock = MonotonicClock()
    preconnect: dict[str, object] = {
        "attempted": False,
        "success": None,
        "elapsed_ms": None,
        "outside_window": True,
        "uploaded_text": False,
    }
    try:
        judge = None
        if args.allow_remote_text:
            transport = HttpxJevTransport(os.environ.get("TYPESAFE_API_KEY", ""))
            judge = JevJudge(transport, clock_ms=clock.now_ms)
            if args.preconnect_jev:
                started_ns = time.perf_counter_ns()
                preconnect["attempted"] = True
                try:
                    await transport.preconnect()
                except JevError:
                    preconnect["success"] = False
                else:
                    preconnect["success"] = True
                preconnect["elapsed_ms"] = round(
                    (time.perf_counter_ns() - started_ns) / 1_000_000, 3
                )
            # HTTP client setup belongs to initialization, not the paced window.
            clock = MonotonicClock()
            judge.clock_ms = clock.now_ms
        result = await run_file(
            args.wav,
            vad,
            asr,
            clock=clock,
            judge=judge,
            allow_remote_text=args.allow_remote_text,
            experimental_stable_partial=args.experimental_stable_partial,
            start_ms=args.start_ms,
            duration_ms=args.duration_ms,
        )
        result.report["model_fingerprints"] = fingerprints
        result.report["jev_preconnect"] = preconnect
        result.report["runtime"] = {
            "python": platform.python_version(),
            "system": platform.system(),
            "architecture": platform.machine(),
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("turnpilot", "numpy", "onnxruntime", "vosk")
            },
        }
        return result
    finally:
        if transport is not None:
            await transport.close()


def main(argv: Sequence[str] | None = None) -> int:
    parser = _SafeParser(description=__doc__)
    parser.add_argument("wav", type=Path)
    parser.add_argument("--vad-model", type=Path, required=True, help="local Silero ONNX file")
    parser.add_argument("--asr-model", type=Path, required=True, help="local Vosk model directory")
    parser.add_argument("--report", type=Path, help="new content-free JSON file; never overwrite")
    parser.add_argument("--trace", type=Path, help="new adaptive-arm JSONL; no words or audio")
    parser.add_argument("--start-ms", type=int, default=0, help="explicit source-window start")
    parser.add_argument("--duration-ms", type=int, help="explicit window length, at most 60000 ms")
    parser.add_argument(
        "--experimental-stable-partial",
        action="store_true",
        help="observe 224-ms stable partials; add a Jev arm only with explicit text-upload consent",
    )
    parser.add_argument(
        "--preconnect-jev",
        action="store_true",
        help="explicit 5-s GET /models before timing; requires --allow-remote-text; no ASR upload",
    )
    parser.add_argument(
        "--allow-remote-text",
        action="store_true",
        help="opt in to uploading this file's ASR text to Jev; requires API key",
    )
    try:
        args = parser.parse_args(argv)
        # Fail before decoding/uploading if an output already exists (including symlinks).
        for path in (args.report, args.trace):
            if path is not None and os.path.lexists(path):
                raise ValueError("output already exists")
        if (
            args.report is not None
            and args.trace is not None
            and args.report.resolve() == args.trace.resolve()
        ):
            raise ValueError("outputs must differ")
        result = asyncio.run(_run_cli(args))
        content = json.dumps(result.report, sort_keys=True, indent=2) + "\n"
        if args.report:
            _write_new(args.report, content)
        if args.trace:
            _write_new(args.trace, result.trace.to_jsonl())
        # Avoid a long timeline on stdout. The explicit report includes it.
        print(
            json.dumps({k: v for k, v in result.report.items() if k != "timeline"}, sort_keys=True)
        )
    except Exception:
        print(
            "File diagnostic failed; check WAV, local models, optional dependencies and outputs.",
            file=sys.stderr,
        )
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
