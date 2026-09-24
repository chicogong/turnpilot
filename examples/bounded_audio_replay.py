"""Causal audio replay of a bounded local-model endpoint candidate.

The scorer is called synchronously only to measure and obtain its output;
its response is delivered to the pure controller at the modeled arrival time.
This is an offline scheduling counterfactual, not a running async microphone
worker or evidence of OVA behavior.
"""

from __future__ import annotations

import hashlib
import heapq
import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from eot_bench_adapter import _prediction_rows
from smartturn_audio_adapter import AudioEOTScorer, LocalSmartTurnScorer

from turnpilot.acoustic import AcousticFrame, AdaptiveSpeechGate
from turnpilot.acoustic_eval import AcousticPoint
from turnpilot.local_audio_eot import (
    LocalAudioEOTConfig,
    LocalAudioEOTController,
    LocalAudioEOTUpdate,
)
from turnpilot.models import TurnRef


@dataclass(frozen=True, slots=True)
class BoundedAudioTrace:
    endpoint_times_ms: tuple[int, ...]
    active_frame_times_ms: tuple[int, ...]
    score_events: tuple[tuple[int, float, float], ...]
    model_timeouts: int
    cancelled_requests: int
    ignored_late_results: int


def replay_bounded_audio_endpoints(
    frames: list[AcousticPoint],
    samples: Any,
    scorer: AudioEOTScorer,
    *,
    sample_rate: int = 16000,
    model_delay_override_ms: int | None = None,
    config: LocalAudioEOTConfig | None = None,
) -> BoundedAudioTrace:
    """Replay one channel with prefix-only model input and bounded fallback."""
    if sample_rate <= 0 or model_delay_override_ms is not None and model_delay_override_ms < 0:
        raise ValueError("sample rate and model delay must be non-negative")
    gate = AdaptiveSpeechGate()
    generation = 0
    ref = TurnRef("bounded-audio-replay", "channel", generation)
    controller = LocalAudioEOTController(ref, config)
    pending: list[tuple[int, int, LocalAudioEOTController, TurnRef, int, float]] = []
    serial = 0
    endpoints: list[int] = []
    active_frames: list[int] = []
    score_events: list[tuple[int, float, float]] = []
    timeouts = 0
    cancellations = 0
    ignored = 0

    def apply_update(update: LocalAudioEOTUpdate, source: LocalAudioEOTController) -> None:
        nonlocal generation, ref, controller, timeouts, cancellations
        if update.reason == "model_timeout":
            timeouts += 1
        if update.cancel_request_id is not None:
            cancellations += 1
        if update.endpoint_candidate and source is controller:
            endpoints.append(update.at_ms)
            generation += 1
            ref = TurnRef("bounded-audio-replay", "channel", generation)
            controller = LocalAudioEOTController(ref, config)

    for point in frames:
        while pending and pending[0][0] < point.observed_at_ms:
            due_ms, _, source, source_ref, request_id, probability = heapq.heappop(pending)
            if source is not controller:
                ignored += 1
                continue
            result = source.receive_score(source_ref, request_id, due_ms, probability)
            if result.reason.startswith("stale_"):
                ignored += 1
            apply_update(result, source)

        signal = gate.observe(
            AcousticFrame(ref, point.observed_at_ms, point.speech_probability, point.energy_dbfs)
        )
        if signal.speech_active:
            active_frames.append(point.observed_at_ms)
        source = controller
        update = source.observe(signal)
        apply_update(update, source)
        if update.request_id is not None:
            observed_samples = min(len(samples), point.observed_at_ms * sample_rate // 1000)
            started = time.perf_counter()
            probability = scorer.score(samples[:observed_samples])
            runtime_ms = (time.perf_counter() - started) * 1000
            score_events.append((point.observed_at_ms, probability, runtime_ms))
            modeled_delay = (
                math.ceil(runtime_ms)
                if model_delay_override_ms is None
                else model_delay_override_ms
            )
            serial += 1
            heapq.heappush(
                pending,
                (
                    point.observed_at_ms + modeled_delay,
                    serial,
                    source,
                    source.ref,
                    update.request_id,
                    probability,
                ),
            )

        while pending and pending[0][0] == point.observed_at_ms:
            due_ms, _, source, source_ref, request_id, probability = heapq.heappop(pending)
            if source is not controller:
                ignored += 1
                continue
            result = source.receive_score(source_ref, request_id, due_ms, probability)
            if result.reason.startswith("stale_"):
                ignored += 1
            apply_update(result, source)

    return BoundedAudioTrace(
        tuple(endpoints),
        tuple(active_frames),
        tuple(score_events),
        timeouts,
        cancellations,
        ignored,
    )


class BoundedSmartTurnEOTAdapter:
    concurrency = 1
    display_name = "TurnPilot bounded local Smart Turn v3.2 (experimental)"

    def __init__(self) -> None:
        silero_path = os.environ.get("TURNPILOT_SILERO_MODEL")
        smartturn_path = os.environ.get("TURNPILOT_SMARTTURN_MODEL")
        if not silero_path or not Path(silero_path).is_file():
            raise RuntimeError("TURNPILOT_SILERO_MODEL must point to a local ONNX file")
        if not smartturn_path or not Path(smartturn_path).is_file():
            raise RuntimeError("TURNPILOT_SMARTTURN_MODEL must point to a local ONNX file")
        self.silero_path = Path(silero_path)
        self.scorer = LocalSmartTurnScorer(Path(smartturn_path))
        model_sha = hashlib.sha256(Path(smartturn_path).read_bytes()).hexdigest()
        self.adapter_id = f"turnpilot-bounded-smartturn-224-300-480-640-740-{model_sha[:12]}"

    def supports_language(self, lang_code: str) -> bool:
        return lang_code in {"zh", "en"}

    async def predict_turn(
        self, row: dict[str, Any], *, inference_interval: float = 0.1
    ) -> dict[str, Any]:
        from eot_harness.io import decode_audio
        from eot_harness.streaming_stt import resample_audio
        from smoothconv_audio_probe import silero_probabilities

        samples, sample_rate = decode_audio(row["audio"])
        samples = resample_audio(samples, sample_rate, 16000)
        whole_frames = samples[: (len(samples) // 512) * 512]
        frames = [
            AcousticPoint(time_ms, probability, energy_dbfs)
            for time_ms, probability, energy_dbfs in silero_probabilities(
                whole_frames, self.silero_path
            )
        ]
        trace = replay_bounded_audio_endpoints(frames, whole_frames, self.scorer)
        return {
            "id": row["id"],
            "audio_sec": len(samples) / 16000,
            "events": [
                {"event": "AcousticEndpoint", "timestamp": time_ms / 1000}
                for time_ms in trace.endpoint_times_ms
            ]
            + [
                {
                    "event": "SmartTurnScore",
                    "timestamp": time_ms / 1000,
                    "probability": probability,
                    "compute_ms": runtime_ms,
                }
                for time_ms, probability, runtime_ms in trace.score_events
            ],
            "prediction_rows": _prediction_rows(
                row, trace.endpoint_times_ms, inference_interval=inference_interval
            ),
        }
