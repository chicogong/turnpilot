"""Experimental causal Silero + local Smart Turn v3.2 eot-bench arm.

The model is downloaded separately and never bundled with TurnPilot. This
adapter uses audio only, invokes Smart Turn once per detected pause, and keeps
the existing 640 ms gate as a separately measured control. It is not a product
policy or an OVA integration.
"""

from __future__ import annotations

import hashlib
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from eot_bench_adapter import _prediction_rows

from turnpilot.acoustic import AcousticFrame, AdaptiveSpeechGate
from turnpilot.acoustic_eval import AcousticPoint
from turnpilot.models import TurnRef

SMARTTURN_V3_2_CPU_SHA256 = "2bb026316b14a660486a75b1733cd3fbab8c2fd0314dc9af7be49f8cca967e4f"


class AudioEOTScorer(Protocol):
    def score(self, samples: Any) -> float: ...


@dataclass(frozen=True, slots=True)
class HybridTrace:
    endpoint_times_ms: tuple[int, ...]
    active_frame_times_ms: tuple[int, ...]
    score_events: tuple[tuple[int, float, float], ...]


def replay_hybrid_endpoints(
    frames: list[AcousticPoint],
    samples: Any,
    scorer: AudioEOTScorer,
    *,
    sample_rate: int = 16000,
    score_pause_ms: int = 224,
    complete_threshold: float = 0.8,
    complete_deadline_ms: int = 480,
    incomplete_deadline_ms: int = 960,
    low_confidence_threshold: float | None = None,
    uncertain_deadline_ms: int = 640,
    model_ready_delay_ms: int | None = None,
) -> HybridTrace:
    """Stream causal frames; never use benchmark spans or future samples.

    Completion time includes measured model runtime, or an explicitly supplied
    fixed latency for deterministic counterfactuals, rounded up to the next
    observed frame. A resumed speech frame cancels any pending endpoint first.
    """
    if sample_rate <= 0 or score_pause_ms <= 0:
        raise ValueError("sample rate and score pause must be positive")
    if not 0 <= complete_threshold <= 1:
        raise ValueError("complete threshold must be in [0, 1]")
    if complete_deadline_ms < score_pause_ms or incomplete_deadline_ms < score_pause_ms:
        raise ValueError("deadlines must follow score pause")
    if low_confidence_threshold is not None:
        if not 0 <= low_confidence_threshold < complete_threshold:
            raise ValueError("low confidence threshold must precede complete threshold")
        if uncertain_deadline_ms < score_pause_ms:
            raise ValueError("uncertain deadline must follow score pause")
    if model_ready_delay_ms is not None and model_ready_delay_ms < 0:
        raise ValueError("model ready delay must be non-negative")
    gate = AdaptiveSpeechGate()
    generation = 0
    selected_deadline: int | None = None
    score_finished_at_ms: float | None = None
    endpoints: list[int] = []
    active_frames: list[int] = []
    score_events: list[tuple[int, float, float]] = []
    for point in frames:
        signal = gate.observe(
            AcousticFrame(
                TurnRef("eot-bench", "channel", generation),
                point.observed_at_ms,
                point.speech_probability,
                point.energy_dbfs,
            )
        )
        pause_ms = signal.pause_duration_ms
        if signal.speech_active:
            active_frames.append(point.observed_at_ms)
        if pause_ms is None or signal.speech_active:
            selected_deadline = None
            score_finished_at_ms = None
            continue
        if pause_ms >= score_pause_ms and selected_deadline is None:
            observed_samples = min(len(samples), point.observed_at_ms * sample_rate // 1000)
            started = time.perf_counter()
            probability = scorer.score(samples[:observed_samples])
            runtime_ms = (time.perf_counter() - started) * 1000
            if not 0 <= probability <= 1:
                raise ValueError("model probability must be in [0, 1]")
            score_events.append((point.observed_at_ms, probability, runtime_ms))
            if probability >= complete_threshold:
                selected_deadline = complete_deadline_ms
            elif low_confidence_threshold is not None and probability >= low_confidence_threshold:
                selected_deadline = uncertain_deadline_ms
            else:
                selected_deadline = incomplete_deadline_ms
            ready_after_ms = runtime_ms if model_ready_delay_ms is None else model_ready_delay_ms
            score_finished_at_ms = point.observed_at_ms + ready_after_ms
        if (
            selected_deadline is not None
            and pause_ms >= selected_deadline
            and score_finished_at_ms is not None
            and point.observed_at_ms >= score_finished_at_ms
        ):
            endpoints.append(point.observed_at_ms)
            generation += 1
            selected_deadline = None
            score_finished_at_ms = None
    return HybridTrace(tuple(endpoints), tuple(active_frames), tuple(score_events))


class LocalSmartTurnScorer:
    """Official Whisper feature extraction + pinned ONNX, CPU-only."""

    def __init__(self, model_path: Path) -> None:
        if hashlib.sha256(model_path.read_bytes()).hexdigest() != SMARTTURN_V3_2_CPU_SHA256:
            raise RuntimeError("Smart Turn model SHA-256 does not match pinned v3.2 CPU")
        import onnxruntime as ort
        from transformers import WhisperFeatureExtractor

        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(model_path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.extractor = WhisperFeatureExtractor(chunk_length=8.0)

    def score(self, samples: Any) -> float:
        import numpy as np

        audio = np.asarray(samples[-128000:], dtype=np.float32)
        if len(audio) < 128000:
            audio = np.pad(audio, (128000 - len(audio), 0))
        features = self.extractor(
            audio,
            sampling_rate=16000,
            return_tensors="np",
            padding="max_length",
            max_length=128000,
            truncation=True,
            do_normalize=True,
        ).input_features.astype(np.float32)
        return float(self.session.run(None, {"input_features": features})[0].reshape(-1)[0])


class SmartTurnHybridEOTAdapter:
    concurrency = 1
    display_name = "TurnPilot Silero + local Smart Turn v3.2 (experimental)"
    profile_id = "224-480-960-0.8"
    replay_kwargs: dict[str, float | int] = {}

    def __init__(self) -> None:
        silero_path = os.environ.get("TURNPILOT_SILERO_MODEL")
        smartturn_path = os.environ.get("TURNPILOT_SMARTTURN_MODEL")
        if not silero_path or not Path(silero_path).is_file():
            raise RuntimeError("TURNPILOT_SILERO_MODEL must point to a local ONNX file")
        if not smartturn_path or not Path(smartturn_path).is_file():
            raise RuntimeError("TURNPILOT_SMARTTURN_MODEL must point to a local ONNX file")
        self.silero_path = Path(silero_path)
        model_sha = hashlib.sha256(Path(smartturn_path).read_bytes()).hexdigest()
        if model_sha != SMARTTURN_V3_2_CPU_SHA256:
            raise RuntimeError("Smart Turn model SHA-256 does not match pinned v3.2 CPU")
        self.scorer = LocalSmartTurnScorer(Path(smartturn_path))
        self.adapter_id = f"turnpilot-smartturn-v3.2-cpu-{self.profile_id}-" + model_sha[:12]

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
        trace = replay_hybrid_endpoints(frames, whole_frames, self.scorer, **self.replay_kwargs)
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


class SmartTurnThreeWayEOTAdapter(SmartTurnHybridEOTAdapter):
    """Leave ambiguous scores on the unchanged 640 ms pause deadline."""

    display_name = "TurnPilot Silero + local Smart Turn v3.2 (three-way experimental)"
    profile_id = "224-480-640-960-0.2-0.8-ready300"
    replay_kwargs = {
        "low_confidence_threshold": 0.2,
        "uncertain_deadline_ms": 640,
        "model_ready_delay_ms": 300,
    }


class SmartTurnFixedLatencyEOTAdapter(SmartTurnHybridEOTAdapter):
    """Original two-way decision with a deterministic 300 ms model-ready delay."""

    display_name = "TurnPilot Silero + local Smart Turn v3.2 (two-way, fixed latency)"
    profile_id = "224-480-960-0.8-ready300"
    replay_kwargs = {"model_ready_delay_ms": 300}
