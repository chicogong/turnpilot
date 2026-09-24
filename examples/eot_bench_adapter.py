"""Opt-in, audio-only TurnPilot adapters for LiveKit's eot-bench harness.

Install the external harness separately and set TURNPILOT_SILERO_MODEL to a
locally obtained Silero v6.2.2 ONNX file. No dataset label, transcript, or
future audio is passed into endpoint detection. The harness labels are attached
only after the full causal acoustic trace has been produced.
"""

from __future__ import annotations

import hashlib
import os
from bisect import bisect_left, bisect_right
from pathlib import Path
from typing import Any

from turnpilot.acoustic import AdaptiveGateConfig
from turnpilot.acoustic_eval import AcousticPoint, replay_acoustic_endpoints


def _grid(start: float, end: float, interval: float) -> list[float]:
    if interval <= 0 or end < start:
        raise ValueError("invalid benchmark time grid")
    count = int((end - start + 1e-6) // interval)
    points = [round(start + index * interval, 6) for index in range(count + 1)]
    if points[-1] < end - 1e-6:
        points.append(round(end, 6))
    return points


def _prediction_rows(
    row: dict[str, Any], endpoint_times_ms: tuple[int, ...], *, inference_interval: float
) -> list[dict[str, Any]]:
    """Map already-computed acoustic events onto benchmark spans for scoring.

    Span boundaries and labels are used only here, never in the gate replay.
    An event becomes visible on the first grid point at or after its timestamp.
    """
    endpoints = sorted(time_ms / 1000 for time_ms in endpoint_times_ms)
    spans = row["silence_spans"]
    result: list[dict[str, Any]] = []
    for span_index, span in enumerate(spans):
        start, end = float(span["start"]), float(span["end"])
        if end - start < 0.1 - 1e-6:
            continue
        first_event = bisect_left(endpoints, start - 1e-6)
        for timestamp in _grid(start, end, inference_interval):
            has_endpoint = bisect_right(endpoints, timestamp + 1e-6) > first_event
            result.append(
                {
                    "id": row["id"],
                    "span_index": span_index,
                    "timestamp": timestamp,
                    "silence_dur": round(timestamp - start, 6),
                    "p_eot": float(has_endpoint),
                    "label": "eot" if span_index == len(spans) - 1 else "hold",
                }
            )
    return result


class _BaseAcousticEOTAdapter:
    """Stream one turn at a time; use the same Silero audio for every arm."""

    concurrency = 1
    pause_deadline_ms = 640
    arm_name = "base"
    gate_config = AdaptiveGateConfig()

    def __init__(self) -> None:
        model_path = os.environ.get("TURNPILOT_SILERO_MODEL", "")
        self.model_path = Path(model_path).expanduser() if model_path else None
        if self.model_path is None or not self.model_path.is_file():
            raise RuntimeError("TURNPILOT_SILERO_MODEL must point to a local Silero ONNX file")
        self.pause_deadline_ms = int(
            os.environ.get("TURNPILOT_EOT_PAUSE_MS", str(self.pause_deadline_ms))
        )
        if not 200 <= self.pause_deadline_ms <= 2000:
            raise ValueError("TURNPILOT_EOT_PAUSE_MS must be between 200 and 2000")
        model_sha = hashlib.sha256(self.model_path.read_bytes()).hexdigest()[:12]
        self.adapter_id = f"turnpilot-{self.arm_name}-{self.pause_deadline_ms}ms-silero-{model_sha}"
        self.display_name = f"TurnPilot {self.arm_name} (acoustic only)"

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
        # The publisher clips are not guaranteed to end on a 32 ms boundary.
        # Do not feed a zero-padded tail as though future microphone audio existed.
        whole_frames = samples[: (len(samples) // 512) * 512]
        acoustic_frames = [
            AcousticPoint(time_ms, probability, energy_dbfs)
            for time_ms, probability, energy_dbfs in silero_probabilities(
                whole_frames, self.model_path
            )
        ]
        trace = replay_acoustic_endpoints(
            acoustic_frames,
            config=self.gate_config,
            pause_deadline_ms=self.pause_deadline_ms,
        )
        return {
            "id": row["id"],
            "audio_sec": len(samples) / 16000,
            "events": [
                {"event": "AcousticEndpoint", "timestamp": time_ms / 1000}
                for time_ms in trace.endpoint_times_ms
            ],
            "prediction_rows": _prediction_rows(
                row, trace.endpoint_times_ms, inference_interval=inference_interval
            ),
        }


class UnchangedGateEOTAdapter(_BaseAcousticEOTAdapter):
    arm_name = "unchanged-gate"


class FixedGateEOTAdapter(_BaseAcousticEOTAdapter):
    arm_name = "fixed-gate"
    gate_config = AdaptiveGateConfig(probability_gain_per_noise_db=0.0)


class FixedRearmEOTAdapter(_BaseAcousticEOTAdapter):
    arm_name = "fixed-rearm"
    gate_config = AdaptiveGateConfig(rearm_during_pause=True, probability_gain_per_noise_db=0.0)


class AdaptiveRearmEOTAdapter(_BaseAcousticEOTAdapter):
    arm_name = "adaptive-rearm"
    gate_config = AdaptiveGateConfig(rearm_during_pause=True)
