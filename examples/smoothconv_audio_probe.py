"""Local acoustic diagnostic on a licensed, publisher-provided 16 kHz mono WAV.

The Silero ONNX model is downloaded separately and must stay outside Git. This
does not run ASR, Jev, speaker attribution, response policy, or OVA. The fixed
arm differs from the adaptive arm only in its noise-threshold adjustment.
"""

from __future__ import annotations

import argparse
import json
import math
import wave
from pathlib import Path

import numpy as np
import onnxruntime as ort  # type: ignore[import-untyped]

from turnpilot.acoustic import AcousticFrame, AdaptiveGateConfig, AdaptiveSpeechGate
from turnpilot.models import TurnRef


def read_pcm(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as audio:
        if (audio.getframerate(), audio.getnchannels(), audio.getsampwidth()) != (16000, 1, 2):
            raise ValueError("WAV must be 16 kHz, mono, signed 16-bit PCM")
        samples = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2")
    return samples.astype(np.float32) / 32768.0


def silero_probabilities(samples: np.ndarray, model: Path) -> list[tuple[int, float, float]]:
    """Stream 32 ms chunks with the model's 64-sample context and recurrent state."""
    options = ort.SessionOptions()
    options.inter_op_num_threads = 1
    options.intra_op_num_threads = 1
    session = ort.InferenceSession(
        str(model), sess_options=options, providers=["CPUExecutionProvider"]
    )
    state = np.zeros((2, 1, 128), dtype=np.float32)
    context = np.zeros((1, 64), dtype=np.float32)
    frames: list[tuple[int, float, float]] = []
    for offset in range(0, len(samples), 512):
        chunk = samples[offset : offset + 512]
        if len(chunk) < 512:
            chunk = np.pad(chunk, (0, 512 - len(chunk)))
        framed = np.concatenate((context, chunk[np.newaxis, :]), axis=1)
        output, state = session.run(
            None,
            {"input": framed, "state": state, "sr": np.array(16000, dtype=np.int64)},
        )
        context = framed[:, -64:]
        probability = float(output[0, 0])
        rms = float(np.sqrt(np.mean(chunk * chunk)))
        energy_dbfs = 20 * math.log10(max(rms, 1e-6))
        frames.append(
            (round((offset + min(512, len(samples) - offset)) / 16), probability, energy_dbfs)
        )
    return frames


def gate_summary(frames: list[tuple[int, float, float]], *, adaptive: bool) -> dict[str, object]:
    config = AdaptiveGateConfig(probability_gain_per_noise_db=0.005 if adaptive else 0.0)
    gate = AdaptiveSpeechGate(config)
    ref = TurnRef("corpus-pilot", "recording", 0)
    speech_frames = 0
    pause_deadline_hits = 0
    previous_pause = False
    thresholds: list[float] = []
    for observed_at_ms, probability, energy_dbfs in frames:
        signal = gate.observe(AcousticFrame(ref, observed_at_ms, probability, energy_dbfs))
        speech_frames += signal.speech_active
        past_deadline = signal.pause_duration_ms is not None and signal.pause_duration_ms >= 640
        if past_deadline and not previous_pause:
            pause_deadline_hits += 1
        previous_pause = past_deadline
        thresholds.append(gate.start_threshold)
    return {
        "speech_active_frames": speech_frames,
        "pause_640ms_episodes": pause_deadline_hits,
        "start_threshold_min": round(min(thresholds), 4),
        "start_threshold_max": round(max(thresholds), 4),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("wav", type=Path, help="publisher WAV resampled to 16 kHz mono")
    parser.add_argument("model", type=Path, help="local Silero v6.2.2 ONNX model")
    args = parser.parse_args()
    samples = read_pcm(args.wav)
    frames = silero_probabilities(samples, args.model)
    result = {
        "audio_duration_ms": round(len(samples) / 16),
        "frames": len(frames),
        "silero_probability_ge_0_5_frames": sum(p >= 0.5 for _, p, _ in frames),
        "fixed_gate": gate_summary(frames, adaptive=False),
        "adaptive_gate": gate_summary(frames, adaptive=True),
        "evidence_level": "one_publisher_audio_acoustic_diagnostic_only",
        "warning": (
            "One TurnRef for the full mixed-mono recording, without speaker/echo truth, "
            "ASR, Jev, policy actions, or OVA. Gate frame counts are not quality rates."
        ),
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
