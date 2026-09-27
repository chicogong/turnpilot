# Examples

These examples are not a deployable voice-agent pipeline. The three manifests/traces below are synthetic and require no model, dataset, API key, or network inference:

```bash
uv run --no-sync turnpilot-eval examples/synthetic-manifest.json
uv run --no-sync turnpilot-evidence-audit examples/synthetic-evidence.json
uv run --no-sync turnpilot-action-replay examples/synthetic-action-trace.jsonl
uv run --no-sync turnpilot-action-replay examples/synthetic-action-trace.jsonl --compare-direct-asr
uv run --no-sync python examples/local_audio_runtime_demo.py
```

The other scripts are optional research probes:

| Scripts | Additional input and boundary |
| --- | --- |
| `turnpilot-file-run` (installed CLI) | Shared, paced local VAD/streaming-ASR input; compare policies without EOF finalization. [Guide](../docs/file-run.md). One window, not a host or a quality benchmark. |
| `live_jev_probe.py`, `easyturn_jev_probe.py` | Explicit opt-in to send authored or licensed text to TypeSafe; never use private transcripts without consent. |
| `smoothconv_annotation_probe.py`, `smoothconv_audio_probe.py`, `smoothconv_event_probe.py` | Separately obtained, licensed SmoothConv annotations/audio and local model dependencies; no corpus is bundled. |
| `eot_bench_adapter.py`, `eot_bench_tune.py` | External benchmark harness and its data; exploratory timing comparisons only. |
| `smartturn_audio_adapter.py`, `bounded_audio_replay.py`, `easy_turn_real_audio_probe.py` | Separately obtained model weights and licensed audio; local classification/replay is not a response-action label. |

The optional inputs have **separate upstream terms** (checked 2026-09-23):

| External input | Upstream terms and boundary |
| --- | --- |
| [Silero VAD](https://github.com/snakers4/silero-vad) ONNX | Upstream [MIT license](https://github.com/snakers4/silero-vad/blob/master/LICENSE); obtain the model separately. |
| [Vosk Chinese small 0.22](https://alphacephei.com/vosk/models) | Upstream model table lists Apache-2.0. Local incremental ASR is a diagnostic input, not a claim of recognition accuracy. Obtain the model separately. |
| [Smart Turn v3.2 CPU](https://huggingface.co/pipecat-ai/smart-turn-v3) ONNX | Model card lists BSD-2-Clause; the probe checks the selected file's SHA-256. Obtain weights separately. |
| [SmoothConv](https://huggingface.co/datasets/qualialabsAI/SmoothConv) | CC BY-NC 4.0 and an upstream research-use statement. Do not assume commercial-use or redistribution rights from TurnPilot's Apache-2.0 license. |
| [Easy Turn Testset](https://huggingface.co/datasets/ASLP-lab/Easy-Turn-Testset) | Dataset card lists Apache-2.0; audio and publisher text are not bundled here. |
| [LiveKit eot-bench](https://github.com/livekit/eot-bench) | Optional harness code is Apache-2.0; its separately published [dataset](https://huggingface.co/datasets/livekit/eot-bench-data) is CC BY 4.0. Neither is vendored here. |
| TypeSafe Jev | An external service, not covered by TurnPilot's license; remote text submission remains per-call opt-in. |

Recheck the exact source and terms before obtaining or using any external asset. Keep downloaded data, weights, recordings, and result manifests out of Git. Probe outputs are not device-level quality claims; see [research status](../docs/research.md) and the [evaluation protocol](../docs/evaluation.md).
