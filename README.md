# TurnPilot

**Help voice agents know when to speak.**

[简体中文](README.zh-CN.md) · [Documentation](docs/README.md) · [Contributing](CONTRIBUTING.md) · [License](LICENSE)

[![Quality](https://github.com/chicogong/turnpilot/actions/workflows/quality.yml/badge.svg)](https://github.com/chicogong/turnpilot/actions/workflows/quality.yml) ![Python](https://img.shields.io/badge/Python-3.10%E2%80%933.13-3776AB) ![License](https://img.shields.io/badge/License-Apache--2.0-blue) ![Jev](https://img.shields.io/badge/Jev-optional-f59e0b)

TurnPilot turns a detected pause into an **evidence-backed, cancelable, deadline-bound conversation decision**. Dynamic acoustic observations and optional endpoint candidates track pauses and resumed speech; optional **Jev** judges the current ASR text for completion, clarification, backchannels, and response need; an action gate combines those signals with turn and playback state to recommend when and how to act.

It is a composable decision layer: your host still owns the microphone, ASR, LLM, TTS, and playback. TurnPilot focuses on **when to act and which action to recommend**.

## Why use it

- **Designed to avoid premature interruptions:** a pause is only a candidate; resumed speech cancels it instead of turning every ASR final into an instruction to respond.
- **Beyond “is it over?”:** Jev adds text-side scores for completion, meaning clarification, backchannels, and response need. Acoustic evidence still decides audio clarity and near-end speech.
- **Bounded real-time behavior:** ASR revisions and Jev results must match the current turn and version and arrive on time. With fresh acoustic input, a maximum-pause local fallback remains available. Fast user barge-in never waits for Jev.
- **Explainable actions:** a pure policy returns `WAIT`, `COMMIT`, `CLARIFY`, `IGNORE`, `YIELD`, and reasons; the host revalidates and applies them.

## Decision path

![TurnPilot decision path](docs/diagrams/turnpilot-decision-flow.png)

[Editable Excalidraw source](docs/diagrams/turnpilot-decision-flow.excalidraw) · [SVG](docs/diagrams/turnpilot-decision-flow.svg). Acoustic, text, and host observations share one clock. The optional action gate only recommends; the host still owns playback and cancellation.

## From ASR final to conversation decisions

ASR tells us what was recognized. A final revision alone cannot establish whether the user is continuing, whether the assistant should answer, or whether it should clarify or yield. TurnPilot treats final ASR as one piece of evidence, not an execution command.

![Direct final-ASR trigger versus TurnPilot gate](docs/diagrams/asr-vs-turnpilot.png)

[Editable Excalidraw source](docs/diagrams/asr-vs-turnpilot.excalidraw) · [SVG](docs/diagrams/asr-vs-turnpilot.svg). The diagram's “direct ASR” is a deliberately minimal comparator: the first nonempty final revision in each turn immediately recommends one response. It does not stand for every ASR-based agent.

## Install and try

Requires Python 3.10–3.13. From this repository:

```bash
python -m pip install -e .
```

```python
from turnpilot import AcousticSignal, AudioQuality, HostState, TranscriptSignal, TurnPolicy, TurnRef

ref = TurnRef("demo-session", "turn-1", 0)
decision = TurnPolicy().decide(
    HostState(ref, now_ms=1000, session_active=True),
    AcousticSignal(
        ref,
        observed_at_ms=1000,
        speech_active=False,
        pause_duration_ms=640,
        audio_quality=AudioQuality.CLEAR,
    ),
    TranscriptSignal(ref, available_at_ms=900, revision=1, text="hello", is_final=True),
)
print(decision.kind.value)  # commit_user_turn
```

The example uses synthetic observations and makes no network call. Before applying any recommendation, the host must recheck the session, turn/generation, expiry, and action-specific playback state.

## Compare on an audio file

`turnpilot-file-run` feeds an authorized WAV in real-time-paced 32 ms chunks to **local Silero VAD and incremental Vosk ASR**, then compares direct final-ASR, fixed 640 ms, dynamic 640 ms, and the provisional action gate on shared observations. Optional Jev is a bounded, cancelable text-side arm.

```bash
python -m pip install -e '.[file]'
turnpilot-file-run corpus/window.wav \
  --vad-model corpus/models/silero.onnx \
  --asr-model corpus/models/vosk-model-small-cn-0.22 \
  --report reports/window.json --trace reports/window.jsonl
```

Obtain licensed models/audio separately and create `reports/` first. Inputs must be 16 kHz mono PCM16, 32 ms–60 s; output files must not already exist. No microphone capture, automatic downloads, cloud calls, EOF-forced final, or added tail silence. Reports contain timings and flags, **not words or audio**. This is one analysis window, not a multi-turn agent; no human labels means no quality-rate claim. [Setup, optional Jev, and interpretation](docs/file-run.md).

For longer recordings, explicitly select `--start-ms 0 --duration-ms 60000`. Reports now split VAD/ASR compute from availability lag. `--experimental-stable-partial` observes 224 ms of unchanged partial ASR on fresh paused audio; only explicit Jev opt-in adds an early-release comparison arm. Optional `--preconnect-jev` records connection setup outside the window without relaxing the 350 ms judgment deadline. All three are diagnostics, not proven conversational gains.

The latest 20-clip profile attributed 96.5% of combined VAD/ASR call time to ASR. A live Jev pass on those clips plus three 60 s public windows changed two cap-time recommendations to meaning clarification, but the stable-partial arm showed **no early-release gain** over the strict gate. [Measured results and limitations](docs/file-run.md#profiling-and-stable-partial-follow-up-2026-09-28).

## Use Jev for text-side judgment

Install the optional dependency with `python -m pip install -e '.[jev]'` and provide `TYPESAFE_API_KEY` through your local environment. This example **explicitly opts in to sending its authored text** to Jev and reads completion and clarification scores:

```python
import asyncio
import os
import time

from turnpilot import TranscriptSignal, TurnRef
from turnpilot.jev import HttpxJevTransport, JevJudge


async def main() -> None:
    transport = HttpxJevTransport(os.environ["TYPESAFE_API_KEY"])
    try:
        now_ms = time.monotonic_ns() // 1_000_000
        transcript = TranscriptSignal(
            TurnRef("demo-session", "turn-1", 0),
            available_at_ms=now_ms,
            revision=1,
            text="Remind me about the meeting at three tomorrow",
            is_final=True,
        )
        signal = await JevJudge(transport, timeout_ms=1500).judge(
            transcript, now_ms=now_ms, allow_remote_text=True
        )
        print(signal.complete_probability, signal.clarification_probability)
    finally:
        await transport.close()


asyncio.run(main())
```

The returned `SemanticSignal` is **text evidence** with a turn, revision, and arrival time. Pass it to `TurnPolicy` or the optional action gate; it neither proves audio clarity nor directly tells the host to speak. The 1,500 ms timeout makes a one-off probe easier; a live path needs a shorter deadline, call budget, and local fallback. [Adapter boundary](docs/architecture.md#jev-adapter-boundary) · [Public-corpus probe](docs/evaluation.md#current-public-data-checks-not-an-end-to-end-ab)

## Evidence so far

- **Cancelable continuation:** on a same-clock, two-turn **synthetic** trace, the direct final-ASR comparator recommended 2 responses, including 1 during resumed speech; the optional gate recommended 1 and canceled the resumed-speech candidate. Its first recommendation was 10 ms later. This validates mechanics, not user experience.
- **Jev adds text-side filtering:** at the current 0.8 threshold, it accepted **0/20** incomplete texts from real [Easy Turn](https://huggingface.co/datasets/ASLP-lab/Easy-Turn-Testset) recordings—but also only **3/20** complete texts. That threshold is too conservative to present as a deployable policy. All 40 calls succeeded; request timings are in the [evaluation report](docs/evaluation.md).
- **Acoustics remain a research path:** on six [SmoothConv](https://huggingface.co/datasets/qualialabsAI/SmoothConv) recordings, both the dynamic candidate and unchanged 640 ms gate cut off 6 of 10 labeled continuation pauses. No acoustic gain is established on this sample.

TurnPilot has a reproducible decision contract, causal replay, and bounded failure behavior. **End-to-end improvement in continuous conversations still requires same-clock streaming ASR, playback events, and human action labels.** See the [evaluation protocol](docs/evaluation.md) and [research status](docs/research.md) for denominators and limits.

## Included

- A provider-neutral [decision contract and architecture](docs/architecture.md), with acoustic evidence kept separate from text/semantic evidence.
- An optional [provisional action gate and content-free replay](docs/standalone-action-replay.md). The gate is opt-in and does not change `TurnPolicy` defaults.
- A bounded local-audio scoring runtime and an opt-in Jev text adapter. Neither includes model weights or permits a text-only score to stand in for audio clarity.
- Offline [evaluation and evidence tools](docs/evaluation.md). Bundled examples are synthetic; [current research status](docs/research.md) records what has and has not been measured.

For local checks, run `uv sync --locked --extra dev` followed by `uv run --no-sync pytest -q`. See [Contributing](CONTRIBUTING.md) for the complete quality gate and privacy rules.

## Publication boundary

TurnPilot's code and original documentation are licensed under [Apache-2.0](LICENSE); the SVG-embedded Comic Shanns font has its own [MIT notice](NOTICE). Neither license grants rights to third-party models, datasets, or the Jev service. No recordings, transcripts, credentials, or third-party model weights are distributed here. Optional Jev use sends text only with explicit per-call permission. Publishing this source does not imply a production-quality or real-conversation improvement claim.
