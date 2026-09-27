# 音频文件对照 / Paced file diagnostic

`turnpilot-file-run` is an optional local diagnostic, separate from the dependency-free core. A single WAV is **one analysis window**, with at most one recommendation per arm. It runs models rather than reading a dataset's final transcript. It neither generates a response nor owns a multi-turn host session.

## Setup

```bash
uv sync --locked --extra dev --extra file
mkdir -p corpus/models reports
```

Separately obtain a 16 kHz [Silero ONNX](https://github.com/snakers4/silero-vad) model (MIT) and a [Vosk model](https://alphacephei.com/vosk/models), such as `vosk-model-small-cn-0.22` (the upstream table lists Apache-2.0). The Silero v6.2.2 state/context interface is the tested target. The `file` extra pins Vosk 0.3.42 on macOS, where it has a universal2 wheel, and 0.3.45 elsewhere; this is a packaging choice, not a quality ranking. Downloaded weights/audio stay under ignored `corpus/`; no model is bundled or automatically fetched.

Use only authorized audio. [Easy Turn](https://huggingface.co/datasets/ASLP-lab/Easy-Turn-Testset) has public real Chinese clips and publisher complete/incomplete labels; its dataset card lists Apache-2.0. A clip label is not a timestamped expected-action label. The runner does not read that label or a publisher transcript. Normalize an authorized file separately if necessary:

```bash
ffmpeg -i corpus/source.wav -ar 16000 -ac 1 -c:a pcm_s16le corpus/window.wav
uv run --no-sync turnpilot-file-run corpus/window.wav \
  --vad-model corpus/models/silero.onnx \
  --asr-model corpus/models/vosk-model-small-cn-0.22 \
  --report reports/window.json --trace reports/window.jsonl
```

The WAV must be mono PCM16, 16 kHz. By default its entire duration must be 32 ms–60 s; longer files are rejected, not silently cropped. For continuous public recordings, explicitly select a window with `--start-ms 0 --duration-ms 60000`. Bounds are integer milliseconds; out-of-source or over-60-s selections are rejected before model loading. Every arm sees only that prefix, with fresh decoder state; this is not a decoder warmed by earlier audio. The report retains source duration, selected start, and whether the selected end is the source EOF. Existing outputs, including symlinks, are never overwritten. No original audio or text is persisted. Timing metadata may still be sensitive: keep real reports/traces outside Git.

## Comparison arms

| Arm | Exact diagnostic behavior |
| --- | --- |
| `direct_asr_final` | First nonempty **decoder-produced** final immediately recommends a commit. Deliberately minimal, not every ASR-based product. |
| `fixed_640` | Existing `TurnPolicy`, with acoustic adaptation gain set to zero; all other acoustic settings are unchanged. |
| `adaptive_640` | Same policy with the existing dynamic acoustic configuration. |
| `gate` | Dynamic acoustic candidate at 224 ms; existing provisional gate, local maximum-pause fallback at 1200 ms. |
| `gate_jev` (opt-in) | Same gate plus actual asynchronous Jev judgments of available ASR revisions. Aligned high scores and final ASR can release early. |
| `gate_stable_partial_jev` (experimental, opt-in) | An unchanged partial over 224 ms of fresh paused audio plus aligned high semantic evidence can release early; the strict gate remains the comparator. |

Every arm consumes the same audio probabilities and ASR revisions. The gate variants intentionally have a different fallback deadline from the 640 ms reference: a later recommendation is **not** evidence of a better model. This diagnostic is not the full fixed/dynamic × Jev factorial evaluation or an unchanged external-host baseline.

Jev is off even when a key is present. To enable it, install `--extra jev`, provide `TYPESAFE_API_KEY` through the local environment, and explicitly add `--allow-remote-text` to authorize uploading **this file's recognized text**. Do not put a key in the command. The existing `JevJudge` uses a 350 ms deadline, two attempts per window, 1200-character limit, and one in-flight task. Resume cancels the task; revised/old results are discarded; errors are counted without provider text. With this short deadline, slow network requests may fall back locally. Cancellation does not guarantee the provider stopped billing or processing a dispatched request.

## Stable partial experiment and connection setup

Add `--experimental-stable-partial` to measure local partial eligibility without any cloud call. The interval requires both advancing acoustic observation time and processed-media pause duration; polling an old snapshot cannot qualify it. Speech resumption, backlog, revisions, invalid audio, playback and stopped/changed turns reset the interval. Stable text is **not** ASR confidence or proof of turn completion. This parameter is uncalibrated; defaults remain unchanged.

With `--allow-remote-text`, the flag also adds the experimental Jev arm. Partial requests now wait for stability; actual final requests need no such interval. Within that run, strict `gate_jev` and experimental `gate_stable_partial_jev` share the **same** request, scores, revision and arrival—not two independently sampled calls. This compares the gate rules under a shared stable-request schedule; do not conflate it with the nonexperimental schedule, which can query partials earlier. The early partial path requires the semantic request's actual start to be inside the current stability interval, as well as the policy's usual turn/revision/age checks. An old pause's score cannot be reused after resume. Partial ASR is never relabeled final; the 1200 ms cap and action kinds remain unchanged.

`--preconnect-jev` explicitly makes one bounded (5 s) [authenticated model-list GET](https://docs.typesafe.ai/models#listing-models) before starting the analysis clock, using the same HTTP client. It uploads no ASR text and performs no judgment. It requires remote opt-in, is off by default, and reports setup time/success separately in `jev_preconnect`. Failure still permits the normal bounded requests and local fallback. An owned client uses a 60 s keep-alive expiry; caller-supplied clients retain their own settings. A server can still close a warmed connection; this option does not guarantee the next request meets 350 ms.

Jev also rejects a parsed result whose recorded arrival exceeds its wall-clock budget, even if a blocked event loop prevented the asynchronous timeout from firing on time. This protects score acceptance, not callback scheduling: synchronous decoder stalls can still delay cancellation/error observation. A production host must keep capture and deadline scheduling responsive.

```bash
# .env is local, ignored, and should be mode 0600; never paste a key into a command.
uv run --env-file .env --no-sync turnpilot-file-run corpus/window.wav \
  --vad-model corpus/models/silero.onnx \
  --asr-model corpus/models/vosk-model-small-cn-0.22 \
  --experimental-stable-partial --allow-remote-text --preconnect-jev \
  --report reports/experiment.json
```

Only the explicit invocation above authorizes text upload. The core does not automatically read `.env`.

## Timing and interpretation

- Audio is delivered in real-time-paced 512-sample chunks. Only that chunk and previous state enter each local decoder. No future publisher transcript is used.
- `media_at_ms` identifies the input prefix; `available_at_ms` records actual monotonic-clock availability **after decoding**, not a simulated ASR delay. Model loading is outside the timed window.
- VAD pause durations count processed media, not wall-clock stalls. If decoding is a frame or more behind, pause-gated recommendations wait for catch-up. Resumed speech still cancels candidates. The intentionally minimal direct-ASR arm can fire while backlogged; its recommendation records that flag.
- `pipeline` reports nearest-rank P95 frame processing/availability lag, maximum lag, and backlog frames. These are machine/run-specific diagnostics, not audio-endpoint or first-audible-response latency.
- `pipeline.component_compute` separately measures each VAD/ASR call with `perf_counter_ns`: frame count, total/mean/P95/max and calls exceeding the 32 ms frame budget. Initialization is excluded, but first-decode costs are included. These are synchronous adapter-call durations, not pure kernel timings; other frame work and scheduling are not charged to either model. Injected-clock fixtures still measure **fixture** compute, not model performance.
- `jev.successful_request_latency_ms` measures request scheduling to parsed result arrival, including client/network/model time but excluding later polling. It includes parsed successes subsequently discarded as stale; errors/cancellations are separate counts. It is not server inference time, and unsuccessful samples are not silently turned into latency successes.
- `vad_resumed_later` says the same VAD subsequently reported activity. It is **not** a human-labeled false cutoff, same-speaker continuation, or wrong response action.
- At EOF the runner stops. It never calls Vosk `FinalResult()` or appends silence. The final fragment shorter than 32 ms is excluded and reported as `unprocessed_tail_samples`. `no_recommendation_by_eof` is censored observation, not automatically a miss or correct wait.
- Audio quality stays `UNKNOWN`; near-end and echo evidence are not invented. No playback events, WER, speaker truth, or clarification/interrupt/nudge acceptance is established.

The JSON report includes the effective acoustic/policy configuration, content-free state/ASR/semantic timeline, and local model fingerprints. Fingerprints hash relative names, lengths and bytes (not plain file SHA-256); no local paths, words or keys are included. Stdout omits the long timeline. Explicit `--trace` exports the existing strict adaptive-arm trace schema; semantic events use poll time, while the report also retains result arrival. The content-free replayer uses placeholder text and lacks the file runner's backlog guard and actual partial content/request-start provenance, so it is **not** an exact model/clock rerun and cannot validate the stable-partial experiment.

`run_file()` accepts injected VAD/ASR/clock fixtures for tests. Such results are marked `injected_clock_fixture`; mock model scores and injected timing must not be published as measured model latency. CLI runs are marked `paced_file_pipeline_only`.

下一步效果验证需要按说话人独立留出、人工审核续说/真实完结时间与预期动作，并锁定 ASR 和延迟预算。短片段、VAD 自己的续说标记和单元测试不能替代这些标签；真实设备与播放验收仍在宿主完成。

## Public file pilot: 2026-09-27

Twenty real Easy Turn clips (10 complete, 10 incomplete; 69.226 s total) were processed with **actual local models**, not fixture probabilities or dataset transcripts. For each `testset/{complete,incomplete}/real` directory, sort WAV basenames by `sha256(basename.encode())`, then take the first 10. Join the selected `label/real/basename` keys in complete-then-incomplete order with `\n`, without a trailing newline: selected-key SHA-256 is `7a0f55ad0ba7e9b920d690dc6ace5e835cde49484ed893f522266d11aaa998c6`. This is an exploratory sample, **not** speaker-held-out acceptance or untouched threshold evaluation.

Two originals, `complete_real_116.wav` and `incomplete_real_114.wav`, are stereo 48 kHz. They were normalized locally with `afconvert -f WAVE -d LEI16@16000 -c 1 input output`; all other inputs were already mono 16 kHz PCM16. No silence was appended and no EOF finalization was performed. Local reports/audio/weights are excluded from Git.

The macOS arm64 final pass used TurnPilot 0.1.0.dev1, Python 3.12.9, Vosk 0.3.42 with Chinese small 0.22, NumPy 2.5.3, ONNX Runtime 1.30.0, and Silero v6.2.2. Silero file SHA-256: `1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3`; official Vosk model ZIP SHA-256: `3af8b0e7e0f835ae9d414ce5df580237a3cfb08d586c9fbbb0f7ff29ad5b14ba`. Those are ordinary asset hashes, unlike the report's model-tree fingerprints. Initialization was excluded; first-decode costs were included. Jev was **disabled**, and no thresholds were tuned on these clips.

| Observation | Complete clips (10) | Incomplete clips (10) |
| --- | ---: | ---: |
| Nonempty decoder final before EOF / direct-ASR recommendation | 0 | 0 |
| Fixed 640 ms recommendation | 2 | 0 |
| Dynamic 640 ms recommendation | 2 | 0 |
| Provisional action-gate recommendation | 1 | 0 |

There were 158 ASR revisions and 469 backlogged frames. Per-file availability-lag P95 ranged from **64 to 367 ms**, with a maximum observed frame lag of **398 ms** across the cohort. These are pipeline lag measurements, not completion/response latency; they combine VAD/ASR compute and scheduling and cannot attribute a bottleneck to either model without further profiling. An earlier pass had 157 revisions and slightly different timings but the same recommendation counts; neither pass is a deterministic model-quality benchmark.

The result exposes a measurement gap, not a win: short clips often end before enough trailing silence exists for a decoder final or the policy deadline. Treating every publisher transcript or EOF flush as a timely ASR final would conceal that gap. Fixed and dynamic behavior matched here; the gate withheld more complete-clip recommendations. There are no human endpoint/action timestamps, so `quality_rates` remains `null`. The next useful evidence is continuous, independently labeled pause/continuation windows and a latency-matched comparison, not a score optimized on these 20 clips.

## Profiling and stable partial follow-up: 2026-09-28

TurnPilot 0.1.0.dev2 repeated the same 20-clip selection and model assets/runtime above. Thresholds were unchanged. Stability now starts **after ASR is actually available**, not at the earlier VAD observation; revision, resume and backlog reset it. These local runs enabled stability diagnostics but not Jev. All reports stayed outside Git.

| Local 20-clip observation | VAD calls | ASR calls |
| --- | ---: | ---: |
| Sum of synchronous adapter-call durations | 0.678 s | 18.492 s |
| Largest individual call | 2.027 ms | 400.213 ms |

ASR represented **96.5% of the combined VAD/ASR call time**, not 96.5% of total pipeline wall time or CPU utilization. First-decode costs were included. There were 472 backlogged frames, 157 real ASR revisions and no nonempty decoder final before EOF. Fixed/dynamic arms still recommended on 2 complete clips, the strict local gate on 1; none acted on the incomplete clips. Hardware, scheduling and decoder runs affect these numbers; they are not a quality delta.

Only 5/20 clips qualified for stable-partial eligibility: **3 complete and 2 incomplete**. Thus stability alone is not a completion label. The experiment needs additional semantic evidence and human endpoint/action labels; changing the default based on these clips would not be justified.

The continuous diagnostic uses the first explicitly declared 60,000 ms of three previously processed, locally normalized SmoothConv channel-0 assets: `1765584008_IyT5DW2WMP_seg9_active_ch0_16k.wav`, `1765799160_cuCZuOweWy_seg59_active_ch0_16k.wav`, and `1766225860_4ggalN9K9m_seg22_active_ch0_16k.wav`. No padding or EOF flush is used. This is a reused research cohort, not speaker/device-held-out validation; the runner still makes at most one recommendation per arm/window and does not recreate a multi-turn agent. SmoothConv's CC BY-NC 4.0 terms remain applicable; no audio or annotations are redistributed.

Across those 180 s of selected audio, local decoding produced 191 revisions and **23 nonempty decoder finals**, unlike the EOF-censored short clips. VAD/ASR call totals were 2.250/21.128 s, with 377 backlogged frames. All three windows had stable partials; all four local arms recommended once in each window. Later VAD activity is expected in continuous conversations and is not an action-error label. The longer windows repair part of the observation gap, not the missing human acceptance evidence.

### Live Jev on the shared streaming observations

A separate final pass on all 23 windows explicitly enabled Jev, stable partials and preconnection, with the same 0.8 threshold, 350 ms deadline and two-attempt/window budget. It uploaded **recognized public-audio text**, not publisher transcripts or audio. Strict and experimental gates shared each request and arrival. No thresholds were tuned, and local setup/report files were excluded from Git.

| Live observation | Final pass |
| --- | ---: |
| Judgment attempts / parsed successes / errors / canceled tasks | 11 / 5 / 1 / 5 |
| Successes subsequently discarded after revision | 1 of the 5 |
| Maximum successful scheduling-to-arrival latency | 317 ms |
| Successful model-list preconnections | 22/23 |
| Preconnection elapsed range, outside the window | 687–5003 ms |
| Early stable-partial recommendations | 0 |
| Windows where strict/experimental Jev-arm summaries differed | 0/23 |

Errors and canceled requests are not included as successful latency samples. Cancellation can occur on resume or window EOF and does not prove the provider stopped processing. The 5 successful calls are too few for a service latency guarantee; network variation and local event-loop scheduling are included.

In two continuous windows, both Jev arms recommended `CLARIFY_MEANING` at the unchanged 1200 ms cap instead of the local gate's `COMMIT`. This exercises **which action** to recommend, but without human action labels it could be useful clarification or unnecessary questioning. Neither arm achieved an early-release benefit in this cohort. Keep the experiment opt-in: the immediate engineering bottleneck is synchronous ASR spikes; the quality-validation bottleneck remains independent, reviewed pause/continuation and action labels.
