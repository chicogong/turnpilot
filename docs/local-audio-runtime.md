# Bounded local-audio runtime / 有界本地音频运行时

`LocalAudioEOTRuntime` connects the pure [`LocalAudioEOTController`](../src/turnpilot/local_audio_eot.py) to a **real background thread** that executes a caller-supplied synchronous scorer. It is an experimental standalone adapter, not a host integration, speaker verifier, response policy, or proof of device latency/quality. It does not include or download model weights.

Run the dependency-free [synthetic demo](../examples/local_audio_runtime_demo.py):

```bash
uv run --no-sync python examples/local_audio_runtime_demo.py
```

The demo returns an **acoustic endpoint candidate** after a fake 50 ms score. It does not ask whether AI should reply. For a real local model, provide a `Callable[[bytes], float]` that decodes an immutable audio-prefix snapshot and returns a finite score in `[0, 1]`. Keep model loading outside the per-request callable. The host owns capture, resampling, near-end/echo evidence, ASR, timers, and actions.

For a host that processes consecutive turns, keep the scorer slot across those turns:

```python
worker = LocalAudioEOTWorker(score_16khz_pcm_prefix)
runtime = LocalAudioEOTRuntime(current_turn_ref, worker=worker)
# Feed observed AcousticSignal events and tick() with one monotonic clock.
runtime.close(now_ms)  # end this turn; the shared worker stays open
# Create the next runtime with the same worker, then at session shutdown:
worker.close(wait=True)
```

`score_16khz_pcm_prefix` is a host-provided function, not a bundled model. Its byte format (for example, mono 16 kHz float32 PCM), window length, and normalization must match the chosen model and must be documented by the host; TurnPilot does not silently resample or reinterpret audio bytes.

## Integration contract

1. Create one runtime for one immutable `TurnRef`; do not reuse it for another turn/generation. For consecutive turns in a session, create one long-lived `LocalAudioEOTWorker(scorer)` and pass `worker=worker` to each runtime. This keeps the physical scorer slot bounded across turns; close each runtime on turn replacement, then close the shared worker at session shutdown. The runtime closes only a worker it created itself. Do not share a worker between sessions unless a single scorer slot across those sessions is intentional. Give all worker users, acoustic observations, and `tick()` calls the **same monotonic millisecond clock**.
2. Feed an observed speech/resume event **before** a timer at the same timestamp. Pass `audio_prefix=bytes(...)` when a pause or tick may dispatch a score; the bytes must contain only audio available by that timestamp. A missing snapshot falls back locally instead of inventing a score.
3. Call `tick(now_ms)` regularly even if no new audio frame arrives. A completed score is delivered at **poll time**, so scheduling delay counts toward the 300 ms request deadline. At or after the deadline, timeout wins even if compute finished but was not polled in time.
4. Inspect every returned `LocalAudioEOTUpdate`. Only `endpoint_candidate=True` is an acoustic candidate; it is **not** a `TurnPolicy` decision. The caller still needs causal transcript/host evidence, and a host must recheck turn identity and playback state before any action.
5. On turn replacement, stop, or disconnect, call `close(now_ms)`. Use `wait=True` only if blocking until the scorer finishes is acceptable.

| Event | Runtime behavior |
| --- | --- |
| Score arrives before deadline | Validate score; the controller may set a 480/640/740 ms acoustic candidate. |
| Timeout, scorer error, invalid score, or missing audio snapshot | Bound the wait and retain the local 640 ms fallback. Exception messages are not exposed. |
| Speech resumes or turn closes | Invalidate the request immediately; late results never become decisions. |
| Old scorer still running when a new pause requests a score | Reject the new score instead of queueing work; use local fallback. |

`AudioWorkerStats` reports submitted, scored, timed-out, canceled, rejected-busy, error, and discarded-result counts, plus the last/max **delivered** computation time. A raw `AudioScoreOutcome` also carries compute time and submission-to-poll elapsed time. These counters are engineering diagnostics, not response-quality metrics; aggregate them without logging audio or transcripts.

## Cancellation limitation and next evidence gate

Python cannot force-stop a scorer already executing in a thread. Cancellation prevents its result from acting and keeps at most **one physical scorer job**, but CPU usage may continue until that function returns. A blocked old call can therefore make a later pause fall back locally. Do not pass an untrusted or indefinitely blocking scorer; hard-kill isolation would require a separate process with an explicit lifecycle and model-loading tradeoff.

The slot limit is **per worker**, not process-global. Creating a fresh owned runtime/worker for every turn can leave several cancelled but still-running threads behind. Reuse a worker per session (or deliberately pool workers under a host-level concurrency limit), and monitor `busy_rejections`/`discarded_results` before deciding whether a different isolation strategy is needed. Shared-worker and controller request timeouts must match.

The [thread-backed tests](../tests/test_local_audio_runtime.py) cover real concurrent execution, timeout, resume, busy rejection, invalid scores, sanitized failures, and lifecycle. The synthetic demo verifies wiring only. Device-specific queue/CPU measurements and a consented, speaker/device-disjoint continuous Chinese corpus with true ASR revision times are still required before claiming any improvement or considering a host shadow adapter. See [research status](research.md) for the separate audio-quality evidence.
