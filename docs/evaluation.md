# Evaluation and test plan

## Dataset and labels

Start with consented, human-reviewed Chinese conversations recorded through real microphone/speaker routes. Include clean, quiet, far-field, noise, changing noise, echo, overlapping voices, short answers, mid-sentence pauses, enumeration, addresses/numbers, backchannels, deliberate barge-ins, ASR mistakes, and long silence after an assistant question. Keep recordings outside Git. Track provenance, checksum, language, hardware/route, consent, classification, and redistribution policy.

For every candidate pause, label the source speech interval, whether the same user thought continues, true end-of-turn timestamp, speaker/echo origin, whether audio is intelligible, expected action (`WAIT`, `RESPOND`, `CLARIFY_AUDIO`, `CLARIFY_MEANING`, `YIELD`, `IGNORE`, or `NUDGE`), and ambiguity/annotator notes. Double-review uncertain cases and adjudicate disagreements. Do not label a model's own output as ground truth.

Split by speaker and device into development, threshold-tuning, and untouched evaluation sets. Fix the split and decision thresholds before the final comparison. Report the number of speakers, sessions, candidate pauses, and positive/negative examples per scenario; calculate final sample size from the observed baseline rate rather than pretending a small pilot can establish a precise percentage.

The separate [continuous-conversation evidence audit](real-conversation-evidence.md) checks a text-free declaration of those sessions and timestamps before A/B/C/D replay. Its counts and ID-overlap check are structural diagnostics, not authentication of consent, human review, clock accuracy, or device behavior. The bundled example is synthetic; no current corpus meets the product-acceptance requirements above.

## Scenario priorities

TurnPilot's proposed value is fewer wrong **actions**, not a higher end-of-turn score in isolation. The following are test hypotheses, not shipped host behaviors or measured gains. Keep the unchanged host reference `A0` separate; run the direct-final-ASR, fixed, and candidate policies on the same causal observations. Pre-register the primary error and latency budget for each scenario before tuning.

| Priority and decision | Required observations and independent labels | Comparison and primary outcome |
| --- | --- | --- |
| P0 — Fluent but wrong ASR: respond, clarify, or wait? | Real audio, timestamped ASR revisions, human-corrected names/dates/numbers, expected action, and host playback onset for audible latency. The current `TranscriptSignal` has no reliability or alternate-hypothesis field; nonempty text with unknown audio quality can still lead to a commit. | Compare direct-final-ASR, fixed, and candidate policies on the same turns, with `A0` reported separately. Count wrong actions on critical values and unnecessary clarifications at matched audible latency. Jev can judge supplied text, not whether the audio said those words. |
| P0 — Missing, late, or revised ASR final: respond or keep listening? | Continuous audio with actual partial/final availability times and human-reviewed continuation and completion times. Short-file EOF must not invent a final; stable partials are not completion labels. | Compare final-only, fixed pause, and the optional gate with/without Jev at matched latency. Count false cutoffs **and** missed eligible responses; report endpoint latency now, and first-audio latency only when host playback events exist. |
| P1 — Interruption intent: yield, ignore, or repair? | Synchronized user and assistant audio, playback acknowledgement, near-end/echo origin, and labels for genuine stop/correction/repetition requests versus backchannels and other speakers. | Compare action confusion, false playback stops, missed genuine interruptions, physical stop latency, and recovery after a false stop. `YIELD_ASSISTANT` is only a recommendation; TurnPilot does not classify every repair intent or control playback. |
| P1 — Correction or cancellation during a tool action | Turn/generation changes, tool-start and result receipts, actual side effects, and human labels for the authorized final action. The current `tool_running` flag only suppresses idle nudges. | Replay stale-result races first, then evaluate in a host. Count wrong or duplicate side effects and late directives applied. The host owns tool cancellation, idempotency, and execution. |
| P2 — Non-speech and impaired channels | Labeled cough/laughter/noise, packet loss, telephone-band audio, and source/clarity evidence rather than text alone. | Count false speech starts, needless “please repeat” actions, and missed real requests. Controlled perturbations are robustness checks, not proof of physical-device performance. |

The existing optional [eot-bench adapters](../examples/README.md) can compare acoustic endpointing on public `hold`/`eot` pauses, but those labels cannot establish clarification, tool, or interruption-action quality. [HumDial-FDBench](https://github.com/ASLP-lab/HumDial-FDBench) offers interruption/rejection scenarios to inspect for a future host-level replay; verify the exact audio rights and usable annotations before use. Do not copy external recordings or labels into this repository. Once the first two P0 cases have same-clock, human-reviewed results, reassess whether a new ASR-reliability signal or policy change is warranted; do not change defaults from synthetic controls alone.

## Comparison design

Keep an unchanged host policy as the **external reference** `A0`; the standalone runner does not execute that host. For a causal 2×2 comparison, give all four experimental arms the *same* provisional-pause event and maximum deadline, then vary only the acoustic policy and Jev:

| Arm | Acoustic policy | Text judgment |
| --- | --- | --- |
| A — controlled baseline | Standalone fixed acoustic reference | none |
| B — acoustic only | Dynamic candidate | none |
| C — Jev only | Same standalone acoustic settings as A | Jev |
| D — combination | Dynamic candidate | Jev |

Report `A0` separately to show whether introducing provisional pauses changes behavior by itself, plus a Smart Turn local-audio reference. Otherwise an apparent model gain may just be a timing change. Jev receives only ASR revisions that existed by each decision timestamp, never the future final transcript. Record remote latency, timeouts, version, request count, and any transcript redaction; never include the key.

### Additional direct-ASR comparison

The direct-ASR-final comparator is **separate** from `A0` and arms A–D. It makes exactly one response recommendation per active turn, at the first nonempty final ASR revision's actual availability time. Same-time stop wins, and old turns/revisions cannot fire. It intentionally has no acoustic-continuation, semantic-completion, clarity, or action-mode check. It is a precise lower-complexity reference, not a representative implementation of every ASR-based agent. The content-free [replayer](standalone-action-replay.md) runs this and the optional TurnPilot gate on the same events:

```bash
uv run --no-sync turnpilot-action-replay examples/synthetic-action-trace.jsonl --compare-direct-asr
```

On the bundled **synthetic** two-turn trace, direct-ASR-final recommends 2 commits (at 1350 and 2600 ms), including 1 after acoustic evidence says speech resumed. The optional gate recommends 1 commit (at 1360 ms), cancels 1 candidate, and recommends none while speech is active. The 10 ms first-turn delay and removed second trigger are properties of this constructed timeline only. There are no independent action labels, so neither an error-rate delta nor a real-device latency delta can be calculated from it.

An additional [content-free synthetic ASR-timing trace](../examples/synthetic-asr-timing-trace.jsonl) is checked by [P0 acceptance tests](../tests/test_asr_timing_acceptance.py). Four separately authored cases exercise no final, final after the 1,200 ms pause cap, a revised final that invalidates an old semantic score, and a final tied with resumed speech. In this constructed trace, the optional gate recommends 3 commits and cancels the resumed-speech candidate; direct-final-ASR recommends 3 responses, including one during speech, but has no response for the missing-final case. The tests assert exact recommendation times and stale-result rejection. These synthetic expected actions verify causal contract behavior only; they are not independent human labels or evidence of a real-world accuracy gain.

### Current public-data checks, not an end-to-end A/B

The [Easy Turn Testset](https://huggingface.co/datasets/ASLP-lab/Easy-Turn-Testset) publisher supplies complete/incomplete labels and real/synthetic speech, but these local text lists contain **static dataset transcripts**, not revisions from a streaming ASR. A deterministic speaker-first, key-hash sample selected 20 `real` complete and 20 `real` incomplete Chinese items (selected-key SHA-256: `32d691834e39a5236218905adf5f022da0f6c68e48c8074670ad38c0a6d981a7`). On 2026-09-24, the opt-in Jev `jev-1.13.0` probe used its existing four questions, no assistant-question context, a pre-existing 0.8 completion threshold, and a 1,500 ms request timeout. All 40 calls succeeded.

| Dataset turn-state label | Nonempty dataset text / idealized final-ASR triggers | Jev completion ≥0.8 | Jev successful-request latency P50 / P95 |
| --- | ---: | ---: | ---: |
| Complete (real) | 20/20 | 3/20 | 326 / 515 ms |
| Incomplete (real) | 20/20 | 0/20 | 328 / 373 ms |

This is a **tradeoff**, not a TurnPilot win: the text threshold would suppress all 20 incomplete-text triggers in this sample, but also withhold 17 of 20 complete-text triggers. The `ProvisionalActionGate` is not run here and can still make a local recommendation at its 1,200 ms pause cap when fresh acoustic evidence arrives. The publisher label describes a turn state, not whether a response is warranted; no ASR recognition errors, revision-arrival times, audio quality, playback, or device route were measured. The Jev latency numbers are HTTP completion times for this run, **not** added end-to-end response delay. The sample is small and exploratory, and no threshold is tuned from it.

Reproduce only after separately obtaining the public test lists under ignored `corpus/`, setting `TYPESAFE_API_KEY` in the local environment, and explicitly allowing remote submission of the public dataset text:

```bash
uv run --no-sync python examples/easyturn_jev_probe.py \
  --complete-list corpus/easy_turn_real/testset/complete/complete_test.list \
  --incomplete-list corpus/easy_turn_real/testset/incomplete/incomplete_real_test.list \
  --origin real --per-stratum 20 --timeout-ms 1500 --allow-remote-text
```

Separately, the locked acoustic candidate and the unchanged 640 ms gate both prematurely ended **6/10** labeled continuation pauses in six previously unseen [SmoothConv](https://huggingface.co/datasets/qualialabsAI/SmoothConv) recordings. Completion P95 was 872 ms for the candidate versus 770 ms for the fixed gate in arm-wise matched windows, which are not identical paired latency samples. This does **not** establish a dynamic-VAD gain. SmoothConv annotations supply segment timing and text but no measured streaming-ASR revision arrivals or assistant playback, so they cannot directly answer the direct-ASR-final comparison.

### Full Chinese eot-bench acoustic replay: 2026-10-13

A local replay used all **344 `zh` validation turns** of [LiveKit eot-bench data](https://huggingface.co/datasets/livekit/eot-bench-data), or 4,029.169 s of audio. The publisher describes these as human-to-agent conversations and labels every intermediate silence `hold` and the last silence `eot`; it does **not** provide speaker or device IDs in these rows. The local dataset SHA-256 was `fdaa024cd74f6f4626db78ef11fd62c211d9adc88dfc69d16f470009695c8173`. The separately obtained Silero v6.2.2 ONNX SHA-256 was `1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3`. Neither asset is distributed here.

Each arm received the **same causal 32 ms Silero probability/energy frames**, dropping a final incomplete frame rather than padding future audio. `replay_acoustic_endpoints` used the same 640 ms pause deadline for all arms. The only ordinary fixed/dynamic difference was `probability_gain_per_noise_db=0` versus the existing default `0.005`; the rearm pair made that same comparison with `rearm_during_pause=True` in both arms. Both sides still used the adaptive noise floor and SNR filter: this isolates the extra *speech-start probability gain*, not all forms of dynamic VAD against a completely fixed detector. The current gain is consulted on entry to speech (and optional pause rearm), not for the fixed speech-end probability or 640 ms deadline. Labels were joined after each trace. Exploratory alignment counted an endpoint within an intermediate `hold` span of 100 ms–5 s, or within the last `eot` span of at least 100 ms; this is **not** the official eot-bench policy sweep, whose operating points and default span filters differ.

The [replay script](../examples/eot_bench_zh_replay.py) accepts locally obtained Chinese Arrow or Parquet data and a separately obtained Silero ONNX model. With the `file` extra installed and `reports/` created, install the two optional readers for this invocation; the script refuses to overwrite an existing output:

```bash
uv run --no-sync --with pyarrow --with soundfile python examples/eot_bench_zh_replay.py \
  --input corpus/eot-bench/zh-validation.arrow \
  --silero-model corpus/models/silero_vad_v6.2.2.onnx \
  --output reports/eot-bench-zh.json
```

The aggregate report records exact script, input, model, and acoustic-code SHA-256 hashes, but no paths, text, row IDs, or audio. Check the hashes against the files used for a replay rather than mixing numbers from different script versions. The script's pure span-alignment and input-boundary checks run without third-party audio assets in the normal test suite.

| Same-input acoustic arm | Endpoint inside any eligible `hold` | Endpoint inside long, speech-observed, full-horizon `hold` | Endpoint inside any final `eot` | Endpoint inside long, speech-observed, full-horizon `eot` | Final-span matched latency P50 / P95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Fixed 640 ms | 98/755 | 87/99 | 313/344 | 309/312 | 612 / 684 ms |
| Dynamic 640 ms | 98/755 | 87/99 | 313/344 | 309/312 | 612 / 684 ms |
| Fixed + rearm | 108/755 | 94/99 | 316/344 | 312/312 | 608 / 652 ms |
| Dynamic + rearm | 108/755 | 94/99 | 316/344 | 312/312 | 608 / 652 ms |

Only 99 of 755 eligible `hold` spans lasted at least 640 ms; both arms observed speech in the preceding labeled speech interval and a complete frame at the labeled deadline for all 99. The 87/99 long-hold count is a **publisher-label span alignment**, not a human-reviewed wrong-response rate. Another 11 ordinary endpoints occurred inside shorter holds: the gate's pause clock can start before the publisher's silence boundary. Of the 344 final spans, 313 lasted at least 640 ms; ordinary arms detected 310 of those, plus 3 short spans. However, only 312 long final spans had a complete frame at the labeled deadline, and ordinary arms detected 309/312 of those. Thus 313/344 happens to equal the number of nominally long spans, but is **not** a 91% completion-accuracy estimate. Short final spans and the one frame-incomplete long span are EOF-censored for the corresponding horizon.

The ordinary fixed and dynamic **endpoint timestamp sequences were identical on all 344 turns**, not merely equal in aggregate; the rearm pair was also endpoint-identical. The ordinary active-frame sequences were identical, while rearm differed by one active frame across the corpus. Of 26,462 shared ordinary speech-start checks, 5,767 passed both SNR checks and just 3 frames landed between the two start thresholds; none formed the required three-frame start opportunity. Rearm created one qualifying opportunity and one start-state difference, still no endpoint difference. This diagnoses why the tested gain has no visible endpoint effect here; it does not test a noise-adaptive speech-end threshold or deadline. There is no streaming ASR in **this acoustic replay**, no Jev, assistant playback, independent action label, or speaker/device-disjoint split, so it cannot establish an improvement over direct ASR or a real conversation benefit. The [dataset card](https://huggingface.co/datasets/livekit/eot-bench-data) marks the data CC BY 4.0 while the [benchmark README](https://github.com/livekit/eot-bench) describes Apache-2.0; the conservative local-only use here avoids redistributing audio or labels.

### Paced Chinese public-audio ASR pilot: 2026-10-13

The [local pilot](../examples/eot_bench_zh_asr_pilot.py) takes the **first four rows in publisher order** from the same cached Chinese Arrow partition, preserving each recording from its start to its original EOF (51.078 s total). Each row gets fresh local Silero and Vosk model state and a real-time-paced monotonic clock; direct-final-ASR, fixed 640 ms, adaptive 640 ms, and the optional action gate share the observations. Jev is disabled, no words or audio appear in the aggregate report, and neither a Vosk final nor tail silence is invented at EOF. This is a first-recommendation-per-file diagnostic, not the official eot-bench metric or a multi-turn host.

With the `file` extra, separately obtained models, and `pyarrow`, reproduce into a **new** ignored report path:

```bash
uv run --no-sync --with pyarrow python examples/eot_bench_zh_asr_pilot.py \
  --input corpus/eot-bench/zh-validation.arrow \
  --silero-model corpus/models/silero_vad_v6.2.2.onnx \
  --vosk-model corpus/models/vosk-model-small-cn-0.22 \
  --max-turns 4 --output reports/zh-asr-pilot.json
```

The four files produced 110 real ASR revision events: 102 nonempty partials and 4 nonempty decoder finals across 3/4 windows. The fourth window had **no nonempty final by EOF**; that observation is censored, not a missed-response label. Direct-final-ASR made a first response recommendation in 3/4 windows, fixed and adaptive policies in 4/4 each (with identical first recommendations), and the optional gate in 2/4. The gate made no recommendation by EOF in two windows: **one had a decoder final and one did not**. This flags a potential missed-response tradeoff but is not a human-labeled missed response. On the three windows where both direct-final and fixed recommended, fixed was earlier twice and direct was earlier once; on the two windows where direct-final and gate both recommended, direct-final was earlier both times. The report records the source-frame silence region separately from actual recommendation availability; a publisher `hold`/`eot` span is not a human expected-action label.

Two independent runs with matching input/model/code hashes reproduced the structural counts and paired directions, but not exact wall-clock timings. The first had 357/1,593 backlogged frames, a 1,789 ms maximum source-frame availability lag, and paired median fixed-minus-direct / gate-minus-direct recommendation times of −51 / +524 ms; its direct-final comparator recommended once while backlogged. The repeat had 202/1,593 backlogged frames, a 345 ms maximum lag, medians of −42 / +536 ms, and no backlogged first recommendation. These are **single-machine paced-file timings under differing load, not device latency SLAs**; neither median is end-of-turn or first-audio latency. The fixed/adaptive policy and gate suppress new recommendations while backlogged; the deliberately minimal direct-final comparator records its first final regardless. Timing and any apparent source-span alignment must be read with that processing constraint. The presence of a final, a recommendation, or a source frame within `eot` does **not** establish a correct response. There is still no WER/adjudicated transcript, independent action label, speaker/device split, assistant playback, or audible latency. The next **product-level** comparison must put direct-ASR-final, the unchanged host baseline, and the optional TurnPilot gate on the **same consented continuous-device trace** with real ASR availability, acoustic/echo events, playback events, and human expected-action labels. Count missed responses as well as premature responses; compare at a matched latency budget and by speaker/device-disjoint held-out split. Until then, there is **no verified end-to-end improvement over direct ASR**.

## Metrics

| Metric | Definition |
| --- | --- |
| False cutoff | Incomplete candidate pauses committed as final / all labeled incomplete candidate pauses |
| End-of-turn latency | Decision time minus human-reviewed true completion time; report P50/P95/P99 and premature negative values separately |
| First audible response latency | Browser-confirmed playback onset minus true completion; report separately from the endpoint decision |
| Acoustic miss / false start | Real user speech not accepted; or noise/echo/non-user speech accepted, per labeled exposure |
| Barge-in stop latency | Physical user speech onset to actual assistant playback stop, P50/P95 |
| False barge-in | Backchannel, echo, or noise that wrongly stops assistant playback / eligible exposures |
| Wrong response action | Recommended action differs from adjudicated expected action, with confusion matrix by scenario |
| Nudge error | Unsolicited or repeated nudge in a disallowed state; measure user-rated annoyance in opt-in trials |
| Reliability | Late directives applied, stale audio, timer/task leaks, Jev timeout/fallback rate, API calls and cost per 1,000 turns |

## Acceptance discipline

Pre-register the primary error and latency criteria before held-out evaluation. Compare false cutoffs, missed responses, acoustic misses and false starts, barge-in behavior, stale actions, and bounded failure handling at a matched latency budget. A production default switch requires real-device, browser, and physical playback evidence; synthetic controls and provider benchmarks are lower evidence levels. This document does not publish a product acceptance threshold or claim that one has been met.

## Test layers

1. Pure policy/unit: every legal state transition; pause/resume, timer cancellation, score boundaries, no transcript vs unreliable transcript, reason codes, and deterministic fallbacks.
2. Causal replay: label-driven 2×2 comparison with clocked ASR revisions; no future-information leakage; identical cases and configuration hashes.
3. Adapter/contract: fake Jev 200/429/5xx/timeout/late response, wrong version, malformed answer; privacy opt-in, redaction, and call budget.
4. Local scorer runtime: actual background execution with a controllable blocking scorer; prove bounded capacity, same-time resume priority, timeout, late-result discard, error sanitization, and lifecycle. This establishes engineering safety only, not a device latency SLA.
5. Browser E2E: fake microphone, real WebSocket event order, interruption and playback acknowledgement, reconnect, stop/disconnect.
6. Physical device: near/far-field, echo, headphone/speaker routes, Mandarin short and interrupted turns, audible barge-in, human listening review.
7. Endurance/faults: repeated turns, slow ASR, jitter, provider cancellation, concurrent sessions, bounded task growth and memory.

Full-session acceptance must run in the actual host. This repository's offline policy result must never be presented as proof that users heard a natural conversation.

## Current offline runner

`uv run turnpilot-eval examples/synthetic-manifest.json` exercises the A/B/C/D comparison without contacting any provider. The JSON schema is demonstrated by that synthetic file: each case has one independently supplied label and decision time, with arm-specific acoustic observations and optional precomputed semantic scores. The manifest contains only transcript availability/revision and `has_text`, never words or audio. The CLI rejects any `transcript.text` field and emits aggregate metrics only. `evidence_level` is a declaration (`synthetic` or `human_reviewed`), not proof that consent, annotations, or device tests exist. Store real manifests under ignored `manifests/` or outside this repository.

The runner checks identical case IDs, labels, speaker/device metadata, turn references, candidate times, and pause-candidate eligibility across arms; it rejects unequal eligible denominators. It counts premature endpoints separately from non-negative endpoint P50/P95/P99 and now emits an aggregate expected-action × actual-action confusion table for each arm. Optional `label.speaker_id` and `label.device_id` are opaque identifiers, never printed in the aggregate output. When every incomplete pause has a speaker ID and at least 10 distinct speakers are present, it also reports deterministic, 2,000-draw **speaker-clustered paired bootstrap** 95% intervals for the false-cutoff rate difference B/C/D minus A. Each bootstrap draw resamples speakers and retains all of their paired candidate pauses. Otherwise the interval is `null`; it must not be interpreted as zero difference. This interval describes only the supplied corpus and does not prove speaker/device-disjoint collection, annotation quality, or generalization. The manifest remains schema version 1; these label fields and output metrics are additive. No transcript text or IDs are emitted.

The runner still lacks acoustic miss/false-start rates, actual playback latency, provider cost, or an external unchanged host `A0` baseline. Those require independently labeled recordings and host/playback evidence. The synthetic example deliberately demonstrates possible disagreement; its rates and any synthetic interval are not estimates of product performance.

See [research status](research.md) for the current evidence and its limitations. Full-session acceptance remains unmet.
