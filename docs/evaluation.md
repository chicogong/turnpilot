# Evaluation and test plan

## Dataset and labels

Start with consented, human-reviewed Chinese conversations recorded through real microphone/speaker routes. Include clean, quiet, far-field, noise, changing noise, echo, overlapping voices, short answers, mid-sentence pauses, enumeration, addresses/numbers, backchannels, deliberate barge-ins, ASR mistakes, and long silence after an assistant question. Keep recordings outside Git. Track provenance, checksum, language, hardware/route, consent, classification, and redistribution policy.

For every candidate pause, label the source speech interval, whether the same user thought continues, true end-of-turn timestamp, speaker/echo origin, whether audio is intelligible, expected action (`WAIT`, `RESPOND`, `CLARIFY_AUDIO`, `CLARIFY_MEANING`, `YIELD`, `IGNORE`, or `NUDGE`), and ambiguity/annotator notes. Double-review uncertain cases and adjudicate disagreements. Do not label a model's own output as ground truth.

Split by speaker and device into development, threshold-tuning, and untouched evaluation sets. Fix the split and decision thresholds before the final comparison. Report the number of speakers, sessions, candidate pauses, and positive/negative examples per scenario; calculate final sample size from the observed baseline rate rather than pretending a small pilot can establish a precise percentage.

The separate [continuous-conversation evidence audit](real-conversation-evidence.md) checks a text-free declaration of those sessions and timestamps before A/B/C/D replay. Its counts and ID-overlap check are structural diagnostics, not authentication of consent, human review, clock accuracy, or device behavior. The bundled example is synthetic; no current corpus meets the product-acceptance requirements above.

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

The next valid comparison must put direct-ASR-final, the unchanged acoustic baseline, and the optional TurnPilot gate on the **same consented continuous-device trace** with real ASR availability, acoustic/echo events, playback events, and human expected-action labels. Count missed responses as well as premature responses; compare at a matched latency budget and by speaker/device-disjoint held-out split. Until then, there is **no verified end-to-end improvement over direct ASR**.

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
