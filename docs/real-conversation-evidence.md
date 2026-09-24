# Continuous-conversation evidence

`turnpilot-evidence-audit` checks whether a **declared** conversation manifest contains the provenance, shared clock, ASR revision times, playback intervals, reviewer labels, and speaker/device split needed for an evaluation. It neither records audio nor verifies that consent, labels, IDs, or timestamps are genuine. It does not call Jev or a host runtime.

```bash
uv run --no-sync turnpilot-evidence-audit examples/synthetic-evidence.json
```

The [example manifest](../examples/synthetic-evidence.json) is synthetic and contains no recording. For real data, keep manifests and media outside Git or in ignored `manifests/` and `corpus/` directories. Never commit recordings, transcripts, consent records, personal identifiers, or private report output.

## What a usable dataset needs

1. Explicit collection and analysis permission, with provenance and redistribution terms recorded separately from this public repository.
2. A single monotonic clock for microphone/acoustic observations, each ASR revision's **availability** time, assistant playback start/stop, and session/turn cancellation. Final text cannot be used before it became available.
3. Candidate-pause labels reviewed independently of model output: actual speaker/echo origin, intelligibility, continuation versus completion, true completion time, expected response action, and an `uncertain` outcome rather than a forced guess. Resolve ambiguous cases with a second reviewer.
4. Development, tuning, and untouched evaluation partitions separated by both speaker and device. Freeze candidate rules, thresholds, and latency budget before viewing the held-out labels.

The auditor rejects extra fields such as `text`, `transcript`, and `audio_path`, inconsistent timestamps, overlapping playback intervals, contradictory continuation labels, and malformed revisions. Its CLI emits aggregate counts only. A clean audit establishes **schema completeness**, not human-review quality, consent validity, or an improvement over the baseline.

Once such data exists, compare the unchanged reference and candidate on the same causal timeline using the [evaluation protocol](evaluation.md). The optional [content-free action trace](standalone-action-replay.md) tests event ordering but cannot create real-device evidence. Current public audio corpora lack at least some of these host/ASR/action fields; see [research status](research.md). There is no accepted device corpus in this repository.
