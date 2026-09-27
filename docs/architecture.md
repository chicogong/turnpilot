# Architecture and safety invariants

TurnPilot is a headless decision prototype. The host supplies time-stamped observations and owns all side effects. The implemented `Decision` contains a turn reference, directive kind, reason, decision/expiry times, and a `used_semantic` flag. Explicit evidence references are a future observability requirement, **not** a field in the current contract. The policy never emits speech, invokes a tool, or stops playback itself.

All `*_ms` timestamps used by a session must share one monotonic clock origin, including the clock injected into `JevJudge`. Durations are relative and non-negative. The host must re-check both the turn/generation and action-specific playback/session state at application time; `decision_is_current()` is only the minimum shared guard. Prototype age and pause thresholds are not calibrated defaults.

## Signals and directives

| Signal family | Examples | What it cannot prove |
| --- | --- | --- |
| Acoustic | speech probability, energy/noise floor, near-end/echo evidence, pause candidate, optional audio end-of-turn score | intent, exact words, semantic completion |
| Transcript | partial/final text, revision, availability timestamp, ASR stability, optional Jev semantic judgments | acoustic clarity or speaker identity by itself |
| Host state | active user/assistant turn, playback acknowledged, last question, idle timer, session lifecycle | whether unseen audio exists |

Initial directives: `WAIT`, `COMMIT_USER_TURN`, `IGNORE_USER_TURN`, `CLARIFY_AUDIO`, `CLARIFY_MEANING`, `YIELD_ASSISTANT`, and `NUDGE_USER`. The host must validate a directive against the current turn before applying it. `CLARIFY_*` selects a response mode, not a generated sentence. `IGNORE_USER_TURN` still closes the observed turn; it is not the same as doing nothing. `NUDGE_USER` and AI-initiated interruption are disabled by default.

## Decision path and host boundary

1. Implemented: `AdaptiveSpeechGate.observe()` turns upstream speech probabilities and external near-end/echo hints into a timestamped `AcousticSignal`. It does not emit named `SPEECH_START`/`PAUSE_CANDIDATE` events or identify a speaker. The separate optional `LocalAudioEOTController` emits an **acoustic endpoint candidate**. A [single-slot thread-backed runtime](local-audio-runtime.md) can now execute a caller-supplied scorer in the background, but has no hard kill for running threads or device latency claim.
2. Implemented: `JevJudge` can score only a causal ASR revision, with explicit remote-text opt-in, a deadline and a per-turn budget. The caller supplies any transcript/context. Its result is text evidence, not audio-quality or speaker evidence.
3. Implemented: `TurnPolicy.decide()` combines the supplied acoustic, transcript, semantic, and host observations under minimum wait and maximum pause bounds. It may recommend waiting, committing, clarifying, ignoring, yielding, or an opt-in nudge; it performs none of those actions.
4. Implemented optional [provisional action gate and content-free replay](standalone-action-replay.md): an explicit acoustic candidate starts a bounded wait. By default only aligned final ASR plus high semantic evidence can release an early terminal recommendation; otherwise the policy is consulted at the maximum pause. A disabled-by-default stable-partial experiment additionally requires fresh paused audio, unchanged real partial content and current request-start provenance; the content-free replayer cannot establish those. This is standalone causal-mechanics coverage, not measured action quality or a real capture adapter.
5. Implemented guards: turn/generation references, observation availability, semantic revision checks, and `decision_is_current()` reject some stale inputs or directives. The host still must own cancellation and re-check action-specific state when applying a directive.
6. Host integration boundary: a caller may provide an early, revocable pause candidate and a bounded scorer, then validate any recommendation against resumed speech, session state, and playback truth before acting. Fast barge-in must not wait for Jev. Playback acknowledgement, false-interruption recovery, and physical-device behavior remain outside this repository.

## Jev adapter boundary

The current Jev HTTP payload contains the causal ASR text, whether that text is final, the last assistant question, whether the assistant is speaking, the pinned model, and four narrow questions: completion, meaning clarification, backchannel, and response need. Turn/revision/timing metadata is retained locally for causal checks; it is **not** sent as remote state. Do not ask Jev to infer acoustic intelligibility from text. Chinese thresholds still need held-out calibration; private transcript upload requires explicit opt-in.

## Non-negotiable invariants

- One session/turn/generation owns each decision; no late result can act on a newer turn.
- A remote result cannot block fast user barge-in or override host cancellation and playback truth.
- Timers are canceled on renewed speech, stop, disconnect, and turn replacement.
- No unsolicited speech after the session is stopped. No repeated nudge loop.
- Before host integration, extend the new content-free event trace with authorized, provenance-checked device timing and aggregate observability for decision reason, elapsed time, and fallback status. Current `Decision` and synthetic replay alone do not supply the complete trace.

This repository does not implement a host adapter. A host must measure its own acoustic endpoint timing before deciding whether an earlier provisional pause is useful; inserting a remote judge only after a final endpoint would add latency.
