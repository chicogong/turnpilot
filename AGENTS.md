# TurnPilot repository instructions

Read `README.md` and the relevant file in `docs/` before changing a contract or policy. Inspect Git status and preserve unrelated work.

- Keep this a provider-neutral decision and evaluation component. Do not duplicate a host's session lifecycle, transport, ASR, LLM, TTS, cancellation, or playback ledger.
- Separate acoustic evidence, transcript/semantic evidence, and action policy. A text-only judge must never be treated as evidence that the audio was clear or spoken by the user.
- Use causal inputs: a decision at time T may only see audio and ASR revisions available by T. Preserve turn identity and reject late results.
- Keep remote inference opt-in, bounded by deadlines, and safe to disable. Never log keys or private audio/transcripts by default.
- Test a proposed improvement against the unchanged baseline on held-out, speaker/device-disjoint data. Record the evidence level; synthetic controls do not establish real-device quality.
- Do not copy third-party code or weights without checking the exact source and model licenses. In particular, LiveKit model weights have framework restrictions.
- Do not commit private recordings, transcripts, reports containing private text, or credentials.
