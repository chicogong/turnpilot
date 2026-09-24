"""Offline, synthetic demonstration of the bounded background audio scorer.

Run with ``uv run --no-sync python examples/local_audio_runtime_demo.py``.
The scorer below is deliberately fake; no model, microphone, ASR, Jev, or OVA
is called. Replace it only with a locally licensed scorer and causal audio
snapshots after reading the runtime safety notes in the documentation.
"""

from __future__ import annotations

import time

from turnpilot.local_audio_runtime import LocalAudioEOTRuntime
from turnpilot.models import AcousticSignal, TurnRef


def main() -> None:
    origin_ns = time.monotonic_ns()

    def now_ms() -> int:
        return (time.monotonic_ns() - origin_ns) // 1_000_000

    def synthetic_scorer(_: bytes) -> float:
        time.sleep(0.05)
        return 0.9

    ref = TurnRef("demo", "synthetic-turn", 0)
    runtime = LocalAudioEOTRuntime(ref, synthetic_scorer)
    try:
        time.sleep(0.224)
        at_ms = now_ms()
        updates = runtime.observe(
            AcousticSignal(ref, at_ms, False, pause_duration_ms=at_ms),
            audio_prefix=b"synthetic-audio-prefix",
        )
        print("initial:", [update.reason for update in updates])
        while now_ms() < 5_000:
            time.sleep(0.01)
            updates = runtime.tick(now_ms())
            if any(update.endpoint_candidate for update in updates):
                print("endpoint candidate:", [update.reason for update in updates])
                break
        else:
            raise RuntimeError("synthetic endpoint deadline was missed")
        print("worker stats:", runtime.stats)
    finally:
        runtime.close(now_ms(), wait=True)


if __name__ == "__main__":
    main()
