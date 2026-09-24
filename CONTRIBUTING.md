# Contributing

TurnPilot is a research-stage, provider-neutral component. Keep session lifecycle, capture, ASR, TTS, playback, and cancellation in the host. A proposed policy change needs an unchanged-baseline comparison; passing synthetic tests alone is not evidence of better conversations.

## Local checks

```bash
uv sync --locked --python 3.12 --extra dev
uv run --no-sync ruff check .
uv run --no-sync ruff format --check .
uv run --no-sync mypy src/turnpilot examples/live_jev_probe.py examples/smoothconv_annotation_probe.py examples/local_audio_runtime_demo.py
uv run --no-sync pytest -q --cov=turnpilot --cov-branch --cov-fail-under=85
uv build
```

CI repeats the checks on Python 3.10–3.13 and runs the offline CLI examples. Do not make tests require an API key, remote provider, model download, or licensed corpus.

## Privacy and provenance

- Do not commit credentials, real recordings, transcripts, personal identifiers, consent records, downloaded data, or model weights.
- Keep optional remote text inference disabled unless a caller explicitly authorizes that request. Never print or include private text in test failures or aggregate reports.
- Identify the exact source and license before reusing third-party code, datasets, or weights. Code licenses do not automatically cover model weights or data.
- Report observed code behavior separately from simulated timing, public-corpus probes, and device/user acceptance. Do not claim quality gains without human-reviewed, speaker/device-disjoint held-out evidence at a matched latency budget.

TurnPilot's code and original documentation are licensed under [Apache-2.0](LICENSE). Focused issues and pull requests are welcome. Discuss policy or API changes in an issue first, include tests for behavior changes, and report the evidence level of any quality claim. Do not attach private recordings, transcripts, credentials, or consent records to an issue or pull request.

## Release checks

Before a release or tag, review the exact commit and distribution files for credentials, private data, and third-party rights. Confirm the README example, package build, and CI on that commit. Keep public-corpus probes and real-device acceptance distinct.
