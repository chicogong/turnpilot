from __future__ import annotations

import asyncio
import json
import struct
import wave
from pathlib import Path
from typing import Any

import pytest

from turnpilot import file_run
from turnpilot.action_trace import replay_action_trace
from turnpilot.file_adapters import AsrUpdate
from turnpilot.file_run import MonotonicClock, inspect_wav, run_file
from turnpilot.jev import JevJudge


class FixtureClock:
    def __init__(self) -> None:
        self.at_ms = 0

    def now_ms(self) -> int:
        return self.at_ms

    async def wait_until(self, at_ms: int) -> None:
        self.at_ms = max(self.at_ms, at_ms)


class FixtureVad:
    def __init__(self, probabilities: list[float]) -> None:
        self.values = iter(probabilities)
        self.inputs: list[bytes] = []

    def probability(self, pcm: bytes) -> float:
        self.inputs.append(pcm)
        return next(self.values)


class FixtureAsr:
    def __init__(
        self,
        updates: dict[int, AsrUpdate],
        clock: FixtureClock | None = None,
        delays: dict[int, int] | None = None,
    ) -> None:
        self.updates = updates
        self.clock = clock
        self.delays = delays or {}
        self.inputs: list[bytes] = []

    def feed(self, pcm: bytes) -> AsrUpdate | None:
        self.inputs.append(pcm)
        if self.clock:
            self.clock.at_ms += self.delays.get(len(self.inputs), 0)
        return self.updates.get(len(self.inputs))


def audio_file(tmp_path: Path, frames: int, *, tail: int = 0) -> Path:
    path = tmp_path / "private-source-name.wav"
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(struct.pack("<h", 5000) * (frames * 512 + tail))
    return path


def run(path: Path, probabilities: list[float], updates: dict[int, AsrUpdate]) -> Any:
    return asyncio.run(
        run_file(path, FixtureVad(probabilities), FixtureAsr(updates), clock=FixtureClock())
    )


def test_shared_causal_frames_and_real_final(tmp_path: Path) -> None:
    path = audio_file(tmp_path, 55)
    vad = FixtureVad([0.9] * 10 + [0.0] * 45)
    asr = FixtureAsr({5: AsrUpdate("secret partial"), 15: AsrUpdate("secret final", True)})
    result = asyncio.run(run_file(path, vad, asr, clock=FixtureClock()))
    assert vad.inputs == asr.inputs
    assert len(vad.inputs) == 55
    arms = result.report["arms"]
    assert arms["direct_asr_final"]["recommendation"]["available_at_ms"] == 480
    assert arms["fixed_640"]["recommendation"]["media_at_ms"] == 960
    assert arms["adaptive_640"] == arms["fixed_640"]
    assert arms["gate"]["recommendation"]["media_at_ms"] == 1536
    assert result.report["evidence_level"] == "injected_clock_fixture"
    serialized = json.dumps(result.report) + result.trace.to_jsonl()
    assert "secret" not in serialized
    assert path.name not in serialized
    assert result.report["quality_rates"] is None
    replay = replay_action_trace(result.trace.events)
    assert replay is not None  # Valid content-free trace, not actual-text re-inference.


def test_resume_cancels_before_gate_cap(tmp_path: Path) -> None:
    result = run(
        audio_file(tmp_path, 90),
        [0.9] * 10 + [0.0] * 25 + [0.9] * 10 + [0.0] * 45,
        {10: AsrUpdate("final", True)},
    )
    arms = result.report["arms"]
    assert arms["direct_asr_final"]["vad_resumed_later"]
    assert arms["fixed_640"]["vad_resumed_later"]
    assert not arms["gate"]["vad_resumed_later"]
    assert result.report["canceled_candidates"]["gate"] == 1
    assert arms["gate"]["recommendation"]["media_at_ms"] == 2656


def test_eof_is_censored_not_finalized_or_padded(tmp_path: Path) -> None:
    vad = FixtureVad([0.9] * 10)
    asr = FixtureAsr({5: AsrUpdate("private pending")})
    result = asyncio.run(
        run_file(audio_file(tmp_path, 10, tail=511), vad, asr, clock=FixtureClock())
    )
    assert len(asr.inputs) == len(vad.inputs) == 10
    assert result.report["unprocessed_tail_samples"] == 511
    assert result.report["eof_forced_final"] is False
    assert result.report["synthetic_tail_silence_ms"] == 0
    assert all(arm["no_recommendation_by_eof"] for arm in result.report["arms"].values())


def test_silence_and_no_words_do_not_invent_transcripts(tmp_path: Path) -> None:
    silent = run(audio_file(tmp_path, 55), [0.0] * 55, {})
    assert silent.report["asr"]["revisions"] == 0
    assert all(arm["no_recommendation_by_eof"] for arm in silent.report["arms"].values())
    unclear = run(audio_file(tmp_path, 55), [0.9] * 10 + [0.0] * 45, {})
    assert unclear.report["arms"]["fixed_640"]["recommendation"]["kind"] == "clarify_audio"
    assert unclear.report["arms"]["direct_asr_final"]["no_recommendation_by_eof"]


def test_backlog_never_authorizes_gated_actions(tmp_path: Path) -> None:
    clock = FixtureClock()
    asr = FixtureAsr({30: AsrUpdate("real final", True)}, clock, {30: 400})
    result = asyncio.run(
        run_file(audio_file(tmp_path, 55), FixtureVad([0.9] * 10 + [0.0] * 45), asr, clock=clock)
    )
    report = result.report
    assert report["pipeline"]["backlog_frames"] > 1
    assert report["pipeline"]["availability_lag_max_ms"] == 400
    assert report["arms"]["direct_asr_final"]["recommendation"]["unprocessed_backlog"]
    assert not report["arms"]["fixed_640"]["recommendation"]["unprocessed_backlog"]
    assert report["arms"]["fixed_640"]["recommendation"]["media_at_ms"] > 960
    assert report["arms"]["gate"]["recommendation"]["media_at_ms"] == 1536


def test_rejected_stale_candidate_retries_when_observations_are_fresh(tmp_path: Path) -> None:
    clock = FixtureClock()
    asr = FixtureAsr({15: AsrUpdate("words", True)}, clock, {17: 400})
    result = asyncio.run(
        run_file(audio_file(tmp_path, 55), FixtureVad([0.9] * 10 + [0.0] * 45), asr, clock=clock)
    )
    candidates = [event for event in result.trace.events if event.kind == "candidate"]
    assert len(candidates) == 1
    assert result.report["arms"]["gate"]["recommendation"]["media_at_ms"] == 1536


class Transport:
    def __init__(self, *, fail: bool = False, hang: bool = False) -> None:
        self.payloads: list[Any] = []
        self.fail = fail
        self.hang = hang

    async def evaluate(self, payload: Any, *, timeout_s: float) -> Any:
        self.payloads.append(payload)
        if self.fail:
            raise RuntimeError("DO NOT PRINT PRIVATE TEXT OR KEY")
        if self.hang:
            await asyncio.Event().wait()
        return {
            "model": "jev-1.13.0",
            "answers": {
                key: {
                    "type": "noul",
                    "noul": 0.9 if key in ("turn_complete", "response_needed") else 0.0,
                }
                for key in (
                    "turn_complete",
                    "response_needed",
                    "needs_meaning_clarification",
                    "backchannel",
                )
            },
        }


def remote_run(
    tmp_path: Path, probabilities: list[float], updates: dict[int, AsrUpdate], transport: Transport
) -> Any:
    clock = FixtureClock()
    judge = JevJudge(transport, clock_ms=clock.now_ms)
    return asyncio.run(
        run_file(
            audio_file(tmp_path, len(probabilities)),
            FixtureVad(probabilities),
            FixtureAsr(updates),
            clock=clock,
            judge=judge,
            allow_remote_text=True,
        )
    )


def test_jev_needs_opt_in(tmp_path: Path) -> None:
    transport = Transport()
    with pytest.raises(ValueError, match="opt-in"):
        asyncio.run(
            run_file(
                audio_file(tmp_path, 5),
                FixtureVad([0.9] * 5),
                FixtureAsr({}),
                clock=FixtureClock(),
                judge=JevJudge(transport),
            )
        )
    assert not transport.payloads


def test_jev_final_aligned_can_release_early(tmp_path: Path) -> None:
    transport = Transport()
    result = remote_run(
        tmp_path, [0.9] * 10 + [0.0] * 45, {15: AsrUpdate("licensed words", True)}, transport
    )
    assert result.report["jev"]["attempts"] == result.report["jev"]["successes"] == 1
    gate = result.report["arms"]["gate_jev"]["recommendation"]
    assert gate["used_semantic"]
    assert gate["media_at_ms"] < 1536
    assert "licensed words" not in json.dumps(result.report)
    assert transport.payloads[0]["state"]["current_user_transcript"] == "licensed words"


def test_jev_partial_is_not_early_release(tmp_path: Path) -> None:
    result = remote_run(
        tmp_path, [0.9] * 10 + [0.0] * 45, {15: AsrUpdate("partial only")}, Transport()
    )
    assert result.report["arms"]["gate_jev"]["recommendation"]["media_at_ms"] == 1536


def test_jev_stale_revision_errors_and_cancellation(tmp_path: Path) -> None:
    result = remote_run(
        tmp_path,
        [0.9] * 10 + [0.0] * 45,
        {15: AsrUpdate("first", True), 18: AsrUpdate("second")},
        Transport(),
    )
    assert result.report["jev"]["attempts"] == 2
    assert result.report["jev"]["discarded"] >= 1
    result = remote_run(
        tmp_path, [0.9] * 10 + [0.0] * 45, {15: AsrUpdate("first", True)}, Transport(fail=True)
    )
    assert result.report["jev"]["errors"] == 1
    assert "PRIVATE" not in json.dumps(result.report)
    result = remote_run(
        tmp_path,
        [0.9] * 10 + [0.0] * 10 + [0.9] * 10,
        {15: AsrUpdate("first", True)},
        Transport(hang=True),
    )
    assert result.report["jev"]["canceled"] == 1
    assert result.report["canceled_candidates"]["gate_jev"] == 1


@pytest.mark.parametrize(
    "rate,channels,width,frames",
    [
        (8000, 1, 2, 512),
        (16000, 2, 2, 512),
        (16000, 1, 1, 512),
        (16000, 1, 2, 0),
        (16000, 1, 2, 960001),
    ],
)
def test_wav_validation(tmp_path: Path, rate: int, channels: int, width: int, frames: int) -> None:
    path = tmp_path / "invalid.wav"
    with wave.open(str(path), "wb") as output:
        output.setnchannels(channels)
        output.setsampwidth(width)
        output.setframerate(rate)
        output.writeframes(b"\0" * frames * channels * width)
    with pytest.raises(ValueError):
        inspect_wav(path)


def test_clock_and_probability_validation(tmp_path: Path) -> None:
    path = audio_file(tmp_path, 5)
    clock = FixtureClock()
    clock.at_ms = 33
    with pytest.raises(ValueError, match="initialization"):
        asyncio.run(run_file(path, FixtureVad([0.9] * 5), FixtureAsr({}), clock=clock))
    with pytest.raises(ValueError, match="probability"):
        run(path, [float("nan")] * 5, {})


def test_cli_sanitized_failure_and_no_overwrite(tmp_path: Path, capsys: Any) -> None:
    report = tmp_path / "existing.json"
    report.write_text("preserve", encoding="utf8")
    args = ["private.wav", "--vad-model", "private.onnx", "--asr-model", "private-model"]
    assert file_run.main(args + ["--report", str(report)]) == 2
    assert report.read_text() == "preserve"
    assert "private" not in capsys.readouterr().err
    assert file_run.main(args) == 2
    assert "private" not in capsys.readouterr().err
    link = tmp_path / "link.json"
    link.symlink_to(report)
    with pytest.raises(FileExistsError):
        file_run._write_new(link, "bad")
    assert report.read_text() == "preserve"


@pytest.mark.parametrize("remote", [False, True])
def test_cli_local_success(tmp_path: Path, monkeypatch: Any, capsys: Any, remote: bool) -> None:
    path = audio_file(tmp_path, 2)
    vad_model = tmp_path / "vad.onnx"
    vad_model.write_bytes(b"model")
    asr_model = tmp_path / "asr"
    asr_model.mkdir()
    (asr_model / "weights").write_bytes(b"asr")
    monkeypatch.setattr(file_run, "SileroOnnxVad", lambda _: FixtureVad([0.9] * 2))
    monkeypatch.setattr(file_run, "VoskStreamingAsr", lambda _: FixtureAsr({}))
    monkeypatch.setattr(file_run.importlib.metadata, "version", lambda _: "fixture-version")
    transports: list[Any] = []

    class CliTransport(Transport):
        closed = False

        async def close(self) -> None:
            self.closed = True

    def transport_factory(key: str) -> CliTransport:
        assert remote  # A key in the environment is not implicit consent.
        assert key == "fixture-only-key"
        transport = CliTransport()
        transports.append(transport)
        return transport

    monkeypatch.setenv("TYPESAFE_API_KEY", "fixture-only-key")
    monkeypatch.setattr(file_run, "HttpxJevTransport", transport_factory)
    report = tmp_path / "new.json"
    trace = tmp_path / "new.jsonl"
    args = [
        str(path),
        "--vad-model",
        str(vad_model),
        "--asr-model",
        str(asr_model),
        "--report",
        str(report),
        "--trace",
        str(trace),
    ]
    if remote:
        args.append("--allow-remote-text")
    assert file_run.main(args) == 0
    value = json.loads(report.read_text())
    assert value["evidence_level"] == "paced_file_pipeline_only"
    assert value["model_fingerprints"]["vad"]
    assert report.stat().st_mode & 0o777 == 0o600
    assert "timeline" not in json.loads(capsys.readouterr().out)
    assert trace.read_text()
    assert value["jev"]["enabled"] == remote
    assert len(transports) == int(remote)
    if remote:
        assert transports[0].closed
    assert "fixture-only-key" not in report.read_text()


def test_monotonic_clock() -> None:
    clock = MonotonicClock()
    asyncio.run(clock.wait_until(2))
    assert clock.now_ms() >= 2


def test_invalid_cli_arguments_are_not_echoed(capsys: Any) -> None:
    assert file_run.main(["--unexpected", "do-not-echo-me"]) == 2
    assert "do-not-echo-me" not in capsys.readouterr().err


def test_jev_call_budget_and_eof_cancel(tmp_path: Path) -> None:
    transport = Transport()
    result = remote_run(
        tmp_path,
        [0.9] * 10 + [0.0] * 45,
        {15: AsrUpdate("one"), 22: AsrUpdate("two"), 30: AsrUpdate("three", True)},
        transport,
    )
    assert result.report["jev"]["attempts"] == len(transport.payloads) == 2
    assert result.report["arms"]["gate_jev"]["recommendation"]["media_at_ms"] == 1536
    result = remote_run(
        tmp_path, [0.9] * 10 + [0.0] * 10, {15: AsrUpdate("one")}, Transport(hang=True)
    )
    assert result.report["jev"]["canceled"] == 1
    assert result.report["arms"]["gate_jev"]["no_recommendation_by_eof"]


def test_component_compute_uses_high_resolution_clock_not_injected_lag(
    tmp_path: Path, monkeypatch: Any
) -> None:
    measurements = iter([0, 1_250_000, 1_250_000, 41_750_000] * 2)
    monkeypatch.setattr(file_run.time, "perf_counter_ns", lambda: next(measurements))
    result = run(audio_file(tmp_path, 2), [0.0] * 2, {})
    compute = result.report["pipeline"]["component_compute"]
    assert compute["vad"]["mean_ms"] == compute["vad"]["p95_ms"] == 1.25
    assert compute["asr"]["total_ms"] == 81.0
    assert compute["asr"]["max_ms"] == 40.5
    assert compute["asr"]["over_frame_budget"] == 2
    assert result.report["pipeline"]["availability_lag_max_ms"] == 0


def test_explicit_window_reads_only_requested_samples(tmp_path: Path) -> None:
    path = tmp_path / "long-source.wav"
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(b"\0\0" * (61 * 16000))
        output.writeframes(struct.pack("<h", 1234) * 512)
    with pytest.raises(ValueError):
        inspect_wav(path)
    assert inspect_wav(path, start_ms=61000, duration_ms=32) == 512
    vad = FixtureVad([0.0])
    result = asyncio.run(
        run_file(path, vad, FixtureAsr({}), clock=FixtureClock(), start_ms=61000, duration_ms=32)
    )
    assert vad.inputs == [struct.pack("<h", 1234) * 512]
    assert result.report["window"] == {
        "start_ms": 61000,
        "source_duration_ms": 61032,
        "ends_at_source_eof": True,
    }
    assert result.report["processed_media_ms"] == 32
    assert "long-source" not in json.dumps(result.report)


@pytest.mark.parametrize(
    "start,duration", [(-1, None), (True, None), (0, 31), (0, 60001), (0, True), (64, 32)]
)
def test_invalid_window_is_rejected(tmp_path: Path, start: int, duration: int | None) -> None:
    with pytest.raises(ValueError):
        inspect_wav(audio_file(tmp_path, 2), start_ms=start, duration_ms=duration)


def test_experimental_partial_diagnostic_is_local_without_judge(tmp_path: Path) -> None:
    result = asyncio.run(
        run_file(
            audio_file(tmp_path, 55),
            FixtureVad([0.9] * 10 + [0.0] * 45),
            FixtureAsr({15: AsrUpdate("only in memory")}),
            clock=FixtureClock(),
            experimental_stable_partial=True,
        )
    )
    assert result.report["partial_stability"]["eligible_frames"] > 0
    assert not result.report["partial_stability"]["semantic_release_tested"]
    assert "gate_stable_partial_jev" not in result.report["arms"]
    assert not result.report["jev"]["enabled"]
    assert "only in memory" not in json.dumps(result.report) + result.trace.to_jsonl()


def test_experimental_partial_and_strict_gate_share_one_semantic_request(tmp_path: Path) -> None:
    clock = FixtureClock()
    transport = Transport()
    result = asyncio.run(
        run_file(
            audio_file(tmp_path, 55),
            FixtureVad([0.9] * 10 + [0.0] * 45),
            FixtureAsr({15: AsrUpdate("partial only")}),
            clock=clock,
            judge=JevJudge(transport, clock_ms=clock.now_ms),
            allow_remote_text=True,
            experimental_stable_partial=True,
        )
    )
    assert len(transport.payloads) == result.report["jev"]["attempts"] == 1
    assert not transport.payloads[0]["state"]["transcript_is_final"]
    arms = result.report["arms"]
    assert arms["gate_jev"]["recommendation"]["media_at_ms"] == 1536
    experiment = arms["gate_stable_partial_jev"]["recommendation"]
    assert experiment["media_at_ms"] < 1536
    assert experiment["reason"] == "semantic_complete_stable_partial"
    assert not experiment["unprocessed_backlog"]
    assert result.report["asr"]["nonempty_decoder_finals"] == 0
    assert result.report["jev"]["successful_request_latency_ms"]["samples"] == 1


@pytest.mark.parametrize("failure", [False, True])
def test_cli_preconnect_is_explicit_outside_window_and_failure_falls_back(
    tmp_path: Path, monkeypatch: Any, capsys: Any, failure: bool
) -> None:
    from turnpilot.jev import JevError

    path = audio_file(tmp_path, 2)
    vad_model = tmp_path / "vad.onnx"
    vad_model.write_bytes(b"model")
    asr_model = tmp_path / "asr"
    asr_model.mkdir()
    (asr_model / "weights").write_bytes(b"asr")
    monkeypatch.setattr(file_run, "SileroOnnxVad", lambda _: FixtureVad([0.0] * 2))
    monkeypatch.setattr(file_run, "VoskStreamingAsr", lambda _: FixtureAsr({}))
    monkeypatch.setattr(file_run.importlib.metadata, "version", lambda _: "fixture-version")
    seen: list[str] = []

    class CliTransport(Transport):
        async def preconnect(self) -> None:
            seen.append("preconnect")
            if failure:
                raise JevError("sanitized failure")

        async def close(self) -> None:
            seen.append("close")

    monkeypatch.setattr(file_run, "HttpxJevTransport", lambda _: CliTransport())
    args = [
        str(path),
        "--vad-model",
        str(vad_model),
        "--asr-model",
        str(asr_model),
        "--preconnect-jev",
        "--experimental-stable-partial",
        "--duration-ms",
        "32",
    ]
    assert file_run.main(args) == 2
    assert not seen
    assert "sanitized failure" not in capsys.readouterr().err
    assert file_run.main(args + ["--allow-remote-text"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert seen == ["preconnect", "close"]
    assert report["jev_preconnect"]["attempted"]
    assert report["jev_preconnect"]["success"] == (not failure)
    assert report["jev_preconnect"]["outside_window"]
    assert report["window"]["ends_at_source_eof"] is False
    assert report["config"]["experimental_stable_partial_ms"] == 224
