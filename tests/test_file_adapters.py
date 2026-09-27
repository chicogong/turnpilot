from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from turnpilot import file_adapters
from turnpilot.file_adapters import AsrUpdate, SileroOnnxVad, VoskStreamingAsr, model_fingerprint


def test_model_fingerprint_names_bytes_and_no_absolute_paths(tmp_path: Path) -> None:
    first, second = tmp_path / "a", tmp_path / "b"
    for root in (first, second):
        (root / "nested").mkdir(parents=True)
        (root / "nested" / "model").write_bytes(b"weights" * 20000)
    assert model_fingerprint(first) == model_fingerprint(second)
    (second / "nested" / "model").rename(second / "nested" / "renamed")
    assert model_fingerprint(first) != model_fingerprint(second)
    with pytest.raises(ValueError, match="no files"):
        model_fingerprint(tmp_path / "missing")
    (first / "link").symlink_to(second / "nested" / "renamed")
    with pytest.raises(ValueError, match="symlinks"):
        model_fingerprint(first)


class Recognizer:
    def __init__(self, results: list[tuple[bool, object]]) -> None:
        self.results = iter(results)
        self.current: object = None
        self.inputs: list[bytes] = []

    def AcceptWaveform(self, pcm: bytes) -> bool:
        self.inputs.append(pcm)
        final, self.current = next(self.results)
        return final

    def Result(self) -> str:
        return json.dumps(self.current)

    def PartialResult(self) -> str:
        return json.dumps(self.current)


def asr_adapter(monkeypatch: Any, results: list[tuple[bool, object]]) -> Any:
    recognizer = Recognizer(results)
    models: list[str] = []
    log_levels: list[int] = []
    vosk = SimpleNamespace(
        SetLogLevel=log_levels.append,
        Model=lambda path: models.append(path),
        KaldiRecognizer=lambda model, rate: recognizer,
    )
    monkeypatch.setattr(file_adapters.importlib, "import_module", lambda name: vosk)
    adapter = VoskStreamingAsr(Path("local-model"))
    assert models == ["local-model"]  # Never Model(lang=...) and no automatic download.
    assert log_levels == [-1]
    return adapter, recognizer


def test_asr_preserves_final_until_real_words_resume(monkeypatch: Any) -> None:
    adapter, recognizer = asr_adapter(
        monkeypatch,
        [
            (False, {"partial": "hello"}),
            (False, {"partial": "hello"}),
            (True, {"text": "hello"}),
            (False, {"partial": ""}),
            (True, {"text": ""}),
            (False, {"partial": "again"}),
            (True, {"text": "again"}),
        ],
    )
    results = [adapter.feed(b"pcm") for _ in range(7)]
    assert results == [
        AsrUpdate("hello"),
        None,
        AsrUpdate("hello", True),
        None,
        None,
        AsrUpdate("hello again"),
        AsrUpdate("hello again", True),
    ]
    assert len(recognizer.inputs) == 7
    assert "hello" not in repr(results[0])


@pytest.mark.parametrize("result", [[], {"partial": 7}, {"partial": None}])
def test_invalid_asr_shapes(monkeypatch: Any, result: object) -> None:
    adapter, _ = asr_adapter(monkeypatch, [(False, result)])
    with pytest.raises(ValueError):
        adapter.feed(b"pcm")


class Array:
    def __getitem__(self, key: Any) -> Any:
        if key == (0, 0):
            return 0.7
        return self

    def astype(self, dtype: Any) -> Array:
        return self

    def __truediv__(self, divisor: float) -> Array:
        assert divisor == 32768.0
        return self


def test_silero_streaming_state_contract_without_optional_dependencies(monkeypatch: Any) -> None:
    calls: list[Any] = []
    zero_shapes: list[Any] = []
    frame = Array()

    def zeros(shape: Any, *, dtype: Any) -> Array:
        zero_shapes.append(shape)
        return Array()

    def frombuffer(pcm: bytes, *, dtype: str) -> Array:
        assert len(pcm) == 1024
        assert dtype == "<i2"
        return frame

    def infer(_outputs: Any, inputs: Any) -> Any:
        calls.append(inputs)
        return frame, "updated-state"

    np = SimpleNamespace(
        zeros=zeros,
        float32="float32",
        int64="int64",
        newaxis=None,
        frombuffer=frombuffer,
        concatenate=lambda arrays, axis: frame,
        array=lambda value, dtype: value,
    )
    ort = SimpleNamespace(
        SessionOptions=SimpleNamespace,
        InferenceSession=lambda *args, **kwargs: SimpleNamespace(run=infer),
    )
    monkeypatch.setattr(
        file_adapters.importlib, "import_module", lambda name: np if name == "numpy" else ort
    )
    vad = SileroOnnxVad(Path("model.onnx"))
    assert zero_shapes == [(2, 1, 128), (1, 64)]
    assert vad.probability(b"\0" * 1024) == 0.7
    assert vad.probability(b"\0" * 1024) == 0.7
    assert calls[1]["state"] == "updated-state"
    assert calls[0]["sr"] == 16000
    with pytest.raises(ValueError, match="512"):
        vad.probability(b"\0" * 100)
