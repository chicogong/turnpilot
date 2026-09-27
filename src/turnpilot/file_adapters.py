"""Optional local models for the paced-file diagnostic, never auto-downloaded."""

from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class AsrUpdate:
    """Current cumulative hypothesis; a final is a decoder endpoint, not EOF."""

    text: str = field(repr=False)
    is_final: bool = False


def model_fingerprint(path: Path) -> str:
    """Hash file bytes and relative names without exporting the model path."""
    files = [path] if path.is_file() else sorted(path.rglob("*"))
    digest = hashlib.sha256()
    count = 0
    for item in files:
        if item.is_symlink():
            raise ValueError("model symlinks are not supported")
        if not item.is_file():
            continue
        name = b"" if path.is_file() else item.relative_to(path).as_posix().encode()
        digest.update(len(name).to_bytes(8, "big"))
        digest.update(name)
        digest.update(item.stat().st_size.to_bytes(8, "big"))
        with item.open("rb") as stream:
            for chunk in iter(lambda: stream.read(65536), b""):
                digest.update(chunk)
        count += 1
    if not count:
        raise ValueError("model has no files")
    return digest.hexdigest()


class SileroOnnxVad:
    """Silero 16 kHz state/context interface, one instance per input file."""

    def __init__(self, model: Path) -> None:
        np = importlib.import_module("numpy")
        ort = importlib.import_module("onnxruntime")
        options = ort.SessionOptions()
        options.inter_op_num_threads = 1
        options.intra_op_num_threads = 1
        options.log_severity_level = 4
        self._np = np
        self._session = ort.InferenceSession(
            str(model), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self._state = np.zeros((2, 1, 128), dtype=np.float32)
        self._context = np.zeros((1, 64), dtype=np.float32)

    def probability(self, pcm: bytes) -> float:
        if len(pcm) != 1024:
            raise ValueError("VAD requires exactly 512 PCM16 samples")
        np = self._np
        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
        framed = np.concatenate((self._context, samples[np.newaxis, :]), axis=1)
        output, self._state = self._session.run(
            None, {"input": framed, "state": self._state, "sr": np.array(16000, dtype=np.int64)}
        )
        self._context = framed[:, -64:]
        return float(output[0, 0])


class VoskStreamingAsr:
    """Incremental local decoder; never calls FinalResult or downloads a model."""

    def __init__(self, model: Path) -> None:
        vosk = importlib.import_module("vosk")
        vosk.SetLogLevel(-1)
        self._recognizer: Any = vosk.KaldiRecognizer(vosk.Model(str(model)), 16000)
        self._segments: list[str] = []
        self._last: AsrUpdate | None = None

    def feed(self, pcm: bytes) -> AsrUpdate | None:
        final = bool(self._recognizer.AcceptWaveform(pcm))
        raw = self._recognizer.Result() if final else self._recognizer.PartialResult()
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError("ASR result must be an object")
        text = value.get("text" if final else "partial")
        if not isinstance(text, str):
            raise ValueError("ASR hypothesis must be text")
        if final:
            if text.strip():
                self._segments.append(text.strip())
            else:
                # An empty endpoint cannot finalize an earlier segment again.
                return None
        elif not text.strip() and self._segments:
            # Keep the last real final until the next segment produces words.
            return None
        hypothesis = " ".join(self._segments + ([] if final else [text])).strip()
        update = AsrUpdate(hypothesis, final)
        if update == self._last:
            return None
        self._last = update
        return update
