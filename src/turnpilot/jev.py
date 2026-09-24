"""Opt-in, bounded TypeSafe Jev text judgment.

This adapter never decides or performs a host action. All transcripts are sent
only when the caller explicitly opts in for that request.
"""

from __future__ import annotations

import asyncio
import math
import time
from collections.abc import Callable, Mapping
from typing import Any, Protocol, cast

from turnpilot.models import SemanticSignal, TranscriptSignal, TurnRef

DEFAULT_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-1.13.0"


class JevError(RuntimeError):
    """A safe-to-display failure; never includes request text or credentials."""


class JevOptInRequired(JevError):
    """Raised when private transcript upload was not explicitly allowed."""


class JevTransport(Protocol):
    async def evaluate(
        self, payload: Mapping[str, object], *, timeout_s: float
    ) -> Mapping[str, object]: ...


def _probability(answers: Mapping[str, object], key: str) -> float:
    raw = answers.get(key)
    if not isinstance(raw, Mapping) or raw.get("type") != "noul":
        raise JevError("Jev response has an invalid answer")
    value = raw.get("noul")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise JevError("Jev response has an invalid probability")
    number = float(value)
    if not math.isfinite(number) or not 0.0 <= number <= 1.0:
        raise JevError("Jev response has an invalid probability")
    return number


def _request_payload(
    transcript: TranscriptSignal,
    *,
    last_assistant_question: str,
    assistant_speaking: bool,
    model: str,
) -> dict[str, object]:
    state = {
        "current_user_transcript": transcript.text,
        "last_assistant_question": last_assistant_question,
        "assistant_speaking": assistant_speaking,
        "transcript_is_final": transcript.is_final,
    }
    questions: dict[str, object] = {
        "turn_complete": {
            "type": "noul",
            "instructions": (
                "Based only on the text and conversation context, is the user's thought "
                "complete enough for the assistant to respond now? A hesitation or unfinished "
                "list is not complete. Do not infer audio clarity."
            ),
        },
        "needs_meaning_clarification": {
            "type": "noul",
            "instructions": (
                "Are the words readable but the user's intended meaning or requested "
                "parameters ambiguous enough to require a specific clarification question? "
                "Do not judge microphone or ASR quality."
            ),
        },
        "backchannel": {
            "type": "noul",
            "instructions": (
                "Is this merely a brief listener acknowledgment that should not take the "
                "floor from an assistant that is currently speaking?"
            ),
        },
        "response_needed": {
            "type": "noul",
            "instructions": (
                "Is this user utterance addressed to the assistant and expecting a response, "
                "rather than incidental speech or a listener acknowledgment?"
            ),
        },
    }
    return {"state": state, "model": model, "questions": questions}


class JevJudge:
    """Compose atomic questions and parse a version-pinned Jev response.

    Create one instance per session. Only turn/revision/timing metadata is
    retained for its in-memory call budget; transcript text is never cached.
    """

    def __init__(
        self,
        transport: JevTransport,
        *,
        model: str = DEFAULT_MODEL,
        timeout_ms: int = 350,
        max_transcript_chars: int = 1200,
        max_calls_per_turn: int = 2,
        min_call_interval_ms: int = 100,
        clock_ms: Callable[[], int] | None = None,
    ) -> None:
        if (
            not model
            or timeout_ms <= 0
            or max_transcript_chars <= 0
            or max_calls_per_turn <= 0
            or min_call_interval_ms < 0
        ):
            raise ValueError("invalid Jev judge configuration")
        self.transport = transport
        self.model = model
        self.timeout_ms = timeout_ms
        self.max_transcript_chars = max_transcript_chars
        self.max_calls_per_turn = max_calls_per_turn
        self.min_call_interval_ms = min_call_interval_ms
        self.clock_ms = clock_ms or (lambda: time.monotonic_ns() // 1_000_000)
        self._budget_ref: TurnRef | None = None
        self._budget_count = 0
        self._last_revision = -1
        self._last_call_ms: int | None = None

    def _reserve_call(self, transcript: TranscriptSignal, now_ms: int) -> None:
        if transcript.ref != self._budget_ref:
            self._budget_ref = transcript.ref
            self._budget_count = 0
            self._last_revision = -1
            self._last_call_ms = None
        if transcript.revision <= self._last_revision:
            raise JevError("Jev transcript revision already evaluated")
        if self._budget_count >= self.max_calls_per_turn:
            raise JevError("Jev call budget exhausted")
        if self._last_call_ms is not None:
            if now_ms < self._last_call_ms:
                raise JevError("Jev call clock moved backward")
            if now_ms - self._last_call_ms < self.min_call_interval_ms:
                raise JevError("Jev call interval too short")
        self._budget_count += 1
        self._last_revision = transcript.revision
        self._last_call_ms = now_ms

    async def judge(
        self,
        transcript: TranscriptSignal,
        *,
        now_ms: int,
        last_assistant_question: str = "",
        assistant_speaking: bool = False,
        allow_remote_text: bool = False,
    ) -> SemanticSignal:
        if not allow_remote_text:
            raise JevOptInRequired("remote transcript evaluation requires explicit opt-in")
        if transcript.available_at_ms > now_ms:
            raise ValueError("future transcript cannot be evaluated")
        if not transcript.text.strip() or len(transcript.text) > self.max_transcript_chars:
            raise ValueError("transcript is empty or exceeds the configured limit")
        if len(last_assistant_question) > self.max_transcript_chars:
            raise ValueError("assistant question exceeds the configured limit")

        self._reserve_call(transcript, now_ms)

        payload = _request_payload(
            transcript,
            last_assistant_question=last_assistant_question,
            assistant_speaking=assistant_speaking,
            model=self.model,
        )
        try:
            response = await asyncio.wait_for(
                self.transport.evaluate(payload, timeout_s=self.timeout_ms / 1000),
                timeout=self.timeout_ms / 1000,
            )
        except asyncio.CancelledError:
            raise
        except JevError:
            raise
        except Exception:
            raise JevError("Jev evaluation unavailable") from None

        response_model = response.get("model")
        raw_answers = response.get("answers")
        if response_model != self.model or not isinstance(raw_answers, Mapping):
            raise JevError("Jev response version or shape mismatch")
        answers = cast(Mapping[str, object], raw_answers)
        received_at_ms = self.clock_ms()
        if received_at_ms < now_ms:
            raise JevError("Jev response clock is incompatible with host clock")
        return SemanticSignal(
            ref=transcript.ref,
            received_at_ms=received_at_ms,
            transcript_revision=transcript.revision,
            complete_probability=_probability(answers, "turn_complete"),
            clarification_probability=_probability(answers, "needs_meaning_clarification"),
            backchannel_probability=_probability(answers, "backchannel"),
            response_probability=_probability(answers, "response_needed"),
            model=self.model,
        )


class HttpxJevTransport:
    """Optional HTTP transport. No retries in the realtime decision path."""

    def __init__(
        self,
        api_key: str,
        *,
        endpoint: str = DEFAULT_ENDPOINT,
        client: Any | None = None,
    ) -> None:
        if not api_key or "\n" in api_key or "\r" in api_key:
            raise ValueError("a valid API key must be supplied")
        try:
            import httpx
        except ImportError:
            raise RuntimeError("install turnpilot[jev] for HTTP transport") from None
        self._api_key = api_key
        self._endpoint = endpoint
        self._owns_client = client is None
        self._client = client if client is not None else httpx.AsyncClient()

    async def evaluate(
        self, payload: Mapping[str, object], *, timeout_s: float
    ) -> Mapping[str, object]:
        try:
            response = await self._client.post(
                self._endpoint,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json=dict(payload),
                timeout=timeout_s,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            raise JevError("Jev transport unavailable") from None
        if response.status_code != 200:
            raise JevError(f"Jev HTTP {response.status_code}")
        try:
            body = response.json()
        except ValueError:
            raise JevError("Jev response is not JSON") from None
        if not isinstance(body, dict):
            raise JevError("Jev response has an invalid shape")
        return cast(Mapping[str, object], body)

    async def close(self) -> None:
        if self._owns_client:
            await self._client.aclose()
