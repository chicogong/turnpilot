from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping

import httpx
import pytest

from turnpilot import TranscriptSignal, TurnRef
from turnpilot.jev import HttpxJevTransport, JevError, JevJudge, JevOptInRequired

REF = TurnRef("session-1", "turn-1", 0)


def answer(value: float) -> dict[str, object]:
    return {"type": "noul", "noul": value}


def valid_response() -> dict[str, object]:
    return {
        "model": "jev-1.13.0",
        "answers": {
            "turn_complete": answer(0.9),
            "needs_meaning_clarification": answer(0.1),
            "backchannel": answer(0.05),
            "response_needed": answer(0.95),
        },
    }


class FakeTransport:
    def __init__(self, response: Mapping[str, object] | None = None) -> None:
        self.response = response or valid_response()
        self.payload: Mapping[str, object] | None = None

    async def evaluate(
        self, payload: Mapping[str, object], *, timeout_s: float
    ) -> Mapping[str, object]:
        self.payload = payload
        assert timeout_s > 0
        return self.response


def transcript(*, available_at_ms: int = 900) -> TranscriptSignal:
    return TranscriptSignal(REF, available_at_ms, 3, "我想订明天下午三点", False)


def test_jev_requires_opt_in_and_causal_transcript() -> None:
    transport = FakeTransport()
    judge = JevJudge(transport, clock_ms=lambda: 1100)
    with pytest.raises(JevOptInRequired):
        asyncio.run(judge.judge(transcript(), now_ms=1000))
    assert transport.payload is None
    with pytest.raises(ValueError, match="future"):
        asyncio.run(
            judge.judge(transcript(available_at_ms=1200), now_ms=1000, allow_remote_text=True)
        )
    assert transport.payload is None


def test_jev_sends_minimal_text_and_parses_atomic_scores() -> None:
    transport = FakeTransport()
    judge = JevJudge(transport, clock_ms=lambda: 1100)
    result = asyncio.run(
        judge.judge(
            transcript(),
            now_ms=1000,
            last_assistant_question="什么时候出发？",
            allow_remote_text=True,
        )
    )
    assert result.complete_probability == 0.9
    assert result.response_probability == 0.95
    assert result.transcript_revision == 3
    assert transport.payload is not None
    assert transport.payload["model"] == "jev-1.13.0"
    assert "session-1" not in str(transport.payload)


def test_jev_rejects_version_mismatch_and_bad_probability() -> None:
    response = valid_response()
    response["model"] = "jev-future"
    judge = JevJudge(FakeTransport(response), clock_ms=lambda: 1100)
    with pytest.raises(JevError, match="version"):
        asyncio.run(judge.judge(transcript(), now_ms=1000, allow_remote_text=True))

    response = valid_response()
    answers = response["answers"]
    assert isinstance(answers, dict)
    answers["turn_complete"] = answer(float("nan"))
    judge = JevJudge(FakeTransport(response), clock_ms=lambda: 1100)
    with pytest.raises(JevError, match="probability"):
        asyncio.run(judge.judge(transcript(), now_ms=1000, allow_remote_text=True))


def test_jev_timeout_is_bounded_and_safe() -> None:
    class SlowTransport:
        async def evaluate(
            self, payload: Mapping[str, object], *, timeout_s: float
        ) -> Mapping[str, object]:
            await asyncio.sleep(0.05)
            return valid_response()

    judge = JevJudge(SlowTransport(), timeout_ms=5, clock_ms=lambda: 1100)
    with pytest.raises(JevError, match="unavailable"):
        asyncio.run(judge.judge(transcript(), now_ms=1000, allow_remote_text=True))


def test_http_transport_uses_bearer_and_hides_error_body() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(429, json={"detail": "sensitive provider error"})

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            transport = HttpxJevTransport("test-key", client=client)
            with pytest.raises(JevError) as error:
                await transport.evaluate({"model": "jev-1.13.0"}, timeout_s=0.1)
            assert "sensitive" not in str(error.value)
            assert "test-key" not in str(error.value)
            assert seen[0].headers["Authorization"] == "Bearer test-key"

    asyncio.run(run())


def test_call_budget_rejects_duplicate_revision_and_excess_requests() -> None:
    transport = FakeTransport()
    judge = JevJudge(transport, clock_ms=lambda: 1500, max_calls_per_turn=2)
    first = transcript()
    asyncio.run(judge.judge(first, now_ms=1000, allow_remote_text=True))
    with pytest.raises(JevError, match="already evaluated"):
        asyncio.run(judge.judge(first, now_ms=1200, allow_remote_text=True))
    second = TranscriptSignal(REF, 1100, 4, "新的部分")
    asyncio.run(judge.judge(second, now_ms=1200, allow_remote_text=True))
    third = TranscriptSignal(REF, 1250, 5, "最终文本")
    with pytest.raises(JevError, match="budget"):
        asyncio.run(judge.judge(third, now_ms=1300, allow_remote_text=True))


def test_call_interval_and_new_generation_budget() -> None:
    transport = FakeTransport()
    judge = JevJudge(transport, clock_ms=lambda: 1500, min_call_interval_ms=100)
    asyncio.run(judge.judge(transcript(), now_ms=1000, allow_remote_text=True))
    second = TranscriptSignal(REF, 1010, 4, "继续")
    with pytest.raises(JevError, match="interval"):
        asyncio.run(judge.judge(second, now_ms=1050, allow_remote_text=True))
    new_ref = TurnRef("session-1", "turn-1", 1)
    fresh = TranscriptSignal(new_ref, 1040, 1, "下一轮")
    result = asyncio.run(judge.judge(fresh, now_ms=1050, allow_remote_text=True))
    assert result.ref == new_ref


def test_http_transport_5xx_error_is_sanitized() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(529, text="private transcript or provider detail")

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            transport = HttpxJevTransport("test-key", client=client)
            with pytest.raises(JevError) as error:
                await transport.evaluate({"state": "synthetic"}, timeout_s=0.1)
            assert str(error.value) == "Jev HTTP 529"

    asyncio.run(run())


def test_judge_http_contract_with_mock_200() -> None:
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        seen.append(payload)
        return httpx.Response(200, json=valid_response())

    async def run() -> None:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            transport = HttpxJevTransport("test-key", client=client)
            judge = JevJudge(transport, clock_ms=lambda: 1100)
            signal = await judge.judge(transcript(), now_ms=1000, allow_remote_text=True)
            assert signal.complete_probability == 0.9

    asyncio.run(run())
    assert len(seen) == 1
    assert seen[0]["model"] == "jev-1.13.0"
    questions = seen[0]["questions"]
    assert isinstance(questions, dict)
    assert set(questions) == {
        "turn_complete",
        "needs_meaning_clarification",
        "backchannel",
        "response_needed",
    }


def test_judge_preserves_sanitized_http_error_code() -> None:
    class RateLimitedTransport:
        async def evaluate(
            self, payload: Mapping[str, object], *, timeout_s: float
        ) -> Mapping[str, object]:
            raise JevError("Jev HTTP 429")

    judge = JevJudge(RateLimitedTransport(), clock_ms=lambda: 1100)
    with pytest.raises(JevError, match="Jev HTTP 429"):
        asyncio.run(judge.judge(transcript(), now_ms=1000, allow_remote_text=True))


def test_jev_rejects_empty_or_oversized_text_before_transport() -> None:
    transport = FakeTransport()
    judge = JevJudge(transport, max_transcript_chars=8)
    for text in (" ", "这是超过八个字符的转写文本"):
        item = TranscriptSignal(REF, 900, 3, text)
        with pytest.raises(ValueError, match="transcript is empty or exceeds"):
            asyncio.run(judge.judge(item, now_ms=1000, allow_remote_text=True))
    with pytest.raises(ValueError, match="assistant question exceeds"):
        asyncio.run(
            judge.judge(
                TranscriptSignal(REF, 900, 3, "短文本"),
                now_ms=1000,
                last_assistant_question="too long question",
                allow_remote_text=True,
            )
        )
    assert transport.payload is None


def test_jev_rejects_missing_answer_and_incompatible_clock() -> None:
    response = valid_response()
    answers = response["answers"]
    assert isinstance(answers, dict)
    answers.pop("response_needed")
    judge = JevJudge(FakeTransport(response), clock_ms=lambda: 1100)
    with pytest.raises(JevError, match="invalid answer"):
        asyncio.run(judge.judge(transcript(), now_ms=1000, allow_remote_text=True))

    judge = JevJudge(FakeTransport(), clock_ms=lambda: 999)
    with pytest.raises(JevError, match="incompatible"):
        asyncio.run(judge.judge(transcript(), now_ms=1000, allow_remote_text=True))


def test_jev_cancellation_propagates_and_never_becomes_fallback_error() -> None:
    class CancelledTransport:
        async def evaluate(
            self, payload: Mapping[str, object], *, timeout_s: float
        ) -> Mapping[str, object]:
            raise asyncio.CancelledError

    judge = JevJudge(CancelledTransport())
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(judge.judge(transcript(), now_ms=1000, allow_remote_text=True))


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (httpx.Response(200, text="private response"), "not JSON"),
        (httpx.Response(200, json=["private response"]), "invalid shape"),
    ],
)
def test_http_transport_rejects_malformed_success_response(
    response: httpx.Response, message: str
) -> None:
    async def run() -> None:
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(lambda _request: response)
        ) as client:
            transport = HttpxJevTransport("test-key", client=client)
            with pytest.raises(JevError, match=message) as error:
                await transport.evaluate({"state": "synthetic"}, timeout_s=0.1)
            assert "private response" not in str(error.value)

    asyncio.run(run())
