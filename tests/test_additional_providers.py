"""Tests for GeminiProvider and OllamaProvider.

Copyright 2026 The OpenAgent Contributors
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from unittest.mock import AsyncMock

import httpx
import pytest

from openagent.core.events import (
    DoneEvent,
    StartEvent,
    TextDelta,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
    UsageEvent,
)
from openagent.core.types import (
    ChatRequest,
    FinishReason,
    Message,
    ToolParam,
    ToolSpec,
)
from openagent.providers.gemini import GeminiProvider
from openagent.providers.ollama import OllamaProvider


class MockAsyncStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk


def make_sse_response(events: list[str]) -> httpx.Response:
    content = "\n\n".join(e.strip() for e in events) + "\n\n"
    return httpx.Response(
        status_code=200,
        headers={"content-type": "text/event-stream"},
        stream=MockAsyncStream([content.encode("utf-8")]),
    )


# ============================================================================ #
# Gemini Provider Tests
# ============================================================================ #


def test_gemini_provider_request_rendering() -> None:
    provider = GeminiProvider(
        model="gemini-1.5-pro",
        api_key="test-api-key",
    )

    req = ChatRequest(
        model="gemini-1.5-pro",
        messages=[
            Message.system("You are a helpful assistant."),
            Message.user("Hello!"),
            Message.assistant("Hi!", tool_calls=[]),
            Message.tool_result("call_1", "read_file", "Result from tool"),
        ],
        tools=[
            ToolSpec(
                name="read_file",
                description="Read a file from disk.",
                params=[ToolParam(name="path", type="string", description="path", required=True)],
            )
        ],
    )

    payload = provider.build_payload(req)
    assert "system_instruction" in payload
    assert payload["system_instruction"]["parts"][0]["text"] == "You are a helpful assistant."

    contents = payload["contents"]
    assert len(contents) == 3  # user, model, user(functionResponse)
    assert contents[0]["role"] == "user"
    assert contents[0]["parts"][0]["text"] == "Hello!"
    assert contents[1]["role"] == "model"
    assert contents[1]["parts"][0]["text"] == "Hi!"
    assert contents[2]["role"] == "user"
    assert "functionResponse" in contents[2]["parts"][0]
    assert contents[2]["parts"][0]["functionResponse"]["name"] == "read_file"

    assert "tools" in payload
    assert "function_declarations" in payload["tools"][0]
    decl = payload["tools"][0]["function_declarations"][0]
    assert decl["name"] == "read_file"


@pytest.mark.asyncio
async def test_gemini_provider_streaming(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = GeminiProvider(
        model="gemini-1.5-pro",
        api_key="test-api-key",
    )

    sse_data = [
        (
            'data: {"candidates": [{"content": {"parts": [{"text": "Hello "}], "role": "model"}, "index": 0}]}'
        ),
        (
            'data: {"candidates": [{"content": {"parts": [{"text": "world!"}], "role": "model"}, "index": 0, "finishReason": "STOP"}], '
            '"usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 10, "totalTokenCount": 15}}'
        ),
    ]
    mock_resp = make_sse_response(sse_data)
    monkeypatch.setattr(provider._transport, "stream_post", AsyncMock(return_value=mock_resp))

    req = ChatRequest(
        model="gemini-1.5-pro",
        messages=[Message.user("Hello")],
    )

    events = []
    async for event in provider.stream(req):
        events.append(event)

    types = [type(e) for e in events]
    assert StartEvent in types
    assert TextDelta in types
    assert UsageEvent in types
    assert DoneEvent in types

    done = next(e for e in events if isinstance(e, DoneEvent))
    assert done.finish_reason == FinishReason.STOP
    assert done.message is not None
    assert done.message.text == "Hello world!"
    assert done.usage.prompt_tokens == 5
    assert done.usage.completion_tokens == 10


@pytest.mark.asyncio
async def test_gemini_provider_tool_calls_streaming(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = GeminiProvider(
        model="gemini-1.5-pro",
        api_key="test-api-key",
    )

    sse_data = [
        (
            'data: {"candidates": [{"content": {"parts": [{'
            '"functionCall": {"name": "read_file", "args": {"path": "main.py"}}'
            '}], "role": "model"}, "index": 0, "finishReason": "STOP"}], '
            '"usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 15}}'
        ),
    ]
    mock_resp = make_sse_response(sse_data)
    monkeypatch.setattr(provider._transport, "stream_post", AsyncMock(return_value=mock_resp))

    req = ChatRequest(
        model="gemini-1.5-pro",
        messages=[Message.user("Read main.py")],
    )

    events = []
    async for event in provider.stream(req):
        events.append(event)

    types = [type(e) for e in events]
    assert StartEvent in types
    assert ToolCallStart in types
    assert ToolCallDelta in types
    assert ToolCallEnd in types
    assert DoneEvent in types

    done = next(e for e in events if isinstance(e, DoneEvent))
    assert done.finish_reason == FinishReason.TOOL_CALLS
    assert done.message is not None
    assert len(done.message.tool_calls) == 1
    call = done.message.tool_calls[0]
    assert call.name == "read_file"
    assert call.arguments == {"path": "main.py"}


# ============================================================================ #
# Ollama Provider Tests
# ============================================================================ #


def test_ollama_provider_request_rendering() -> None:
    provider = OllamaProvider(
        model="llama3.2",
    )

    req = ChatRequest(
        model="llama3.2",
        messages=[
            Message.system("Be concise."),
            Message.user("Hello!"),
        ],
        tools=[
            ToolSpec(
                name="bash",
                description="Run shell command",
                params=[ToolParam(name="cmd", type="string", description="command", required=True)],
            )
        ],
    )

    payload = provider.build_payload(req)
    assert payload["model"] == "llama3.2"
    assert len(payload["messages"]) == 2
    assert payload["messages"][0]["role"] == "system"
    assert payload["messages"][0]["content"] == "Be concise."
    assert payload["messages"][1]["role"] == "user"
    assert payload["messages"][1]["content"] == "Hello!"
    assert "tools" in payload
    assert payload["tools"][0]["function"]["name"] == "bash"


@pytest.mark.asyncio
async def test_ollama_provider_streaming(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = OllamaProvider(
        model="llama3.2",
    )

    ndjson_body = (
        json.dumps({"message": {"role": "assistant", "content": "Hello "}, "done": False}) + "\n"
        + json.dumps({"message": {"role": "assistant", "content": "from Ollama!"}, "done": False}) + "\n"
        + json.dumps({
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 8,
            "eval_count": 16,
        }) + "\n"
    )

    mock_resp = httpx.Response(
        status_code=200,
        headers={"content-type": "application/x-ndjson"},
        stream=MockAsyncStream([ndjson_body.encode("utf-8")]),
    )
    monkeypatch.setattr(provider.transport, "stream_post", AsyncMock(return_value=mock_resp))

    req = ChatRequest(
        model="llama3.2",
        messages=[Message.user("Hi")],
    )

    events = []
    async for event in provider.stream(req):
        events.append(event)

    types = [type(e) for e in events]
    assert StartEvent in types
    assert TextDelta in types
    assert UsageEvent in types
    assert DoneEvent in types

    done = next(e for e in events if isinstance(e, DoneEvent))
    assert done.finish_reason == FinishReason.STOP
    assert done.message is not None
    assert done.message.text == "Hello from Ollama!"
    assert done.usage.prompt_tokens == 8
    assert done.usage.completion_tokens == 16


@pytest.mark.asyncio
async def test_ollama_provider_tool_calls_streaming(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = OllamaProvider(
        model="llama3.2",
    )

    ndjson_body = (
        json.dumps({
            "message": {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "function": {
                            "name": "calc",
                            "arguments": {"expr": "2+2"},
                        }
                    }
                ],
            },
            "done": False,
        }) + "\n"
        + json.dumps({
            "message": {"role": "assistant", "content": ""},
            "done": True,
            "done_reason": "stop",
            "prompt_eval_count": 14,
            "eval_count": 28,
        }) + "\n"
    )

    mock_resp = httpx.Response(
        status_code=200,
        headers={"content-type": "application/x-ndjson"},
        stream=MockAsyncStream([ndjson_body.encode("utf-8")]),
    )
    monkeypatch.setattr(provider.transport, "stream_post", AsyncMock(return_value=mock_resp))

    req = ChatRequest(
        model="llama3.2",
        messages=[Message.user("Compute 2+2")],
    )

    events = []
    async for event in provider.stream(req):
        events.append(event)

    types = [type(e) for e in events]
    assert StartEvent in types
    assert ToolCallStart in types
    assert ToolCallDelta in types
    assert ToolCallEnd in types
    assert DoneEvent in types

    done = next(e for e in events if isinstance(e, DoneEvent))
    assert done.finish_reason == FinishReason.TOOL_CALLS
    assert done.message is not None
    assert len(done.message.tool_calls) == 1
    assert done.message.tool_calls[0].name == "calc"
    assert done.message.tool_calls[0].arguments == {"expr": "2+2"}
