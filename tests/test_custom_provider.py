"""Tests for CustomJsonPathProvider.

Copyright 2026 The OpenAgent Contributors
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import AsyncMock

import httpx
import pytest

from openagent.core.events import (
    DoneEvent,
    StartEvent,
    TextDelta,
    ToolCallEnd,
    ToolCallStart,
    UsageEvent,
)
from openagent.core.types import ChatRequest, FinishReason, Message
from openagent.providers.custom import CustomJsonPathProvider


class MockAsyncStream(httpx.AsyncByteStream):
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk


def test_custom_provider_parse_chunk_text() -> None:
    provider = CustomJsonPathProvider(
        base_url="https://api.custom.ai",
        model="custom-v1",
        text_path="output.text",
    )
    chunk = {"output": {"text": "hello from custom API"}}
    events = provider.parse_chunk(chunk)
    assert len(events) == 1
    assert isinstance(events[0], TextDelta)
    assert events[0].text == "hello from custom API"


def test_custom_provider_parse_chunk_tool_calls() -> None:
    provider = CustomJsonPathProvider(
        base_url="https://api.custom.ai",
        model="custom-v1",
        text_path="result.delta.text",
        tool_calls_path="result.delta.tools",
    )
    chunk = {
        "result": {
            "delta": {
                "text": "Using tool",
                "tools": [
                    {
                        "id": "t1",
                        "name": "lookup",
                        "arguments": {"key": "openagent"},
                    }
                ],
            }
        }
    }
    events = provider.parse_chunk(chunk)
    text_deltas = [e for e in events if isinstance(e, TextDelta)]
    call_starts = [e for e in events if isinstance(e, ToolCallStart)]
    call_ends = [e for e in events if isinstance(e, ToolCallEnd)]

    assert len(text_deltas) == 1
    assert text_deltas[0].text == "Using tool"
    assert len(call_starts) == 1
    assert call_starts[0].name == "lookup"
    assert len(call_ends) == 1
    assert call_ends[0].call is not None
    assert call_ends[0].call.arguments == {"key": "openagent"}


def test_custom_provider_parse_chunk_usage() -> None:
    provider = CustomJsonPathProvider(
        base_url="https://api.custom.ai",
        model="custom-v1",
        usage_prompt_path="meta.usage.in",
        usage_completion_path="meta.usage.out",
    )
    chunk = {"meta": {"usage": {"in": 42, "out": 99}}}
    events = provider.parse_chunk(chunk)
    usage_events = [e for e in events if isinstance(e, UsageEvent)]
    assert len(usage_events) == 1
    assert usage_events[0].usage.prompt_tokens == 42
    assert usage_events[0].usage.completion_tokens == 99


@pytest.mark.asyncio
async def test_custom_provider_streaming(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = CustomJsonPathProvider(
        base_url="https://api.custom.ai",
        model="custom-v1",
        text_path="choices[0].delta.content",
        usage_prompt_path="usage.prompt",
        usage_completion_path="usage.completion",
    )

    sse_data = (
        'data: {"choices": [{"delta": {"content": "Hello "}}]}\n\n'
        'data: {"choices": [{"delta": {"content": "world!"}}]}\n\n'
        'data: {"usage": {"prompt": 12, "completion": 24}}\n\n'
        'data: [DONE]\n\n'
    )
    mock_resp = httpx.Response(
        status_code=200,
        headers={"content-type": "text/event-stream"},
        stream=MockAsyncStream([sse_data.encode("utf-8")]),
    )
    monkeypatch.setattr(provider.transport, "stream_post", AsyncMock(return_value=mock_resp))

    req = ChatRequest(
        model="custom-v1",
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
    assert done.usage.prompt_tokens == 12
    assert done.usage.completion_tokens == 24
