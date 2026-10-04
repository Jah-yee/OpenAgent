"""Tests for Provider implementations and Text Protocol Codec.

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
    ThinkingDelta,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
    UsageEvent,
)
from openagent.core.provider import (
    AuthenticationError,
    ContextWindowError,
    ModelNotFoundError,
    RateLimitError,
)
from openagent.core.types import (
    ChatRequest,
    FinishReason,
    Message,
    ToolParam,
    ToolProtocol,
    ToolSpec,
)
from openagent.providers.anthropic import AnthropicProvider
from openagent.providers.openai_compat import OpenAICompatProvider
from openagent.providers.text_protocol import TextToolCodec
from openagent.utils.http import HttpTransport


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


def test_text_tool_codec_render_and_parse() -> None:
    codec = TextToolCodec()
    req = ChatRequest(
        model="test-model",
        messages=[Message.user("Hello")],
        tools=[
            ToolSpec(
                name="read_file",
                description="Read contents of a file.",
                params=[
                    ToolParam(name="path", type="string", description="File path", required=True),
                ],
            )
        ],
    )
    system = codec.render(req)
    assert "read_file" in system
    assert "```tool_code" in system

    raw_response = (
        "Let me inspect that.\n\n"
        "```tool_code\n"
        '{"name": "read_file", "arguments": {"path": "main.py"}}\n'
        "```\n"
        "Done."
    )
    calls, remaining = codec.parse(raw_response)
    assert len(calls) == 1
    assert calls[0].name == "read_file"
    assert calls[0].arguments == {"path": "main.py"}
    assert "Let me inspect that." in remaining
    assert "Done." in remaining


def test_text_tool_codec_streaming_parser() -> None:
    codec = TextToolCodec()
    parser = codec.parse_stream()

    chunk1 = "Thinking...\n```tool_code\n"
    chunk2 = '{"name": "test_tool", "arguments": {"x": 1}}\n```'
    chunk3 = "\nFinished."

    parser.feed(chunk1)
    parser.feed(chunk2)
    parser.feed(chunk3)

    calls, remaining = parser.finish()
    assert len(calls) == 1
    assert calls[0].name == "test_tool"
    assert calls[0].arguments == {"x": 1}
    assert "Thinking..." in remaining
    assert "Finished." in remaining


def test_text_tool_codec_line_syntax() -> None:
    codec = TextToolCodec()
    raw = (
        "<<<TOOL>>>\n"
        "search_code\n"
        '{"query": "def foo"}\n\n'
        "Some remaining text"
    )
    calls, remaining = codec.parse(raw)
    assert len(calls) == 1
    assert calls[0].name == "search_code"
    assert calls[0].arguments == {"query": "def foo"}
    assert "Some remaining text" in remaining


@pytest.mark.asyncio
async def test_openai_compat_provider_streaming(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = OpenAICompatProvider(
        base_url="https://api.openai.com/v1",
        model="gpt-4o",
        api_key="test-key",
    )

    sse_data = [
        'data: {"choices": [{"delta": {"reasoning_content": "plan steps"}}]}',
        'data: {"choices": [{"delta": {"content": "Hello "}}]}',
        'data: {"choices": [{"delta": {"content": "world"}}]}',
        (
            'data: {"choices": [{"delta": {"tool_calls": ['
            '{"index": 0, "id": "call_1", "function": {"name": "read_file", "arguments": "{\\"path\\": "}}'
            ']}}]}'
        ),
        (
            'data: {"choices": [{"delta": {"tool_calls": ['
            '{"index": 0, "function": {"arguments": "\\"a.txt\\"}"}}'
            ']}}]}'
        ),
        (
            'data: {"choices": [{"finish_reason": "tool_calls"}], '
            '"usage": {"prompt_tokens": 10, "completion_tokens": 20}}'
        ),
        'data: [DONE]',
    ]
    mock_resp = make_sse_response(sse_data)
    monkeypatch.setattr(provider.transport, "stream_post", AsyncMock(return_value=mock_resp))

    req = ChatRequest(
        model="gpt-4o",
        messages=[Message.user("Hi")],
    )

    events = []
    async for event in provider.stream(req):
        events.append(event)

    types = [type(e) for e in events]
    assert StartEvent in types
    assert ThinkingDelta in types
    assert TextDelta in types
    assert ToolCallStart in types
    assert ToolCallDelta in types
    assert ToolCallEnd in types
    assert DoneEvent in types

    done = next(e for e in events if isinstance(e, DoneEvent))
    assert done.finish_reason == FinishReason.TOOL_CALLS
    assert done.usage.prompt_tokens == 10
    assert done.usage.completion_tokens == 20
    assert done.message is not None
    assert len(done.message.tool_calls) == 1
    assert done.message.tool_calls[0].name == "read_file"
    assert done.message.tool_calls[0].arguments == {"path": "a.txt"}
    assert mock_resp.is_closed


@pytest.mark.asyncio
async def test_anthropic_provider_streaming(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = AnthropicProvider(
        model="claude-3-5-sonnet",
        api_key="test-key",
    )

    sse_data = [
        'event: message_start\ndata: {"type": "message_start", "message": {"usage": {"input_tokens": 50, "output_tokens": 5}}}',
        'event: content_block_start\ndata: {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking"}}',
        'event: content_block_delta\ndata: {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "Let me think"}}',
        'event: content_block_stop\ndata: {"type": "content_block_stop", "index": 0}',
        'event: content_block_start\ndata: {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}}',
        'event: content_block_delta\ndata: {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Hello there"}}',
        'event: content_block_stop\ndata: {"type": "content_block_stop", "index": 1}',
        'event: message_delta\ndata: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 15}}',
        'event: message_stop\ndata: {"type": "message_stop"}',
    ]
    mock_resp = make_sse_response(sse_data)
    monkeypatch.setattr(provider._transport, "stream_post", AsyncMock(return_value=mock_resp))

    req = ChatRequest(
        model="claude-3-5-sonnet",
        messages=[Message.user("Hi")],
    )

    events = []
    async for event in provider.stream(req):
        events.append(event)

    types = [type(e) for e in events]
    assert StartEvent in types
    assert ThinkingDelta in types
    assert TextDelta in types
    assert UsageEvent in types
    assert DoneEvent in types

    done = next(e for e in events if isinstance(e, DoneEvent))
    assert done.finish_reason == FinishReason.STOP
    assert done.usage.prompt_tokens == 50
    assert done.usage.completion_tokens == 15
    assert done.message is not None
    assert done.message.text == "Hello there"
    assert done.message.reasoning == "Let me think"
    assert mock_resp.is_closed


@pytest.mark.asyncio
async def test_anthropic_provider_tool_calls_streaming(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = AnthropicProvider(
        model="claude-3-5-sonnet",
        api_key="test-key",
    )

    sse_data = [
        'event: message_start\ndata: {"type": "message_start", "message": {"usage": {"input_tokens": 100, "output_tokens": 10}}}',
        'event: content_block_start\ndata: {"type": "content_block_start", "index": 0, "content_block": {"type": "tool_use", "id": "call_123", "name": "edit_file"}}',
        'event: content_block_delta\ndata: {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": "{\\"path\\": \\"test.txt\\""}}',
        'event: content_block_delta\ndata: {"type": "content_block_delta", "index": 0, "delta": {"type": "input_json_delta", "partial_json": ", \\"content\\": \\"hello\\"}"}}',
        'event: content_block_stop\ndata: {"type": "content_block_stop", "index": 0}',
        'event: message_delta\ndata: {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 25}}',
        'event: message_stop\ndata: {"type": "message_stop"}',
    ]
    mock_resp = make_sse_response(sse_data)
    monkeypatch.setattr(provider._transport, "stream_post", AsyncMock(return_value=mock_resp))

    req = ChatRequest(
        model="claude-3-5-sonnet",
        messages=[Message.user("Edit file")],
    )

    events = []
    async for event in provider.stream(req):
        events.append(event)

    types = [type(e) for e in events]
    assert ToolCallStart in types
    assert ToolCallDelta in types
    assert ToolCallEnd in types
    assert DoneEvent in types

    deltas = [e for e in events if isinstance(e, ToolCallDelta)]
    assert len(deltas) == 2
    assert deltas[0].arguments_delta == '{"path": "test.txt"'
    assert deltas[1].arguments_delta == ', "content": "hello"}'

    ends = [e for e in events if isinstance(e, ToolCallEnd)]
    assert len(ends) == 1
    assert ends[0].call is not None
    assert ends[0].call.id == "call_123"
    assert ends[0].call.name == "edit_file"
    assert ends[0].call.arguments == {"path": "test.txt", "content": "hello"}

    done = next(e for e in events if isinstance(e, DoneEvent))
    assert done.finish_reason == FinishReason.TOOL_CALLS
    assert done.usage.completion_tokens == 25
    assert done.message is not None
    assert len(done.message.tool_calls) == 1
    assert done.message.tool_calls[0].name == "edit_file"
    assert mock_resp.is_closed


@pytest.mark.asyncio
async def test_http_transport_post_stream_and_errors() -> None:
    mock_client = AsyncMock()
    mock_client.build_request = lambda method, url, **kwargs: httpx.Request(method, url)

    # Success case for post_stream
    mock_res = httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=MockAsyncStream([b"data: ok\n\n"]))
    mock_client.send = AsyncMock(return_value=mock_res)
    transport = HttpTransport("https://api.example.com", provider="test-provider", client=mock_client)

    res = await transport.post_stream("/test", json={"foo": "bar"})
    assert res.status_code == 200

    # 401 Unauthorized -> AuthenticationError
    err_res_401 = httpx.Response(401, text='{"error": {"message": "invalid api key"}}')
    mock_client.send = AsyncMock(return_value=err_res_401)
    with pytest.raises(AuthenticationError):
        await transport.post_stream("/test", json={})

    # 429 RateLimitError
    err_res_429 = httpx.Response(429, text='{"error": {"message": "rate limit reached"}}')
    mock_client.send = AsyncMock(return_value=err_res_429)
    with pytest.raises(RateLimitError):
        await transport.post_stream("/test", json={})

    # 404 ModelNotFoundError
    err_res_404 = httpx.Response(404, text='{"error": {"message": "model not found"}}')
    mock_client.send = AsyncMock(return_value=err_res_404)
    with pytest.raises(ModelNotFoundError):
        await transport.post_stream("/test", json={})

    # 400 ContextWindowError
    err_res_400 = httpx.Response(400, text='{"error": {"message": "context length exceeded"}}')
    mock_client.send = AsyncMock(return_value=err_res_400)
    with pytest.raises(ContextWindowError):
        await transport.post_stream("/test", json={})


@pytest.mark.asyncio
async def test_openai_compat_text_protocol(monkeypatch: pytest.MonkeyPatch) -> None:
    provider = OpenAICompatProvider(
        base_url="https://api.openai.com/v1",
        model="gpt-4o",
        api_key="test-key",
    )
    provider.tool_protocol = ToolProtocol.TEXT

    sse_data = [
        'data: {"choices": [{"delta": {"content": "Checking file:\\n```tool_code\\n"}}]}',
        'data: {"choices": [{"delta": {"content": "{\\"name\\": \\"read_file\\", \\"arguments\\": {\\"path\\": \\"test.py\\"}}\\n```"}}]}',
        'data: {"choices": [{"delta": {"content": "\\nAll done."}}]}',
        'data: {"choices": [{"finish_reason": "stop"}]}',
        'data: [DONE]',
    ]
    mock_resp = make_sse_response(sse_data)
    monkeypatch.setattr(provider.transport, "stream_post", AsyncMock(return_value=mock_resp))

    req = ChatRequest(
        model="gpt-4o",
        messages=[Message.user("Check test.py")],
        tools=[
            ToolSpec(
                name="read_file",
                description="Read a file",
                params=[ToolParam(name="path", type="string", description="path")],
            )
        ],
    )

    events = []
    async for event in provider.stream(req):
        events.append(event)

    types = [type(e) for e in events]
    assert ToolCallEnd in types
    assert DoneEvent in types

    tool_ends = [e for e in events if isinstance(e, ToolCallEnd)]
    assert len(tool_ends) == 1
    assert tool_ends[0].call is not None
    assert tool_ends[0].call.name == "read_file"
    assert tool_ends[0].call.arguments == {"path": "test.py"}

    done = next(e for e in events if isinstance(e, DoneEvent))
    assert done.message is not None
    assert len(done.message.tool_calls) == 1
    assert done.message.tool_calls[0].name == "read_file"
    assert "Checking file:" in done.message.text
    assert "All done." in done.message.text
    assert mock_resp.is_closed


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("https://api.openai.com", "https://api.openai.com/v1"),
        ("https://api.openai.com/", "https://api.openai.com/v1"),
        ("https://api.openai.com/v1", "https://api.openai.com/v1"),
        (
            "https://api.openai.com/v1/chat/completions",
            "https://api.openai.com/v1",
        ),
        ("http://localhost:11434", "http://localhost:11434/v1"),
        ("http://localhost:11434/v1", "http://localhost:11434/v1"),
        ("https://api.deepseek.com/v1/chat/completions", "https://api.deepseek.com/v1"),
    ],
)
def test_normalise_base_url(raw: str, expected: str) -> None:
    """Test that _normalise_base_url correctly handles api.openai.com.

    Regression test for: https://github.com/mj10612/OpenAgent/issues/14
    The /v1 suffix must be appended to api.openai.com bare hosts.
    """
    assert OpenAICompatProvider._normalise_base_url(raw) == expected
