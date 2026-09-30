"""Tests for Server-Sent Events (SSE) parsing and decoding.

Copyright 2026 The OpenAgent Contributors
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import httpx
import pytest

from openagent.utils.sse import ErrorInfo, SSEvent, decode_sse_lines, iter_sse, parse_json_payload


class MockAsyncStream(httpx.AsyncByteStream):
    """Simulates an asynchronous HTTP byte stream."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = chunks

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for chunk in self._chunks:
            yield chunk


def make_mock_response(chunks: list[bytes]) -> httpx.Response:
    return httpx.Response(
        status_code=200,
        headers={"content-type": "text/event-stream"},
        stream=MockAsyncStream(chunks),
    )


def test_decode_sse_lines_basic() -> None:
    lines = [
        "event: custom",
        "id: 123",
        "data: hello world",
        "",
    ]
    events = list(decode_sse_lines(iter(lines)))
    assert len(events) == 1
    assert events[0] == SSEvent(event="custom", data="hello world", id="123")


def test_decode_sse_lines_multiline_data_and_comments() -> None:
    lines = [
        ": comment / keepalive",
        "data: first line",
        "data: second line",
        "",
        ": another comment",
        "data: third line",
        "",
    ]
    events = list(decode_sse_lines(iter(lines)))
    assert len(events) == 2
    assert events[0].data == "first line\nsecond line"
    assert events[1].data == "third line"


@pytest.mark.asyncio
async def test_iter_sse_async_stream() -> None:
    raw_payload = (
        b": keepalive\n\n"
        b"event: message\n"
        b"data: chunk 1\n\n"
        b"event: delta\n"
        b"data: chunk 2\n"
        b"id: 42\n\n"
        b"data: [DONE]\n\n"
    )
    # Split into multiple arbitrary chunks
    chunks = [raw_payload[:15], raw_payload[15:40], raw_payload[40:]]
    response = make_mock_response(chunks)

    events: list[SSEvent] = []
    async for event in iter_sse(response):
        events.append(event)

    assert len(events) == 3
    assert events[0] == SSEvent(event="message", data="chunk 1")
    assert events[1] == SSEvent(event="delta", data="chunk 2", id="42")
    assert events[2].data == "[DONE]"


def test_parse_json_payload() -> None:
    assert parse_json_payload('{"key": "value"}') == {"key": "value"}
    assert parse_json_payload('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_json_payload("") == {}


def test_error_info_from_body() -> None:
    info1 = ErrorInfo.from_body({"error": {"message": "rate limit", "type": "rate_limit_error"}})
    assert info1.message == "rate limit"
    assert info1.type == "rate_limit_error"

    info2 = ErrorInfo.from_body("raw error string")
    assert info2.message == "raw error string"


def test_decode_sse_lines_retry_and_trailing() -> None:
    lines = [
        "retry: 5000",
        "data: trailing event without newline",
    ]
    events = list(decode_sse_lines(iter(lines)))
    assert len(events) == 1
    assert events[0].retry == 5000
    assert events[0].data == "trailing event without newline"


def test_parse_json_payload_invalid() -> None:
    with pytest.raises(Exception) as exc_info:
        parse_json_payload("not valid json at all")
    assert "invalid JSON" in str(exc_info.value)
