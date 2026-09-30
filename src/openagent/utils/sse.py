"""Server-Sent Events parsing and tolerant JSON decoding.

Streaming is the part of the stack most likely to be fed malformed bytes, so
everything here is deliberately forgiving: a truncated line is emitted rather
than dropped, and unknown event types are ignored rather than fatal.

Copyright 2026 The OpenAgent Contributors
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Iterator
from dataclasses import dataclass, field
from typing import Any

import httpx

from ..core.provider import ProviderError


@dataclass(slots=True)
class SSEvent:
    """A single decoded Server-Sent Event."""

    event: str = "message"
    data: str = ""
    id: str | None = None
    retry: int | None = None


def decode_sse_lines(lines: Iterator[str]) -> Iterator[SSEvent]:
    """Fold a stream of raw lines into events.

    Spec-driven: fields are ``field: value``, a blank line dispatches the
    buffer, and multiple ``data:`` lines are joined with newlines per the SSE
    specification.
    """
    event_name = "message"
    data_lines: list[str] = []
    event_id: str | None = None
    retry: int | None = None

    def flush() -> SSEvent | None:
        nonlocal event_name, data_lines, event_id, retry
        if not data_lines and event_name == "message" and event_id is None:
            return None
        evt = SSEvent(
            event=event_name,
            data="\n".join(data_lines),
            id=event_id,
            retry=retry,
        )
        event_name = "message"
        data_lines = []
        event_id = None
        retry = None
        return evt

    for raw in lines:
        line = raw.rstrip("\n").rstrip("\r")

        if not line:
            # Blank line: dispatch. Comment-only keep-alives produce nothing.
            evt = flush()
            if evt is not None and evt.data.strip():
                yield evt
            continue

        if line.startswith(":"):
            # Comment / heartbeat (": ping").
            continue

        if ":" not in line:
            # A bare field name with no value is legal but useless.
            continue

        field_name, _, value = line.partition(":")
        if value.startswith(" "):
            value = value[1:]

        match field_name:
            case "event":
                event_name = value
            case "data":
                data_lines.append(value)
            case "id":
                event_id = value
            case "retry":
                try:
                    retry = int(value)
                except ValueError:
                    retry = None
            case _:
                continue

    # Flush a trailing event that arrived without a final blank line.
    evt = flush()
    if evt is not None and evt.data.strip():
        yield evt


async def iter_sse(response: httpx.Response) -> AsyncIterator[SSEvent]:
    """Decode an open streaming HTTP response into events.

    ``response.aiter_lines`` is used rather than ``iter_bytes`` so the framing
    is handled by the HTTP layer rather than by us.
    """
    async for event in decode_sse_lines(_async_to_sync_lines(response)):
        yield event


async def _async_to_sync_lines(response: httpx.Response) -> AsyncIterator[str]:
    async for line in response.aiter_lines():
        yield line


def parse_json_payload(text: str, *, provider: str = "") -> Any:
    """Decode a JSON body, mapping failures to a provider error."""
    text = text.strip()
    if not text:
        return {}
    try:
        return json.loads(text)
    except json.JSONDecodeError as exc:
        # Some gateways double-encode or wrap payloads in ```json fences.
        cleaned = text.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            snippet = text[:200]
            raise ProviderError(
                f"invalid JSON in response: {exc}; body starts with: {snippet!r}",
                provider=provider,
            ) from exc


@dataclass(slots=True)
class ErrorInfo:
    """Normalised error extracted from a provider response body."""

    message: str
    type: str = ""
    code: str = ""
    raw: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_body(cls, body: Any, *, provider: str = "") -> ErrorInfo:
        """Dig a message out of the many error shapes in circulation.

        Recognises OpenAI (``error.message``), Anthropic, Gemini
        (``error.message``), Ollama (``error``), and a bare string body.
        """
        if isinstance(body, str):
            return cls(message=body or "unknown error")

        if not isinstance(body, dict):
            return cls(message=str(body))

        err = body.get("error", body.get("detail", body.get("message")))
        if isinstance(err, str):
            return cls(message=err, raw=body)
        if isinstance(err, dict):
            return cls(
                message=str(
                    err.get("message")
                    or err.get("detail")
                    or err.get("status")
                    or err
                ),
                type=str(err.get("type", "")),
                code=str(err.get("code", "")),
                raw=body,
            )

        # Gemini: {"error": {"message": ...}} is covered above; Ollama: {"error": "..."}.
        return cls(message=json.dumps(body)[:500], raw=body)