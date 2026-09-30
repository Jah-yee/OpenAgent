"""Custom JSONPath HTTP provider adapter.

Extracts text deltas, tool calls, and token usage from arbitrary HTTP JSON / SSE / NDJSON
APIs using configurable jsonpath-ng expressions.

Copyright 2026 The OpenAgent Contributors
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from typing import Any

import jsonpath_ng

from ..core.events import (
    DoneEvent,
    ErrorEvent,
    StartEvent,
    StreamEvent,
    TextDelta,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
    UsageEvent,
)
from ..core.provider import (
    ChatProvider,
    ProviderError,
    coerce_arguments,
)
from ..core.types import (
    ChatRequest,
    FinishReason,
    Message,
    ModelInfo,
    ToolCall,
    ToolProtocol,
    Usage,
)
from ..utils.http import HttpTransport
from ..utils.sse import iter_sse


class CustomJsonPathProvider(ChatProvider):
    """Adapter for arbitrary JSON APIs using jsonpath-ng expressions."""

    name = "custom"
    tool_protocol = ToolProtocol.JSON_SCHEMA
    supports_vision = False
    supports_reasoning = True
    supports_native_tool_results = True
    supports_system_role = True
    default_context_window = 128_000

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        text_path: str | None = "choices[0].delta.content",
        tool_calls_path: str | None = None,
        usage_prompt_path: str | None = None,
        usage_completion_path: str | None = None,
        chat_endpoint: str = "/chat/completions",
        api_key: str | None = None,
        auth_header: str = "authorization",
        auth_prefix: str = "Bearer ",
        headers: Mapping[str, str] | None = None,
        timeout: float = 600.0,
        max_retries: int = 3,
        context_window: int | None = None,
        client: Any = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.chat_endpoint = chat_endpoint
        self.context_window = context_window or 128_000

        self.text_path = text_path
        self.tool_calls_path = tool_calls_path
        self.usage_prompt_path = usage_prompt_path
        self.usage_completion_path = usage_completion_path

        self._text_expr = jsonpath_ng.parse(text_path) if text_path else None
        self._tool_calls_expr = jsonpath_ng.parse(tool_calls_path) if tool_calls_path else None
        self._usage_prompt_expr = jsonpath_ng.parse(usage_prompt_path) if usage_prompt_path else None
        self._usage_completion_expr = (
            jsonpath_ng.parse(usage_completion_path) if usage_completion_path else None
        )

        request_headers = dict(headers or {})
        if api_key:
            header_name = auth_header.lower()
            if header_name == "authorization":
                request_headers["Authorization"] = f"{auth_prefix}{api_key}"
            else:
                request_headers[header_name] = api_key

        self._transport = HttpTransport(
            self.base_url,
            provider=self.name,
            headers=request_headers,
            timeout=timeout,
            max_retries=max_retries,
            client=client,
        )

    @property
    def transport(self) -> HttpTransport:
        return self._transport

    # -- chunk parsing ------------------------------------------------------ #

    def parse_chunk(self, chunk: dict[str, Any]) -> list[StreamEvent]:
        """Tolerantly extract events from a parsed JSON chunk."""
        events: list[StreamEvent] = []

        # 1. Text deltas
        if self._text_expr:
            try:
                matches = self._text_expr.find(chunk)
                for m in matches:
                    if m.value is not None:
                        val = str(m.value)
                        if val:
                            events.append(TextDelta(text=val))
            except Exception:
                pass

        # 2. Tool calls
        if self._tool_calls_expr:
            try:
                matches = self._tool_calls_expr.find(chunk)
                for m in matches:
                    val = m.value
                    if isinstance(val, dict):
                        calls = [val]
                    elif isinstance(val, list):
                        calls = [c for c in val if isinstance(c, dict)]
                    else:
                        continue

                    for idx, c in enumerate(calls):
                        fn = c.get("function")
                        if isinstance(fn, dict):
                            call_name = fn.get("name", "")
                            call_args = fn.get("arguments") or {}
                        else:
                            call_name = c.get("name", "")
                            call_args = c.get("arguments") or c.get("args") or {}

                        call_id = c.get("id") or f"call_{idx}"
                        raw_args = (
                            json.dumps(call_args)
                            if isinstance(call_args, (dict, list))
                            else str(call_args)
                        )
                        parsed_args = coerce_arguments(call_args)

                        tool_index = c.get("index", idx)
                        events.append(ToolCallStart(index=tool_index, id=call_id, name=call_name))
                        events.append(ToolCallDelta(index=tool_index, arguments_delta=raw_args))
                        tc = ToolCall(
                            id=call_id,
                            name=call_name,
                            arguments=parsed_args,
                            raw_arguments=raw_args,
                        )
                        events.append(ToolCallEnd(index=tool_index, call=tc))
            except Exception:
                pass

        # 3. Usage
        prompt_tokens: int | None = None
        completion_tokens: int | None = None

        if self._usage_prompt_expr:
            try:
                matches = self._usage_prompt_expr.find(chunk)
                if matches and matches[0].value is not None:
                    prompt_tokens = int(matches[0].value)
            except Exception:
                pass

        if self._usage_completion_expr:
            try:
                matches = self._usage_completion_expr.find(chunk)
                if matches and matches[0].value is not None:
                    completion_tokens = int(matches[0].value)
            except Exception:
                pass

        if prompt_tokens is not None or completion_tokens is not None:
            events.append(
                UsageEvent(
                    usage=Usage(
                        prompt_tokens=prompt_tokens or 0,
                        completion_tokens=completion_tokens or 0,
                    )
                )
            )

        return events

    # -- streaming ---------------------------------------------------------- #

    def build_payload(self, request: ChatRequest) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": [{"role": m.role, "content": m.text} for m in request.messages],
            "stream": True,
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.top_p is not None:
            payload["top_p"] = request.top_p
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.tools:
            payload["tools"] = [t.to_openai_schema() for t in request.tools]
        return payload

    async def stream(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        payload = self.build_payload(request)
        try:
            response = await self._transport.stream_post(self.chat_endpoint, payload)
        except ProviderError as exc:
            yield ErrorEvent(error=exc)
            return

        try:
            yield StartEvent(model=request.model, provider=self.name)

            text_parts: list[str] = []
            calls: list[ToolCall] = []
            usage = Usage()

            content_type = response.headers.get("content-type", "").lower()
            is_sse = "text/event-stream" in content_type

            if is_sse:
                async for event in iter_sse(response):
                    if event.data == "[DONE]":
                        break
                    if not event.data.strip():
                        continue
                    try:
                        chunk = json.loads(event.data)
                    except json.JSONDecodeError:
                        continue
                    for ev in self.parse_chunk(chunk):
                        match ev:
                            case TextDelta(text=t):
                                text_parts.append(t)
                                yield ev
                            case ToolCallEnd(call=c) if c is not None:
                                calls.append(c)
                                yield ev
                            case UsageEvent(usage=u):
                                usage = usage + u
                                yield ev
                            case _:
                                yield ev
            else:
                async for line in response.aiter_lines():
                    line = line.strip()
                    if not line:
                        continue
                    if line.startswith("data:"):
                        line = line[5:].strip()
                        if line == "[DONE]":
                            break
                    try:
                        chunk = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    for ev in self.parse_chunk(chunk):
                        match ev:
                            case TextDelta(text=t):
                                text_parts.append(t)
                                yield ev
                            case ToolCallEnd(call=c) if c is not None:
                                calls.append(c)
                                yield ev
                            case UsageEvent(usage=u):
                                usage = usage + u
                                yield ev
                            case _:
                                yield ev

            finish = FinishReason.TOOL_CALLS if calls else FinishReason.STOP
            yield DoneEvent(
                finish_reason=finish,
                message=Message.assistant("".join(text_parts), tool_calls=calls),
                usage=usage,
            )
        finally:
            await response.aclose()

    async def list_models(self) -> list[ModelInfo]:
        return []

    async def close(self) -> None:
        await self._transport.aclose()
