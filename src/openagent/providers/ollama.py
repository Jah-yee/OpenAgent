"""Ollama native REST API adapter.

Streams newline-delimited JSON (NDJSON) chunks from local Ollama /api/chat endpoint.
Supports tools, images, and model discovery.

Copyright 2026 The OpenAgent Contributors
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator, Mapping
from typing import Any

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
    ImagePart,
    Message,
    ModelInfo,
    TextPart,
    ToolCall,
    ToolProtocol,
    Usage,
)
from ..utils.http import HttpTransport


class OllamaProvider(ChatProvider):
    """Native adapter for local Ollama `/api/chat` service."""

    name = "ollama"
    tool_protocol = ToolProtocol.JSON_SCHEMA
    supports_vision = True
    supports_reasoning = True
    supports_native_tool_results = True
    supports_system_role = True
    default_context_window = 32_768

    def __init__(
        self,
        model: str,
        *,
        base_url: str = "http://localhost:11434",
        api_key: str | None = None,
        timeout: float = 600.0,
        max_retries: int = 3,
        extra_headers: Mapping[str, str] | None = None,
        context_window: int | None = None,
        client: Any = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        if self.base_url.endswith("/v1"):
            self.base_url = self.base_url[:-3]
        self.context_window = context_window or 32_768

        headers = dict(extra_headers or {})
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        self._transport = HttpTransport(
            self.base_url,
            provider=self.name,
            headers=headers,
            timeout=timeout,
            max_retries=max_retries,
            client=client,
        )

    @property
    def transport(self) -> HttpTransport:
        return self._transport

    # -- request rendering -------------------------------------------------- #

    def _render_message(self, msg: Message) -> dict[str, Any]:
        match msg.role:
            case "tool":
                return {
                    "role": "tool",
                    "content": msg.tool_text(),
                }
            case "assistant":
                out: dict[str, Any] = {
                    "role": "assistant",
                    "content": msg.text,
                }
                if msg.tool_calls:
                    out["tool_calls"] = [
                        {
                            "function": {
                                "name": c.name,
                                "arguments": c.arguments or {},
                            }
                        }
                        for c in msg.tool_calls
                    ]
                return out
            case "system":
                return {
                    "role": "system",
                    "content": msg.text,
                }
            case _:
                images = [p.data for p in msg.content if isinstance(p, ImagePart) and p.data]
                text_parts = [p.text for p in msg.content if isinstance(p, TextPart) and p.text]
                out = {
                    "role": "user",
                    "content": "\n".join(text_parts) if text_parts else msg.text,
                }
                if images:
                    out["images"] = images
                return out

    def build_payload(self, request: ChatRequest) -> dict[str, Any]:
        messages = [self._render_message(m) for m in request.messages]
        payload: dict[str, Any] = {
            "model": request.model,
            "messages": messages,
            "stream": True,
        }
        options: dict[str, Any] = {}
        if request.temperature is not None:
            options["temperature"] = request.temperature
        if request.top_p is not None:
            options["top_p"] = request.top_p
        if request.stop:
            options["stop"] = list(request.stop)
        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens

        if options:
            payload["options"] = options

        if request.tools:
            payload["tools"] = [spec.to_openai_schema() for spec in request.tools]

        return payload

    # -- streaming ---------------------------------------------------------- #

    async def stream(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        payload = self.build_payload(request)
        headers = {"accept": "application/x-ndjson, */*"}

        try:
            response = await self._transport.stream_post("/api/chat", payload, headers=headers)
        except ProviderError as exc:
            yield ErrorEvent(error=exc)
            return

        try:
            yield StartEvent(model=request.model, provider=self.name)

            text_parts: list[str] = []
            calls: list[ToolCall] = []
            usage = Usage()
            finish_reason = FinishReason.STOP
            done_received = False

            async for line in response.aiter_lines():
                line = line.strip()
                if not line:
                    continue

                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if "error" in chunk:
                    err = chunk["error"]
                    msg = err.get("message", "Unknown error") if isinstance(err, dict) else str(err)
                    yield ErrorEvent(error=ProviderError(msg, provider=self.name))
                    return

                msg_obj = chunk.get("message") or {}
                content = msg_obj.get("content")
                if content:
                    text_parts.append(content)
                    yield TextDelta(text=content, index=len(text_parts) - 1)

                tool_calls = msg_obj.get("tool_calls")
                if tool_calls and isinstance(tool_calls, list):
                    for tc in tool_calls:
                        fn = tc.get("function") or {}
                        call_name = fn.get("name", "")
                        raw_args = fn.get("arguments") or {}
                        call_id = tc.get("id") or f"call_{len(calls)}"
                        raw_args_str = (
                            json.dumps(raw_args) if isinstance(raw_args, (dict, list)) else str(raw_args)
                        )
                        parsed_args = coerce_arguments(raw_args)

                        yield ToolCallStart(index=len(calls), id=call_id, name=call_name)
                        yield ToolCallDelta(index=len(calls), arguments_delta=raw_args_str)
                        call = ToolCall(
                            id=call_id,
                            name=call_name,
                            arguments=parsed_args,
                            raw_arguments=raw_args_str,
                        )
                        yield ToolCallEnd(index=len(calls), call=call)
                        calls.append(call)

                if chunk.get("done"):
                    done_received = True
                    prompt_tokens = int(chunk.get("prompt_eval_count", 0))
                    completion_tokens = int(chunk.get("eval_count", 0))
                    usage = Usage(prompt_tokens=prompt_tokens, completion_tokens=completion_tokens)
                    yield UsageEvent(usage=usage)

                    done_reason = chunk.get("done_reason", "stop")
                    if calls:
                        finish_reason = FinishReason.TOOL_CALLS
                    elif done_reason == "length":
                        finish_reason = FinishReason.LENGTH
                    else:
                        finish_reason = FinishReason.STOP

                    yield DoneEvent(
                        finish_reason=finish_reason,
                        message=Message.assistant("".join(text_parts), tool_calls=calls),
                        usage=usage,
                    )
                    break

            if not done_received:
                if calls:
                    finish_reason = FinishReason.TOOL_CALLS
                yield DoneEvent(
                    finish_reason=finish_reason,
                    message=Message.assistant("".join(text_parts), tool_calls=calls),
                    usage=usage,
                )
        finally:
            await response.aclose()

    async def list_models(self) -> list[ModelInfo]:
        try:
            data = await self._transport.get_json("/api/tags")
            models = data.get("models", [])
            return [
                ModelInfo(
                    id=m.get("name", ""),
                    provider=self.name,
                )
                for m in models
                if isinstance(m, dict) and "name" in m
            ]
        except Exception:
            return []

    async def close(self) -> None:
        await self._transport.aclose()
