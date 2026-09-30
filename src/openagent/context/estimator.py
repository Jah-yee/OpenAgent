"""Token estimation engine for text, messages, tool calls, tool specs, and requests.

Fast rule-based token counter without heavyweight external dependencies.
Accounts for per-message framing overhead, tool definition schema overhead,
image parts, and non-ASCII / CJK language variations.

Copyright 2026 The OpenAgent Contributors
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from typing import Any

from openagent.core.types import (
    ChatRequest,
    ContentPart,
    ImagePart,
    Message,
    TextPart,
    ThinkingPart,
    ToolCall,
    ToolResultPart,
    ToolSpec,
)


class TokenEstimator:
    """Rule-based token counter for estimating conversation and request budgets."""

    def __init__(
        self,
        *,
        chars_per_token: float = 4.0,
        cjk_factor: float = 1.2,
        message_overhead: int = 4,
        tool_call_overhead: int = 8,
        tool_spec_overhead: int = 16,
        request_overhead: int = 3,
        image_low_tokens: int = 85,
        image_high_tokens: int = 800,
    ) -> None:
        self.chars_per_token = chars_per_token
        self.cjk_factor = cjk_factor
        self.message_overhead = message_overhead
        self.tool_call_overhead = tool_call_overhead
        self.tool_spec_overhead = tool_spec_overhead
        self.request_overhead = request_overhead
        self.image_low_tokens = image_low_tokens
        self.image_high_tokens = image_high_tokens

    def estimate_text(self, text: str | None) -> int:
        """Estimate token count for a text string.

        Accounts for CJK (Chinese, Japanese, Korean) characters which have higher
        token density (~1-1.5 tokens per character) compared to Latin text
        (~4 characters per token).
        """
        if not text:
            return 0

        cjk_count = 0
        ascii_count = 0
        other_count = 0

        for char in text:
            cp = ord(char)
            # CJK Unified Ideographs, Extension A, Hiragana, Katakana, Hangul Syllables, Hangul Jamo
            if (
                0x4E00 <= cp <= 0x9FFF
                or 0x3400 <= cp <= 0x4DBF
                or 0x3040 <= cp <= 0x309F
                or 0x30A0 <= cp <= 0x30FF
                or 0xAC00 <= cp <= 0xD7AF
                or 0x1100 <= cp <= 0x11FF
            ):
                cjk_count += 1
            elif cp < 128:
                ascii_count += 1
            else:
                other_count += 1

        tokens = (
            (ascii_count / self.chars_per_token)
            + (cjk_count * self.cjk_factor)
            + (other_count / 2.0)
        )
        return max(1, math.ceil(tokens))

    def estimate_part(self, part: ContentPart) -> int:
        """Estimate token count for a single content part."""
        match part:
            case TextPart(text=t):
                return self.estimate_text(t)
            case ThinkingPart(text=t):
                # Thinking tags and framing overhead (~4 tokens) + text
                return self.estimate_text(t) + 4
            case ImagePart(detail=det):
                return self.image_low_tokens if det == "low" else self.image_high_tokens
            case ToolResultPart(content=c, name=n):
                # Framing overhead + content + name
                return self.estimate_text(c) + self.estimate_text(n) + 4
            case _:
                return 0

    def estimate_tool_call(self, call: ToolCall) -> int:
        """Estimate token count for an assistant tool invocation."""
        name_tokens = self.estimate_text(call.name)
        if call.raw_arguments:
            args_tokens = self.estimate_text(call.raw_arguments)
        elif call.arguments:
            try:
                args_json = json.dumps(call.arguments, ensure_ascii=False)
            except Exception:
                args_json = str(call.arguments)
            args_tokens = self.estimate_text(args_json)
        else:
            args_tokens = 0

        return name_tokens + args_tokens + self.tool_call_overhead

    def estimate_message(self, msg: Message) -> int:
        """Estimate token count for a conversation message.

        Includes role framing overhead, all content parts, tool calls, and reasoning.
        """
        tokens = self.message_overhead
        if msg.name:
            tokens += self.estimate_text(msg.name)
        if msg.tool_call_id:
            tokens += self.estimate_text(msg.tool_call_id)
        if msg.reasoning:
            tokens += self.estimate_text(msg.reasoning) + 4

        for part in msg.content:
            tokens += self.estimate_part(part)

        for call in msg.tool_calls:
            tokens += self.estimate_tool_call(call)

        return tokens

    def estimate_messages(self, messages: Iterable[Message]) -> int:
        """Estimate total tokens for a sequence of messages."""
        return sum(self.estimate_message(m) for m in messages)

    def estimate_tool_spec(self, spec: ToolSpec) -> int:
        """Estimate token overhead of providing a tool specification schema to the model."""
        schema_dict = spec.to_openai_schema()
        schema_json = json.dumps(schema_dict, ensure_ascii=False)
        return self.estimate_text(schema_json) + self.tool_spec_overhead

    def estimate_tools(self, tools: Iterable[ToolSpec]) -> int:
        """Estimate total tokens for a collection of tool specifications."""
        return sum(self.estimate_tool_spec(t) for t in tools)

    def estimate_request(self, req: ChatRequest) -> int:
        """Estimate total input tokens for a complete ChatRequest."""
        tokens = self.request_overhead
        tokens += self.estimate_messages(req.messages)
        tokens += self.estimate_tools(req.tools)
        return tokens

    def estimate(self, item: Any) -> int:
        """Polymorphic estimation helper."""
        if isinstance(item, str):
            return self.estimate_text(item)
        if isinstance(item, ContentPart):
            return self.estimate_part(item)
        if isinstance(item, Message):
            return self.estimate_message(item)
        if isinstance(item, ToolCall):
            return self.estimate_tool_call(item)
        if isinstance(item, ToolSpec):
            return self.estimate_tool_spec(item)
        if isinstance(item, ChatRequest):
            return self.estimate_request(item)
        if isinstance(item, Sequence):
            if all(isinstance(x, Message) for x in item):
                return self.estimate_messages(item)
            if all(isinstance(x, ToolSpec) for x in item):
                return self.estimate_tools(item)
            return sum(self.estimate(x) for x in item)
        return 0
