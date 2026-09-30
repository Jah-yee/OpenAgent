"""The Core Agent Execution Loop.

Orchestrates multi-step agent reasoning, tool call execution, dynamic prompts,
context window management, streaming event delivery, and session persistence.

Copyright 2026 The OpenAgent Contributors
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from pathlib import Path

from openagent.context.compactor import Compactor
from openagent.context.messages import MessageManager
from openagent.core.events import (
    DoneEvent,
    ErrorEvent,
    StartEvent,
    StreamEvent,
    TextDelta,
    ThinkingDelta,
    ToolCallDelta,
    ToolCallEnd,
    ToolCallStart,
    ToolResultEvent,
    UsageEvent,
)
from openagent.core.provider import ChatProvider
from openagent.core.types import (
    ChatRequest,
    FinishReason,
    Message,
    TextPart,
    ToolCall,
    Usage,
)
from openagent.prompts.base import PromptBuilder
from openagent.session.store import SessionStore
from openagent.tools.registry import AskCallback, ToolRegistry

__all__ = ["AgentRunner"]


class AgentRunner:
    """Core orchestrator driving the turn-based agent execution loop."""

    def __init__(
        self,
        provider: ChatProvider,
        *,
        model: str | None = None,
        tools: ToolRegistry | None = None,
        messages: MessageManager | None = None,
        prompt_builder: PromptBuilder | None = None,
        compactor: Compactor | None = None,
        session_store: SessionStore | None = None,
        session_id: str | None = None,
        max_tool_iterations: int = 25,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float | None = None,
        workspace_root: str | Path | None = None,
        extra_instructions: str | Sequence[str] | None = None,
    ) -> None:
        self.provider = provider
        self.model = (
            model
            or getattr(provider, "model", None)
            or getattr(provider, "default_model", None)
            or "gpt-4o"
        )
        self.tools = tools if tools is not None else ToolRegistry()
        self.prompt_builder = prompt_builder if prompt_builder is not None else PromptBuilder()
        self.compactor = compactor if compactor is not None else Compactor()
        self.session_store = session_store
        self.session_id = session_id
        self.max_tool_iterations = max_tool_iterations
        self.max_tokens = (
            max_tokens
            if max_tokens is not None
            else getattr(provider, "context_window", None)
            or getattr(provider, "default_context_window", 128_000)
        )
        self.temperature = temperature
        self.top_p = top_p
        self.workspace_root = workspace_root
        self.extra_instructions = extra_instructions

        if messages is not None:
            self.messages = messages
        elif self.session_store is not None and self.session_id is not None:
            try:
                _, loaded_msgs = self.session_store.load_session(self.session_id)
                self.messages = MessageManager(loaded_msgs)
            except FileNotFoundError:
                self.messages = MessageManager()
        else:
            self.messages = MessageManager()

        self._ensure_system_prompt()

    def _ensure_system_prompt(self) -> None:
        """Ensure an initial system prompt is present in messages."""
        if self.messages.system_message is None and self.prompt_builder is not None:
            sys_text = self.prompt_builder.build_system_prompt(
                workspace_root=self.workspace_root,
                tools=self.tools,
                extra_instructions=self.extra_instructions,
            )
            self.messages.add_system(sys_text)

    def _persist_session(self) -> None:
        """Persist current conversation history to SessionStore if configured."""
        if self.session_store is not None:
            if self.session_id is None:
                self.session_id = self.session_store.create_session(model=self.model)
            self.session_store.save_messages(self.session_id, self.messages.messages)

    async def run_turn(
        self,
        user_input: str | Message,
        ask_callback: AskCallback | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Execute a conversational turn across streaming model calls and tools."""
        self._ensure_system_prompt()
        initial_msg_count = len(self.messages)

        user_msg = Message.user(user_input) if isinstance(user_input, str) else user_input
        self.messages.append(user_msg)

        cumulative_usage = Usage()

        for _ in range(self.max_tool_iterations):
            # Check token budget and invoke Compactor if budget is exceeded
            if (
                self.max_tokens is not None
                and self.compactor is not None
                and self.messages.total_tokens() > self.max_tokens
            ):
                compacted = await self.compactor.compact(
                    self.messages.messages,
                    self.max_tokens,
                    provider=self.provider,
                )
                self.messages.clear()
                self.messages.extend(compacted)

            # Window messages maintaining Atomic Tool Pair Invariant
            if self.max_tokens is not None:
                req_messages = self.messages.window(self.max_tokens)
            else:
                req_messages = self.messages.messages

            tool_specs = self.tools.list_specs() if self.tools else []
            request = ChatRequest(
                messages=req_messages,
                model=self.model,
                tools=tool_specs,
                temperature=self.temperature,
                top_p=self.top_p,
            )

            text_parts: list[str] = []
            reasoning_parts: list[str] = []
            calls: list[ToolCall] = []
            done_event: DoneEvent | None = None
            error_encountered: ErrorEvent | None = None
            step_usage = Usage()

            try:
                async for event in self.provider.stream(request):
                    match event:
                        case TextDelta(text=t):
                            text_parts.append(t)
                            yield event
                        case ThinkingDelta(text=t):
                            reasoning_parts.append(t)
                            yield event
                        case ToolCallStart() | ToolCallDelta():
                            yield event
                        case ToolCallEnd(call=c):
                            if c is not None:
                                calls.append(c)
                            yield event
                        case UsageEvent(usage=u):
                            step_usage = step_usage + u
                            yield event
                        case DoneEvent() as done:
                            done_event = done
                        case ErrorEvent() as err:
                            error_encountered = err
                            yield event
                        case StartEvent():
                            yield event
                        case _:
                            yield event
            except Exception as exc:
                err_ev = ErrorEvent(error=exc)
                yield err_ev
                while len(self.messages) > initial_msg_count:
                    self.messages.pop()
                return

            if error_encountered is not None:
                while len(self.messages) > initial_msg_count:
                    self.messages.pop()
                return

            if done_event is not None and done_event.usage.total_tokens > 0:
                cumulative_usage = cumulative_usage + done_event.usage
            else:
                cumulative_usage = cumulative_usage + step_usage

            # Record assistant turn in messages
            if done_event is not None and done_event.message is not None:
                assistant_msg = done_event.message
                if calls and not assistant_msg.tool_calls:
                    assistant_msg.tool_calls = list(calls)
                if not assistant_msg.text and text_parts:
                    assistant_msg.content = [TextPart("".join(text_parts))]
                if not assistant_msg.reasoning and reasoning_parts:
                    assistant_msg.reasoning = "".join(reasoning_parts)
            else:
                assistant_msg = Message.assistant(
                    text="".join(text_parts) if text_parts else None,
                    tool_calls=calls,
                    reasoning="".join(reasoning_parts) if reasoning_parts else None,
                )

            self.messages.append(assistant_msg)

            # If no tool calls: yield DoneEvent, persist session, and end turn
            if not assistant_msg.tool_calls:
                final_done = (
                    done_event
                    if done_event is not None
                    else DoneEvent(
                        finish_reason=FinishReason.STOP,
                        message=assistant_msg,
                        usage=cumulative_usage,
                    )
                )
                final_done.message = assistant_msg
                final_done.usage = cumulative_usage
                yield final_done
                self._persist_session()
                return

            # Tool calls present: execute tools concurrently via ToolRegistry
            tool_results = await self.tools.execute_parallel(
                assistant_msg.tool_calls,
                ask_callback=ask_callback,
            )

            for call, result in zip(assistant_msg.tool_calls, tool_results, strict=True):
                yield ToolResultEvent(
                    call_id=result.call_id,
                    tool_name=call.name,
                    output=result.output,
                    is_error=result.is_error,
                )
                tool_msg = result.to_tool_message(name=call.name)
                self.messages.append(tool_msg)

        else:
            # Reached max_tool_iterations without model concluding
            terminal_event = DoneEvent(
                finish_reason=FinishReason.LENGTH,
                message=Message.assistant("Reached maximum tool call iterations without concluding."),
                usage=cumulative_usage,
            )
            yield terminal_event
            self._persist_session()

    async def run_turn_to_completion(
        self,
        user_input: str | Message,
        ask_callback: AskCallback | None = None,
    ) -> str:
        """Run a turn to completion and return the final text response."""
        text_chunks: list[str] = []
        done_text: str = ""
        async for event in self.run_turn(user_input, ask_callback=ask_callback):
            if isinstance(event, TextDelta):
                text_chunks.append(event.text)
            elif isinstance(event, DoneEvent) and event.message and event.message.text:
                done_text = event.message.text

        if done_text.strip():
            return done_text
        return "".join(text_chunks)

    def reset(self) -> None:
        """Clear active turn state while preserving or re-initializing configuration."""
        self.messages.clear()
        self.session_id = None
        self._ensure_system_prompt()
