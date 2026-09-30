"""Tests for Context Management: TokenEstimator, MessageManager, and Compactor.

Copyright 2026 The OpenAgent Contributors
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

from collections.abc import AsyncIterator

import pytest

from openagent.context import Compactor, MessageManager, TokenEstimator
from openagent.core.events import StreamEvent
from openagent.core.provider import ChatProvider
from openagent.core.types import (
    ChatRequest,
    ChatResponse,
    ImagePart,
    Message,
    TextPart,
    ThinkingPart,
    ToolCall,
    ToolParam,
    ToolResultPart,
    ToolSpec,
)

# =========================================================================== #
# Dummy Provider for LLM Compaction Tests
# =========================================================================== #


class MockSummaryProvider(ChatProvider):
    name = "mock-summary"

    def __init__(self, summary_text: str = "Mocked LLM summary of conversation.") -> None:
        self.summary_text = summary_text
        self.last_request: ChatRequest | None = None
        self.should_fail = False

    async def stream(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        if False:
            yield  # type: ignore[unreachable]
        raise NotImplementedError

    async def complete(self, request: ChatRequest) -> ChatResponse:
        self.last_request = request
        if self.should_fail:
            raise RuntimeError("Provider connection failed")
        return ChatResponse(
            message=Message.assistant(self.summary_text),
            model=request.model,
        )


# =========================================================================== #
# 1. TokenEstimator Tests
# =========================================================================== #


def test_estimator_text_empty_and_ascii() -> None:
    estimator = TokenEstimator()
    assert estimator.estimate_text("") == 0
    assert estimator.estimate_text("a") == 1
    # 40 ascii characters at ~4 chars per token -> ~10 tokens
    text = "a" * 40
    assert estimator.estimate_text(text) == 10


def test_estimator_language_variations_cjk() -> None:
    estimator = TokenEstimator()
    # 5 Korean characters: should estimate higher token density than 5 ascii characters
    korean_text = "안녕하세요"  # 5 characters
    ascii_text = "hello"  # 5 characters

    korean_tokens = estimator.estimate_text(korean_text)
    ascii_tokens = estimator.estimate_text(ascii_text)

    # Korean chars typically take ~1.2 tokens per char, so ~6 tokens
    # ASCII "hello" takes ceil(5/4) = 2 tokens
    assert korean_tokens > ascii_tokens
    assert korean_tokens == 6
    assert ascii_tokens == 2


def test_estimator_content_parts() -> None:
    estimator = TokenEstimator()

    # TextPart
    text_part = TextPart(text="hello world")
    assert estimator.estimate_part(text_part) == estimator.estimate_text("hello world")

    # ThinkingPart
    thinking_part = ThinkingPart(text="reasoning steps")
    assert estimator.estimate_part(thinking_part) > estimator.estimate_text("reasoning steps")

    # ImagePart (low vs auto/high)
    img_low = ImagePart(url="https://example.com/img.png", detail="low")
    img_high = ImagePart(url="https://example.com/img.png", detail="high")
    assert estimator.estimate_part(img_low) == 85
    assert estimator.estimate_part(img_high) > estimator.estimate_part(img_low)

    # ToolResultPart
    tool_res_part = ToolResultPart(call_id="call_1", name="read_file", content="file content here")
    assert estimator.estimate_part(tool_res_part) > estimator.estimate_text("file content here")


def test_estimator_tool_call() -> None:
    estimator = TokenEstimator()
    call = ToolCall(name="grep_search", arguments={"query": "TokenEstimator", "path": "src"})
    tokens = estimator.estimate_tool_call(call)
    # Must account for name, arguments JSON, and framing overhead
    assert tokens > estimator.estimate_text("grep_search")
    assert tokens >= 15


def test_estimator_message() -> None:
    estimator = TokenEstimator()

    # User message
    msg_user = Message.user("Hello agent!")
    user_tokens = estimator.estimate_message(msg_user)
    # Must include message overhead + text
    assert user_tokens > estimator.estimate_text("Hello agent!")

    # Assistant message with tool calls
    call = ToolCall(name="read_file", arguments={"path": "a.txt"})
    msg_assistant = Message.assistant(text="Let me check", tool_calls=[call])
    assistant_tokens = estimator.estimate_message(msg_assistant)
    assert assistant_tokens > user_tokens

    # Tool result message
    msg_tool = Message.tool_result(call_id=call.id, name="read_file", content="file contents")
    tool_tokens = estimator.estimate_message(msg_tool)
    assert tool_tokens > 0


def test_estimator_tool_spec() -> None:
    estimator = TokenEstimator()
    spec = ToolSpec(
        name="edit_file",
        description="Edit a file by replacing old_str with new_str",
        params=[
            ToolParam(name="path", type="string", description="Path to file"),
            ToolParam(name="old_str", type="string", description="Existing content"),
            ToolParam(name="new_str", type="string", description="Replacement content"),
        ],
    )
    tokens = estimator.estimate_tool_spec(spec)
    # Must account for schema keys, descriptions, parameter names, and overhead
    assert tokens > 30


def test_estimator_chat_request_and_polymorphic() -> None:
    estimator = TokenEstimator()
    spec = ToolSpec(name="test_tool", description="Test")
    req = ChatRequest(
        model="gpt-4o",
        messages=[
            Message.system("You are helpful."),
            Message.user("Do something"),
        ],
        tools=[spec],
    )
    req_tokens = estimator.estimate_request(req)
    assert req_tokens > 0

    # Polymorphic entry point
    assert estimator.estimate("hello") == estimator.estimate_text("hello")
    assert estimator.estimate(req.messages[0]) == estimator.estimate_message(req.messages[0])
    assert estimator.estimate(spec) == estimator.estimate_tool_spec(spec)
    assert estimator.estimate(req) == req_tokens


# =========================================================================== #
# 2. MessageManager Tests (Atomic Tool Pair Invariant)
# =========================================================================== #


def test_message_manager_basic_operations() -> None:
    mgr = MessageManager()
    assert len(mgr) == 0
    assert mgr.system_message is None

    mgr.add_system("System instructions")
    assert len(mgr) == 1
    assert mgr.system_message is not None
    assert mgr.system_message.role == "system"

    mgr.add_user("User prompt")
    mgr.add_assistant("Assistant response")
    assert len(mgr) == 3
    assert mgr[-1].role == "assistant"
    assert mgr.total_tokens() > 0


def test_message_manager_atomic_groups_partitioning() -> None:
    call1 = ToolCall(id="call_1", name="tool1")
    call2 = ToolCall(id="call_2", name="tool2")

    messages = [
        Message.system("sys"),
        Message.user("u1"),
        Message.assistant(tool_calls=[call1, call2]),
        Message.tool_result("call_1", "tool1", "res1"),
        Message.tool_result("call_2", "tool2", "res2"),
        Message.assistant("final answer"),
        Message.user("u2"),
    ]

    groups = MessageManager.get_atomic_groups(messages)
    # Group 0: [system]
    # Group 1: [u1]
    # Group 2: [assistant(tools), tool_result1, tool_result2]  <- ATOMIC GROUP
    # Group 3: [assistant(final answer)]
    # Group 4: [u2]
    assert len(groups) == 5
    assert len(groups[0]) == 1
    assert len(groups[1]) == 1
    assert len(groups[2]) == 3
    assert groups[2][0].role == "assistant"
    assert groups[2][1].role == "tool"
    assert groups[2][2].role == "tool"
    assert len(groups[3]) == 1
    assert len(groups[4]) == 1


def test_message_manager_window_within_budget() -> None:
    mgr = MessageManager()
    mgr.add_system("sys")
    mgr.add_user("u1")
    mgr.add_assistant("a1")

    # High budget keeps all messages
    windowed = mgr.window(max_tokens=10_000)
    assert len(windowed) == 3
    assert [m.role for m in windowed] == ["system", "user", "assistant"]


def test_message_manager_window_preserves_system_prompt() -> None:
    estimator = TokenEstimator()
    mgr = MessageManager(estimator=estimator)
    mgr.add_system("System prompt that must be retained")
    mgr.add_user("Old user message 1")
    mgr.add_assistant("Old assistant message 1")
    mgr.add_user("New user message 2")

    # Budget fits system + latest user message, but not older messages
    sys_tokens = estimator.estimate_message(mgr[0])
    new_user_tokens = estimator.estimate_message(mgr[3])
    budget = sys_tokens + new_user_tokens + 2  # slightly more than system + new_user

    windowed = mgr.window(max_tokens=budget)
    assert len(windowed) == 2
    assert windowed[0].role == "system"
    assert windowed[0].text == "System prompt that must be retained"
    assert windowed[1].role == "user"
    assert windowed[1].text == "New user message 2"


def test_message_manager_window_atomic_tool_pair_preservation() -> None:
    """CRITICAL INVARIANT: Windowing must NEVER split an assistant(tool_calls) from its tool results."""
    estimator = TokenEstimator()
    mgr = MessageManager(estimator=estimator)

    mgr.add_system("sys")
    mgr.add_user("Run tool")

    call = ToolCall(id="c1", name="shell", arguments={"cmd": "ls"})
    msg_assist_tool = Message.assistant(text="Running shell", tool_calls=[call])
    msg_tool_res = Message.tool_result(call_id="c1", name="shell", content="file1.txt file2.txt")
    msg_final = Message.assistant("Finished listing files")

    mgr.add(msg_assist_tool)
    mgr.add(msg_tool_res)
    mgr.add(msg_final)

    sys_tokens = estimator.estimate_message(mgr[0])
    final_tokens = estimator.estimate_message(msg_final)
    tool_group_tokens = (
        estimator.estimate_message(msg_assist_tool) + estimator.estimate_message(msg_tool_res)
    )

    # Case A: Budget allows sys + final + tool_group, but not the older "Run tool" user message
    budget_a = sys_tokens + final_tokens + tool_group_tokens + 5
    win_a = mgr.window(max_tokens=budget_a)
    roles_a = [m.role for m in win_a]
    assert roles_a == ["system", "assistant", "tool", "assistant"]
    assert win_a[1].tool_calls[0].id == "c1"
    assert win_a[2].tool_call_id == "c1"

    # Case B: Budget allows sys + final, but NOT the full tool_group
    # Tool result MUST NOT be kept alone as an orphan!
    budget_b = sys_tokens + final_tokens + (tool_group_tokens // 2)
    win_b = mgr.window(max_tokens=budget_b)
    roles_b = [m.role for m in win_b]
    assert roles_b == ["system", "assistant"]
    assert win_b[1].text == "Finished listing files"
    # Verify no tool messages and no orphaned assistant tool calls
    assert not any(m.role == "tool" for m in win_b)
    assert not any(m.tool_calls for m in win_b)


def test_message_manager_window_multiple_tool_calls_atomic() -> None:
    """Windowing must not split between multiple tool responses of the same assistant turn."""
    estimator = TokenEstimator()
    mgr = MessageManager(estimator=estimator)

    mgr.add_system("sys")
    c1 = ToolCall(id="c1", name="t1")
    c2 = ToolCall(id="c2", name="t2")
    mgr.add(Message.assistant(tool_calls=[c1, c2]))
    mgr.add(Message.tool_result("c1", "t1", "result 1"))
    mgr.add(Message.tool_result("c2", "t2", "result 2"))
    mgr.add(Message.user("follow up"))

    # If the budget fits sys + follow_up + c2_result, but NOT c1_result + assistant(c1,c2),
    # it must NOT include c2_result alone!
    follow_up_tokens = estimator.estimate_message(mgr[-1])
    sys_tokens = estimator.estimate_message(mgr[0])
    c2_tokens = estimator.estimate_message(mgr[3])

    budget = sys_tokens + follow_up_tokens + c2_tokens + 2
    windowed = mgr.window(max_tokens=budget)

    # Should only contain system and follow-up, never partial tool results
    assert [m.role for m in windowed] == ["system", "user"]
    assert windowed[-1].text == "follow up"


def test_message_manager_window_orphaned_tool_in_history_skipped() -> None:
    """If history somehow contains an orphaned tool message, windowing must never include it without assistant."""
    mgr = MessageManager()
    mgr.add_system("sys")
    # Corrupted history: tool message without assistant tool_call
    mgr.add(Message.tool_result("orphaned_call", "bad_tool", "data"))
    mgr.add(Message.user("hello"))

    windowed = mgr.window(max_tokens=10_000)
    # Windowing should filter out the orphaned tool message
    assert not any(m.role == "tool" for m in windowed)
    assert [m.role for m in windowed] == ["system", "user"]


# =========================================================================== #
# 3. Compactor Tests
# =========================================================================== #


@pytest.mark.asyncio
async def test_compactor_no_compression_when_within_budget() -> None:
    compactor = Compactor()
    messages = [
        Message.system("sys"),
        Message.user("hello"),
        Message.assistant("hi"),
    ]
    # Huge max_tokens -> no compaction
    compacted = await compactor.compact(messages, max_tokens=10_000)
    assert len(compacted) == 3
    assert compacted == messages


@pytest.mark.asyncio
async def test_compactor_heuristic_summarization_trigger() -> None:
    compactor = Compactor()
    call = ToolCall(id="call_1", name="fs_read", arguments={"path": "main.py"})

    messages = [
        Message.system("System prompt instructions"),
        Message.user("Build a web scraper in main.py"),
        Message.assistant("Checking main.py", tool_calls=[call]),
        Message.tool_result("call_1", "fs_read", "import requests\n" + "# code line\n" * 40),
        Message.assistant("File exists. Now let's implement the scraper."),
        Message.user("Add error handling to scraper"),
        Message.assistant("I have added error handling."),
    ]

    # Force compaction with max_tokens smaller than uncompressed (~170 tokens) but enough for compacted (~130 tokens)
    compacted = await compactor.compact(messages, max_tokens=140)
    assert len(compacted) < len(messages)

    # Context structure: [system_message, summary_message, ...recent_messages]
    assert compacted[0].role == "system"
    assert compacted[0].text == "System prompt instructions"

    summary_msg = compacted[1]
    assert summary_msg.metadata.get("is_summary") is True
    # Verify structured summary sections
    summary_text = summary_msg.text
    assert "Goals & User Requests" in summary_text
    assert "Tool Actions & Results" in summary_text
    assert "Key Decisions & Findings" in summary_text
    assert "Build a web scraper" in summary_text
    assert "fs_read" in summary_text

    # Recent messages should be preserved at the end
    assert compacted[-1].role == "assistant"
    assert compacted[-1].text == "I have added error handling."


@pytest.mark.asyncio
async def test_compactor_llm_summarization_with_provider() -> None:
    mock_provider = MockSummaryProvider(
        summary_text="Goals: Scrape news.\nTools: Read main.py.\nDecisions: Added error handling."
    )
    compactor = Compactor()

    messages = [
        Message.system("System instructions"),
        Message.user("Old user goal " + "x" * 200),
        Message.assistant("Old assistant decision " + "y" * 200),
        Message.user("Latest user goal"),
        Message.assistant("Latest assistant reply"),
    ]

    compacted = await compactor.compact(messages, max_tokens=80, provider=mock_provider)
    assert mock_provider.last_request is not None

    # Verify summary message contains the LLM output
    summary_msg = compacted[1]
    assert summary_msg.metadata.get("is_summary") is True
    assert "Goals: Scrape news" in summary_msg.text
    assert compacted[-1].text == "Latest assistant reply"


@pytest.mark.asyncio
async def test_compactor_provider_failure_fallback_to_heuristic() -> None:
    mock_provider = MockSummaryProvider()
    mock_provider.should_fail = True

    compactor = Compactor()
    messages = [
        Message.system("System instructions"),
        Message.user("Initial goal " + "x" * 200),
        Message.assistant("Working on it " + "y" * 200),
        Message.user("Next step"),
        Message.assistant("Done with next step"),
    ]

    # Should not raise exception; must gracefully fallback to heuristic summary
    compacted = await compactor.compact(messages, max_tokens=80, provider=mock_provider)
    assert len(compacted) >= 2
    summary_msg = compacted[1]
    assert summary_msg.metadata.get("is_summary") is True
    assert "Goals & User Requests" in summary_msg.text


def test_compactor_heuristic_sync_method() -> None:
    compactor = Compactor()
    messages = [
        Message.user("Goal 1: " + "a" * 100),
        Message.assistant("Action 1: " + "b" * 100),
        Message.user("Goal 2"),
        Message.assistant("Action 2"),
    ]
    compacted = compactor.compact_heuristic(messages, max_tokens=60)
    assert len(compacted) >= 2
    assert any(m.metadata.get("is_summary") for m in compacted)


@pytest.mark.asyncio
async def test_compactor_iterative_compaction_preserves_previous_summary() -> None:
    """Iterative compaction must include previous summary information without dropping it."""
    compactor = Compactor()
    prev_summary_msg = Message(
        role="user",
        content=[TextPart("Initial phase completed. Project initialized.")],
        metadata={"is_summary": True},
    )

    messages = [
        Message.system("System prompt"),
        prev_summary_msg,
        Message.user("Phase 2: Add API endpoints: " + "p2 " * 20),
        Message.assistant("Endpoints added: " + "resp " * 20),
        Message.user("Phase 3: Add unit tests"),
        Message.assistant("Unit tests added."),
    ]

    compacted = await compactor.compact(messages, max_tokens=100)
    summary_msg = compacted[1]
    assert "Initial phase completed" in summary_msg.text
    assert "Phase 2: Add API endpoints" in summary_msg.text
