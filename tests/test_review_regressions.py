"""Regression coverage for GitHub review issues #1 through #8."""

from __future__ import annotations

import asyncio
import io
import json
import sys
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from rich.console import Console

from openagent.cli import _build_runner, main
from openagent.config import CustomProviderConfig, load_config, save_config
from openagent.context.compactor import Compactor
from openagent.context.messages import MessageManager
from openagent.core.events import DoneEvent, ErrorEvent, StreamEvent, TextDelta, ThinkingDelta
from openagent.core.provider import ChatProvider, ProviderError
from openagent.core.types import ChatRequest, Message
from openagent.providers.anthropic import AnthropicProvider
from openagent.providers.custom import CustomJsonPathProvider
from openagent.providers.ollama import OllamaProvider
from openagent.providers.openai_compat import OpenAICompatProvider
from openagent.runner import AgentRunner
from openagent.tools.mcp.client import MCPServerConfig
from openagent.tools.mcp.manager import MCPManager
from openagent.tools.registry import PermissionAction, ToolRegistry
from openagent.tools.shell import MAX_OUTPUT_BYTES, ShellTool, _collect_output


@pytest.fixture
def isolated_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("OPENAGENT_GLOBAL_CONFIG", str(tmp_path / "global.toml"))
    monkeypatch.setenv("OPENAGENT_SESSION_DIR", str(tmp_path / "sessions"))
    for name in ("MODEL", "BASE_URL", "API_KEY", "MAX_TOKENS", "CONTEXT_WINDOW", "WORKSPACE"):
        monkeypatch.delenv(f"OPENAGENT_{name}", raising=False)
    return tmp_path


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "reference,expected",
    [
        ("deepseek/deepseek-chat", "deepseek-chat"),
        ("anthropic/claude-3-5-sonnet", "claude-3-5-sonnet"),
        ("ollama/llama3.2", "llama3.2"),
        ("openrouter/anthropic/claude-3.5-sonnet", "anthropic/claude-3.5-sonnet"),
        ("deepseek", "deepseek-chat"),
    ],
)
async def test_cli_resolved_model_payload(
    isolated_config: Path, reference: str, expected: str
) -> None:
    runner, _ = _build_runner(load_config(model=reference), Console(file=io.StringIO()))
    try:
        request = ChatRequest(model=runner.model, messages=[Message.user("hello")])
        provider = runner.provider
        assert isinstance(provider, (OpenAICompatProvider, AnthropicProvider, OllamaProvider))
        payload = (
            provider.build_payload(request, stream=True)
            if isinstance(provider, OpenAICompatProvider)
            else provider.build_payload(request)
        )
        assert payload["model"] == expected
    finally:
        await runner.provider.close()


def test_policy_inheritance_across_all_layers(isolated_config: Path) -> None:
    (isolated_config / "global.toml").write_text(
        '[danger_policy]\nexecute="deny"\nnetwork="deny"\nwrite="deny"\n',
        encoding="utf-8",
    )
    (isolated_config / "openagent.toml").write_text(
        '[danger_policy]\nwrite="ask"\n', encoding="utf-8"
    )
    custom = isolated_config / "override.toml"
    custom.write_text('[danger_policy]\nnone="deny"\n', encoding="utf-8")
    cfg = load_config(custom, danger_policy={"write": "allow"})
    assert cfg.danger_policy == {
        "none": PermissionAction.DENY,
        "execute": PermissionAction.DENY,
        "network": PermissionAction.DENY,
        "write": PermissionAction.ALLOW,
    }


@pytest.mark.asyncio
@pytest.mark.parametrize("model", ["internal", "custom/internal"])
@pytest.mark.parametrize("content_type", ["application/x-ndjson", "text/event-stream"])
async def test_custom_config_roundtrip_and_stream(
    isolated_config: Path,
    model: str,
    content_type: str,
) -> None:
    config_path = isolated_config / "custom.toml"
    config_path.write_text(
        f'''model = "{model}"
base_url = "https://example.invalid/generate"
api_key = "test-key"
context_window = 32000
[custom_provider]
chat_endpoint = ""
auth_scheme = "api_key"
auth_header = "x-api-key"
jsonpath_text = "response.text"
jsonpath_thinking = "response.reasoning"
jsonpath_tool_calls = "response.actions[*]"
jsonpath_input_tokens = "metrics.input"
jsonpath_output_tokens = "metrics.output"
''',
        encoding="utf-8",
    )
    config = load_config(config_path)
    saved = isolated_config / "saved.toml"
    save_config(config, saved)
    reloaded = load_config(saved)
    assert reloaded.custom_provider == config.custom_provider
    assert reloaded.context_window == 32000
    runner, _ = _build_runner(reloaded, Console(file=io.StringIO()))
    assert isinstance(runner.provider, CustomJsonPathProvider)
    provider = runner.provider
    chunk = {
        "response": {
            "text": "hello",
            "reasoning": "plan",
            "actions": [
                {"id": "call1", "name": "think", "arguments": {"thought": "ok"}},
            ],
        },
        "metrics": {"input": 10, "output": 2},
    }

    def respond(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://example.invalid/generate"
        assert request.headers["x-api-key"] == "test-key"
        assert json.loads(request.content)["model"] == "internal"
        data = json.dumps(chunk)
        if content_type == "text/event-stream":
            data = f"data: {data}\n\ndata: [DONE]\n\n"
        return httpx.Response(200, headers={"content-type": content_type}, text=data)

    await provider.transport.aclose()
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        provider.transport._client = client
        events = [
            event
            async for event in provider.stream(
                ChatRequest(model=runner.model, messages=[Message.user("hello")]),
            )
        ]
    assert any(isinstance(event, TextDelta) and event.text == "hello" for event in events)
    assert any(isinstance(event, ThinkingDelta) and event.text == "plan" for event in events)
    done = events[-1]
    assert isinstance(done, DoneEvent) and done.message is not None
    assert done.message.reasoning == "plan"
    assert done.message.tool_calls[0].name == "think"
    assert done.usage.prompt_tokens == 10 and done.usage.completion_tokens == 2


class RecordingProvider(ChatProvider):
    model = "mock-model"
    context_window = 2000

    def __init__(self, failure: str | None = None) -> None:
        self.failure = failure
        self.requests: list[ChatRequest] = []

    async def stream(self, request: ChatRequest) -> AsyncIterator[StreamEvent]:
        self.requests.append(request)
        if self.failure == "exception":
            raise ProviderError("simulated failure")
        if self.failure == "event":
            yield ErrorEvent(error=ProviderError("simulated failure"))
        else:
            yield DoneEvent(message=Message.assistant("success"))


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["event", "exception"])
async def test_rollback_after_real_compaction(failure: str) -> None:
    history = MessageManager(
        [
            Message.system("system"),
            Message.user("old " * 1000),
            Message.assistant("previous answer"),
        ]
    )
    before = history.to_dict()
    compactor = Compactor()
    with patch.object(
        compactor, "extract_llm_summary", AsyncMock(return_value="old history")
    ) as summary:
        runner = AgentRunner(
            RecordingProvider(failure), messages=history, compactor=compactor, context_window=200
        )
        events = [event async for event in runner.run_turn("failed request")]
    summary.assert_awaited_once()
    assert any(isinstance(event, ErrorEvent) for event in events)
    assert history.to_dict() == before


@pytest.mark.asyncio
@pytest.mark.parametrize("stage", ["connect", "list_tools"])
async def test_mcp_start_failure_cleanup(stage: str) -> None:
    client = AsyncMock()
    failure = RuntimeError(f"failed {stage}")
    getattr(client, stage).side_effect = failure
    registry = ToolRegistry()
    manager = MCPManager(
        registry=registry,
        configs=[MCPServerConfig(name="bad", command="invalid")],
        client_factory=lambda cfg: client,
    )
    with pytest.raises(RuntimeError, match=f"failed {stage}"):
        await manager.start_server("bad")
    client.close.assert_awaited_once()
    assert manager.failed_servers["bad"] is failure
    assert not manager.clients and not registry.list_tools()


@pytest.mark.parametrize("failure,expected", [("event", 1), ("exception", 1), (None, 0)])
def test_cli_terminal_exit_status(
    isolated_config: Path, failure: str | None, expected: int
) -> None:
    console = Console(file=io.StringIO())
    runner = AgentRunner(RecordingProvider(failure))
    with (
        patch("openagent.cli.get_console", return_value=console),
        patch(
            "openagent.cli._build_runner",
            return_value=(runner, None),
        ),
    ):
        assert main(["run", "hello"]) == expected


@pytest.mark.asyncio
async def test_output_limit_and_input_budget_are_independent() -> None:
    provider = RecordingProvider()
    runner = AgentRunner(provider, max_tokens=100)
    prompt = "hello " * 150
    _ = [event async for event in runner.run_turn(prompt)]
    request = provider.requests[-1]
    assert request.max_tokens == 100
    assert request.messages[-1].text == prompt
    estimator = runner.messages.estimator
    assert estimator.estimate_request(request) + 100 <= provider.context_window


@pytest.mark.asyncio
async def test_tool_schema_and_output_reserved_before_windowing() -> None:
    registry = ToolRegistry([ShellTool()])
    provider = RecordingProvider()
    history = MessageManager(
        [Message.system("system"), Message.user("old " * 1000), Message.assistant("answer")]
    )
    runner = AgentRunner(
        provider, messages=history, tools=registry, max_tokens=100, context_window=400
    )
    _ = [event async for event in runner.run_turn("new question")]
    request = provider.requests[-1]
    assert history.estimator.estimate_request(request) + 100 <= 400
    assert request.messages[-1].text == "new question"


@pytest.mark.asyncio
async def test_collect_output_retains_combined_cap_and_drains_to_eof() -> None:
    class ChunkStream:
        def __init__(self) -> None:
            self.reads = 0

        async def read(self, size: int) -> bytes:
            self.reads += 1
            await asyncio.sleep(0)
            return b"x" * size if self.reads <= 500 else b""

    class Process:
        stdout = ChunkStream()
        stderr = ChunkStream()

        async def wait(self) -> int:
            return 0

    process: Any = Process()
    stdout, stderr, truncated = await _collect_output(process)
    assert len(stdout) + len(stderr) == MAX_OUTPUT_BYTES
    assert truncated and process.stdout.reads == process.stderr.reads == 501


@pytest.mark.asyncio
async def test_shell_large_parallel_streams_finish_and_mark_truncation(tmp_path: Path) -> None:
    command = f'"{sys.executable}" -c "import sys; sys.stdout.write(chr(120)*2097152); sys.stderr.write(chr(121)*2097152)"'
    result = await ShellTool(workspace_root=tmp_path).execute(command=command, timeout=10)
    assert not result.is_error
    assert "[Output truncated at 100KB]" in result.output
    assert len(result.output.encode()) <= MAX_OUTPUT_BYTES + 40


@pytest.mark.asyncio
@pytest.mark.parametrize("scheme,expected", [("bearer", "Bearer test-key"), ("none", None)])
async def test_custom_auth_and_namespaced_model(
    isolated_config: Path,
    scheme: str,
    expected: str | None,
) -> None:
    cfg = load_config(model="organization/model", api_key="test-key")
    cfg.custom_provider = CustomProviderConfig(auth_scheme=scheme)
    runner, _ = _build_runner(cfg, Console(file=io.StringIO()))
    assert isinstance(runner.provider, CustomJsonPathProvider)
    try:
        assert runner.model == "organization/model"
        assert runner.provider.transport.extra_headers.get("Authorization") == expected
    finally:
        await runner.provider.close()


@pytest.mark.asyncio
async def test_impossible_context_budget_emits_error_and_restores_history() -> None:
    provider = RecordingProvider()
    runner = AgentRunner(provider, context_window=100, max_tokens=100)
    before = runner.messages.to_dict()
    events = [event async for event in runner.run_turn("hello")]
    assert len(events) == 1 and isinstance(events[0], ErrorEvent)
    assert runner.messages.to_dict() == before
    assert not provider.requests


@pytest.mark.asyncio
async def test_verbose_shell_timeout_cleans_up(tmp_path: Path) -> None:
    command = f'"{sys.executable}" -c "import sys,time; sys.stdout.write(chr(120)*2097152); sys.stdout.flush(); time.sleep(10)"'
    result = await ShellTool(workspace_root=tmp_path).execute(command=command, timeout=0.3)
    assert result.is_error and "timed out" in result.output
