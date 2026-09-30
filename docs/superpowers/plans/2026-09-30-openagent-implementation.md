# OpenAgent Full Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and deliver OpenAgent, an Apache-2.0 licensed, model-agnostic terminal AI coding agent that can work with any LLM provider (OpenAI-compatible, Anthropic, Gemini, Ollama, custom JSONPath HTTP), featuring streaming REPL/CLI, robust tool calling with sandboxing, context compaction, session persistence, and MCP support.

**Architecture:** A layered architecture with a central `AgentRunner` orchestrating a unified streaming event pipeline. The bottom layer handles HTTP/SSE transport; the provider layer normalises all LLM APIs (native, OpenAI-compat, and JSONPath mapping) into canonical types and `StreamEvent`s; the tool layer executes sandboxed fs/shell/MCP tools; the context manager ensures token budget and turn atomicity; and the top layer provides interactive Rich TUI and CLI.

**Tech Stack:** Python 3.11+, `httpx`, `rich`, `prompt-toolkit`, `jsonpath-ng`, `mcp`, `pytest`, `pytest-asyncio`, `ruff`, `mypy`.

## Global Constraints

- License: Apache-2.0 on all files with notice headers.
- Python compatibility: 3.11+ (use `enum.StrEnum`, modern type annotations).
- Zero vendor lock-in: The agent runner and UI only consume `StreamEvent` and canonical `Message`/`ToolCall` types.
- Safety: Path sandboxing for file writes/reads, array-based non-shell process spawning, secret masking in logs, permission levels (`allow`, `ask`, `deny`).
- Quality gates: `ruff check src tests` passes cleanly, `mypy src/openagent` passes with 0 errors, full `pytest` suite passes.
- Final step: Git commit and push to `origin main` as requested by user.

---

### Task 1: Fix Core Transport, SSE Parser, and Existing Providers

**Files:**
- Modify: `src/openagent/utils/sse.py`
- Modify: `src/openagent/utils/http.py`
- Modify: `src/openagent/providers/text_protocol.py`
- Modify: `src/openagent/providers/openai_compat.py`
- Modify: `src/openagent/providers/anthropic.py`
- Create: `tests/test_sse.py`
- Create: `tests/test_providers.py`

**Interfaces:**
- Consumes: `src/openagent/core/types.py`, `src/openagent/core/events.py`
- Produces: Working async `HttpTransport.post_stream()`, async `iter_sse()`, `TextToolCodec._StreamingParser`, `OpenAICompatProvider.stream()`, `AnthropicProvider.stream()`.

- [ ] **Step 1: Write tests for SSE parsing, text protocol parser, and provider streaming**

Write `tests/test_sse.py` and `tests/test_providers.py` testing:
- SSE stream parsing with multi-chunk events and line splits.
- Text tool codec parsing ```tool_code fenced blocks and JSON extraction.
- OpenAI-compatible response translation to `StreamEvent`s.
- Anthropic response translation to `StreamEvent`s.

- [ ] **Step 2: Run tests to verify initial failure**

Run: `.venv\Scripts\pytest.exe tests/test_sse.py tests/test_providers.py -v`
Expected: FAIL due to existing type and attribute errors.

- [ ] **Step 3: Fix SSE, HTTP transport, text protocol, and provider code**

- In `src/openagent/utils/sse.py`: Ensure `iter_sse(response: httpx.Response)` is an `async def` generator that reads `response.aiter_lines()`, accumulates SSE event fields (`event`, `data`, `id`, `retry`), and yields `SSEvent`.
- In `src/openagent/utils/http.py`: Fix `post_stream` to properly yield or return the open streaming response context manager so callers can iterate SSE events.
- In `src/openagent/providers/text_protocol.py`: Fix `_StreamingParser` to define `consume(text)` (which calls `feed` and returns `self`), `finish()`, and proper marker detection.
- In `src/openagent/providers/openai_compat.py`: Fix type annotations and `ToolCall` conversions.
- In `src/openagent/providers/anthropic.py`: Fix `Usage` attribute usage (`prompt_tokens` instead of `input_tokens`), and stream handling.

- [ ] **Step 4: Run tests and type checks to verify pass**

Run: `.venv\Scripts\pytest.exe tests/test_sse.py tests/test_providers.py -v`
Run: `.venv\Scripts\mypy.exe src/openagent/utils src/openagent/providers`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/openagent/utils src/openagent/providers tests/test_sse.py tests/test_providers.py
git commit -m "fix(providers): resolve sse, text protocol, and adapter streaming bugs"
```

---

### Task 2: Provider Router and Additional Providers (Gemini, Ollama, Custom JSONPath)

**Files:**
- Create: `src/openagent/core/router.py`
- Create: `src/openagent/providers/gemini.py`
- Create: `src/openagent/providers/ollama.py`
- Create: `src/openagent/providers/custom.py`
- Create: `tests/test_router.py`
- Create: `tests/test_custom_provider.py`

**Interfaces:**
- Consumes: `core/provider.py`, `core/presets.py`, `core/types.py`, `utils/http.py`
- Produces:
  - `ProviderRouter.resolve(model_ref: str, config: Config | None) -> ChatProvider`
  - `GeminiProvider(ChatProvider)`: native Google Gemini `generateContent` & streaming adapter.
  - `OllamaProvider(ChatProvider)`: local Ollama `/api/chat` adapter.
  - `CustomJsonPathProvider(ChatProvider)`: generic adapter mapping arbitrary JSON API using `jsonpath-ng`.

- [ ] **Step 1: Write tests for Router, Gemini, Ollama, and CustomJsonPath provider**

In `tests/test_router.py`:
- Test resolving presets like `gpt-4o`, `claude-3-5-sonnet`, `gemini-1.5-pro`, `deepseek-chat`, `ollama/llama3`.
- Test fallback behavior and custom base_url overriding.
In `tests/test_custom_provider.py`:
- Test parsing arbitrary JSON responses and streaming chunks via JSONPath expressions.

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv\Scripts\pytest.exe tests/test_router.py tests/test_custom_provider.py -v`
Expected: FAIL (modules not found).

- [ ] **Step 3: Implement Router, Gemini, Ollama, and Custom provider**

- Implement `src/openagent/core/router.py`:
  - `ModelReference.parse(ref: str)` (handles `provider/model` or alias).
  - Match against `PRESETS` from `presets.py`.
  - Instantiate appropriate provider class (`OpenAICompatProvider`, `AnthropicProvider`, `GeminiProvider`, `OllamaProvider`, `CustomJsonPathProvider`).
- Implement `src/openagent/providers/gemini.py`:
  - Gemini REST format: `contents: [{role: "user"|"model", parts: [...]}]`.
  - SSE streaming via `:streamGenerateContent?alt=sse`.
- Implement `src/openagent/providers/ollama.py`:
  - `/api/chat` with newline-delimited JSON chunks.
- Implement `src/openagent/providers/custom.py`:
  - Uses `jsonpath-ng` to extract text delta, tool calls, and usage from arbitrary JSON schemas.

- [ ] **Step 4: Run tests and verify**

Run: `.venv\Scripts\pytest.exe tests/test_router.py tests/test_custom_provider.py -v`
Run: `.venv\Scripts\mypy.exe src/openagent/core/router.py src/openagent/providers`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/openagent/core/router.py src/openagent/providers tests/
git commit -m "feat(providers): add router, gemini, ollama, and custom jsonpath adapters"
```

---

### Task 3: Tool System, Sandboxed Filesystem, Shell, and Agent Tools

**Files:**
- Create: `src/openagent/tools/base.py`
- Create: `src/openagent/tools/registry.py`
- Create: `src/openagent/tools/fs.py`
- Create: `src/openagent/tools/shell.py`
- Create: `src/openagent/tools/agent_tools.py`
- Create: `tests/test_tools.py`

**Interfaces:**
- Consumes: `core/types.py` (`ToolCall`, `ToolSpec`, `Permission`)
- Produces:
  - `Tool(ABC)`: abstract definition with `name`, `description`, `parameters`, and `execute(params) -> str`.
  - `ToolRegistry`: registry for tools, permission verification, schema export for providers, parallel execution.
  - `fs` tools: `read_file`, `write_file`, `edit_file`, `list_directory`, `glob_find`, `grep_search` with path sandboxing against workspace root.
  - `shell` tool: array-based command execution with timeout and output limit.
  - `think`, `todo`, `web_fetch` tools.

- [ ] **Step 1: Write tests for ToolRegistry and built-in tools**

In `tests/test_tools.py`:
- Test tool registration and JSON schema extraction.
- Test sandboxing: attempt to read/write outside workspace root (must raise error).
- Test `edit_file` replacing target strings accurately.
- Test `shell` running commands and capturing stdout/stderr/exit codes safely.
- Test permission checks (`ALLOW`, `ASK`, `DENY`).

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv\Scripts\pytest.exe tests/test_tools.py -v`
Expected: FAIL (modules not found).

- [ ] **Step 3: Implement Tool base, Registry, FS, Shell, and Agent tools**

- Implement `src/openagent/tools/base.py`:
  - `Tool` abstract base class with docstring/type introspection or explicit `ToolSpec`.
  - `ToolResult` container with text content and error flag.
- Implement `src/openagent/tools/registry.py`:
  - `ToolRegistry` supporting `register(tool)`, `get(name)`, `list_specs()`, `execute(call, permission_callback)`.
  - Parallel execution via `asyncio.gather`.
- Implement `src/openagent/tools/fs.py`:
  - Root path confinement (`os.path.commonpath` / `resolve()`).
  - `read_file(path, offset, limit)`, `write_file(path, content)`, `edit_file(path, old_str, new_str)`, `list_directory(path)`, `glob_find(pattern)`, `grep_search(query, path)`.
- Implement `src/openagent/tools/shell.py`:
  - `execute_shell(command, timeout_seconds=120)`. Spawns process safely without shell injection where applicable or via platform shell (`powershell` / `cmd` on Windows, `bash` on Unix) with output capture up to 100KB limit.
- Implement `src/openagent/tools/agent_tools.py`:
  - `think(thought: str)`: Record reasoning.
  - `todo(action: str, tasks: list[str])`: Maintain task list.
  - `web_fetch(url: str)`: Fetch web page text via httpx.

- [ ] **Step 4: Run tests and verify**

Run: `.venv\Scripts\pytest.exe tests/test_tools.py -v`
Run: `.venv\Scripts\mypy.exe src/openagent/tools`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/openagent/tools tests/test_tools.py
git commit -m "feat(tools): implement sandboxed fs, safe shell, registry, and agent tools"
```

---

### Task 4: MCP (Model Context Protocol) Client Integration

**Files:**
- Create: `src/openagent/tools/mcp/client.py`
- Create: `src/openagent/tools/mcp/manager.py`
- Create: `tests/test_mcp.py`

**Interfaces:**
- Consumes: `mcp` library, `tools/base.py`, `tools/registry.py`
- Produces:
  - `MCPClient`: stdio and SSE client connecting to external MCP servers.
  - `MCPManager`: lifecycle management, auto-registering discovered MCP tools into `ToolRegistry`.

- [ ] **Step 1: Write tests for MCP client and manager**

In `tests/test_mcp.py`:
- Mock MCP session discovery `tools/list` and `tools/call`.
- Test wrapping MCP tools as OpenAgent `Tool` objects and registering them into `ToolRegistry`.

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv\Scripts\pytest.exe tests/test_mcp.py -v`
Expected: FAIL (modules not found).

- [ ] **Step 3: Implement MCP client and manager**

- Implement `src/openagent/tools/mcp/client.py`: stdio transport client and SSE transport client wrapping `mcp` protocol.
- Implement `src/openagent/tools/mcp/manager.py`: reads MCP server configs, connects, lists tools, registers them into `ToolRegistry`, handles graceful shutdown.

- [ ] **Step 4: Run tests and verify**

Run: `.venv\Scripts\pytest.exe tests/test_mcp.py -v`
Run: `.venv\Scripts\mypy.exe src/openagent/tools/mcp`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/openagent/tools/mcp tests/test_mcp.py
git commit -m "feat(mcp): implement mcp client and server manager"
```

---

### Task 5: Context Management (Estimation, Windowing, Compaction)

**Files:**
- Create: `src/openagent/context/estimator.py`
- Create: `src/openagent/context/messages.py`
- Create: `src/openagent/context/compactor.py`
- Create: `tests/test_context.py`

**Interfaces:**
- Consumes: `core/types.py`, `core/provider.py`
- Produces:
  - `TokenEstimator`: fast rule-based token estimation for text, tools, and message overhead.
  - `MessageManager`: manages message history, sliding window, and preserves atomic tool call/result pairs.
  - `Compactor`: triggers summarization when token count reaches threshold.

- [ ] **Step 1: Write tests for context manager and compactor**

In `tests/test_context.py`:
- Test token estimation accuracy.
- Test sliding window: verify it never cuts midway between a `tool_calls` assistant message and its `tool` responses.
- Test compactor substituting old turns with a structured summary message.

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv\Scripts\pytest.exe tests/test_context.py -v`
Expected: FAIL (modules not found).

- [ ] **Step 3: Implement context estimator, message manager, and compactor**

- Implement `src/openagent/context/estimator.py`: ~4 chars per token heuristic + tool spec schema token overhead.
- Implement `src/openagent/context/messages.py`: maintains list of `Message`s, enforces turn boundary invariants.
- Implement `src/openagent/context/compactor.py`: compacts messages older than window threshold using provider or heuristic fallback.

- [ ] **Step 4: Run tests and verify**

Run: `.venv\Scripts\pytest.exe tests/test_context.py -v`
Run: `.venv\Scripts\mypy.exe src/openagent/context`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/openagent/context tests/test_context.py
git commit -m "feat(context): implement token estimation, turn-safe sliding window, and compactor"
```

---

### Task 6: Prompt Builder and Session Store

**Files:**
- Create: `src/openagent/prompts/base.py`
- Create: `src/openagent/session/store.py`
- Create: `tests/test_prompts_session.py`

**Interfaces:**
- Consumes: `core/types.py`, `tools/registry.py`
- Produces:
  - `PromptBuilder.build_system_prompt()`: includes workspace info, OS, current date, tool specifications, and guidelines.
  - `SessionStore`: saves conversation turns to JSONL format, lists past sessions, reloads history for `--resume`.

- [ ] **Step 1: Write tests for PromptBuilder and SessionStore**

In `tests/test_prompts_session.py`:
- Verify system prompt includes workspace directory, date, and tools.
- Test session serialization to JSONL and deserialization back into canonical `Message` objects.

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv\Scripts\pytest.exe tests/test_prompts_session.py -v`
Expected: FAIL (modules not found).

- [ ] **Step 3: Implement PromptBuilder and SessionStore**

- Implement `src/openagent/prompts/base.py`: dynamic system prompt generator with customizable templates.
- Implement `src/openagent/session/store.py`: `SessionStore` with atomic file writes, session ID generation, indexing, and loading.

- [ ] **Step 4: Run tests and verify**

Run: `.venv\Scripts\pytest.exe tests/test_prompts_session.py -v`
Run: `.venv\Scripts\mypy.exe src/openagent/prompts src/openagent/session`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/openagent/prompts src/openagent/session tests/test_prompts_session.py
git commit -m "feat(session): add prompt builder and jsonl session persistence"
```

---

### Task 7: AgentRunner (The Core Agent Execution Loop)

**Files:**
- Create: `src/openagent/runner.py`
- Create: `tests/test_runner.py`

**Interfaces:**
- Consumes: All core components (`PromptBuilder`, `MessageManager`, `ToolRegistry`, `ProviderRouter`, `SessionStore`, `StreamEvent`)
- Produces:
  - `AgentRunner`:
    - `run_turn(user_input: str) -> AsyncIterator[StreamEvent]`
    - Manages tool call loop (model call -> execute tools -> model call ...) until stop.
    - Permission callback hook for interactive confirmation.

- [ ] **Step 1: Write integration tests with a Mock/Fake Provider**

In `tests/test_runner.py`:
- Mock provider yielding `TextDelta`, then `ToolCallStart`/`ToolCallEnd` for `write_file`, then final `TextDelta` confirming completion.
- Verify `AgentRunner` executes the tool, registers the result, makes the second LLM call, and yields appropriate events.
- Test permission denial handling (agent receives denial error and adapts).

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv\Scripts\pytest.exe tests/test_runner.py -v`
Expected: FAIL (runner not found).

- [ ] **Step 3: Implement AgentRunner**

- Implement `src/openagent/runner.py`:
  - Manage turn lifecycle: user message -> context update -> provider stream.
  - Streaming event dispatch.
  - Detect tool calls upon stream completion.
  - Check permissions (`ALLOW`, `ASK`, `DENY`).
  - Execute tools in parallel (`asyncio.gather`), format `ToolResultPart` into `tool` messages.
  - Recurse or loop back to step 1 until `FinishReason.STOP` or max turns reached.
  - Record tokens to `SessionStore`.

- [ ] **Step 4: Run tests and verify**

Run: `.venv\Scripts\pytest.exe tests/test_runner.py -v`
Run: `.venv\Scripts\mypy.exe src/openagent/runner.py`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/openagent/runner.py tests/test_runner.py
git commit -m "feat(runner): implement complete agent loop with tool orchestration"
```

---

### Task 8: Configuration System and CLI / TUI

**Files:**
- Create: `src/openagent/config.py`
- Create: `src/openagent/tui/app.py`
- Create: `src/openagent/cli.py`
- Create: `tests/test_cli_config.py`

**Interfaces:**
- Consumes: `runner.py`, `rich`, `prompt-toolkit`
- Produces:
  - `Config`: parses TOML config files and environment variables.
  - `cli.py`: main entry point with commands: `run`, `chat` (interactive REPL), `models`, `sessions`, `config`.
  - `tui/app.py`: Rich live streaming console UI displaying text markdown, collapsible thinking blocks, tool calls, and syntax highlighting.

- [ ] **Step 1: Write tests for Config loading and CLI argument parsing**

In `tests/test_cli_config.py`:
- Test config parsing with default values and custom toml file.
- Test CLI arg parsing for `openagent run "prompt" --model gpt-4o` and `--dry-run`.

- [ ] **Step 2: Run tests to verify failure**

Run: `.venv\Scripts\pytest.exe tests/test_cli_config.py -v`
Expected: FAIL (modules not found).

- [ ] **Step 3: Implement Config, CLI, and TUI**

- Implement `src/openagent/config.py`: Load config from `~/.config/openagent/config.toml`, `./openagent.toml`, or env vars (`OPENAGENT_*`, `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`).
- Implement `src/openagent/tui/app.py`: Rich console app with live streaming token rendering, thinking folding, tool call syntax highlighting, user input prompt.
- Implement `src/openagent/cli.py`: Click or argparse CLI entrypoint with subcommands `run`, `chat`, `models`, `sessions`.

- [ ] **Step 4: Run tests and verify**

Run: `.venv\Scripts\pytest.exe tests/test_cli_config.py -v`
Run: `.venv\Scripts\mypy.exe src/openagent/config.py src/openagent/cli.py src/openagent/tui`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/openagent/config.py src/openagent/cli.py src/openagent/tui tests/test_cli_config.py
git commit -m "feat(cli): implement configuration system, rich streaming tui, and cli entrypoint"
```

---

### Task 9: Full Quality Verification, Git Commit and Push

**Files:**
- Modify: `docs/DESIGN.md` (update status)
- Modify: `README.md` (usage examples)
- Run full verification suite across all files

- [ ] **Step 1: Run comprehensive tests**

Run: `.venv\Scripts\pytest.exe -v`
Expected: All tests PASS.

- [ ] **Step 2: Run linter and type checker**

Run: `.venv\Scripts\ruff.exe check src tests`
Run: `.venv\Scripts\mypy.exe src/openagent`
Expected: 0 errors.

- [ ] **Step 3: Test CLI execution directly**

Run: `.venv\Scripts\python.exe -m openagent.cli --help`
Run: `.venv\Scripts\python.exe -m openagent.cli models`
Expected: Exit code 0, cleanly displaying help and model presets.

- [ ] **Step 4: Git commit and push to origin main**

```bash
git add -A
git commit -m "feat: complete openagent model-agnostic coding agent implementation"
git push origin main
```
Verify push succeeds.
