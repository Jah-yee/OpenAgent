"""Rich streaming terminal interface and REPL for OpenAgent.

Provides styled thinking blocks, tool syntax highlighting, interactive confirmation
prompts, and full-featured conversation loop.

Copyright 2026 The OpenAgent Contributors
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from prompt_toolkit import PromptSession
from prompt_toolkit.history import InMemoryHistory
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm

from openagent import __version__
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
from openagent.core.types import ToolCall
from openagent.tools.registry import AskCallback


class TUIApp:
    """Manages rich terminal display for streaming agent events and tool execution."""

    def __init__(self, console: Console | None = None) -> None:
        self.console = console if console is not None else Console()
        self._in_thinking = False
        self._in_text = False
        self._thinking_buffer: list[str] = []
        self._text_buffer: list[str] = []

    def reset_turn(self) -> None:
        """Reset state tracking before starting a new conversation turn."""
        self._in_thinking = False
        self._in_text = False
        self._thinking_buffer.clear()
        self._text_buffer.clear()

    def render_event(self, event: StreamEvent) -> None:
        """Process and render a single streaming agent event."""
        match event:
            case StartEvent():
                self.reset_turn()

            case ThinkingDelta(text=t):
                if not self._in_thinking:
                    self._in_thinking = True
                    self.console.print("\n[dim]Thinking...[/dim]", highlight=False)
                self.console.print(t, style="dim", end="", markup=False, highlight=False)
                self._thinking_buffer.append(t)

            case TextDelta(text=t):
                if self._in_thinking:
                    self.console.print()  # End thinking block newline
                    self._in_thinking = False
                if not self._in_text:
                    self._in_text = True
                self.console.print(t, end="", markup=False, highlight=False)
                self._text_buffer.append(t)

            case ToolCallStart():
                if self._in_thinking or self._in_text:
                    self.console.print()
                    self._in_thinking = False
                    self._in_text = False

            case ToolCallDelta():
                pass

            case ToolCallEnd(call=call):
                if self._in_thinking or self._in_text:
                    self.console.print()
                    self._in_thinking = False
                    self._in_text = False

                if call is not None:
                    name = call.name
                    args = call.arguments
                    if args:
                        formatted_args = json.dumps(args, indent=2)
                        if "\n" in formatted_args or len(formatted_args) > 60:
                            call_str = f"[bold blue]Tool: {name}[/bold blue] (args:\n[dim]{formatted_args}[/dim])"
                        else:
                            call_str = f"[bold blue]Tool: {name}[/bold blue] ({json.dumps(args)})"
                    else:
                        call_str = f"[bold blue]Tool: {name}[/bold blue] ()"
                    self.console.print(call_str, highlight=False)

            case ToolResultEvent(tool_name=tool_name, output=output, is_error=is_error):
                trimmed_output = output
                if len(output) > 2000:
                    trimmed_output = (
                        output[:2000]
                        + f"\n... [dim](output truncated, {len(output) - 2000} more characters)[/dim]"
                    )

                if is_error:
                    self.console.print(
                        f"[bold red]Tool Error ({tool_name}):[/bold red]\n{trimmed_output}",
                        highlight=False,
                    )
                else:
                    self.console.print(
                        f"[bold green]Tool Result ({tool_name}):[/bold green]\n{trimmed_output}",
                        highlight=False,
                    )

            case UsageEvent(usage=usage):
                pass

            case ErrorEvent(error=error):
                if self._in_thinking or self._in_text:
                    self.console.print()
                    self._in_thinking = False
                    self._in_text = False
                self.console.print(f"\n[bold red]Error:[/bold red] {error}", highlight=False)

            case DoneEvent(usage=usage):
                if self._in_thinking or self._in_text:
                    self.console.print()
                    self._in_thinking = False
                    self._in_text = False

                if usage and usage.total_tokens > 0:
                    self.console.print(
                        f"[dim]Tokens: prompt={usage.prompt_tokens}, completion={usage.completion_tokens}, total={usage.total_tokens}[/dim]",
                        highlight=False,
                    )

            case _:
                pass


def make_ask_callback(console: Console | None = None, auto_approve: bool = False) -> AskCallback:
    """Create an interactive ask_callback confirming tool execution with the user."""
    active_console = console if console is not None else Console()

    async def _ask_callback(tool_call: ToolCall) -> bool:
        if auto_approve:
            return True

        args_str = json.dumps(tool_call.arguments, indent=2) if tool_call.arguments else "{}"
        active_console.print(
            f"\n[bold yellow]⚠️  Permission Request:[/bold yellow] Tool [cyan]{tool_call.name}[/cyan]"
        )
        if tool_call.arguments:
            active_console.print(f"[dim]Arguments:\n{args_str}[/dim]")

        try:
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(
                None,
                lambda: Confirm.ask(
                    "Allow tool execution?",
                    console=active_console,
                    default=True,
                ),
            )
        except RuntimeError:
            return Confirm.ask(
                "Allow tool execution?",
                console=active_console,
                default=True,
            )
        except (EOFError, KeyboardInterrupt):
            return False

    return _ask_callback


async def run_repl(
    runner: Any,
    console: Console | None = None,
    auto_approve: bool = False,
) -> None:
    """Interactive REPL chat loop."""
    active_console = console if console is not None else Console()

    model_name = getattr(runner, "model", "default")
    workspace = getattr(runner, "workspace_root", ".")
    session_id = getattr(runner, "session_id", None) or "new"

    header_text = (
        f"[bold cyan]OpenAgent[/bold cyan] [dim]v{__version__}[/dim]\n"
        f"Model: [bold green]{model_name}[/bold green] | "
        f"Workspace: [dim]{workspace}[/dim] | "
        f"Session: [dim]{session_id[:8] if len(session_id) > 8 else session_id}[/dim]\n"
        f"[dim]Commands: /exit to quit, /clear to reset context, /help for help[/dim]"
    )
    active_console.print(Panel(header_text, border_style="blue", expand=False))

    try:
        session: PromptSession[str] | None = PromptSession(history=InMemoryHistory())
    except Exception:
        session = None

    app = TUIApp(console=active_console)
    ask_cb = make_ask_callback(console=active_console, auto_approve=auto_approve)

    while True:
        try:
            if session is not None:
                user_input = await session.prompt_async("\nopenagent> ")
            else:
                user_input = await asyncio.to_thread(input, "\nopenagent> ")
        except (KeyboardInterrupt, EOFError):
            active_console.print("\n[yellow]Exiting OpenAgent. Goodbye![/yellow]")
            break

        stripped = user_input.strip()
        if not stripped:
            continue

        if stripped.lower() in ("/exit", "/quit", "exit", "quit"):
            active_console.print("[yellow]Exiting OpenAgent. Goodbye![/yellow]")
            break

        if stripped.lower() == "/clear":
            runner.reset()
            active_console.print("[green]✔ Context cleared and session reset.[/green]")
            continue

        if stripped.lower() == "/help":
            active_console.print(
                "[bold]OpenAgent Help:[/bold]\n"
                "  /clear  - Clear conversation history and reset session\n"
                "  /exit   - Exit the interactive REPL session\n"
                "  /quit   - Exit the interactive REPL session\n"
            )
            continue

        try:
            async for event in runner.run_turn(user_input, ask_callback=ask_cb):
                app.render_event(event)
        except KeyboardInterrupt:
            active_console.print("\n[yellow]Turn cancelled by user.[/yellow]")
        except Exception as exc:
            active_console.print(f"\n[bold red]✖ Unexpected error: {exc}[/bold red]")
