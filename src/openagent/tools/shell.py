"""Safe asynchronous shell execution tool with timeout and output bounding.

Copyright 2026 The OpenAgent Contributors
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import asyncio
import contextlib
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Literal

from openagent.core.types import ToolParam
from openagent.tools.base import DangerLevel, Tool, ToolResult

MAX_OUTPUT_BYTES = 100 * 1024  # 100 KB cap


def _truncate_output(text: str, cap: int = MAX_OUTPUT_BYTES) -> str:
    """Cap output string to limit and append truncation indicator if exceeded."""
    if len(text.encode("utf-8", errors="replace")) <= cap:
        return text
    truncated = text[:cap]
    while len(truncated.encode("utf-8", errors="replace")) > cap and truncated:
        truncated = truncated[:-10]
    return f"{truncated}\n... [Output truncated at 100KB]"


class ShellTool(Tool):
    """Executes terminal commands asynchronously with timeout and bounded output."""

    name = "execute_shell"
    description = (
        "Execute a terminal command asynchronously with timeout enforcement and bounded output."
    )
    danger: DangerLevel = "execute"

    def __init__(
        self,
        workspace_root: Path | str | None = None,
        shell_type: Literal["auto", "powershell", "cmd", "bash", "sh"] = "auto",
    ) -> None:
        super().__init__()
        self.workspace_root = Path(workspace_root).resolve() if workspace_root else None
        self.shell_type = shell_type
        self.params = [
            ToolParam(name="command", type="string", description="Shell command line to execute"),
            ToolParam(
                name="timeout",
                type="number",
                description="Timeout in seconds (default 120.0)",
                required=False,
                default=120.0,
            ),
            ToolParam(
                name="cwd",
                type="string",
                description="Optional working directory for the command",
                required=False,
            ),
        ]

    async def execute(self, **kwargs: Any) -> ToolResult:
        call_id = str(kwargs.get("call_id", ""))
        command = str(kwargs.get("command", ""))
        timeout = float(kwargs.get("timeout", 120.0))
        cwd = kwargs.get("cwd")

        if not command:
            return ToolResult(
                call_id=call_id,
                output="Parameter 'command' is required.",
                is_error=True,
            )

        # Determine effective working directory
        work_dir: Path | None = None
        if cwd is not None:
            work_dir = Path(cwd).resolve()
        elif self.workspace_root is not None:
            work_dir = self.workspace_root

        if work_dir and self.workspace_root and not work_dir.is_relative_to(self.workspace_root):
            return ToolResult(
                call_id=call_id,
                output=f"Access denied: working directory '{work_dir}' is outside workspace root '{self.workspace_root}'.",
                is_error=True,
            )

        # Prepare subprocess invocation
        try:
            if self.shell_type == "powershell":
                proc = await asyncio.create_subprocess_exec(
                    "powershell.exe",
                    "-NoProfile",
                    "-NonInteractive",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(work_dir) if work_dir else None,
                )
            elif self.shell_type == "cmd":
                proc = await asyncio.create_subprocess_exec(
                    os.environ.get("COMSPEC", "cmd.exe"),
                    "/c",
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(work_dir) if work_dir else None,
                )
            elif self.shell_type in ("bash", "sh"):
                sh_binary = shutil.which(self.shell_type) or "/bin/sh"
                proc = await asyncio.create_subprocess_exec(
                    sh_binary,
                    "-c",
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(work_dir) if work_dir else None,
                )
            else:
                proc = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=str(work_dir) if work_dir else None,
                )
        except Exception as exc:
            return ToolResult(
                call_id=call_id,
                output=f"Failed to start command process: {exc}",
                is_error=True,
            )

        # Wait with timeout
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(),
                timeout=timeout,
            )
        except TimeoutError:
            with contextlib.suppress(Exception):
                proc.kill()
            with contextlib.suppress(Exception):
                await proc.wait()
            return ToolResult(
                call_id=call_id,
                output=f"Command timed out after {timeout} seconds.",
                is_error=True,
            )

        stdout_text = stdout_bytes.decode(sys.getdefaultencoding(), errors="replace")
        stderr_text = stderr_bytes.decode(sys.getdefaultencoding(), errors="replace")

        # Bound output
        stdout_text = _truncate_output(stdout_text)
        stderr_text = _truncate_output(stderr_text)

        returncode = proc.returncode if proc.returncode is not None else -1
        is_error = returncode != 0

        if not is_error:
            if stderr_text.strip():
                output = f"{stdout_text.strip()}\n{stderr_text.strip()}" if stdout_text.strip() else stderr_text.strip()
            else:
                output = stdout_text.strip() or "(command completed with no output)"
        else:
            pieces = [f"Command failed with exit code {returncode}"]
            if stdout_text.strip():
                pieces.append(f"Stdout:\n{stdout_text.strip()}")
            if stderr_text.strip():
                pieces.append(f"Stderr:\n{stderr_text.strip()}")
            output = "\n".join(pieces)

        return ToolResult(call_id=call_id, output=output, is_error=is_error)


async def execute_shell(
    command: str,
    timeout: float = 120.0,
    cwd: str | Path | None = None,
    call_id: str = "",
) -> ToolResult:
    """Helper function to execute shell command asynchronously."""
    tool = ShellTool(workspace_root=cwd)
    return await tool.execute(command=command, timeout=timeout, cwd=cwd, call_id=call_id)
