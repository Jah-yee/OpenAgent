"""Dynamic system prompt generation for OpenAgent.

Builds structured, contextual system prompts incorporating operating system
details, workspace path, available tools, behavioral guidelines, and project-specific
instructions.

Copyright 2026 The OpenAgent Contributors
Licensed under the Apache License, Version 2.0.
"""

from __future__ import annotations

import platform
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from openagent.core.types import ToolSpec
from openagent.tools.base import Tool
from openagent.tools.registry import ToolRegistry

DEFAULT_ROLE_DESCRIPTION = (
    "You are OpenAgent, an autonomous, highly capable AI software engineering assistant. "
    "Your objective is to solve technical problems, write high-quality code, inspect project "
    "files, and execute tasks safely and effectively."
)

DEFAULT_GUIDELINES: list[str] = [
    "Analyze problems step-by-step and formulate a plan before making changes.",
    "Inspect existing code and files thoroughly before proposing or applying modifications.",
    "Make precise, minimal changes that solve the problem without unnecessary rewrites or collateral changes.",
    "Verify your work using tests, type checks, or shell execution whenever possible.",
    "Handle errors and edge cases defensively, reporting failures clearly and concisely.",
    "Adhere strictly to project conventions, formatting, and file organization.",
]


class PromptBuilder:
    """Dynamic system prompt generator with customizable templates."""

    def __init__(
        self,
        name: str = "OpenAgent",
        role_description: str | None = None,
        guidelines: Sequence[str] | None = None,
        template: str | None = None,
    ) -> None:
        self.name = name
        self.role_description = role_description or DEFAULT_ROLE_DESCRIPTION
        self.guidelines = list(guidelines) if guidelines is not None else list(DEFAULT_GUIDELINES)
        self.template = template

    def format_environment(self, workspace_root: str | Path | None = None) -> str:
        """Format the current execution environment information."""
        resolved_ws = Path(workspace_root).resolve() if workspace_root is not None else Path.cwd().resolve()
        os_info = f"{platform.system()} {platform.release()} ({platform.machine()})"
        now_str = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")

        lines = [
            "## Environment",
            f"- Operating System: {os_info}",
            f"- Workspace Directory: {resolved_ws}",
            f"- Current Date/Time: {now_str}",
        ]
        return "\n".join(lines)

    def format_tools(
        self,
        tools: Sequence[Tool | ToolSpec] | ToolRegistry | None = None,
    ) -> str:
        """Format available tools into structured documentation."""
        if tools is None:
            return ""

        specs: list[ToolSpec] = []
        if isinstance(tools, ToolRegistry):
            specs = tools.list_specs()
        else:
            for item in tools:
                if isinstance(item, Tool):
                    specs.append(item.spec)
                elif isinstance(item, ToolSpec):
                    specs.append(item)

        if not specs:
            return ""

        lines = ["## Available Tools", ""]
        for spec in specs:
            lines.append(f"### `{spec.name}`")
            lines.append(spec.description.strip())
            if spec.params:
                lines.append("Parameters:")
                for param in spec.params:
                    req = "required" if param.required else "optional"
                    desc = f": {param.description}" if param.description else ""
                    lines.append(f"  - `{param.name}` ({param.type}, {req}){desc}")
            lines.append("")

        return "\n".join(lines).strip()

    def format_guidelines(self, guidelines: Sequence[str] | None = None) -> str:
        """Format behavioral guidelines into markdown list."""
        active_guidelines = guidelines if guidelines is not None else self.guidelines
        if not active_guidelines:
            return ""

        lines = ["## Behavioral Guidelines"]
        for g in active_guidelines:
            lines.append(f"- {g}")
        return "\n".join(lines)

    def format_extra_instructions(
        self,
        extra_instructions: str | Sequence[str] | None = None,
    ) -> str:
        """Format extra project-specific instructions."""
        if not extra_instructions:
            return ""

        if isinstance(extra_instructions, str):
            text = extra_instructions.strip()
        else:
            text = "\n".join(line.strip() for line in extra_instructions if line.strip())

        if not text:
            return ""

        return f"## Extra Instructions\n{text}"

    def build_system_prompt(
        self,
        workspace_root: str | Path | None = None,
        tools: Sequence[Tool | ToolSpec] | ToolRegistry | None = None,
        extra_instructions: str | Sequence[str] | None = None,
    ) -> str:
        """Build the dynamic system prompt string.

        Args:
            workspace_root: Root directory of the target project/workspace.
            tools: Sequence of Tools/ToolSpecs or a ToolRegistry instance.
            extra_instructions: Custom instructions, project guidelines, or rules.

        Returns:
            The complete system prompt string.
        """
        resolved_ws = Path(workspace_root).resolve() if workspace_root is not None else Path.cwd().resolve()
        env_section = self.format_environment(workspace_root=resolved_ws)
        tools_section = self.format_tools(tools)
        guidelines_section = self.format_guidelines()
        extra_section = self.format_extra_instructions(extra_instructions)

        if self.template:
            # Substitute known variables
            replacements: dict[str, Any] = {
                "name": self.name,
                "role_description": self.role_description,
                "workspace": str(resolved_ws),
                "workspace_root": str(resolved_ws),
                "environment": env_section,
                "tools": tools_section,
                "guidelines": guidelines_section,
                "extra_instructions": extra_section,
            }
            res = self.template
            for key, val in replacements.items():
                res = res.replace(f"{{{key}}}", str(val))
            return res.strip()

        sections = [
            f"# {self.name}",
            self.role_description,
            env_section,
        ]

        if tools_section:
            sections.append(tools_section)

        if guidelines_section:
            sections.append(guidelines_section)

        if extra_section:
            sections.append(extra_section)

        return "\n\n".join(s.strip() for s in sections if s.strip())
